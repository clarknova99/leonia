"""Declarative mitigation scenarios applied on top of the baseline World.

Design
------
Scenarios mutate the OSM (nodes, links) tuple **before** it is loaded
into a UXsim World. This is the cleanest extension point because:

  - it avoids deep-copying a populated World (UXsim's internal state is
    not trivially copyable)
  - the same demand model is applied to both baseline and scenario
    Worlds, so any volume delta is attributable to the network change
  - one-way conversions and closures are naturally expressible as
    deletions/additions of links

Usage
-----
::

    from leonia_traffic.simulation.scenarios import (
        OneWayConversion, Closure, SpeedHumpCalming, apply_scenarios,
        run_scenario,
    )
    sc = OneWayConversion(osm_way_ids=[11586338], allowed_bearing_deg=180)
    result = run_scenario(
        baseline_build,
        [sc],
        name="broad_ave_oneway_southbound",
    )
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
from uxsim import World

from leonia_traffic.network.osm_builder import (
    OSMBuildConfig,
    build_or_load_network,
    make_world,
    parse_uxsim_link_name,
)
from leonia_traffic.simulation.calibration import (
    extract_simulated_flows,
    score_simulation,
)
from leonia_traffic.simulation.demand import apply_gateway_demand
from leonia_traffic.network.calibration_match import match_segments_to_links
from leonia_traffic.data.streetlight_loader import load_cached, restrict_to_study_area
from leonia_traffic.config import SIM_DEFAULTS
from uxsim.OSMImporter import OSMImporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario types
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """Base class. Subclasses override :meth:`apply`."""

    name: str = "scenario"

    def apply(self, nodes: list, links: list) -> tuple[list, list]:
        raise NotImplementedError


def _link_bearing_deg(node_xy: dict, link: list) -> float:
    """Compass bearing of a link, 0° = north, 90° = east."""
    a = node_xy.get(link[1])
    b = node_xy.get(link[2])
    if a is None or b is None:
        return float("nan")
    dx = b[0] - a[0]  # lon
    dy = b[1] - a[1]  # lat
    # atan2(east, north) gives bearing measured clockwise from north.
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    return bearing


@dataclass
class OneWayConversion(Scenario):
    """Convert a two-way OSM way to one-way.

    Keeps only the direction whose bearing is within ``tolerance_deg``
    of ``allowed_bearing_deg``. The "other-direction" link (if any) is
    removed.
    """

    osm_way_ids: list[int] = field(default_factory=list)
    allowed_bearing_deg: float = 0.0
    tolerance_deg: float = 90.0

    def apply(self, nodes: list, links: list) -> tuple[list, list]:
        node_xy = {n[0]: (n[1], n[2]) for n in nodes}
        ids = set(int(i) for i in self.osm_way_ids)
        if not ids:
            return nodes, links

        def keep(link):
            _, osmid, _ = parse_uxsim_link_name(link[0])
            if osmid not in ids:
                return True
            bearing = _link_bearing_deg(node_xy, link)
            if math.isnan(bearing):
                return True
            delta = abs(((bearing - self.allowed_bearing_deg + 180) % 360) - 180)
            return delta <= self.tolerance_deg

        new_links = [l for l in links if keep(l)]
        used = {l[1] for l in new_links} | {l[2] for l in new_links}
        new_nodes = [n for n in nodes if n[0] in used]
        return new_nodes, new_links


@dataclass
class SpeedHumpCalming(Scenario):
    """Reduce free-flow speed on listed OSM ways to mimic traffic calming."""

    osm_way_ids: list[int] = field(default_factory=list)
    free_flow_speed_factor: float = 0.5
    min_free_flow_speed_ms: float = 4.5  # ~10 mph

    def apply(self, nodes: list, links: list) -> tuple[list, list]:
        ids = set(int(i) for i in self.osm_way_ids)
        if not ids:
            return nodes, links
        new_links: list[list] = []
        for link in links:
            l = list(link)
            _, osmid, _ = parse_uxsim_link_name(l[0])
            if osmid in ids:
                l[4] = max(self.min_free_flow_speed_ms, l[4] * self.free_flow_speed_factor)
            new_links.append(l)
        return nodes, new_links


@dataclass
class Closure(Scenario):
    """Fully close listed OSM ways (delete all matching links)."""

    osm_way_ids: list[int] = field(default_factory=list)

    def apply(self, nodes: list, links: list) -> tuple[list, list]:
        ids = set(int(i) for i in self.osm_way_ids)
        if not ids:
            return nodes, links
        new_links = []
        for link in links:
            _, osmid, _ = parse_uxsim_link_name(link[0])
            if osmid in ids:
                continue
            new_links.append(link)
        used = {l[1] for l in new_links} | {l[2] for l in new_links}
        new_nodes = [n for n in nodes if n[0] in used]
        return new_nodes, new_links


@dataclass
class LaneReduction(Scenario):
    """Reduce ``lanes`` to ``target_lanes`` on listed OSM ways."""

    osm_way_ids: list[int] = field(default_factory=list)
    target_lanes: int = 1

    def apply(self, nodes: list, links: list) -> tuple[list, list]:
        ids = set(int(i) for i in self.osm_way_ids)
        new_links = []
        for link in links:
            l = list(link)
            _, osmid, _ = parse_uxsim_link_name(l[0])
            if osmid in ids:
                l[3] = max(1, min(l[3], self.target_lanes))
            new_links.append(l)
        return nodes, new_links


# ---------------------------------------------------------------------------
# Apply + run
# ---------------------------------------------------------------------------


def apply_scenarios(
    nodes: list, links: list, scenarios: Iterable[Scenario]
) -> tuple[list, list]:
    """Apply a sequence of scenarios in order, returning new (nodes, links)."""
    cur_nodes = [list(n) for n in nodes]
    cur_links = [list(l) for l in links]
    for sc in scenarios:
        cur_nodes, cur_links = sc.apply(cur_nodes, cur_links)
    return cur_nodes, cur_links


@dataclass
class ScenarioRunResult:
    name: str
    world: World
    matched: pd.DataFrame
    sim_flow: pd.Series
    score: object
    scoring_df: pd.DataFrame


def run_scenario(
    scenarios: Iterable[Scenario],
    *,
    name: str = "scenario",
    network_cfg: OSMBuildConfig | None = None,
    streetlight_source: str = "weekdays",
    duration_hours: float = 2.0,
    tmax: int = 2 * 3600,
    deltan: int = 20,
    daily_to_peak_factor: float = 0.10,
    gwb_share: float = 0.6,
    min_gateway_volume: float = 500.0,
    print_mode: int = 0,
) -> ScenarioRunResult:
    """Build a fresh World with the scenario applied, run simulation, score.

    Reuses the cached OSM network so all scenarios start from the same
    raw (nodes, links). The scenario list is applied to a deep copy of
    the lists before loading into the World.
    """
    base_nodes, base_links, cfg = build_or_load_network(network_cfg)
    new_nodes, new_links = apply_scenarios(base_nodes, base_links, scenarios)

    W = make_world(
        name=name, tmax=tmax, deltan=deltan, print_mode=print_mode,
    )
    OSMImporter.osm_network_to_World(
        W,
        new_nodes,
        new_links,
        default_jam_density=SIM_DEFAULTS.default_jam_density,
        coef_degree_to_meter=SIM_DEFAULTS.coef_degree_to_meter,
    )

    gdf = restrict_to_study_area(load_cached())
    matched = match_segments_to_links(gdf, new_links, source_label=streetlight_source)

    apply_gateway_demand(
        W,
        matched,
        duration_hours=duration_hours,
        gwb_share=gwb_share,
        min_volume=min_gateway_volume,
        daily_to_peak_factor=daily_to_peak_factor,
    )

    W.exec_simulation()

    sim_flow = extract_simulated_flows(W, t_start_s=0, t_end_s=W.TMAX)
    score, scoring_df = score_simulation(
        sim_flow,
        matched,
        observed_to_hourly_factor=daily_to_peak_factor / duration_hours,
    )
    return ScenarioRunResult(
        name=name,
        world=W,
        matched=matched,
        sim_flow=sim_flow,
        score=score,
        scoring_df=scoring_df,
    )


def run_scenario_v2(
    scenarios: Iterable[Scenario],
    *,
    name: str = "scenario_v2",
    network_cfg: OSMBuildConfig | None = None,
    streetlight_source: str = "weekdays",
    day_type_code: int = 1,
    day_part_code: int = 2,
    duration_hours: float = 4.0,
    tmax: int = 4 * 3600,
    deltan: int = 20,
    demand_scale: float = 1.0,
    jam_density_factor: float = 1.0,
    intersection_capacity_factor: float = 1.0,
    apply_speed_overrides: bool = True,
    include_za_streets_in_match: bool = False,
    za_day_type_code: int = 4,
    za_day_part_code: int = 0,
    print_mode: int = 0,
) -> ScenarioRunResult:
    """Pass-B equivalent of :func:`run_scenario`.

    Uses real bridge OD demand and congestion-derived link speed
    overrides on top of the scenario's mutated network.
    """
    from leonia_traffic.analysis.congestion import link_speed_overrides
    from leonia_traffic.config import DATA_NETWORK_DIR
    from leonia_traffic.data.bridge_od_loader import (
        load_bridge_od, load_bridge_zone_shapes,
    )
    from leonia_traffic.data.congestion_loader import (
        load_congestion, load_congestion_zones,
    )
    from leonia_traffic.network.calibration_match import (
        build_osm_to_uxsim_index, index_uxsim_links, spatial_resolve_osm_way_ids,
    )
    from leonia_traffic.network.osm_builder import apply_congestion_overrides
    from leonia_traffic.simulation.demand import apply_bridge_od_demand

    base_nodes, base_links, cfg = build_or_load_network(network_cfg)
    new_nodes, new_links = apply_scenarios(base_nodes, base_links, scenarios)

    W = make_world(
        name=name, tmax=tmax, deltan=deltan, print_mode=print_mode,
    )
    OSMImporter.osm_network_to_World(
        W,
        new_nodes,
        new_links,
        default_jam_density=SIM_DEFAULTS.default_jam_density,
        coef_degree_to_meter=SIM_DEFAULTS.coef_degree_to_meter,
    )

    if jam_density_factor != 1.0:
        for link in W.LINKS:
            link.kappa = link.kappa * jam_density_factor

    gdf = restrict_to_study_area(load_cached())
    matched = match_segments_to_links(gdf, new_links, source_label=streetlight_source)
    if not matched.empty:
        matched = matched.copy()
        matched["source"] = "scanner"

    if include_za_streets_in_match:
        from leonia_traffic.data.za_streets_loader import load_za_main
        from leonia_traffic.network.calibration_match import (
            match_za_streets_to_links,
        )
        za_matched = match_za_streets_to_links(
            W=None,
            za_main_df=load_za_main(),
            uxsim_links=new_links,
            line_gdf=None,
            day_type_code=za_day_type_code,
            day_part_code=za_day_part_code,
        )
        if not za_matched.empty:
            za_aligned = za_matched.reindex(columns=list(matched.columns), fill_value=pd.NA)
            za_aligned = za_aligned.loc[~za_aligned.index.isin(matched.index)]
            matched = pd.concat([matched, za_aligned], axis=0)

    refs = index_uxsim_links(new_links)
    osm_to_uxsim = build_osm_to_uxsim_index(refs)

    od_zone_to_link = spatial_resolve_osm_way_ids(
        load_bridge_zone_shapes(kind="line"),
        W, name_col="name", max_distance_m=200.0,
    )

    if apply_speed_overrides:
        cdf = load_congestion()
        if not cdf.empty:
            overrides = link_speed_overrides(
                cdf, day_type_code=1, day_part_code=9,
            )
            congestion_zone_to_link = spatial_resolve_osm_way_ids(
                load_congestion_zones(), W, name_col="name", max_distance_m=100.0,
            )
            apply_congestion_overrides(
                W, overrides,
                osm_to_uxsim=osm_to_uxsim,
                zone_to_link_name=congestion_zone_to_link,
            )

    if intersection_capacity_factor != 1.0:
        for link in W.LINKS:
            cap = getattr(link, "capacity_out", None)
            if cap is None or not math.isfinite(cap):
                continue
            try:
                link.capacity_out = float(cap) * intersection_capacity_factor
            except Exception:
                pass

    od_df = load_bridge_od()
    apply_bridge_od_demand(
        W, od_df, osm_to_uxsim,
        day_type_code=day_type_code,
        day_part_code=day_part_code,
        duration_hours=duration_hours,
        demand_scale=demand_scale,
        zone_to_link_name=od_zone_to_link,
    )

    W.exec_simulation()

    sim_flow = extract_simulated_flows(W, t_start_s=0, t_end_s=W.TMAX)
    # For V2 we score against the same Street Scanner observations the V1
    # path uses, since the bridge OD only covers gate-to-bridge trips and
    # would not yield a per-link comparable count.
    score, scoring_df = score_simulation(
        sim_flow, matched,
        observed_to_hourly_factor=0.10 / duration_hours,
    )
    return ScenarioRunResult(
        name=name,
        world=W,
        matched=matched,
        sim_flow=sim_flow,
        score=score,
        scoring_df=scoring_df,
    )


def compare_scenarios(
    baseline: ScenarioRunResult,
    scenario: ScenarioRunResult,
    *,
    min_observed: float = 50.0,
) -> pd.DataFrame:
    """Per-link comparison of two ScenarioRunResults.

    Joins on UXsim link name. Returns a DataFrame with baseline and
    scenario simulated flows, the delta, and a flag for residential
    spillover (positive delta on a low-baseline-flow link).
    """
    base = baseline.sim_flow.rename("sim_flow_baseline_vph")
    scen = scenario.sim_flow.rename("sim_flow_scenario_vph")
    df = pd.concat([base, scen], axis=1).fillna(0.0)
    df["delta_vph"] = df["sim_flow_scenario_vph"] - df["sim_flow_baseline_vph"]
    df["abs_delta_vph"] = df["delta_vph"].abs()
    df["spillover_flag"] = (
        (df["sim_flow_baseline_vph"] < 200) & (df["delta_vph"] > 50)
    )
    obs = baseline.matched[["observed_volume", "osm_way_id"]].rename(
        columns={"observed_volume": "observed_volume_daily"}
    )
    df = df.join(obs, how="left")
    return df.sort_values("abs_delta_vph", ascending=False)


__all__ = [
    "Closure",
    "LaneReduction",
    "OneWayConversion",
    "Scenario",
    "ScenarioRunResult",
    "SpeedHumpCalming",
    "apply_scenarios",
    "compare_scenarios",
    "run_scenario",
    "run_scenario_v2",
]
