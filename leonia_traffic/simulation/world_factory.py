"""High-level helpers that produce a ready-to-simulate UXsim ``World``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from uxsim import World

from leonia_traffic.config import SIM_DEFAULTS
from leonia_traffic.data.streetlight_loader import load_cached, restrict_to_study_area
from leonia_traffic.network.calibration_match import match_segments_to_links
from leonia_traffic.network.osm_builder import (
    OSMBuildConfig,
    build_or_load_network,
    world_from_osm,
)
from leonia_traffic.simulation.demand import (
    BridgeODDemandSummary,
    DemandSummary,
    apply_bridge_od_demand,
    apply_gateway_demand,
)

logger = logging.getLogger(__name__)


@dataclass
class BaselineBuild:
    world: World
    matched: pd.DataFrame
    demand_summary: DemandSummary


def build_baseline(
    *,
    name: str = "leonia_baseline",
    tmax: int = SIM_DEFAULTS.tmax_seconds,
    deltan: int = SIM_DEFAULTS.deltan,
    network_cfg: OSMBuildConfig | None = None,
    streetlight_source: str = "weekdays",
    duration_hours: float = 4.0,
    daily_to_peak_factor: float = 0.10,
    gwb_share: float = 0.6,
    min_gateway_volume: float = 100.0,
    print_mode: int = 1,
) -> BaselineBuild:
    """Build a fully-loaded ``World`` ready for ``exec_simulation``.

    Steps:
      1. Build/load the OSM network and create a fresh World.
      2. Load StreetLight observed volumes for the chosen source.
      3. Match StreetLight segments to UXsim links via OSM way ID.
      4. Apply gateway-based placeholder demand.
    """
    W, nodes, links = world_from_osm(
        cfg=network_cfg, name=name, tmax=tmax, deltan=deltan
    )
    if print_mode <= 0:
        W.print_mode = 0

    gdf = restrict_to_study_area(load_cached())
    matched = match_segments_to_links(gdf, links, source_label=streetlight_source)

    summary = apply_gateway_demand(
        W,
        matched,
        duration_hours=duration_hours,
        gwb_share=gwb_share,
        min_volume=min_gateway_volume,
        daily_to_peak_factor=daily_to_peak_factor,
    )
    logger.info(
        "Baseline demand: %d demands, total %.0f veh/h",
        summary.n_demands_added,
        summary.total_flow_vph,
    )
    return BaselineBuild(world=W, matched=matched, demand_summary=summary)


@dataclass
class CalibratedBaselineBuild:
    world: World
    matched: pd.DataFrame
    demand_summary: BridgeODDemandSummary
    override_counts: dict


def build_calibrated_baseline(
    *,
    name: str = "leonia_calibrated",
    tmax: int = SIM_DEFAULTS.tmax_seconds,
    deltan: int = SIM_DEFAULTS.deltan,
    network_cfg: OSMBuildConfig | None = None,
    streetlight_source: str = "weekdays",
    day_type_code: int = 1,
    day_part_code: int = 2,
    duration_hours: float = 4.0,
    demand_scale: float = 1.0,
    jam_density_factor: float = 1.0,
    apply_speed_overrides: bool = True,
    include_za_streets_in_match: bool = False,
    za_day_type_code: int = 4,
    za_day_part_code: int = 0,
    print_mode: int = 1,
) -> CalibratedBaselineBuild:
    """Build a Pass-B baseline that uses real OD + observed link speeds.

    Steps:
      1. Build / load the OSM network and create a fresh World.
      2. Load StreetLight Street Scanner observations for calibration scoring.
      3. Match scanner segments to UXsim links.
      4. Load congestion data, derive per-link speed overrides, and apply
         them to the World.
      5. Load the bridge OD matrix, resolve origin/destination OSM way
         IDs to UXsim nodes, and inject demand via
         :func:`apply_bridge_od_demand`.
    """
    from leonia_traffic.analysis.congestion import link_speed_overrides
    from leonia_traffic.config import DATA_NETWORK_DIR
    from leonia_traffic.data.bridge_od_loader import (
        load_bridge_od,
        load_bridge_zone_shapes,
    )
    from leonia_traffic.data.congestion_loader import (
        load_congestion,
        load_congestion_zones,
    )
    from leonia_traffic.network.calibration_match import (
        build_osm_to_uxsim_index,
        index_uxsim_links,
        spatial_resolve_osm_way_ids,
    )
    from leonia_traffic.network.osm_builder import apply_congestion_overrides

    W, nodes, links = world_from_osm(
        cfg=network_cfg, name=name, tmax=tmax, deltan=deltan
    )
    if print_mode <= 0:
        W.print_mode = 0

    # Optional global jam-density tweak.
    if jam_density_factor != 1.0:
        for link in W.LINKS:
            link.kappa = link.kappa * jam_density_factor

    gdf = restrict_to_study_area(load_cached())
    matched = match_segments_to_links(gdf, links, source_label=streetlight_source)
    # Tag the Street Scanner matches so per-source logging in
    # `score_simulation_by_source` can distinguish them from the
    # ZA-streets matches added below.
    if not matched.empty:
        matched = matched.copy()
        matched["source"] = "scanner"

    # Optionally union in the Pass-C ZA-streets observations so
    # calibration scoring also covers Leonia's residential tertiaries.
    if include_za_streets_in_match:
        from leonia_traffic.data.za_streets_loader import (
            load_za_line_shapes,
            load_za_main,
        )
        from leonia_traffic.network.calibration_match import (
            match_za_streets_to_links,
        )

        za_main = load_za_main()
        line_gdf = load_za_line_shapes()
        za_matched = match_za_streets_to_links(
            W=None,  # spatial fallback not needed for in-network OSM ids
            za_main_df=za_main,
            uxsim_links=links,
            line_gdf=None,
            day_type_code=za_day_type_code,
            day_part_code=za_day_part_code,
        )
        if not za_matched.empty:
            logger.info(
                "ZA-streets matched %d UXsim links (Pass C.3 residential).",
                len(za_matched),
            )
            # Outer-concat. Where a UXsim link appears in both sources
            # (rare — residential tertiaries usually only exist in the
            # ZA export), the Street Scanner row wins to preserve the
            # historical Pass-B behavior; comment out the dedup line to
            # average the two instead.
            keep_cols = list(matched.columns)
            za_aligned = za_matched.reindex(columns=keep_cols, fill_value=pd.NA)
            # Drop Pass-C rows whose link is already in the Scanner-matched frame
            za_aligned = za_aligned.loc[~za_aligned.index.isin(matched.index)]
            matched = pd.concat([matched, za_aligned], axis=0)

    refs = index_uxsim_links(links)
    osm_to_uxsim = build_osm_to_uxsim_index(refs)

    # Spatial fallback for OD origin/destination zones whose OSM way IDs
    # are stale relative to the current OSM extract.
    od_zone_to_link = spatial_resolve_osm_way_ids(
        load_bridge_zone_shapes(kind="line"),
        W,
        name_col="name",
        max_distance_m=200.0,
    )
    logger.info("OD spatial fallback resolved %d zones -> UXsim links", len(od_zone_to_link))

    override_counts: dict = {}
    if apply_speed_overrides:
        cdf = load_congestion()
        if not cdf.empty:
            overrides = link_speed_overrides(
                cdf,
                day_type_code=1,
                day_part_code=9,  # Peak AM aggregate in the congestion schema
                cache_path=DATA_NETWORK_DIR / "speed_overrides_weekday_peak_am.parquet",
            )
            congestion_zone_to_link = spatial_resolve_osm_way_ids(
                load_congestion_zones(),
                W,
                name_col="name",
                max_distance_m=100.0,
            )
            override_counts = apply_congestion_overrides(
                W, overrides,
                osm_to_uxsim=osm_to_uxsim,
                zone_to_link_name=congestion_zone_to_link,
                cache_path=DATA_NETWORK_DIR / "speed_overrides_applied.parquet",
            )

    od_df = load_bridge_od()
    demand_summary = apply_bridge_od_demand(
        W, od_df, osm_to_uxsim,
        day_type_code=day_type_code,
        day_part_code=day_part_code,
        duration_hours=duration_hours,
        demand_scale=demand_scale,
        zone_to_link_name=od_zone_to_link,
    )

    logger.info(
        "Calibrated baseline: demands=%d origin_ways=%d dest_ways=%d skipped=%d "
        "link_speed_overrides_applied=%s",
        demand_summary.n_demands_added,
        demand_summary.n_origin_ways_resolved,
        demand_summary.n_destination_ways_resolved,
        demand_summary.n_pairs_skipped_no_match,
        override_counts.get("n_links_changed", 0),
    )
    return CalibratedBaselineBuild(
        world=W, matched=matched,
        demand_summary=demand_summary,
        override_counts=override_counts,
    )


__all__ = [
    "BaselineBuild",
    "CalibratedBaselineBuild",
    "build_baseline",
    "build_calibrated_baseline",
]
