"""Build SUMO ``<routes>`` XML files from the canonical StreetLight data.

Public surface
--------------

* :class:`DemandSource` — enum of the five built-in demand strategies.
* :func:`build_routes` — turn one or more :class:`DemandSource` values
  into a single SUMO routes file. Returns the count of ``<flow>``
  entries written.

The builder is deliberately decoupled from the libsumo runtime: it only
needs the canonical parquets + the SUMO ``.net.xml`` (for the OSM →
edge id lookup). That makes it easy to test (no SUMO process needed)
and easy to reuse from notebooks.

Demand sources
--------------

``BRIDGE_OD_FULL``
    The 49 Bridge OD pairs spread across the 5 named StreetLight
    windows (Early AM, Peak AM, Mid-Day, Peak PM, Late PM). Same
    output as :mod:`scripts.11_export_sumo` produces today.

``BRIDGE_OD_PEAK_AM``
    Slice ``BRIDGE_OD_FULL`` to Peak AM only (``day_part_code == 2``,
    06:00 – 10:00). Useful when calibrating against AM-peak GEH.

``ZA_HOURLY``
    Synthesise per-segment flows from
    ``data/processed/derived/hourly_profiles.parquet`` (152 zones × 24
    hours of Visitor volume). Each zone with non-zero data contributes
    one ``<flow>`` per hour from the zone centroid edge to its nearest
    GWB-bound destination edge — a heuristic but defensible "drive
    onto Leonia, leave via the GWB" path that puts measured volumes
    on residential streets the Bridge OD doesn't touch.

``BRIDGE_OD_PLUS_ZA``
    Concatenation of ``BRIDGE_OD_FULL`` and ``ZA_HOURLY`` — the most
    complete demand picture currently possible from the canonical
    lake.

``PEAK_AM_SLICE``
    A single 07:00 – 08:00 hour slice of ``BRIDGE_OD_FULL``, scaled
    so the original per-hour vehicles-per-hour rate is preserved.
    Designed for fast iteration: a 1-hour run completes in tens of
    seconds even on a slow machine.

``BRIDGE_OD_WEEKDAY_24H`` / ``BRIDGE_OD_SUNDAY_24H``
    Full 24-hour Bridge OD demand, but pulled from the **weekday
    average** (mean of Mon–Fri, ``day_type_code`` ∈ {1,2,3,4,5})
    or **Sunday** (``day_type_code == 7``) cohorts in the
    StreetLight 2036064 export instead of the All-Days
    (``day_type_code == 0``) default. Used by the stakeholder
    one-pager to show side-by-side commuter-day vs. quiet-day
    patterns: weekday morning peak around Broad/Grand/Fort Lee Rd
    vs. dispersed midday flows on Sunday with very different
    cut-through pressure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import SUMO_BASE_DIR
from leonia_traffic.data.dataset_io import (
    CANONICAL_DIR,
    DERIVED_DIR,
    CanonicalFiles,
    DerivedFiles,
)
from leonia_traffic.sumo.net_lookup import (
    load_osm_to_sumo_lookup,
    load_sumo_edge_geometries,
    spatial_resolve_zones,
)

logger = logging.getLogger(__name__)


SUMO_DIR = SUMO_BASE_DIR
DEFAULT_NET_PATH = SUMO_DIR / "leonia.net.xml"


# ---------------------------------------------------------------------------
# Bridge OD windows (matches scripts/11_export_sumo.py)
# ---------------------------------------------------------------------------


# (day_part_code → (window label, begin_hour, end_hour))
#
# The 2036064_Destinations export uses 24 hourly day parts. Each
# code N (1…24) covers the hour ``[N-1, N)`` of the day. Code 0 is
# the All-Day total and is intentionally excluded — including it
# would double-count traffic.
#
# We synthesize a short label per hour ("H07" for 7am–8am etc.) so
# flow ids in the SUMO XML stay readable. Legacy window names
# (PeakAM, MidDay, …) are no longer the primary axis of demand;
# they survive as named *ranges* below for users who still want to
# slice by "peak AM" rather than by individual hours.
BRIDGE_OD_WINDOWS: dict[int, tuple[str, int, int]] = {
    code: (f"H{code - 1:02d}", code - 1, code) for code in range(1, 25)
}


# Named-range presets for callers that want to slice the hourly
# windows into the legacy 5 buckets (e.g. ``BRIDGE_OD_PEAK_AM``
# DemandSource still means "Peak AM = 6am-10am" but is now backed
# by codes [7, 8, 9, 10] rather than the single legacy code 2).
BRIDGE_OD_HOUR_RANGES: dict[str, list[int]] = {
    "EarlyAM": list(range(1, 7)),    # 0am–6am  (codes 1..6)
    "PeakAM":  list(range(7, 11)),   # 6am–10am (codes 7..10)
    "MidDay":  list(range(11, 16)),  # 10am–3pm (codes 11..15)
    "PeakPM":  list(range(16, 20)),  # 3pm–7pm  (codes 16..19)
    "LatePM":  list(range(20, 25)),  # 7pm–12am (codes 20..24)
}


# ---------------------------------------------------------------------------
# DemandSource enum
# ---------------------------------------------------------------------------


class DemandSource(str, Enum):
    """Built-in demand strategies for :func:`build_routes`."""

    BRIDGE_OD_FULL = "bridge_od_full"
    BRIDGE_OD_PEAK_AM = "bridge_od_peak_am"
    ZA_HOURLY = "za_hourly"
    BRIDGE_OD_PLUS_ZA = "bridge_od_plus_za"
    PEAK_AM_SLICE = "peak_am_slice"
    BRIDGE_OD_WEEKDAY_24H = "bridge_od_weekday_24h"
    BRIDGE_OD_SUNDAY_24H = "bridge_od_sunday_24h"


# ---------------------------------------------------------------------------
# Day-type cohorts (StreetLight 2036064 schema)
# ---------------------------------------------------------------------------


# StreetLight ``day_type_code`` values (see DAY_TYPE_CODES in
# leonia_traffic.data.bridge_od_loader): 0=All Days, 1=Mon, 2=Tue,
# 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun. Codes 1–5 are individual
# weekdays — when we want a representative "weekday" pattern we
# average across them per (origin, destination, day_part).
WEEKDAY_DAY_TYPE_CODES: tuple[int, ...] = (1, 2, 3, 4, 5)
SUNDAY_DAY_TYPE_CODE: int = 7
ALL_DAYS_DAY_TYPE_CODE: int = 0


@dataclass
class _Flow:
    """A single ``<flow>`` row in the output XML."""

    flow_id: str
    from_edge: str
    to_edge: str
    begin_s: int
    end_s: int
    veh_per_hour: float
    vtype: str = "passenger"

    def to_xml(self) -> str:
        return (
            f'  <flow id="{escape(self.flow_id)}" type="{self.vtype}" '
            f'from="{escape(self.from_edge)}" to="{escape(self.to_edge)}" '
            f'begin="{self.begin_s}" end="{self.end_s}" '
            f'vehsPerHour="{self.veh_per_hour:.2f}" '
            'departLane="best" departSpeed="max"/>\n'
        )


# ---------------------------------------------------------------------------
# Bridge OD demand
# ---------------------------------------------------------------------------


def _load_bridge_od_lookup(net_path: Path) -> dict[int, list[str]]:
    """Direct OSM-id lookup augmented with Bridge OD spatial fallback."""
    lookup = load_osm_to_sumo_lookup(net_path)
    bz_path = CANONICAL_DIR / CanonicalFiles.bridge_od_zones
    if not bz_path.exists():
        return lookup
    zones = gpd.read_parquet(bz_path)
    if zones.empty:
        return lookup
    if "osm_way_id" not in zones.columns:
        zones["osm_way_id"] = (
            zones["name"].str.extract(r"/ (\d+)").astype("Int64")
        )
    edges = load_sumo_edge_geometries(net_path)
    return spatial_resolve_zones(lookup, zones, edges, max_distance_m=300.0)


def _bridge_od_flows(
    osm_lookup: dict[int, list[str]],
    *,
    day_part_codes: Iterable[int] | None = None,
    day_type_codes: Iterable[int] | None = None,
    rate_scale: float = 1.0,
    time_offset_s: int = 0,
    label_suffix: str = "",
) -> list[_Flow]:
    """Pull rows from ``bridge_od.parquet`` and emit one ``_Flow`` each.

    Parameters
    ----------
    osm_lookup
        OSM-way → SUMO-edge mapping (Bridge OD-augmented).
    day_part_codes
        Subset of ``BRIDGE_OD_WINDOWS`` keys to emit. Defaults to all.
    day_type_codes
        Subset of ``day_type_code`` values to use. Defaults to
        ``(0,)`` (All Days). When multiple codes are given (e.g.
        ``(1,2,3,4,5)`` for Mon–Fri), per-(origin, destination,
        day_part) volumes are averaged across the codes — this is
        how we synthesise an "average weekday" demand from the
        five weekday cohorts in the StreetLight export.
    rate_scale
        Multiplier on each flow's ``vehsPerHour`` (used by
        ``PEAK_AM_SLICE`` to stretch the 4-hour rate over a 1-hour
        window).
    time_offset_s
        Optional shift applied to begin/end (used by ``PEAK_AM_SLICE``
        to centre on 07–08am).
    label_suffix
        Appended to each flow id, e.g. ``"slice"`` for the peak slice.
    """
    od_path = CANONICAL_DIR / CanonicalFiles.bridge_od
    if not od_path.exists():
        logger.warning("bridge_od.parquet not found at %s; skipping", od_path)
        return []
    od = pd.read_parquet(od_path)
    codes = (
        set(BRIDGE_OD_WINDOWS.keys())
        if day_part_codes is None
        else set(int(c) for c in day_part_codes)
    )
    dt_codes = (
        {ALL_DAYS_DAY_TYPE_CODE}
        if day_type_codes is None
        else set(int(c) for c in day_type_codes)
    )
    sub = od[
        (od["day_type_code"].isin(dt_codes))
        & (od["day_part_code"].isin(codes))
        & (od["origin_osm_way_id"].notna())
        & (od["destination_osm_way_id"].notna())
        & (od["od_volume"] > 0)
    ]
    # When multiple day_type_codes are selected (e.g. Mon–Fri),
    # collapse to a single representative row per (origin, dest,
    # day_part) by averaging od_volume. This avoids emitting 5
    # near-identical flows for every weekday cohort.
    if len(dt_codes) > 1 and not sub.empty:
        sub = (
            sub.groupby(
                ["origin_osm_way_id", "destination_osm_way_id",
                 "day_part_code"],
                as_index=False, dropna=False,
            )
            .agg(od_volume=("od_volume", "mean"))
        )
    flows: list[_Flow] = []
    skipped_origin = 0
    skipped_dest = 0
    for _, row in sub.iterrows():
        o = int(row["origin_osm_way_id"])
        d = int(row["destination_osm_way_id"])
        o_edges = osm_lookup.get(o, [])
        d_edges = osm_lookup.get(d, [])
        if not o_edges:
            skipped_origin += 1
            continue
        if not d_edges:
            skipped_dest += 1
            continue
        code = int(row["day_part_code"])
        label, hr_start, hr_end = BRIDGE_OD_WINDOWS[code]
        window_hours = max(hr_end - hr_start, 1)
        vph = float(row["od_volume"]) / window_hours * rate_scale
        if vph <= 0:
            continue
        begin = hr_start * 3600 + time_offset_s
        end = hr_end * 3600 + time_offset_s
        suffix = f"_{label_suffix}" if label_suffix else ""
        flows.append(_Flow(
            flow_id=f"od_{o}_to_{d}_{label}{suffix}",
            from_edge=o_edges[0],
            to_edge=d_edges[0],
            begin_s=begin,
            end_s=end,
            veh_per_hour=vph,
        ))
    if skipped_origin or skipped_dest:
        logger.info(
            "Bridge OD: skipped %d unmapped origins, %d unmapped destinations",
            skipped_origin, skipped_dest,
        )
    return flows


def _peak_am_slice_flows(osm_lookup: dict[int, list[str]]) -> list[_Flow]:
    """One-hour 07–08am slice for fast-iterating SUMO smoke tests.

    Produces total simulated time = 3600 s, flows compressed into a
    single hour beginning at 07:00:00. With the 24-hour daypart
    schema we can pull the **exact** 7am–8am hourly rate
    (``day_part_code == 8``) instead of averaging across the legacy
    4-hour Peak AM window — so the simulated volumes match the
    real 7–8am peak rather than the smoothed 6–10am mean.
    """
    base = _bridge_od_flows(
        osm_lookup,
        day_part_codes=[8],          # 7am–8am (hourly code 8)
        rate_scale=1.0,
        time_offset_s=0,
        label_suffix="slice",
    )
    out: list[_Flow] = []
    for f in base:
        f2 = _Flow(
            flow_id=f.flow_id,
            from_edge=f.from_edge,
            to_edge=f.to_edge,
            begin_s=7 * 3600,
            end_s=8 * 3600,
            veh_per_hour=f.veh_per_hour,
            vtype=f.vtype,
        )
        out.append(f2)
    return out


# ---------------------------------------------------------------------------
# ZA hourly demand
# ---------------------------------------------------------------------------


def _gwb_destination_edges(
    osm_lookup: dict[int, list[str]],
    edges_gdf: gpd.GeoDataFrame | None = None,
) -> list[str]:
    """Return the SUMO edge ids of the Bridge OD destination zones (GWB).

    The Bridge OD ``zones`` parquet stores destination zones as
    line/polygon geometries with ``zone_role == "destination"`` but
    *without* an ``osm_way_id`` column (only the gate centroid and
    polygon are available). To resolve them to SUMO edges we pick
    the nearest network edge to each gate centroid in a metric CRS.

    We use this set as the sink for ZA-hourly demand on the
    heuristic that Leonia residential cut-through is overwhelmingly
    heading toward the bridge approach.
    """
    bz_path = CANONICAL_DIR / CanonicalFiles.bridge_od_zones
    if not bz_path.exists():
        return []
    zones = gpd.read_parquet(bz_path)
    if zones.empty:
        return []

    if "zone_role" in zones.columns:
        dests = zones[zones["zone_role"] == "destination"]
        if dests.empty:
            dests = zones
    else:
        dests = zones

    # Legacy path: if the zones export carried osm_way_id, prefer it.
    if "osm_way_id" in dests.columns and dests["osm_way_id"].notna().any():
        edge_ids: list[str] = []
        for way in dests["osm_way_id"].dropna().astype(int).unique():
            for eid in osm_lookup.get(int(way), []):
                if eid not in edge_ids:
                    edge_ids.append(eid)
        if edge_ids:
            return edge_ids

    # Spatial fallback: snap each destination zone's centroid (or
    # gate centroid when present) to the nearest SUMO edge.
    if edges_gdf is None:
        # Late import to avoid a circular dep at module load.
        edges_gdf = load_sumo_edge_geometries(DEFAULT_NET_PATH)
    if edges_gdf.empty:
        return []
    try:
        edges_m = edges_gdf.to_crs(3857)
    except Exception:
        edges_m = edges_gdf

    edge_ids = []
    for _, row in dests.iterrows():
        target = None
        gate_lat = row.get("gate_lat") if "gate_lat" in dests.columns else None
        gate_lon = row.get("gate_lon") if "gate_lon" in dests.columns else None
        try:
            if pd.notna(gate_lat) and pd.notna(gate_lon):
                lat_f, lon_f = float(gate_lat), float(gate_lon)
                from shapely.geometry import Point as _Point
                target = _Point(lon_f, lat_f)
        except (ValueError, TypeError):
            target = None
        if target is None:
            geom = row.get("geometry")
            if geom is None or getattr(geom, "is_empty", True):
                continue
            target = geom.centroid
        try:
            target_m = (
                gpd.GeoSeries([target], crs=edges_gdf.crs)
                .to_crs(3857)
                .iloc[0]
            )
        except Exception:
            target_m = target
        dists = edges_m.geometry.distance(target_m)
        nearest = edges_m.loc[dists.idxmin(), "edge_id"]
        if nearest not in edge_ids:
            edge_ids.append(nearest)
    return edge_ids


def _nearest_destination_edge(
    origin_geom,
    destination_edges: list[str],
    edges_gdf: gpd.GeoDataFrame,
) -> str | None:
    """Pick the nearest GWB-bound edge to a ZA zone centroid."""
    if not destination_edges or origin_geom is None or origin_geom.is_empty:
        return None
    candidates = edges_gdf[edges_gdf["edge_id"].isin(destination_edges)]
    if candidates.empty:
        return None
    target = origin_geom.centroid
    # Reproject to a metric CRS for accurate distance.
    try:
        candidates_m = candidates.to_crs(3857)
        target_m = (
            gpd.GeoSeries([target], crs=edges_gdf.crs).to_crs(3857).iloc[0]
        )
    except Exception:
        candidates_m = candidates
        target_m = target
    dists = candidates_m.geometry.distance(target_m)
    return candidates.loc[dists.idxmin(), "edge_id"]


def _za_hourly_profile_for_day_types(
    day_type_codes: tuple[int, ...] | None,
) -> pd.DataFrame:
    """Return a wide ``zone_name × h00..h23`` profile for one day-type cohort.

    Defaults to the cached ``hourly_profiles.parquet`` (All-Days
    Visitor cohort) for backward-compat. When a non-default
    ``day_type_codes`` is passed (e.g. Mon-Thu for "weekday avg"
    or Sunday-only), rebuilds the profile on the fly from
    ``za_volume.parquet`` so it carries the correct cohort. The
    rebuild path mirrors
    :func:`leonia_traffic.analysis.cutthrough_streets.weekday_hourly_profile`
    exactly; the difference is only the ``day_types`` filter.

    Note: the ZA day-type schema has 7 codes (0=All Days, 1-4=Mon-Thu,
    5=Sat, 6=Sun) and *no Friday*. The Bridge OD schema has 8 codes
    (0=All Days, 1-5=Mon-Fri, 6=Sat, 7=Sun). Callers must translate
    between the two — see :data:`ZA_WEEKDAY_DAY_TYPE_CODES` and
    :data:`ZA_SUNDAY_DAY_TYPE_CODE`.
    """
    if not day_type_codes or set(day_type_codes) == {0}:
        # Use the prebuilt All-Days cache when possible (no need to
        # repeat the pivot at every demand build).
        hp_path = DERIVED_DIR / DerivedFiles.hourly_profiles
        if hp_path.exists():
            return pd.read_parquet(hp_path)

    # Rebuild from za_volume.parquet for the requested day-type cohort.
    za_path = CANONICAL_DIR / CanonicalFiles.za_volume
    if not za_path.exists():
        logger.warning(
            "za_volume.parquet not found at %s; cannot derive day-type-"
            "specific ZA hourly profile",
            za_path,
        )
        return pd.DataFrame()

    from leonia_traffic.analysis.cutthrough_streets import (
        weekday_hourly_profile,
    )
    za = pd.read_parquet(za_path)
    if za.empty:
        return pd.DataFrame()
    return weekday_hourly_profile(
        za, day_types=tuple(day_type_codes),
    )


# ZA day-type constants (mirror cutthrough_streets.py — different
# schema from the Bridge OD's WEEKDAY_DAY_TYPE_CODES, see
# ``_za_hourly_profile_for_day_types`` for the why).
ZA_WEEKDAY_DAY_TYPE_CODES: tuple[int, ...] = (1, 2, 3, 4)  # Mon-Thu
ZA_SUNDAY_DAY_TYPE_CODE: int = 6


# Default scaling for ZA-derived flows in 24h whole-borough demand
# sources. ZA Visitor counts are *measurements at every observation
# point along a corridor*, so summing them as new OD demand leads to
# severe double-counting (one trip = N segment crossings) AND
# saturates the network: at full scale the simulation stalls in
# mid-day gridlock with thousands of teleporting vehicles per hour.
# For the stakeholder visualization, we want residential streets to
# *light up* without overloading the network. Empirically:
#
# * 1.00 (raw) — gridlock; sim takes >2hr and never reaches steady state
# * 0.15 — still over capacity; teleport storms after hour ~12
# * 0.05 — sim runs cleanly; ~28k extra vehicles, every ZA segment
#   sees several vehicles per hour → visible coverage of every
#   tracked residential street
ZA_VISIBILITY_SCALE_DEFAULT: float = 0.05


def _za_hourly_flows(
    osm_lookup: dict[int, list[str]],
    net_path: Path,
    *,
    day_type_codes: tuple[int, ...] | None = None,
    label_suffix: str = "",
    scale: float = 1.0,
) -> list[_Flow]:
    """Synthesise per-segment hourly flows from the ZA Visitor cohort.

    Each ZA segment with non-zero hourly volume contributes one
    ``<flow>`` per hour with a non-zero ``h<HH>`` value. Origin = the
    SUMO edge id mapped from that segment's OSM way (with spatial
    fallback). Destination = the nearest GWB-bound edge.

    Parameters
    ----------
    osm_lookup
        OSM-way → SUMO-edge mapping.
    net_path
        Path to the SUMO ``.net.xml``.
    day_type_codes
        Subset of ZA ``day_type_code`` values to filter on
        (``None`` = All-Days cached profile). Use
        :data:`ZA_WEEKDAY_DAY_TYPE_CODES` for a Mon-Thu mean and
        ``(ZA_SUNDAY_DAY_TYPE_CODE,)`` for Sunday-only.
    label_suffix
        Appended to each flow id (e.g. ``"wkd"``) so flows from
        different day-type runs don't collide when concatenated.
    scale
        Multiplier applied to every ``vehsPerHour`` value. Pass
        :data:`ZA_VISIBILITY_SCALE_DEFAULT` (0.15) when these flows
        will be combined with Bridge OD in a 24h whole-borough demand
        — the raw ZA Visitor counts triple- or quadruple-count any
        trip that crosses multiple measurement zones, so emitting
        them un-scaled produces a network-wide gridlock that masks
        the very effect we want to visualize. Leave at 1.0 when
        running ZA-only or for analytical use of the raw Visitor
        counts (e.g. ``DemandSource.ZA_HOURLY``).
    """
    shapes_path = CANONICAL_DIR / CanonicalFiles.za_line_shapes
    if not shapes_path.exists():
        logger.warning("za_line_shapes.parquet not found; skipping ZA demand")
        return []

    hp = _za_hourly_profile_for_day_types(day_type_codes)
    shapes = gpd.read_parquet(shapes_path)
    if hp.empty or shapes.empty:
        return []

    # Augment the lookup with ZA spatial fallback, and grab edge
    # geometries once for nearest-destination selection.
    edges_gdf = load_sumo_edge_geometries(net_path)
    augmented_lookup = spatial_resolve_zones(
        osm_lookup, shapes[["osm_way_id", "geometry"]].copy(),
        edges_gdf, max_distance_m=300.0,
    )

    destination_edges = _gwb_destination_edges(augmented_lookup, edges_gdf)
    if not destination_edges:
        logger.warning("No GWB destination edges resolved; skipping ZA demand")
        return []

    shapes_idx = shapes.set_index("osm_way_id")

    flows: list[_Flow] = []
    skipped_no_origin = 0
    skipped_no_dest = 0
    for _, row in hp.iterrows():
        way = row.get("osm_way_id")
        if pd.isna(way):
            continue
        way_int = int(way)
        origin_edges = augmented_lookup.get(way_int, [])
        if not origin_edges:
            skipped_no_origin += 1
            continue
        if way_int not in shapes_idx.index:
            continue
        origin_geom = shapes_idx.loc[way_int, "geometry"]
        if isinstance(origin_geom, pd.Series):
            origin_geom = origin_geom.iloc[0]
        to_edge = _nearest_destination_edge(
            origin_geom, destination_edges, edges_gdf,
        )
        if not to_edge:
            skipped_no_dest += 1
            continue
        from_edge = origin_edges[0]
        if from_edge == to_edge:
            # Would route an origin to itself — skip.
            continue
        suffix = f"_{label_suffix}" if label_suffix else ""
        for hr in range(24):
            col = f"h{hr:02d}"
            val = row.get(col)
            if val is None or pd.isna(val) or float(val) <= 0:
                continue
            vph = float(val) * float(scale)
            if vph <= 0:
                continue
            flows.append(_Flow(
                flow_id=f"za_{way_int}_h{hr:02d}{suffix}",
                from_edge=from_edge,
                to_edge=to_edge,
                begin_s=hr * 3600,
                end_s=(hr + 1) * 3600,
                veh_per_hour=vph,
            ))
    if skipped_no_origin or skipped_no_dest:
        logger.info(
            "ZA hourly: skipped %d unmapped origins, %d zones with no "
            "reachable destination",
            skipped_no_origin, skipped_no_dest,
        )
    return flows


# ---------------------------------------------------------------------------
# build_routes — public entry point
# ---------------------------------------------------------------------------


def build_routes(
    sources: DemandSource | Iterable[DemandSource],
    out: Path,
    *,
    net_path: Path = DEFAULT_NET_PATH,
) -> int:
    """Write a SUMO ``<routes>`` XML for the requested demand source(s).

    Parameters
    ----------
    sources
        A single :class:`DemandSource` or any iterable of them. When
        multiple sources are supplied their flows are concatenated.
    out
        Destination path. Parent directory is created if missing.
    net_path
        Path to the SUMO ``.net.xml``. Used for OSM-id resolution.

    Returns
    -------
    int
        Number of ``<flow>`` entries written.
    """
    if isinstance(sources, (DemandSource, str)):
        source_list: list[DemandSource] = [DemandSource(sources)]
    else:
        source_list = [DemandSource(s) for s in sources]

    osm_lookup = _load_bridge_od_lookup(net_path)
    if not osm_lookup:
        logger.warning("Empty OSM→SUMO lookup; the routes file will likely "
                       "have zero flows.")

    all_flows: list[_Flow] = []
    for src in source_list:
        if src is DemandSource.BRIDGE_OD_FULL:
            all_flows.extend(_bridge_od_flows(osm_lookup))
        elif src is DemandSource.BRIDGE_OD_PEAK_AM:
            # Peak AM is 6am-10am → hourly codes 7..10 in the new
            # 24-window schema. Was a single code (2) in the legacy
            # 5-window schema; we keep the same wall-clock semantics.
            all_flows.extend(
                _bridge_od_flows(
                    osm_lookup,
                    day_part_codes=BRIDGE_OD_HOUR_RANGES["PeakAM"],
                )
            )
        elif src is DemandSource.ZA_HOURLY:
            all_flows.extend(_za_hourly_flows(osm_lookup, net_path))
        elif src is DemandSource.BRIDGE_OD_PLUS_ZA:
            all_flows.extend(_bridge_od_flows(osm_lookup))
            all_flows.extend(_za_hourly_flows(osm_lookup, net_path))
        elif src is DemandSource.PEAK_AM_SLICE:
            all_flows.extend(_peak_am_slice_flows(osm_lookup))
        elif src is DemandSource.BRIDGE_OD_WEEKDAY_24H:
            # Bridge OD covers gateway-to-gateway flows (the high-vph
            # arterials). Add ZA hourly flows for the same cohort
            # (Mon-Thu Visitor mean) so residential streets light up
            # too — without this, the 24h animation only paints
            # Broad / Grand / Fort Lee Rd and the rest of Leonia
            # is dark. ZA flows are scaled by ZA_VISIBILITY_SCALE
            # to avoid double-counting (a single trip is observed at
            # multiple ZA segments).
            all_flows.extend(
                _bridge_od_flows(
                    osm_lookup,
                    day_type_codes=WEEKDAY_DAY_TYPE_CODES,
                    label_suffix="wkd",
                )
            )
            all_flows.extend(
                _za_hourly_flows(
                    osm_lookup, net_path,
                    day_type_codes=ZA_WEEKDAY_DAY_TYPE_CODES,
                    label_suffix="wkd",
                    scale=ZA_VISIBILITY_SCALE_DEFAULT,
                )
            )
        elif src is DemandSource.BRIDGE_OD_SUNDAY_24H:
            all_flows.extend(
                _bridge_od_flows(
                    osm_lookup,
                    day_type_codes=(SUNDAY_DAY_TYPE_CODE,),
                    label_suffix="sun",
                )
            )
            all_flows.extend(
                _za_hourly_flows(
                    osm_lookup, net_path,
                    day_type_codes=(ZA_SUNDAY_DAY_TYPE_CODE,),
                    label_suffix="sun",
                    scale=ZA_VISIBILITY_SCALE_DEFAULT,
                )
            )

    # SUMO requires ``<flow>`` entries sorted ascending by
    # ``begin``; otherwise it silently drops every flow whose
    # ``begin`` is earlier than the most recent one it parsed
    # (you'll see "Route file should be sorted by departure time"
    # warnings and ``n_inserted`` collapses to a small handful).
    # The peak_am_slice path used a single begin-time so the issue
    # didn't surface, but the 24-hour weekday/sunday paths emit
    # 24 flows per OD pair grouped by pair — without this sort,
    # SUMO drops ~95% of vehicles.
    all_flows.sort(key=lambda f: (f.begin_s, f.end_s, f.flow_id))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write(
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '        xsi:noNamespaceSchemaLocation="'
            'http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        )
        fh.write(
            '  <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" '
            'length="5.0" maxSpeed="22.22" guiShape="passenger"/>\n'
        )
        for flow in all_flows:
            fh.write(flow.to_xml())
        fh.write('</routes>\n')

    return len(all_flows)


def default_routes_path(source: DemandSource) -> Path:
    """Conventional output path for a given demand source."""
    return SUMO_DIR / f"leonia.routes_{source.value}.xml"
