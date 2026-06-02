"""Build the static-map JSON artefacts for the stakeholder webapp.

The webapp's "Static Maps" tab renders non-animated deck.gl maps that
summarise average conditions for the day, driven by a unified control
bar. This module emits the precomputed artefacts that tab consumes:

* ``_static/traffic_weekday.json`` / ``_static/traffic_sunday.json`` —
  per-edge average vehicles/hour for five windows (All Day, Peak AM,
  Peak PM, Off-peak early, Off-peak late), derived from the StreetLight
  hourly overlays (``_overlays/streetlight_<demand>.json``) joined to
  SUMO edge geometry.
* ``_static/crashes.json`` — borough-filtered NJDOT crash points with
  per-point year/severity, plus the list of years available for the
  Year filter, over a grey street skeleton.

The peak windows differ by day type because Leonia's weekday traffic
is bimodal (AM + PM commute peaks) while Sunday is a midday plateau
(see the hourly profiles in ``streetlight_<demand>.json``). The chosen
windows are data-derived and written into the catalog so they are easy
to adjust later without touching the front-end.

Outputs are served by the existing ``/precache/{path}`` route and are
referenced from ``catalog.json``'s ``static`` block (see
:func:`static_catalog_block`).

Usage
-----

::

    venv/bin/python webapp/scripts/build_static_maps.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from leonia_traffic.config import SUMO_BASE_DIR, SUMO_PRECACHE_DIR
from leonia_traffic.sumo.net_lookup import (
    load_sumo_edge_geometries,
    load_sumo_edge_junctions,
    load_sumo_junction_coords,
)
from leonia_traffic.sumo.visualizations import (
    _filter_crash_rows_to_borough,
    _is_state_system_street,
    _round_coords,
    load_crash_points_if_available,
    load_crash_segments_if_available,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SUMO_DIR = SUMO_BASE_DIR
NET_PATH = SUMO_DIR / "leonia.net.xml"
PRECACHE_DIR = SUMO_PRECACHE_DIR
OVERLAY_DIR = PRECACHE_DIR / "_overlays"
STATIC_DIR = PRECACHE_DIR / "_static"

# Demand profiles that have a StreetLight overlay. "weekend" intentionally
# maps to Sunday only — the StreetLight ZA export has no Saturday data
# (Friday rolls into All-Days), so "Sunday" is the only weekend cohort we
# can represent today.
DAY_TYPES: tuple[str, ...] = ("weekday", "sunday")

# Day-part windows as ``[start_hour, end_hour)`` (24h clock). Weekday is
# bimodal (AM + PM commute peaks) with a midday lull and a quiet evening;
# Sunday is a flat midday plateau bracketed by a quiet morning and evening.
# Catalog-driven so the front-end labels each option with its hours without
# hard-coding them. Each window key here becomes a selectable day part and an
# averaged value on every edge (plus the implicit "all_day").
PEAK_WINDOWS: dict[str, dict[str, list[int]]] = {
    "weekday": {
        "peak_am": [7, 10],
        "peak_pm": [15, 18],
        "off_peak_early": [10, 15],  # midday lull between the commute peaks
        "off_peak_late": [19, 23],   # evening, after the PM peak fades
        "overnight": [0, 6],         # 12am–6am quiet overnight window
    },
    "sunday": {
        "peak_am": [10, 13],
        "peak_pm": [13, 16],
        "off_peak_early": [6, 10],   # early morning before the midday plateau
        "off_peak_late": [16, 21],   # late afternoon into evening
        "overnight": [0, 6],         # 12am–6am quiet overnight window
    },
}

# Windows whose values drive the colour ramp's vmax. Restricting this to the
# busier windows keeps the scale stable so the lower-volume off-peak windows
# correctly read as lighter/greener instead of depressing vmax (which would
# tint every window redder).
VMAX_WINDOWS: tuple[str, ...] = ("peak_am", "peak_pm")

MAX_SKELETON = 4000
ZOOM = 14

# How far outside the borough polygon an edge may sit and still count as
# "in Leonia". netconvert trims lane shapes back from junctions and the
# OSM borough polygon is only accurate to a few tens of metres, so a
# strict (0 m) test drops legitimate border streets (Bergen Boulevard,
# the Leonia ends of Broad Avenue / Fort Lee Road). 50 m recovers that
# border ring while still excluding same-named streets in neighbouring
# towns (Englewood's Grand Avenue, far Park Avenue) which sit 800 m+ out.
BORDER_BUFFER_M = 50.0

# The George Washington Bridge approach sits just outside Leonia in Fort
# Lee but carries the through-traffic the study is about, so it's
# explicitly included even though it's out of the borough. The region is
# derived at build time from the actual approach facilities (Route
# 46/US-1-9-46, Bergen Boulevard and the ramps feeding them) rather than
# a hand-drawn box, which keeps it tight enough to exclude the
# neighbouring residential streets (Park/Grand/Glenwood Avenue) that a
# loose bounding box would sweep back in.
APPROACH_NAMES = {"US 1;US 9;US 46", "Bergen Boulevard"}
RAMP_NAMES = {"motorway_link", "trunk_link"}
APPROACH_SEED_MAX_M = 600.0   # only the approach segments near Leonia
RAMP_NEAR_HULL_M = 120.0      # ramps hugging the approach corridor
APPROACH_REGION_BUFFER_M = 60.0


# ---------------------------------------------------------------------------
# Traffic static builder
# ---------------------------------------------------------------------------


# Street names that are part of the GWB approach corridor / limited-access
# facilities, NOT Leonia local roads. The static traffic map still draws
# them for context, but the "top roads" table is Leonia-only.
# ``_is_state_system_street`` already flags the turnpike / I-95 / express
# lanes / motorway-link / GWB names; this set adds the surface approach
# corridor and the OSM highway-type placeholders that the borough's 50 m
# border buffer grazes (e.g. US-1-9-46 and Bergen Boulevard clipping the
# southern tip). Matched case-insensitively. NJ-93 (= Grand Avenue) and
# Broad Avenue / Fort Lee Road are intentionally NOT here — their
# in-borough segments are exactly the Leonia main roads to keep.
_NON_LOCAL_TABLE_NAMES = {
    "us 1;us 9;us 46", "us 1;us 9", "us 9;us 1",
    "bergen boulevard", "mackay highway", "bruce reynolds boulevard",
    "bridge plaza north", "south bridge plaza",
    "george washington bridge plaza", "fletcher avenue",
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
}


def _is_leonia_local_road(name: object) -> bool:
    """True for a real Leonia local-road name (for the top-roads table).

    Excludes state-system facilities (via ``_is_state_system_street``),
    the GWB approach corridor (``_NON_LOCAL_TABLE_NAMES``), and the OSM
    highway-type placeholders / edge-id fallbacks that have no street name.
    """
    if not isinstance(name, str) or not name.strip():
        return False
    if _is_state_system_street(name):
        return False
    return name.strip().lower() not in _NON_LOCAL_TABLE_NAMES


def _window_mean(hourly: list[float], lo: int, hi: int) -> float:
    """Mean vehicles/hour across the half-open hour window ``[lo, hi)``."""
    vals = [hourly[h] for h in range(lo, hi) if 0 <= h < len(hourly)]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _hourly_24(hourly: list[float]) -> list[int]:
    """Coerce a measured profile to a fixed 24-slot integer vph array.

    The "Hourly (24 hrs)" day-part animates these directly, one frame per
    hour from midnight to midnight, so they must be exactly 24 long.
    """
    out = [int(round(float(hourly[h]))) if h < len(hourly) else 0
           for h in range(24)]
    return out


def _vals_to_hourly(vals: dict, windows: dict) -> list[int]:
    """Reconstruct a 24-slot hourly profile from window means.

    Used for the gap-filled collinear segments (which only carry window
    means, not a measured curve) so they still pulse in step with their
    parent street during the hourly playback instead of sitting static.
    Each named window paints its hours; hours no window covers fall back
    to the all-day average.
    """
    base = int(vals.get("all_day", 0))
    hourly = [base] * 24
    for wkey, (lo, hi) in windows.items():
        v = vals.get(wkey)
        if v is None:
            continue
        for h in range(max(0, int(lo)), min(24, int(hi))):
            hourly[h] = int(v)
    return hourly


def snapped_coords_by_edge(geo: pd.DataFrame) -> dict[str, list[list[float]]]:
    """Per-edge WGS84 coords with endpoints extended to their junctions.

    netconvert trims each edge's lane shape back from the junction box,
    which leaves a visible grey gap at every intersection when edges are
    drawn directly (a path stops a few metres short of each cross
    street). Prepending the ``from`` junction centre and appending the
    ``to`` junction centre closes those gaps so the network reads as a
    continuous grid. Falls back to the raw shape when junction data is
    unavailable.
    """
    junctions = load_sumo_junction_coords(NET_PATH)
    edge_junctions = load_sumo_edge_junctions(NET_PATH)
    out: dict[str, list[list[float]]] = {}
    for eid, srow in geo.iterrows():
        geom = srow.geometry
        if geom is None or geom.is_empty:
            continue
        cs = list(geom.coords)
        ft = edge_junctions.get(str(eid))
        if ft:
            f, t = ft
            if f in junctions:
                cs = [junctions[f]] + cs
            if t in junctions:
                cs = cs + [junctions[t]]
        rounded = _round_coords(cs)
        dedup: list[list[float]] = []
        for pt in rounded:
            if not dedup or dedup[-1] != pt:
                dedup.append(pt)
        out[str(eid)] = dedup
    return out


def _heading_away(coords: list[list[float]], at_start: bool) -> tuple[float, float] | None:
    """Unit vector pointing away from one end of a polyline (lon/lat).

    Longitude is scaled by ``cos(lat)`` so the angle is metric-accurate
    at Leonia's latitude. ``at_start`` picks the from-end; otherwise the
    to-end.
    """
    if len(coords) < 2:
        return None
    if at_start:
        (x0, y0), (x1, y1) = coords[0], coords[1]
    else:
        (x0, y0), (x1, y1) = coords[-1], coords[-2]
    dx = (x1 - x0) * math.cos(math.radians(y0))
    dy = y1 - y0
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n else None


def _angle_deg(u: tuple[float, float], v: tuple[float, float]) -> float:
    d = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.degrees(math.acos(d))


# Minimum angle (deg) between two edges' away-vectors at a shared junction
# for them to count as "the road continues straight through" (180° = a
# perfectly straight continuation). 150° tolerates gentle curves while
# rejecting actual turns onto a cross street.
STRAIGHT_THROUGH_DEG = 150.0


def fill_unnamed_gaps(
    colored: dict[str, dict],
    coords_by_edge: dict[str, list[list[float]]],
    name_of: dict[str, str],
    edge_junctions: dict[str, tuple[str, str]],
    in_borough: set[str],
) -> dict[str, dict]:
    """Colour unnamed in-borough edges that continue a measured street.

    Some short OSM ways in the middle of a street (e.g. Broad Avenue
    between Harrison Street and Beechwood Place) carry no ``name`` tag,
    so the StreetLight overlay — which matches on street name — never
    assigns them a volume and they render as grey gaps. Here we
    propagate a coloured edge's volume across connected *unnamed* edges
    whenever the road continues roughly straight through the shared
    junction, iterating so a multi-segment gap fills from both ends.
    Bearing continuity keeps the fill on the corridor instead of turning
    onto a named cross street. Returns ``{edge_id: vals}`` for the newly
    filled edges (street name inherited from the source).
    """
    def is_unnamed(eid: str) -> bool:
        n = name_of.get(eid)
        return n is None or (isinstance(n, float) and math.isnan(n)) or str(n).strip() == ""

    junction_edges: dict[str, list[str]] = defaultdict(list)
    for eid in in_borough:
        ft = edge_junctions.get(eid)
        if ft:
            junction_edges[ft[0]].append(eid)
            junction_edges[ft[1]].append(eid)

    candidates = [
        eid for eid in in_borough
        if eid not in colored and is_unnamed(eid)
        and eid in edge_junctions and len(coords_by_edge.get(eid, [])) >= 2
    ]

    value: dict[str, dict] = dict(colored)
    filled: dict[str, dict] = {}
    changed = True
    iterations = 0
    while changed and iterations < 12:
        changed = False
        iterations += 1
        for E in candidates:
            if E in filled:
                continue
            fE, tE = edge_junctions[E]
            cE = coords_by_edge[E]
            best: dict | None = None
            for j, at_start in ((fE, True), (tE, False)):
                e_away = _heading_away(cE, at_start)
                if e_away is None:
                    continue
                for C in junction_edges.get(j, []):
                    if C == E or C not in value:
                        continue
                    cC = coords_by_edge.get(C)
                    ftC = edge_junctions.get(C)
                    if not cC or not ftC or len(cC) < 2:
                        continue
                    c_away = _heading_away(cC, at_start=(ftC[0] == j))
                    if c_away is None:
                        continue
                    if _angle_deg(e_away, c_away) >= STRAIGHT_THROUGH_DEG:
                        best = value[C]
                        break
                if best is not None:
                    break
            if best is not None:
                filled[E] = best
                value[E] = best
                changed = True
    return filled


def in_borough_edge_ids(geo: pd.DataFrame) -> set[str] | None:
    """Set of SUMO edge ids whose geometry is within Leonia.

    The SUMO network extends past the borough to keep through-traffic
    realistic, so several streets share a name with a neighbour
    (Englewood's Grand Avenue, Teaneck's Glenwood Avenue, etc.). The
    StreetLight overlay maps ``street_name -> edge_id`` without a
    borough filter, so without this guard the static traffic map would
    colour same-named segments outside Leonia.

    Membership is "within ``BORDER_BUFFER_M`` of the borough polygon",
    measured in a metric CRS. A strict (0 m) test drops legitimate
    border streets because netconvert trims lane shapes back from
    junctions and the OSM polygon is only good to a few tens of metres;
    the buffer is still far tighter than the 800 m+ separation to the
    same-named streets in neighbouring towns. Returns ``None`` if the
    polygon can't be loaded (caller then leaves edges unfiltered rather
    than dropping everything).

    ``geo`` is expected to be indexed by ``edge_id``.
    """
    try:
        import geopandas as gpd
        from leonia_traffic.config import load_leonia_polygon
    except Exception as exc:
        logger.warning("Borough filter imports unavailable (%s); skipping.", exc)
        return None
    try:
        borough = load_leonia_polygon()
    except Exception as exc:
        logger.warning("Could not load Leonia borough polygon (%s); skipping.", exc)
        return None
    if geo.empty:
        return None
    try:
        metric_crs = "EPSG:32118"  # NAD83 / New Jersey
        borough_m = (
            gpd.GeoSeries([borough], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
        )
        geo_m = gpd.GeoSeries(
            geo.geometry.values, index=geo.index, crs="EPSG:4326"
        ).to_crs(metric_crs)
        dist = geo_m.distance(borough_m)
    except Exception as exc:
        logger.warning("Could not measure borough distance (%s); skipping.", exc)
        return None

    return {str(eid) for eid in dist.index[dist <= BORDER_BUFFER_M]}


def bridge_approach_edge_ids(geo: pd.DataFrame) -> set[str]:
    """SUMO edge ids in the Fort Lee GWB-approach corridor (out of borough).

    The bridge approach (Route 46 / US-1-9-46, Bergen Boulevard and the
    ramps feeding the George Washington Bridge) is just outside Leonia
    but carries the through-traffic the project studies, so the static
    traffic map should highlight it. We build a tight include region
    from the approach facilities themselves — the convex hull of the
    near-borough US-46 / Bergen Boulevard edges plus the ramps hugging
    them, buffered ~60 m — so it captures the corridor without sweeping
    in the surrounding residential streets a bounding box would. Returns
    an empty set if the inputs can't be loaded (degrade to borough-only).

    ``geo`` is expected to be indexed by ``edge_id``.
    """
    try:
        import geopandas as gpd
        from shapely.ops import unary_union

        from leonia_traffic.config import load_leonia_polygon
        from leonia_traffic.sumo.net_lookup import load_meta_lookup
    except Exception as exc:
        logger.warning("Approach-region imports unavailable (%s); skipping.", exc)
        return set()
    if geo.empty:
        return set()
    try:
        borough = load_leonia_polygon()
        metric_crs = "EPSG:32118"
        borough_m = (
            gpd.GeoSeries([borough], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
        )
        geo_m = gpd.GeoSeries(
            geo.geometry.values, index=geo.index.astype(str), crs="EPSG:4326"
        ).to_crs(metric_crs)
        dist = geo_m.distance(borough_m)

        meta = load_meta_lookup(SUMO_DIR / "leonia.edgedata.meta.csv")
        if meta.empty:
            return set()
        meta = meta.copy()
        meta["sumo_edge_id"] = meta["sumo_edge_id"].astype(str)
        name_of = dict(zip(meta["sumo_edge_id"], meta["street_name"]))

        seeds = [
            e for e in geo_m.index
            if name_of.get(e) in APPROACH_NAMES and dist.loc[e] <= APPROACH_SEED_MAX_M
        ]
        if not seeds:
            return set()
        hull = unary_union([geo_m.loc[e] for e in seeds]).convex_hull
        ramps = [
            e for e in geo_m.index
            if name_of.get(e) in RAMP_NAMES
            and dist.loc[e] <= APPROACH_SEED_MAX_M
            and geo_m.loc[e].distance(hull) <= RAMP_NEAR_HULL_M
        ]
        region = unary_union(
            [geo_m.loc[e] for e in seeds] + [geo_m.loc[e] for e in ramps]
        ).convex_hull.buffer(APPROACH_REGION_BUFFER_M)
        return {e for e in geo_m.index if geo_m.loc[e].intersects(region)}
    except Exception as exc:
        logger.warning("Could not build bridge-approach region (%s); skipping.", exc)
        return set()


def build_traffic_static(
    demand: str,
    geo: pd.DataFrame,
    in_borough: set[str] | None = None,
    coords_by_edge: dict[str, list[list[float]]] | None = None,
    vmax_ids: set[str] | None = None,
    name_of: dict[str, str] | None = None,
    edge_junctions: dict[str, tuple[str, str]] | None = None,
    leonia_ids: set[str] | None = None,
) -> dict:
    """Build the static traffic payload for one day type.

    Reads ``_overlays/streetlight_<demand>.json`` (per-edge 24h vph),
    averages each edge over the All-Day / Peak-AM / Peak-PM windows, and
    joins to SUMO edge geometry. Edges without geometry are skipped;
    every other edge in the network becomes grey skeleton for context.
    """
    overlay_path = OVERLAY_DIR / f"streetlight_{demand}.json"
    if not overlay_path.is_file():
        raise FileNotFoundError(
            f"StreetLight overlay missing: {overlay_path}. Run "
            "webapp/scripts/build_streetlight_overlay.py first."
        )
    overlay = json.loads(overlay_path.read_text())
    by_edge = overlay.get("by_edge", {})
    windows = PEAK_WINDOWS[demand]

    edges_out: list[dict] = []
    # Two separate colour-scale pools: Leonia local roads vs. the
    # surrounding highways / GWB approach. A volume that saturates a local
    # street (≈800 vph) is light traffic on the turnpike, so colouring both
    # off one vmax paints every highway red. Each class gets its own vmax
    # so the green→red gradient is meaningful within its own road class.
    local_values: list[float] = []
    highway_values: list[float] = []
    # Separate pools for the hourly-playback colour scale: a single peak
    # hour runs higher than the 4-hour Peak-AM/PM window means, so reusing
    # the static vmax would clip every busy hour to red. These hold each
    # edge's peak single-hour vph instead.
    local_hourly_peaks: list[float] = []
    highway_hourly_peaks: list[float] = []
    covered: set[str] = set()
    for eid, rec in by_edge.items():
        if eid not in geo.index:
            continue
        # Same-named streets in neighbouring towns share a name but sit
        # outside the borough; don't colour them (they stay grey skeleton).
        if in_borough is not None and eid not in in_borough:
            continue
        geom = geo.loc[eid, "geometry"]
        if isinstance(geom, pd.Series):
            geom = geom.iloc[0]
        if geom is None or geom.is_empty:
            continue
        hourly = rec.get("hourly_vph") or []
        if not hourly:
            continue
        all_day = float(sum(hourly) / len(hourly))
        vals = {"all_day": int(round(all_day))}
        for wkey, (lo, hi) in windows.items():
            vals[wkey] = int(round(_window_mean(hourly, lo, hi)))
        covered.add(eid)
        nm = rec.get("street") or str(eid)
        # Leonia-local flag: strictly in-borough AND a local-road name
        # (drives the Leonia-only top-roads table AND which colour scale
        # the edge uses; the map draws all edges either way).
        is_leonia = (
            (leonia_ids is None or str(eid) in leonia_ids)
            and _is_leonia_local_road(nm)
        )
        # Only the busy windows feed each vmax (see VMAX_WINDOWS).
        pool = local_values if is_leonia else highway_values
        pool.append(all_day)
        pool.extend(float(vals[w]) for w in VMAX_WINDOWS if w in vals)
        hourly_int = _hourly_24(hourly)
        peak_pool = local_hourly_peaks if is_leonia else highway_hourly_peaks
        peak_pool.append(float(max(hourly_int)) if hourly_int else 0.0)
        coords = (
            coords_by_edge.get(str(eid)) if coords_by_edge is not None else None
        ) or _round_coords(list(geom.coords))
        edges_out.append({
            "id": str(eid),
            "name": nm,
            "coords": coords,
            "vals": vals,
            "hourly": hourly_int,
            "in_leonia": is_leonia,
        })

    def _q95(values: list[float]) -> float:
        pos = [v for v in values if v > 0]
        if not pos:
            return 50.0
        return max(50.0, round(float(np.quantile(pos, 0.95))))

    vmax = _q95(local_values)
    # Highways scale to their own 95th percentile, floored at the local
    # vmax so a quiet-data day never makes them scale lower than streets.
    vmax_highway = max(vmax, _q95(highway_values)) if highway_values else vmax

    # Hourly-playback scales: 95th percentile of per-edge peak-hour vph
    # (floored at the static vmax so the animation never reads cooler than
    # the matching static window).
    vmax_hourly = max(vmax, _q95(local_hourly_peaks))
    vmax_highway_hourly = (
        max(vmax_highway, _q95(highway_hourly_peaks))
        if highway_hourly_peaks else vmax_hourly
    )

    # Fill collinear gaps: unnamed mid-street segments OSM never tagged
    # (e.g. Broad Avenue through the Harrison/Beechwood block) get the
    # neighbouring measured volume so the street reads continuously.
    if (
        in_borough is not None and coords_by_edge is not None
        and name_of is not None and edge_junctions is not None
    ):
        colored = {
            e["id"]: {"vals": e["vals"], "name": e["name"]}
            for e in edges_out
        }
        filled = fill_unnamed_gaps(
            colored, coords_by_edge, name_of, edge_junctions, in_borough,
        )
        for eid, payload in filled.items():
            covered.add(eid)
            nm = payload.get("name") or str(eid)
            edges_out.append({
                "id": str(eid),
                "name": nm,
                "coords": coords_by_edge[eid],
                "vals": dict(payload["vals"]),
                "hourly": _vals_to_hourly(payload["vals"], windows),
                "in_leonia": (
                    (leonia_ids is None or str(eid) in leonia_ids)
                    and _is_leonia_local_road(nm)
                ),
            })
        if filled:
            logger.info("%s: filled %d unnamed gap segments.", demand, len(filled))

    skeleton: list[list[list[float]]] = []
    for eid, srow in geo.iterrows():
        if eid in covered:
            continue
        geom = srow.geometry
        if geom is None or geom.is_empty:
            continue
        coords = (
            coords_by_edge.get(str(eid)) if coords_by_edge is not None else None
        ) or _round_coords(list(geom.coords))
        skeleton.append(coords)
        if len(skeleton) >= MAX_SKELETON:
            break

    all_pts = [pt for e in edges_out for pt in e["coords"]]
    if all_pts:
        center = [
            round(sum(p[0] for p in all_pts) / len(all_pts), 5),
            round(sum(p[1] for p in all_pts) / len(all_pts), 5),
        ]
    else:
        center = [-73.99, 40.86]

    return {
        "meta": {
            "title": f"Leonia measured traffic · {demand}",
            "demand": demand,
            "metric": "avg_vph",
            "vmax_vph": int(vmax),
            "vmax_highway_vph": int(vmax_highway),
            "vmax_vph_hourly": int(vmax_hourly),
            "vmax_highway_vph_hourly": int(vmax_highway_hourly),
            "center": center,
            "zoom": ZOOM,
            "n_edges": len(edges_out),
            "peak_windows": windows,
        },
        "skeleton": skeleton,
        "edges": edges_out,
    }


# ---------------------------------------------------------------------------
# Crash static builder
# ---------------------------------------------------------------------------


def build_crash_static(
    geo: pd.DataFrame,
    coords_by_edge: dict[str, list[list[float]]] | None = None,
) -> dict | None:
    """Build the static crash payload (borough-filtered points + years).

    Returns ``None`` if the crash parquet hasn't been built yet (run
    ``scripts/14_build_crash_overlay.py`` first).
    """
    crashes = load_crash_points_if_available()
    if crashes is None or crashes.empty:
        logger.warning(
            "No crash parquet available; skipping crash static map. Run "
            "scripts/14_build_crash_overlay.py to populate it."
        )
        return None
    segments = load_crash_segments_if_available()

    df = crashes.copy()
    if "geocoded_lat" in df.columns and "geocoded_lon" in df.columns:
        df["lat"] = df["geocoded_lat"]
        df["lon"] = df["geocoded_lon"]
    elif "latitude" in df.columns and "longitude" in df.columns:
        df["lat"] = df["latitude"]
        df["lon"] = df["longitude"]
    else:
        logger.warning("Crash parquet has no coordinate columns; skipping.")
        return None

    df = _filter_crash_rows_to_borough(
        df, drop_state_system=True, crash_segments=segments,
    )
    if df.empty:
        logger.warning("All crashes filtered out of borough; skipping.")
        return None

    # Canonical road name per crash from the OSM-snapped way. NJDOT's
    # on-road text is noisy (state-route numbers, county codes, compound
    # "A / B" labels, case drift), which fragments one physical street
    # into several rows. The OSM ``street_name`` for the geocoded way
    # collapses them onto a single road so the webapp's top-roads table
    # groups cleanly; rows without an OSM snap fall back to ``None`` and
    # the front-end parses the on-road from the label instead.
    df = df.copy()
    way_to_name: dict = {}
    if (
        segments is not None and not segments.empty
        and "osm_way_id" in segments.columns
        and "street_name" in segments.columns
    ):
        way_to_name = (
            segments[["osm_way_id", "street_name"]]
            .dropna(subset=["osm_way_id"])
            .drop_duplicates("osm_way_id")
            .set_index("osm_way_id")["street_name"]
            .to_dict()
        )
    if way_to_name and "geocoded_osm_way_id" in df.columns:
        df["_road"] = pd.to_numeric(
            df["geocoded_osm_way_id"], errors="coerce",
        ).map(way_to_name)
    else:
        df["_road"] = None

    points: list[dict] = []
    years: set[int] = set()
    for _, r in df.iterrows():
        sev = r.get("severity_code")
        sev = sev.strip().upper() if isinstance(sev, str) and sev else "O"
        yr = r.get("year")
        try:
            yr = int(yr) if pd.notna(yr) else None
        except (TypeError, ValueError):
            yr = None
        if yr is not None:
            years.add(yr)
        date = r.get("crash_date")
        try:
            date_str = (
                pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else ""
            )
        except Exception:
            date_str = ""
        loc = r.get("crash_location") or ""
        cross = r.get("cross_street") or ""
        label = (loc + (" \u00d7 " + cross if cross else "")).strip() or "(unknown)"
        epdo = r.get("epdo")
        try:
            epdo = float(epdo) if pd.notna(epdo) else None
        except (TypeError, ValueError):
            epdo = None
        road = r.get("_road")
        road = road.strip() if isinstance(road, str) and road.strip() else None
        points.append({
            "lat": round(float(r["lat"]), 6),
            "lon": round(float(r["lon"]), 6),
            "year": yr,
            "severity": sev,
            "severity_label": (
                r.get("severity_label") if isinstance(r.get("severity_label"), str)
                else sev
            ),
            "epdo": epdo,
            "ped": bool(r.get("ped_involved", False)),
            "label": label,
            "road": road,
            "date": date_str,
        })

    skeleton: list[list[list[float]]] = []
    for _eid, srow in geo.iterrows():
        geom = srow.geometry
        if geom is None or geom.is_empty:
            continue
        coords = (
            coords_by_edge.get(str(_eid)) if coords_by_edge is not None else None
        ) or _round_coords(list(geom.coords))
        skeleton.append(coords)
        if len(skeleton) >= MAX_SKELETON:
            break

    if points:
        center = [
            round(sum(p["lon"] for p in points) / len(points), 5),
            round(sum(p["lat"] for p in points) / len(points), 5),
        ]
    else:
        center = [-73.99, 40.86]

    return {
        "meta": {
            "title": "Leonia crashes (NJDOT)",
            "center": center,
            "zoom": ZOOM,
            "n_points": len(points),
        },
        "years": sorted(years),
        "skeleton": skeleton,
        "points": points,
    }


# ---------------------------------------------------------------------------
# Catalog block
# ---------------------------------------------------------------------------


def static_catalog_block(precache_dir: Path = PRECACHE_DIR) -> dict:
    """Return the ``static`` block for catalog.json based on what's on disk."""
    static_dir = precache_dir / "_static"

    def _rel(name: str) -> str | None:
        return f"_static/{name}" if (static_dir / name).is_file() else None

    crash_years: list[int] = []
    crashes_path = static_dir / "crashes.json"
    if crashes_path.is_file():
        try:
            crash_years = json.loads(crashes_path.read_text()).get("years", [])
        except Exception:
            crash_years = []

    return {
        "traffic": {dt: _rel(f"traffic_{dt}.json") for dt in DAY_TYPES},
        "crash": _rel("crashes.json"),
        "crash_years": crash_years,
        "peak_windows": PEAK_WINDOWS,
    }


def refresh_catalog_static_block(precache_dir: Path = PRECACHE_DIR) -> bool:
    """Patch the ``static`` block of an existing ``catalog.json`` in place.

    ``build_precache.py`` owns full catalog generation, but rebuilding only
    the static maps (overlays + ``_static/``) leaves the catalog's ``static``
    block — including ``peak_windows`` the front-end labels day parts from —
    stale. This refreshes just that block (everything else is preserved) so a
    standalone ``build_static_maps`` run keeps the catalog consistent. Returns
    ``True`` if the catalog was updated.
    """
    catalog_path = precache_dir / "catalog.json"
    if not catalog_path.is_file():
        logger.info("No catalog.json at %s; skipping static-block refresh.",
                    catalog_path)
        return False
    try:
        catalog = json.loads(catalog_path.read_text())
    except Exception as exc:
        logger.warning("Could not read catalog.json (%s); skipping refresh.", exc)
        return False
    catalog["static"] = static_catalog_block(precache_dir)
    catalog_path.write_text(json.dumps(catalog, separators=(",", ":")))
    logger.info("Refreshed catalog.json static block (paths + peak_windows).")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_static_maps(out_dir: Path = STATIC_DIR) -> dict[str, Path]:
    """Build all static-map artefacts. Returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    geo = load_sumo_edge_geometries(NET_PATH)
    if geo.empty:
        raise FileNotFoundError(
            f"No SUMO edge geometry from {NET_PATH}; run scripts/11_export_sumo.py."
        )
    geo = geo.set_index("edge_id")
    borough_only = in_borough_edge_ids(geo)
    include = borough_only
    if borough_only is not None:
        approach = bridge_approach_edge_ids(geo)
        if approach:
            include = borough_only | approach
            logger.info("Bridge-approach region adds %d edges (GWB corridor).",
                        len(approach - borough_only))
        logger.info("Static traffic filter: %d of %d edges (Leonia + approach).",
                    len(include), len(geo))
    coords_by_edge = snapped_coords_by_edge(geo)

    from leonia_traffic.sumo.net_lookup import load_meta_lookup
    edge_junctions = load_sumo_edge_junctions(NET_PATH)
    _meta = load_meta_lookup(SUMO_DIR / "leonia.edgedata.meta.csv")
    name_of: dict[str, str] = {}
    if not _meta.empty:
        name_of = dict(
            zip(_meta["sumo_edge_id"].astype(str), _meta["street_name"])
        )

    written: dict[str, Path] = {}
    for demand in DAY_TYPES:
        try:
            payload = build_traffic_static(
                demand, geo, include, coords_by_edge, vmax_ids=borough_only,
                name_of=name_of, edge_junctions=edge_junctions,
                leonia_ids=borough_only,
            )
        except FileNotFoundError as exc:
            logger.warning("traffic %s: %s", demand, exc)
            continue
        path = out_dir / f"traffic_{demand}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        written[f"traffic_{demand}"] = path
        logger.info(
            "Wrote %s — %d edges (vmax %d vph).",
            path, payload["meta"]["n_edges"], payload["meta"]["vmax_vph"],
        )

    crash = build_crash_static(geo, coords_by_edge)
    if crash is not None:
        path = out_dir / "crashes.json"
        path.write_text(json.dumps(crash, separators=(",", ":")))
        written["crashes"] = path
        logger.info(
            "Wrote %s — %d crash points across years %s.",
            path, crash["meta"]["n_points"],
            f"{crash['years'][0]}-{crash['years'][-1]}" if crash["years"] else "(none)",
        )

    # Keep the catalog's static block (paths + peak_windows) in sync so the
    # front-end picks up the off-peak windows without a full precache rebuild.
    if out_dir == STATIC_DIR:
        refresh_catalog_static_block(PRECACHE_DIR)

    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out-dir", type=Path, default=STATIC_DIR,
        help=f"Output directory (default: {STATIC_DIR}).",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    build_static_maps(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
