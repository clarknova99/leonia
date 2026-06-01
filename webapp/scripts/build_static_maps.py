"""Build the static-map JSON artefacts for the stakeholder webapp.

The webapp's "Static Maps" tab renders non-animated deck.gl maps that
summarise average conditions for the day, driven by a unified control
bar. This module emits the precomputed artefacts that tab consumes:

* ``_static/traffic_weekday.json`` / ``_static/traffic_sunday.json`` —
  per-edge average vehicles/hour for three windows (All Day, Peak AM,
  Peak PM), derived from the StreetLight hourly overlays
  (``_overlays/streetlight_<demand>.json``) joined to SUMO edge
  geometry.
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

from leonia_traffic.config import DATA_PROCESSED_DIR
from leonia_traffic.sumo.net_lookup import (
    load_sumo_edge_geometries,
    load_sumo_edge_junctions,
    load_sumo_junction_coords,
)
from leonia_traffic.sumo.visualizations import (
    _filter_crash_rows_to_borough,
    _round_coords,
    load_crash_points_if_available,
    load_crash_segments_if_available,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SUMO_DIR = DATA_PROCESSED_DIR / "sumo"
NET_PATH = SUMO_DIR / "leonia.net.xml"
PRECACHE_DIR = SUMO_DIR / "runs_precache"
OVERLAY_DIR = PRECACHE_DIR / "_overlays"
STATIC_DIR = PRECACHE_DIR / "_static"

# Demand profiles that have a StreetLight overlay. "weekend" intentionally
# maps to Sunday only — the StreetLight ZA export has no Saturday data
# (Friday rolls into All-Days), so "Sunday" is the only weekend cohort we
# can represent today.
DAY_TYPES: tuple[str, ...] = ("weekday", "sunday")

# Data-derived peak windows as ``[start_hour, end_hour)`` (24h clock).
# Weekday is bimodal; Sunday is a flat midday plateau split into a
# late-morning ("AM") and afternoon ("PM") side. Catalog-driven so the
# front-end labels these as Peak AM / Peak PM without hard-coding hours.
PEAK_WINDOWS: dict[str, dict[str, list[int]]] = {
    "weekday": {"peak_am": [7, 10], "peak_pm": [15, 18]},
    "sunday": {"peak_am": [10, 13], "peak_pm": [13, 16]},
}

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


def _window_mean(hourly: list[float], lo: int, hi: int) -> float:
    """Mean vehicles/hour across the half-open hour window ``[lo, hi)``."""
    vals = [hourly[h] for h in range(lo, hi) if 0 <= h < len(hourly)]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


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
    am_lo, am_hi = windows["peak_am"]
    pm_lo, pm_hi = windows["peak_pm"]

    edges_out: list[dict] = []
    all_values: list[float] = []
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
        peak_am = _window_mean(hourly, am_lo, am_hi)
        peak_pm = _window_mean(hourly, pm_lo, pm_hi)
        covered.add(eid)
        # Scale colours off Leonia's own streets so the much higher-volume
        # GWB approach saturates at max-red instead of washing out the
        # in-town gradient. Falls back to all edges when no subset given.
        if vmax_ids is None or str(eid) in vmax_ids:
            all_values.extend([all_day, peak_am, peak_pm])
        coords = (
            coords_by_edge.get(str(eid)) if coords_by_edge is not None else None
        ) or _round_coords(list(geom.coords))
        edges_out.append({
            "id": str(eid),
            "name": rec.get("street") or str(eid),
            "coords": coords,
            "vals": {
                "all_day": int(round(all_day)),
                "peak_am": int(round(peak_am)),
                "peak_pm": int(round(peak_pm)),
            },
        })

    positive = [v for v in all_values if v > 0]
    vmax = (
        max(50.0, round(float(np.quantile(positive, 0.95)))) if positive else 50.0
    )

    # Fill collinear gaps: unnamed mid-street segments OSM never tagged
    # (e.g. Broad Avenue through the Harrison/Beechwood block) get the
    # neighbouring measured volume so the street reads continuously.
    if (
        in_borough is not None and coords_by_edge is not None
        and name_of is not None and edge_junctions is not None
    ):
        colored = {
            e["id"]: {
                "all_day": e["vals"]["all_day"],
                "peak_am": e["vals"]["peak_am"],
                "peak_pm": e["vals"]["peak_pm"],
                "name": e["name"],
            }
            for e in edges_out
        }
        filled = fill_unnamed_gaps(
            colored, coords_by_edge, name_of, edge_junctions, in_borough,
        )
        for eid, vals in filled.items():
            covered.add(eid)
            edges_out.append({
                "id": str(eid),
                "name": vals.get("name") or str(eid),
                "coords": coords_by_edge[eid],
                "vals": {
                    "all_day": int(vals["all_day"]),
                    "peak_am": int(vals["peak_am"]),
                    "peak_pm": int(vals["peak_pm"]),
                },
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
