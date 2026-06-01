"""Borough-of-Leonia jurisdictional filtering.

Leonia's borough council can only act on streets that are (a) located
inside the Borough of Leonia and (b) under municipal jurisdiction.
State and federal facilities (NJ Turnpike, GWB Plaza, US 1/9/46, NJ 4,
motorway ramps) crossing the borough are governed by NJDOT, the Port
Authority, or the federal government — so even if they appear in the
StreetLight congestion data, they cannot be the target of a borough
recommendation.

This module provides the spatial and name-based filtering used by the
evidence report and the recommendation engine to enforce that scope.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from leonia_traffic.config import (
    COUNTY_STATE_ARTERIALS,
    NON_BOROUGH_ROAD_CLASSES,
    NON_BOROUGH_ROAD_NAMES,
    load_leonia_polygon,
)

logger = logging.getLogger(__name__)


def _matches_non_borough_name(name: object) -> bool:
    """True if ``name`` mentions a state/federal facility."""
    if not isinstance(name, str) or not name:
        return False
    for fragment in NON_BOROUGH_ROAD_NAMES:
        if fragment.lower() in name.lower():
            return True
    return False


def is_county_state_arterial(name: object) -> bool:
    """True if ``name`` is one of Broad Ave / Grand Ave / Fort Lee Rd.

    These are Bergen-County-owned arterials passing through Leonia. The
    borough cannot modify them, but they ARE the desired channels for
    through-traffic — the recommendation engine downgrades direct
    interventions on these to "monitor / petition" status and explicitly
    encourages routing local cut-through ONTO them.
    """
    if not isinstance(name, str) or not name:
        return False
    low = name.lower()
    for fragment in COUNTY_STATE_ARTERIALS:
        if fragment.lower() in low:
            return True
    return False


def _matches_non_borough_class(road_class: object) -> bool:
    if not isinstance(road_class, str) or not road_class:
        return False
    return road_class in NON_BOROUGH_ROAD_CLASSES


def is_under_borough_jurisdiction(
    osm_name: object,
    road_class: object = None,
) -> bool:
    """Name + class filter: is this segment plausibly municipal?

    Returns False for:
    * State/federal facilities (NJ Turnpike, GWB, US routes, NJ 4 …)
    * Motorway / on-off-ramp road classes
    * Bergen County arterials (Broad Ave, Grand Ave, Fort Lee Rd /
      Main Street) — the borough cannot act on these.
    """
    if _matches_non_borough_name(osm_name):
        return False
    if _matches_non_borough_class(road_class):
        return False
    if is_county_state_arterial(osm_name):
        return False
    return True


def filter_segments_to_leonia(
    summary_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame | None,
    *,
    borough_polygon: BaseGeometry | None = None,
    boundary_buffer_deg: float = 0.0,
) -> pd.DataFrame:
    """Return rows of ``summary_df`` whose segments lie inside Leonia and
    are under borough jurisdiction.

    Parameters
    ----------
    summary_df:
        Per-segment table (e.g. ``summarize_link_reliability`` or
        ``delay_hotspot_ranking``). Must contain ``zone_name`` or
        ``osm_way_id`` plus ``osm_name`` and (optionally) ``road_class``.
    zones_gdf:
        GeoDataFrame from ``load_congestion_zones`` providing the
        segment geometries. If ``None`` the filter falls back to
        name/class filtering only.
    borough_polygon:
        Shapely polygon defining the borough. If ``None`` the canonical
        Leonia boundary from ``config.load_leonia_polygon`` is used.
    boundary_buffer_deg:
        Optional buffer (in degrees ≈ 111km) to include segments that
        clip the border. Default 0 = strict.
    """
    if summary_df is None or summary_df.empty:
        return summary_df

    out = summary_df.copy()

    name_col = "osm_name" if "osm_name" in out.columns else None
    class_col = "road_class" if "road_class" in out.columns else None

    if name_col is not None:
        keep_name = out[name_col].apply(_matches_non_borough_name)
        out = out[~keep_name]
        arterial_name = out[name_col].apply(is_county_state_arterial)
        out = out[~arterial_name]
    if class_col is not None:
        keep_class = out[class_col].apply(_matches_non_borough_class)
        out = out[~keep_class]

    if out.empty:
        return out

    if zones_gdf is None or zones_gdf.empty:
        return out

    if borough_polygon is None:
        borough_polygon = load_leonia_polygon()

    poly = borough_polygon
    if boundary_buffer_deg > 0:
        poly = poly.buffer(boundary_buffer_deg)

    zones = zones_gdf.to_crs(4326) if zones_gdf.crs is not None else zones_gdf
    zones = zones.copy()
    zones["_in_leonia"] = zones.geometry.intersects(poly)

    join_key = None
    for cand in ("zone_name", "osm_way_id"):
        if cand in out.columns and cand in zones.columns:
            join_key = cand
            break
    if join_key is None:
        logger.warning(
            "filter_segments_to_leonia: no join key (zone_name / osm_way_id) "
            "in summary_df; falling back to name/class filter only"
        )
        return out

    inside = zones.loc[zones["_in_leonia"], [join_key]].drop_duplicates()
    return out.merge(inside, on=join_key, how="inner")


def annotate_in_leonia(
    df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame | None,
    *,
    borough_polygon: BaseGeometry | None = None,
    boundary_buffer_deg: float = 0.0,
    name_col: str = "osm_name",
    class_col: str = "road_class",
    new_col: str = "in_leonia_jurisdiction",
) -> pd.DataFrame:
    """Like ``filter_segments_to_leonia`` but returns the full input with
    a new boolean column instead of filtering rows. Useful for tables
    we want to keep verbatim (e.g. "all corridors observed") but where
    some rows should be flagged as out-of-scope for recommendations.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out[new_col] = True

    if name_col in out.columns:
        out.loc[out[name_col].apply(_matches_non_borough_name), new_col] = False
        out.loc[out[name_col].apply(is_county_state_arterial), new_col] = False
    if class_col in out.columns:
        out.loc[out[class_col].apply(_matches_non_borough_class), new_col] = False

    if zones_gdf is None or zones_gdf.empty:
        return out

    if borough_polygon is None:
        borough_polygon = load_leonia_polygon()

    poly = borough_polygon
    if boundary_buffer_deg > 0:
        poly = poly.buffer(boundary_buffer_deg)

    zones = zones_gdf.to_crs(4326) if zones_gdf.crs is not None else zones_gdf
    zones = zones.copy()
    zones["_in_leonia"] = zones.geometry.intersects(poly)

    join_key = None
    for cand in ("zone_name", "osm_way_id"):
        if cand in out.columns and cand in zones.columns:
            join_key = cand
            break
    if join_key is None:
        return out

    inside = zones[[join_key, "_in_leonia"]].drop_duplicates(subset=[join_key])
    out = out.merge(inside, on=join_key, how="left")
    out.loc[~out["_in_leonia"].fillna(False), new_col] = False
    out = out.drop(columns=["_in_leonia"])
    return out


__all__ = [
    "annotate_in_leonia",
    "filter_segments_to_leonia",
    "is_county_state_arterial",
    "is_under_borough_jurisdiction",
]
