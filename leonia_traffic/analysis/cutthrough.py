"""Heuristics that rank segments by how cut-through-y they look.

This module operates purely on observed StreetLight data. The full
cut-through identification (Phase 6) layers UXsim trajectory analysis
on top of these signals.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from leonia_traffic.analysis.jurisdiction import is_county_state_arterial
from leonia_traffic.config import SUSPECTED_CUTTHROUGH_STREETS


def _normalize_name(name: str | None) -> str:
    return (name or "").strip().lower()


_SUSPECT_LC = {_normalize_name(n) for n in SUSPECTED_CUTTHROUGH_STREETS}


def add_weekday_weekend_signals(piv: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add weekday/weekend ratio columns to a pivoted GeoDataFrame.

    Expects ``avg_volume__weekdays`` and ``avg_volume__weekend`` columns
    (output of ``pivot_by_source``). Adds:

      - ``weekday_weekend_ratio``: avg_volume__weekdays / avg_volume__weekend
        (clipped to handle zero/missing).
      - ``weekday_minus_weekend``: absolute difference.
      - ``is_suspected_cutthrough_street``: bool, by ``road_name`` match.
    """
    out = piv.copy()
    wd = out.get("avg_volume__weekdays")
    we = out.get("avg_volume__weekend")
    if wd is None or we is None:
        out["weekday_weekend_ratio"] = np.nan
        out["weekday_minus_weekend"] = np.nan
    else:
        wd_arr = pd.to_numeric(wd, errors="coerce")
        we_arr = pd.to_numeric(we, errors="coerce")
        denom = we_arr.where(we_arr > 0, np.nan)
        out["weekday_weekend_ratio"] = wd_arr / denom
        out["weekday_minus_weekend"] = wd_arr - we_arr

    out["is_suspected_cutthrough_street"] = (
        out["road_name"].fillna("").map(_normalize_name).isin(_SUSPECT_LC)
    )
    return out


def rank_cutthrough_suspects(
    piv: gpd.GeoDataFrame,
    *,
    road_classes: tuple[str, ...] = ("residential", "tertiary"),
    min_weekday_volume: float = 200.0,
    top_n: int | None = 50,
) -> gpd.GeoDataFrame:
    """Rank residential/tertiary segments by weekday/weekend asymmetry.

    County-owned arterials (Broad Ave, Grand Ave, Fort Lee Rd / Main St)
    are excluded — they carry through-traffic by design and are not
    targets for local-road intervention.
    """
    g = piv.copy()
    if "road_class" in g.columns:
        g = g[g["road_class"].isin(road_classes)]
    if "avg_volume__weekdays" in g.columns:
        g = g[g["avg_volume__weekdays"].fillna(0) >= min_weekday_volume]

    # Exclude county arterials — they should not appear in the local-road
    # cut-through suspect list.
    name_col = next((c for c in ("road_name", "osm_name", "street_name", "name")
                     if c in g.columns), None)
    if name_col:
        g = g[~g[name_col].fillna("").apply(is_county_state_arterial)]

    g = g.sort_values("weekday_weekend_ratio", ascending=False, na_position="last")
    if top_n is not None:
        g = g.head(top_n)
    return g


def rank_speed_over_limit(
    gdf: gpd.GeoDataFrame,
    *,
    source: str = "weekdays",
    road_classes: tuple[str, ...] = ("residential", "tertiary"),
    min_speed_limit: float = 15.0,
    top_n: int | None = 50,
) -> pd.DataFrame:
    """Rank segments by average speed above the posted limit."""
    g = gdf[gdf["source"] == source].copy()
    if "road_class" in g.columns:
        g = g[g["road_class"].isin(road_classes)]
    g = g[g["speed_limit_mph"] >= min_speed_limit]
    g["speed_over_limit"] = g["avg_speed_mph"] - g["speed_limit_mph"]
    g = g.sort_values("speed_over_limit", ascending=False, na_position="last")
    if top_n is not None:
        g = g.head(top_n)
    return g


def residential_volume_percentiles(
    gdf: gpd.GeoDataFrame,
    *,
    source: str = "weekdays",
) -> pd.DataFrame:
    """Compute volume percentiles by road class for one source."""
    g = gdf[gdf["source"] == source]
    return (
        g.groupby("road_class")["avg_volume"]
        .describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        .round(1)
    )


def detect_simulated_cutthrough(
    W,
    *,
    min_trip_length_m: float = 500.0,
    leonia_bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Identify UXsim links that carry vehicles which neither originated
    nor terminated near the link.

    A trip's *origin* and *destination* are taken from its first and
    last node coordinates. A link is flagged as carrying cut-through
    flow if at least one vehicle's trip:

      - was longer than ``min_trip_length_m``
      - had both endpoints outside a small radius around the link's
        midpoint (defined as ``min_trip_length_m / 2``)

    Returns a DataFrame indexed by link name with:

      - ``n_cutthrough_vehicles``: # passing vehicles meeting the criteria.
      - ``n_total_vehicles``: # passing vehicles.
      - ``cutthrough_share``: ratio in [0, 1].
    """
    import math

    veh_routes: dict[str, dict] = {}
    for vid, v in W.VEHICLES.items():
        try:
            log_t = v.log_t
            log_x = v.log_x
            log_y = v.log_y
            log_link = v.log_link
        except AttributeError:
            continue
        if not log_t or len(log_t) < 2:
            continue
        veh_routes[vid] = {
            "origin": (log_x[0], log_y[0]),
            "dest": (log_x[-1], log_y[-1]),
            "links": [str(ln) for ln in log_link if ln],
        }

    link_counts: dict[str, dict] = {}
    for vid, info in veh_routes.items():
        ox, oy = info["origin"]
        dx, dy = info["dest"]
        trip_len = math.hypot((dx - ox) * 111000, (dy - oy) * 111000)
        is_long = trip_len >= min_trip_length_m
        for link_name in set(info["links"]):
            try:
                link = W.get_link(link_name)
            except (KeyError, AttributeError):
                continue
            mx = (link.start_node.x + link.end_node.x) / 2.0
            my = (link.start_node.y + link.end_node.y) / 2.0
            radius_deg = min_trip_length_m / 2 / 111000.0
            o_far = math.hypot(ox - mx, oy - my) > radius_deg
            d_far = math.hypot(dx - mx, dy - my) > radius_deg
            is_cutthrough = is_long and o_far and d_far
            rec = link_counts.setdefault(
                link_name, {"n_cutthrough_vehicles": 0, "n_total_vehicles": 0}
            )
            rec["n_total_vehicles"] += 1
            if is_cutthrough:
                rec["n_cutthrough_vehicles"] += 1

    rows = []
    for name, rec in link_counts.items():
        total = rec["n_total_vehicles"]
        share = rec["n_cutthrough_vehicles"] / total if total else 0.0
        rows.append(
            {
                "uxsim_link_name": name,
                "n_total_vehicles": total,
                "n_cutthrough_vehicles": rec["n_cutthrough_vehicles"],
                "cutthrough_share": share,
            }
        )
    return pd.DataFrame(rows).set_index("uxsim_link_name")


__all__ = [
    "add_weekday_weekend_signals",
    "detect_simulated_cutthrough",
    "rank_cutthrough_suspects",
    "rank_speed_over_limit",
    "residential_volume_percentiles",
]
