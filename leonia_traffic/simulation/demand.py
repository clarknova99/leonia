"""Demand-model builders for the Leonia UXsim simulation.

Two backends:

  - :func:`apply_gateway_demand` — placeholder that drives the World
    from observed StreetLight gateway volumes (Phase 3 today).
  - :func:`apply_od_demand` — drives the World from a StreetLight OD
    Analysis export (Phase 3 when the OD export lands).

Both functions return a small summary structure for logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from uxsim import World

from leonia_traffic.config import GWB_APPROACH_POLYGON, LEONIA_BBOX_WGS84
from leonia_traffic.data.od_loader import classify_zone_category

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gateway-based placeholder demand
# ---------------------------------------------------------------------------


@dataclass
class DemandSummary:
    n_demands_added: int = 0
    total_flow_vph: float = 0.0
    notes: list[str] = field(default_factory=list)


def _link_midpoint(W: World, link_name: str) -> tuple[float, float] | None:
    """Return the (lon, lat) midpoint of a UXsim link, or None."""
    try:
        l = W.get_link(link_name)
    except (KeyError, AttributeError):
        return None
    start = l.start_node
    end = l.end_node
    return ((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)


def _identify_gateway_links(
    matched: pd.DataFrame,
    W: World,
    *,
    leonia_polygon: Polygon | None,
    leonia_buffer_deg: float,
    road_classes: tuple[str, ...] = (
        "primary",
        "secondary",
        "tertiary",
        "residential",
    ),
    min_volume: float = 100.0,
    edge_band_deg: float = 0.003,   # ~330 m
    max_gateways: int = 40,
) -> pd.DataFrame:
    """Find UXsim links that cross the study-area border with non-trivial flow.

    Two modes:

    1. If ``leonia_polygon`` is provided, a link qualifies when exactly
       one of its endpoints lies inside the polygon (or its buffer).
    2. Otherwise (placeholder behavior), a link qualifies when its
       midpoint sits in a thin band along the *outer* bbox edge — i.e.
       the link is on the inbound perimeter of the study area.

    In both modes we then keep only the top ``max_gateways`` links by
    observed volume to keep the demand model tractable.
    """
    if matched.empty:
        return pd.DataFrame()

    minx, miny, maxx, maxy = LEONIA_BBOX_WGS84
    rows: list[dict] = []
    for link_name, row in matched.iterrows():
        vol = row.get("observed_volume")
        if pd.isna(vol) or vol < min_volume:
            continue
        try:
            l = W.get_link(link_name)
        except (KeyError, AttributeError):
            continue
        sx, sy = l.start_node.x, l.start_node.y
        ex, ey = l.end_node.x, l.end_node.y
        mid_x = (sx + ex) / 2.0
        mid_y = (sy + ey) / 2.0

        is_gateway = False
        inbound = True
        if leonia_polygon is not None:
            start_in = _pt_in_poly(leonia_polygon, (sx, sy))
            end_in = _pt_in_poly(leonia_polygon, (ex, ey))
            buf = leonia_polygon.buffer(leonia_buffer_deg)
            in_band = _pt_in_poly(buf, (mid_x, mid_y))
            is_gateway = (start_in != end_in) and in_band
            inbound = (not start_in) and end_in
        else:
            # Distance from midpoint to the nearest bbox edge.
            d = min(mid_x - minx, maxx - mid_x, mid_y - miny, maxy - mid_y)
            is_gateway = d <= edge_band_deg
        if not is_gateway:
            continue
        rows.append(
            {
                **row.to_dict(),
                "uxsim_link_name": link_name,
                "inbound": inbound,
                "mid_lon": mid_x,
                "mid_lat": mid_y,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("observed_volume", ascending=False).head(max_gateways)
    df = df.drop_duplicates(subset="osm_way_id", keep="first")
    return df.reset_index(drop=True)


def _pt_in_poly(poly: Polygon, pt: tuple[float, float]) -> bool:
    from shapely.geometry import Point
    return poly.contains(Point(pt))


def apply_gateway_demand(
    W: World,
    matched: pd.DataFrame,
    *,
    duration_hours: float = 4.0,
    t_start_s: float = 0.0,
    gwb_share: float = 0.6,
    gwb_polygon: Polygon = GWB_APPROACH_POLYGON,
    internal_destinations: list[tuple[float, float]] | None = None,
    internal_radius_deg: float = 0.0015,
    gateway_radius_deg: float = 0.0005,
    leonia_polygon: Polygon | None = None,
    leonia_buffer_deg: float = 0.002,
    min_volume: float = 100.0,
    daily_to_peak_factor: float = 0.10,
    max_gateways: int = 25,
) -> DemandSummary:
    """Add gateway-origin → destination demand to ``W`` from observed data.

    ``matched`` is the DataFrame returned by
    ``calibration_match.match_segments_to_links`` (indexed by UXsim link
    name, with an ``observed_volume`` column).

    Each gateway's observed daily volume is multiplied by
    ``daily_to_peak_factor`` (defaults to 10 %, a literature value for
    AM-peak share of AADT on urban arterials) to convert from a 24-h
    AADT-style value to a total trip volume injected over the simulated
    window. UXsim's ``adddemand_area2area2`` takes ``volume`` directly
    (the total number of trips between the two areas), which is what we
    pass in.
    """
    if internal_destinations is None:
        internal_destinations = [
            (-73.985, 40.864),   # Leonia core
            (-73.992, 40.870),   # Leonia north / Englewood line
        ]

    gw = _identify_gateway_links(
        matched,
        W,
        leonia_polygon=leonia_polygon,
        leonia_buffer_deg=leonia_buffer_deg,
        min_volume=min_volume,
        max_gateways=max_gateways,
    )
    if gw.empty:
        return DemandSummary(notes=["No gateway links found."])

    duration_s = duration_hours * 3600.0
    t_end_s = t_start_s + duration_s

    gwb_cx = gwb_polygon.centroid.x
    gwb_cy = gwb_polygon.centroid.y
    gwb_radius_deg = max(
        (gwb_polygon.bounds[2] - gwb_polygon.bounds[0]) / 2.0,
        (gwb_polygon.bounds[3] - gwb_polygon.bounds[1]) / 2.0,
    )

    summary = DemandSummary()
    leonia_share = max(1.0 - gwb_share, 0.0)
    n_dests = len(internal_destinations)
    per_dest_share = leonia_share / n_dests if n_dests > 0 else 0.0

    for _, row in gw.iterrows():
        daily = float(row["observed_volume"])
        # Total trip volume during the simulated window.
        window_volume = daily * daily_to_peak_factor
        if window_volume <= 0:
            continue
        origin_x, origin_y = float(row["mid_lon"]), float(row["mid_lat"])

        gwb_volume = window_volume * gwb_share
        if gwb_volume > 0:
            try:
                W.adddemand_area2area2(
                    origin_x, origin_y, gateway_radius_deg,
                    gwb_cx, gwb_cy, gwb_radius_deg,
                    t_start_s, t_end_s,
                    volume=gwb_volume,
                )
                summary.n_demands_added += 1
                summary.total_flow_vph += gwb_volume / duration_hours
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "Gateway→GWB demand failed for %s: %s",
                    row.get("uxsim_link_name"), exc,
                )
                summary.notes.append(
                    f"GWB path failed at {row.get('uxsim_link_name')}: {exc}"
                )

        for (dx, dy) in internal_destinations:
            dest_volume = window_volume * per_dest_share
            if dest_volume <= 0:
                continue
            try:
                W.adddemand_area2area2(
                    origin_x, origin_y, gateway_radius_deg,
                    dx, dy, internal_radius_deg,
                    t_start_s, t_end_s,
                    volume=dest_volume,
                )
                summary.n_demands_added += 1
                summary.total_flow_vph += dest_volume / duration_hours
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "Gateway→Leonia demand failed for %s: %s",
                    row.get("uxsim_link_name"), exc,
                )

    summary.notes.append(
        f"Processed {len(gw)} gateways with daily_to_peak_factor="
        f"{daily_to_peak_factor:.3f} over {duration_hours:.1f} h."
    )
    return summary


# ---------------------------------------------------------------------------
# OD-based demand (drop-in for when the StreetLight OD export lands)
# ---------------------------------------------------------------------------


@dataclass
class ODZone:
    name: str
    centroid_x: float
    centroid_y: float
    radius_deg: float
    category: str
    is_origin: bool = True
    is_destination: bool = True


def od_zones_from_shapefile(
    zones: gpd.GeoDataFrame,
    *,
    name_col: str = "zone_name",
    default_radius_deg: float = 0.002,
) -> list[ODZone]:
    """Convert an OD zone shapefile into ``ODZone`` records."""
    zones_geo = zones.to_crs(4326) if zones.crs and zones.crs.to_epsg() != 4326 else zones
    out: list[ODZone] = []
    for _, row in zones_geo.iterrows():
        name = str(row.get(name_col, "")) or str(row.get("name", ""))
        if not name:
            continue
        c = row.geometry.centroid
        bounds = row.geometry.bounds
        radius = max((bounds[2] - bounds[0]) / 2.0, (bounds[3] - bounds[1]) / 2.0, default_radius_deg)
        category = classify_zone_category(name)
        is_origin = name.startswith("IN_") or name.startswith("ORIG_")
        is_destination = name.startswith("DEST_")
        if not is_origin and not is_destination:
            is_origin = is_destination = True
        out.append(
            ODZone(
                name=name,
                centroid_x=float(c.x),
                centroid_y=float(c.y),
                radius_deg=float(radius),
                category=category,
                is_origin=is_origin,
                is_destination=is_destination,
            )
        )
    return out


def apply_od_demand(
    W: World,
    od_long: pd.DataFrame,
    zones: list[ODZone],
    *,
    duration_hours: float = 4.0,
    t_start_s: float = 0.0,
    day_part: str | None = None,
    day_type: str | None = None,
    volume_col: str = "avg_daily_od_volume",
    daily_to_peak_factor: float = 1.0,
) -> DemandSummary:
    """Apply a StreetLight OD matrix to ``W``."""
    df = od_long.copy()
    if day_part:
        df = df[df["day_part"] == day_part]
    if day_type:
        df = df[df["day_type"] == day_type]
    if df.empty:
        return DemandSummary(notes=["No OD rows after filtering."])

    zone_by_name = {z.name: z for z in zones}
    duration_s = duration_hours * 3600.0
    t_end_s = t_start_s + duration_s

    summary = DemandSummary()
    for _, row in df.iterrows():
        oz = zone_by_name.get(row["origin_zone"])
        dz = zone_by_name.get(row["destination_zone"])
        if oz is None or dz is None:
            continue
        daily_v = float(row.get(volume_col, 0.0))
        if not (daily_v > 0):
            continue
        window_volume = daily_v * daily_to_peak_factor
        try:
            W.adddemand_area2area2(
                oz.centroid_x, oz.centroid_y, oz.radius_deg,
                dz.centroid_x, dz.centroid_y, dz.radius_deg,
                t_start_s, t_end_s,
                volume=window_volume,
            )
            summary.n_demands_added += 1
            summary.total_flow_vph += window_volume / duration_hours
        except Exception as exc:  # pragma: no cover
            logger.debug("OD demand %s->%s failed: %s", oz.name, dz.name, exc)
            summary.notes.append(f"{oz.name}->{dz.name}: {exc}")

    return summary


# ---------------------------------------------------------------------------
# Bridge-destination OD demand (real observed per-day-part volumes)
# ---------------------------------------------------------------------------


@dataclass
class BridgeODDemandSummary(DemandSummary):
    n_origin_ways_resolved: int = 0
    n_destination_ways_resolved: int = 0
    n_pairs_skipped_no_match: int = 0


def apply_bridge_od_demand(
    W: World,
    od_df: pd.DataFrame,
    osm_to_uxsim: dict[int, list],
    *,
    day_type_code: int = 1,
    day_part_code: int = 2,
    duration_hours: float = 4.0,
    t_start_s: float = 0.0,
    demand_scale: float = 1.0,
    volume_col: str = "od_volume",
    zone_to_link_name: dict[str, str] | None = None,
) -> BridgeODDemandSummary:
    """Apply real Bridge-destination OD demand to a UXsim World.

    Uses observed per-day-part OD volumes from
    :func:`leonia_traffic.data.bridge_od_loader.load_bridge_od`. Each
    origin / destination gate's OSM way ID is resolved to one or more
    UXsim links via ``osm_to_uxsim`` (built by
    :func:`leonia_traffic.network.calibration_match.build_osm_to_uxsim_index`),
    and demand is added via ``W.adddemand(orig_node, dest_node, ...)``
    using the link's downstream node as origin and upstream node as
    destination. That avoids the "no node in radius" failure mode of
    ``adddemand_area2area2`` for tight gateway geometries.

    Parameters
    ----------
    W
        UXsim ``World`` (built fresh, before ``W.exec_simulation()``).
    od_df
        Long-format OD DataFrame from ``load_bridge_od``.
    osm_to_uxsim
        Mapping of ``osm_way_id`` → list of ``UXsimLinkRef``.
    day_type_code
        Day Type to simulate. Defaults to 1 (Monday); pass 0 for all-days.
    day_part_code
        Day Part to simulate. Defaults to 2 (Peak AM).
    duration_hours
        Simulation horizon for which the OD volumes are valid. The
        Peak-AM bucket is 4 h.
    t_start_s
        Simulation time at which the demand starts.
    demand_scale
        Global multiplier for sensitivity / calibration sweeps.
    volume_col
        Column in ``od_df`` carrying the trip volume. Defaults to
        ``"od_volume"``.
    zone_to_link_name
        Optional spatial-fallback mapping from StreetLight zone name
        (e.g. ``"Fort Lee Road / 590576"``) to the closest UXsim link
        name. Used when an OD row's OSM way ID is stale and no longer
        appears in the current OSM extract. Build via
        :func:`leonia_traffic.network.calibration_match.spatial_resolve_osm_way_ids`.
    """
    df = od_df.copy()
    df = df[(df["day_type_code"] == day_type_code)
            & (df["day_part_code"] == day_part_code)]
    if df.empty:
        return BridgeODDemandSummary(notes=[
            f"No OD rows for day_type_code={day_type_code} day_part_code={day_part_code}.",
        ])

    duration_s = duration_hours * 3600.0
    t_end_s = t_start_s + duration_s

    summary = BridgeODDemandSummary()
    origins_seen: set[int] = set()
    dests_seen: set[int] = set()

    zone_to_link_name = zone_to_link_name or {}

    for _, row in df.iterrows():
        ow = row.get("origin_osm_way_id")
        dw = row.get("destination_osm_way_id")
        volume = float(row.get(volume_col, 0.0) or 0.0) * demand_scale
        if not (volume > 0):
            continue

        # Try OSM way ID first, then fall back to the spatial zone-name
        # mapping (handles stale OSM way IDs in the StreetLight export).
        origin_link_name: str | None = None
        dest_link_name: str | None = None
        if not pd.isna(ow):
            links = osm_to_uxsim.get(int(ow), [])
            if links:
                origin_link_name = links[0].name
        if origin_link_name is None:
            origin_link_name = zone_to_link_name.get(str(row.get("origin_zone", "")))

        if not pd.isna(dw):
            links = osm_to_uxsim.get(int(dw), [])
            if links:
                dest_link_name = links[0].name
        if dest_link_name is None:
            dest_link_name = zone_to_link_name.get(str(row.get("destination_zone", "")))

        if origin_link_name is None or dest_link_name is None:
            summary.n_pairs_skipped_no_match += 1
            continue
        try:
            orig_node = W.get_link(origin_link_name).end_node.name
            dest_node = W.get_link(dest_link_name).start_node.name
        except (KeyError, AttributeError) as exc:  # pragma: no cover
            logger.debug("Bridge OD %s->%s failed link lookup: %s", ow_int, dw_int, exc)
            summary.n_pairs_skipped_no_match += 1
            continue

        try:
            W.adddemand(orig_node, dest_node, t_start_s, t_end_s, volume=volume)
            summary.n_demands_added += 1
            summary.total_flow_vph += volume / duration_hours
            if not pd.isna(ow):
                origins_seen.add(int(ow))
            if not pd.isna(dw):
                dests_seen.add(int(dw))
        except Exception as exc:  # pragma: no cover
            logger.debug("Bridge OD demand %s→%s failed: %s", orig_node, dest_node, exc)
            summary.notes.append(f"{orig_node}->{dest_node}: {exc}")

    summary.n_origin_ways_resolved = len(origins_seen)
    summary.n_destination_ways_resolved = len(dests_seen)
    summary.notes.append(
        f"day_type_code={day_type_code} day_part_code={day_part_code} "
        f"scale={demand_scale:.3f} duration_h={duration_hours:.1f} "
        f"origins={len(origins_seen)} destinations={len(dests_seen)} "
        f"skipped={summary.n_pairs_skipped_no_match}"
    )
    return summary


__all__ = [
    "BridgeODDemandSummary",
    "DemandSummary",
    "ODZone",
    "apply_bridge_od_demand",
    "apply_gateway_demand",
    "apply_od_demand",
    "od_zones_from_shapefile",
]
