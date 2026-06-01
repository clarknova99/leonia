"""Translate Bridge-OD trips into an assignment-ready demand dict.

Workflow:

1. Filter ``bridge_od.parquet`` to one ``(day_type_code, day_part_code)``
   slice. The remaining rows give the daily-trip count per OD pair.
2. Convert daily volumes to **trips per hour** by dividing by the
   day-part's nominal duration (e.g. Peak-AM ≈ 4 h, see ``DAY_PART_HOURS``).
3. Map each origin / destination zone to the nearest graph node using
   the zone's centroid from ``bridge_od_zones.parquet``. The 300 m
   spatial fallback is the same trick used by ``scripts/11_export_sumo.py``.

Returns a ``{(origin_node_id, destination_node_id): trips_per_hour}``
mapping suitable for the Frank–Wolfe loop in
:mod:`leonia_traffic.assignment.assignment`.
"""

from __future__ import annotations

import logging
import math
from typing import Mapping

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Nominal hours per day-part code (Bridge OD product). Matches the
# labels in `leonia_traffic.data.bridge_od_loader.DAY_PART_CODES`.
DAY_PART_HOURS: Mapping[int, float] = {
    0: 24.0,   # All Day (12am-12am)
    1: 6.0,    # Early AM (12am-6am)
    2: 4.0,    # Peak AM (6am-10am)
    3: 5.0,    # Mid-Day (10am-3pm)
    4: 4.0,    # Peak PM (3pm-7pm)
    5: 5.0,    # Late PM (7pm-12am)
}


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def map_zones_to_graph_nodes(
    zones_gdf: gpd.GeoDataFrame,
    G: nx.DiGraph,
    *,
    max_distance_m: float = 1500.0,
    name_col: str = "name",
) -> dict[str, str]:
    """Return ``{zone_name: nearest_graph_node_id}``.

    Uses a projected nearest-neighbour search (EPSG:3857) for accurate
    meter distances. Zones farther than ``max_distance_m`` from any node
    are dropped (and logged). The default radius (1500 m) is wide enough
    to attach the Bridge-OD perimeter gates, including GWB-area zones
    whose centroids sit on highway approaches outside the residential
    network.

    Note
    ----
    For the Bridge-OD product, prefer :func:`map_bridge_od_to_graph_edges`
    which uses ``origin_osm_way_id`` / ``destination_osm_way_id`` to
    pick *distinct* graph nodes per gate, avoiding the overlap problem
    where co-located origin and destination zones collapse to the same
    node.
    """
    if zones_gdf.empty:
        return {}

    # Build a GeoDataFrame of graph nodes.
    node_rows = []
    for nid, data in G.nodes(data=True):
        if "lon" in data and "lat" in data:
            node_rows.append({
                "node_id": str(nid),
                "geometry": Point(float(data["lon"]), float(data["lat"])),
            })
    if not node_rows:
        logger.warning("Graph has no nodes with lon/lat — cannot map zones.")
        return {}
    nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")

    zones = zones_gdf.copy()
    if name_col not in zones.columns:
        logger.warning("Zone GeoDataFrame missing '%s' column.", name_col)
        return {}

    # Project first, then take centroids (centroids on geographic CRS
    # trigger a UserWarning and are technically wrong).
    zones_proj = zones.to_crs(3857)
    zones_proj["geometry"] = zones_proj.geometry.centroid
    nodes_proj = nodes_gdf.to_crs(3857)

    joined = gpd.sjoin_nearest(
        zones_proj[[name_col, "geometry"]],
        nodes_proj[["node_id", "geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="_match_distance_m",
    )

    mapping: dict[str, str] = {}
    dropped = 0
    for _, row in joined.iterrows():
        zname = row.get(name_col)
        node_id = row.get("node_id")
        if pd.isna(zname) or pd.isna(node_id):
            dropped += 1
            continue
        mapping[str(zname)] = str(node_id)

    if dropped:
        logger.info(
            "map_zones_to_graph_nodes: %d zones farther than %dm from any node",
            dropped,
            int(max_distance_m),
        )
    return mapping


def _build_osm_way_to_edge(G: nx.DiGraph) -> dict[int, list[tuple[str, str]]]:
    """Return ``{osm_way_id: [(u, v), ...]}`` across all graph edges."""
    out: dict[int, list[tuple[str, str]]] = {}
    for u, v, d in G.edges(data=True):
        osm_id = d.get("osm_way_id")
        if osm_id is None:
            continue
        out.setdefault(int(osm_id), []).append((u, v))
    return out


def map_bridge_od_to_graph_edges(
    bridge_od_df: pd.DataFrame,
    G: nx.DiGraph,
    *,
    role: str = "origin",
) -> dict[str, str]:
    """Return ``{zone_name: graph_node_id}`` for Bridge-OD origins/destinations.

    Resolves zones via the OSM way id encoded in the Bridge OD CSV
    (``origin_osm_way_id`` or ``destination_osm_way_id``). For origins
    we use the edge's downstream node (where vehicles depart into the
    network); for destinations we use the upstream node (where they
    arrive at the gate before exiting).

    This avoids the overlap problem in
    :func:`map_zones_to_graph_nodes` where co-located origin and
    destination zones snap to the same graph node and get dropped as
    self-loops.

    Falls back to spatial lookup via :func:`map_zones_to_graph_nodes`
    only for zones whose ``osm_way_id`` is missing from the graph (OSM
    id drift between the StreetLight export and the current OSM
    snapshot — same problem documented in ``scripts/11_export_sumo.py``).
    """
    if role not in ("origin", "destination"):
        raise ValueError("role must be 'origin' or 'destination'")

    id_col = f"{role}_osm_way_id"
    zone_col = f"{role}_zone"
    if id_col not in bridge_od_df.columns or zone_col not in bridge_od_df.columns:
        logger.warning(
            "bridge_od_df missing %s / %s columns; mapping returned empty.",
            id_col, zone_col,
        )
        return {}

    way_to_edges = _build_osm_way_to_edge(G)

    sub = bridge_od_df[[zone_col, id_col]].drop_duplicates()
    mapping: dict[str, str] = {}
    missing_ids: list[str] = []
    for _, row in sub.iterrows():
        zname = row[zone_col]
        osm_id = row[id_col]
        if pd.isna(zname):
            continue
        if pd.notna(osm_id) and int(osm_id) in way_to_edges:
            edges = way_to_edges[int(osm_id)]
            # Pick the first edge; for origins use its v (downstream
            # node), for destinations use its u (upstream node). This
            # gives different nodes for co-located gates pointing in
            # opposite directions.
            u, v = edges[0]
            mapping[str(zname)] = v if role == "origin" else u
        else:
            missing_ids.append(str(zname))

    if missing_ids:
        logger.info(
            "map_bridge_od_to_graph_edges[%s]: %d zones have stale/missing "
            "osm_way_id (will need spatial fallback)",
            role, len(missing_ids),
        )
    return mapping


def bridge_od_to_demand(
    bridge_od_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    G: nx.DiGraph,
    *,
    day_type_code: int = 1,
    day_part_code: int = 2,
    max_zone_match_m: float = 1500.0,
    drop_self_loops: bool = True,
) -> dict[tuple[str, str], float]:
    """Return ``{(origin_node, dest_node): trips_per_hour}``.

    Parameters
    ----------
    bridge_od_df
        DataFrame from ``bridge_od.parquet`` (long format, one row per
        OD x day_type x day_part).
    zones_gdf
        GeoDataFrame of zone geometries from ``bridge_od_zones.parquet``.
        Must include a ``name`` column matching ``origin_zone`` /
        ``destination_zone`` in ``bridge_od_df``, plus a polygon
        geometry.
    G
        Assignment graph from :func:`build_assignment_graph`.
    day_type_code
        Bridge-OD day-type filter (default 1 = Monday-like weekday).
    day_part_code
        Day-part filter (default 2 = Peak AM, 4h window).
    max_zone_match_m
        Max distance from a zone centroid to the nearest graph node.
    drop_self_loops
        Skip pairs where origin and destination map to the same node.

    Notes
    -----
    Volumes are converted from average **daily** trips for the chosen
    day-part to **trips per hour** by dividing by
    ``DAY_PART_HOURS[day_part_code]``. The static UE assignment
    consumes hourly flow rates because its BPR cost function and link
    capacities are also expressed in vehicles per hour.
    """
    sub = bridge_od_df[
        (bridge_od_df["day_type_code"] == day_type_code)
        & (bridge_od_df["day_part_code"] == day_part_code)
    ]
    if sub.empty:
        logger.warning(
            "bridge_od_to_demand: no rows for day_type=%d day_part=%d",
            day_type_code,
            day_part_code,
        )
        return {}

    hours = DAY_PART_HOURS.get(int(day_part_code), 4.0)

    # Primary mapping: via origin_osm_way_id / destination_osm_way_id.
    # This gives *distinct* nodes per gate, which is critical because
    # the Bridge-OD origin and destination zones are physically
    # co-located perimeter gates — if we used polygon centroids, many
    # OD pairs would collapse to self-loops.
    origin_map = map_bridge_od_to_graph_edges(bridge_od_df, G, role="origin")
    dest_map = map_bridge_od_to_graph_edges(bridge_od_df, G, role="destination")

    # Spatial fallback for zones whose osm_way_id is stale.
    if zones_gdf is not None and not zones_gdf.empty:
        fallback = map_zones_to_graph_nodes(
            zones_gdf, G, max_distance_m=max_zone_match_m
        )
        for zname, node in fallback.items():
            origin_map.setdefault(str(zname), node)
            dest_map.setdefault(str(zname), node)

    if not origin_map and not dest_map:
        logger.warning("bridge_od_to_demand: no zones could be mapped to graph nodes")
        return {}

    demand: dict[tuple[str, str], float] = {}
    unmapped = 0
    self_loops = 0
    for _, row in sub.iterrows():
        o_zone = row["origin_zone"]
        d_zone = row["destination_zone"]
        o_node = origin_map.get(str(o_zone))
        d_node = dest_map.get(str(d_zone))
        if o_node is None or d_node is None:
            unmapped += 1
            continue
        if o_node == d_node:
            self_loops += 1
            if drop_self_loops:
                continue
        vol_daily = float(row["od_volume"])
        if vol_daily <= 0 or not math.isfinite(vol_daily):
            continue
        trips_per_hour = vol_daily / hours
        # Sum if multiple zone pairs collapse to the same node pair.
        demand[(o_node, d_node)] = demand.get((o_node, d_node), 0.0) + trips_per_hour

    if unmapped:
        logger.info("bridge_od_to_demand: %d OD rows with unmapped zones", unmapped)
    if self_loops:
        logger.info(
            "bridge_od_to_demand: %d OD rows collapsed to self-loops%s",
            self_loops, " (dropped)" if drop_self_loops else "",
        )
    logger.info(
        "bridge_od_to_demand: %d OD pairs, %.0f total trips/hour",
        len(demand),
        sum(demand.values()),
    )
    return demand


__all__ = [
    "DAY_PART_HOURS",
    "bridge_od_to_demand",
    "map_zones_to_graph_nodes",
]
