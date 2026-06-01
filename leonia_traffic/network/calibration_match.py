"""Match StreetLight observed segments to UXsim simulation links.

The bridge is the OpenStreetMap way ID, which is:

  - extractable from a StreetLight zone name via ``parse_zone_name`` (the
    string is ``"[OSM name] / [osm_way_id] / [split]"``), and
  - encoded in UXsim's link ``name`` field by ``OSMImporter`` as
    ``"<street>-<osm_id>"`` (parseable via
    ``leonia_traffic.network.osm_builder.parse_uxsim_link_name``).

Because UXsim post-processing merges short OSM ways into longer links,
the relationship is many-to-one: many StreetLight zones can map to the
same UXsim link. We aggregate observed volumes by summing across the
zones that share a UXsim link (weighted by segment length if shapefile
geometry is available, otherwise straight average).
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import SIM_DEFAULTS
from leonia_traffic.network.osm_builder import parse_uxsim_link_name


@dataclass(frozen=True)
class UXsimLinkRef:
    """A reference to a single UXsim link as returned by OSMImporter."""

    name: str
    from_node: str
    to_node: str
    lanes: int
    free_flow_speed: float
    length: float
    osm_way_id: int | None
    is_reverse: bool


def index_uxsim_links(uxsim_links: list) -> list[UXsimLinkRef]:
    """Wrap the raw UXsim links list (post ``osm_network_postprocessing``).

    The post-processing pipeline stores ``length`` in WGS84 degrees (that
    is what ``OSMImporter.osm_network_to_World`` expects so it can apply
    ``coef_degree_to_meter`` when adding to a World). We convert to
    meters here so the matcher can be reasoned about in physical units.
    """
    coef = SIM_DEFAULTS.coef_degree_to_meter
    out: list[UXsimLinkRef] = []
    for l in uxsim_links:
        name = str(l[0])
        road_name, osmid, is_rev = parse_uxsim_link_name(name)
        length_deg = float(l[5]) if len(l) >= 6 else float("nan")
        out.append(
            UXsimLinkRef(
                name=name,
                from_node=str(l[1]),
                to_node=str(l[2]),
                lanes=int(l[3]),
                free_flow_speed=float(l[4]),
                length=length_deg * coef,
                osm_way_id=osmid,
                is_reverse=is_rev,
            )
        )
    return out


def build_osm_to_uxsim_index(
    refs: list[UXsimLinkRef],
) -> dict[int, list[UXsimLinkRef]]:
    """Return ``{osm_way_id: [UXsimLinkRef, ...]}`` for matchable links."""
    index: dict[int, list[UXsimLinkRef]] = {}
    for r in refs:
        if r.osm_way_id is None:
            continue
        index.setdefault(r.osm_way_id, []).append(r)
    return index


def match_segments_to_links(
    streetlight_gdf: gpd.GeoDataFrame,
    uxsim_links: list,
    *,
    value_col: str = "avg_volume",
    source_label: str = "weekdays",
) -> pd.DataFrame:
    """Join StreetLight observed volumes to UXsim links via OSM way ID.

    Each StreetLight "split segment" is one row in the input frame and
    represents the volume on that *piece* of an OSM way. After UXsim's
    node-merging, one OSM way can correspond to several UXsim links. We
    therefore:

      1. Compute one **observed flow** (avg volume across the zones on
         this OSM way), interpreted as the steady-state flow rate along
         the way. ``avg_volume`` is a per-segment count for the
         observation period, so averaging is the right aggregation —
         every segment on the same way should record approximately the
         same flow.
      2. Compute the length-weighted average speed across zones.
      3. Apply that single flow and speed to each UXsim link sharing
         the OSM way. The UXsim link's own ``length_m`` is preserved.

    Returns a DataFrame indexed by UXsim link ``name``.
    """
    refs = index_uxsim_links(uxsim_links)
    osm_to_uxsim = build_osm_to_uxsim_index(refs)

    sub = streetlight_gdf[streetlight_gdf["source"] == source_label].copy()
    sub = sub.dropna(subset=["osm_way_id"])
    sub["osm_way_id"] = sub["osm_way_id"].astype(int)

    sub_proj = (
        sub.to_crs(3857)
        if sub.crs is not None and sub.crs.to_epsg() != 3857
        else sub
    )
    sub["zone_length_m"] = sub_proj.geometry.length

    records: list[dict] = []
    for osm_id, group in sub.groupby("osm_way_id"):
        if int(osm_id) not in osm_to_uxsim:
            continue
        # Flow is conserved along a way: average across the zones rather
        # than sum (each zone is a *redundant* observation of the same
        # flow at a different point along the way).
        flow = group[value_col].mean()
        total_len = group["zone_length_m"].sum()
        if total_len > 0:
            weights = group["zone_length_m"] / total_len
            speed = (group["avg_speed_mph"] * weights).sum()
        else:
            speed = group["avg_speed_mph"].mean()
        for r in osm_to_uxsim[int(osm_id)]:
            records.append(
                {
                    "uxsim_link_name": r.name,
                    "osm_way_id": r.osm_way_id,
                    "observed_volume": float(flow) if pd.notna(flow) else None,
                    "observed_avg_speed_mph": float(speed) if pd.notna(speed) else None,
                    "n_streetlight_zones": len(group),
                    "length_m": r.length,
                    "free_flow_speed_ms": r.free_flow_speed,
                    "lanes": r.lanes,
                    "from_node": r.from_node,
                    "to_node": r.to_node,
                    "is_reverse": r.is_reverse,
                }
            )

    return pd.DataFrame.from_records(records).set_index("uxsim_link_name")


# ---------------------------------------------------------------------------
# Spatial fallback: nearest UXsim link by geometry
# ---------------------------------------------------------------------------


def _link_midpoint_lonlat(W, link_name: str) -> tuple[float, float] | None:
    try:
        l = W.get_link(link_name)
    except Exception:  # pragma: no cover
        return None
    sn, en = l.start_node, l.end_node
    return ((sn.x + en.x) / 2.0, (sn.y + en.y) / 2.0)


def build_link_midpoint_gdf(W) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of UXsim link midpoints in EPSG:4326.

    Used by :func:`spatial_resolve_osm_way_ids` to find the nearest
    UXsim link to a StreetLight zone whose OSM way ID is stale.
    """
    from shapely.geometry import Point

    rows: list[dict] = []
    for link in getattr(W, "LINKS", []):
        sn, en = link.start_node, link.end_node
        mx, my = (sn.x + en.x) / 2.0, (sn.y + en.y) / 2.0
        rows.append({
            "uxsim_link_name": link.name,
            "geometry": Point(mx, my),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def spatial_resolve_osm_way_ids(
    zone_gdf: gpd.GeoDataFrame,
    W,
    *,
    name_col: str = "name",
    max_distance_m: float = 100.0,
) -> dict[str, str]:
    """Return ``{zone_name: nearest_uxsim_link_name}``.

    Spatial-fallback matcher for StreetLight zones whose advertised OSM
    way ID is no longer present in the current network (because the OSM
    way was split / renumbered after the StreetLight export was made).
    Projects to EPSG:3857 for meter-accurate nearest-neighbor matching.

    Parameters
    ----------
    zone_gdf
        GeoDataFrame of StreetLight zones (must include ``name`` column
        and a geometry).
    W
        UXsim ``World``.
    max_distance_m
        Skip zones farther than this distance from any UXsim link.

    Notes
    -----
    Uses ``geopandas.sjoin_nearest`` which is O(n_zones·log n_links)
    via a spatial index. Distances are computed in projected meters.
    """
    if zone_gdf.empty:
        return {}

    links_gdf = build_link_midpoint_gdf(W)
    if links_gdf.empty:
        return {}

    zones = zone_gdf.copy()
    if name_col not in zones.columns:
        return {}

    # Both to projected CRS for meter distances.
    zones_proj = zones.to_crs(3857)
    links_proj = links_gdf.to_crs(3857)

    joined = gpd.sjoin_nearest(
        zones_proj[[name_col, "geometry"]],
        links_proj[["uxsim_link_name", "geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="match_distance_m",
    )

    # Deduplicate: keep the closest UXsim link per zone.
    joined = joined.sort_values("match_distance_m").drop_duplicates(name_col)

    out: dict[str, str] = {}
    for _, row in joined.iterrows():
        zone_name = row[name_col]
        link_name = row.get("uxsim_link_name")
        if pd.isna(link_name):
            continue
        out[str(zone_name)] = str(link_name)
    return out


# ---------------------------------------------------------------------------
# ZA-streets matcher (Pass C.3)
# ---------------------------------------------------------------------------


def match_za_streets_to_links(
    W,
    za_main_df: pd.DataFrame,
    uxsim_links: list,
    *,
    line_gdf: gpd.GeoDataFrame | None = None,
    day_type_code: int = 4,
    day_part_code: int = 0,
    spatial_fallback_max_distance_m: float = 100.0,
) -> pd.DataFrame:
    """Join the Leonia-streets ZA Visitor volumes to UXsim links.

    Pass-C residential-street observations complement the broader
    Street Scanner export by adding direct measurements on ~150 tertiary
    segments inside Leonia. This matcher converts those into the same
    ``matched`` DataFrame shape that
    :func:`leonia_traffic.network.calibration_match.match_segments_to_links`
    produces, so calibration scoring can union the two sources without
    reshaping.

    Parameters
    ----------
    W:
        UXsim ``World`` — only used by the spatial fallback for zones
        whose advertised OSM way ID is stale.
    za_main_df:
        Output of :func:`leonia_traffic.data.za_streets_loader.load_za_main`.
    uxsim_links:
        The links list returned by ``world_from_osm``.
    line_gdf:
        Optional GeoDataFrame from
        :func:`leonia_traffic.data.za_streets_loader.load_za_line_shapes`,
        used for spatial-nearest fallback. If omitted, only direct OSM-id
        matches are returned.
    day_type_code:
        Day Type filter (default 4 = Thursday, the project's canonical
        weekday-typical day; the export's day types are All / Mon-Thu /
        Sat / Sun).
    day_part_code:
        Day Part filter (default 0 = All Day; pass 8/9/10 for AM-peak
        hours).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``uxsim_link_name`` (matching the Street Scanner
        matched-frame index), with columns ``osm_way_id``,
        ``observed_volume``, ``observed_avg_speed_mph``,
        ``street_name``, ``source`` (= ``"za_streets"``), and any
        passthrough UXsim link metadata (``length_m``, ``lanes``,
        ``from_node``, ``to_node``, ``is_reverse``,
        ``free_flow_speed_ms``).
    """
    if za_main_df is None or za_main_df.empty:
        return pd.DataFrame()

    df = za_main_df
    if "filter" in df.columns:
        df = df[df["filter"] == "Visitors"]
    df = df[(df["day_type_code"] == day_type_code)
            & (df["day_part_code"] == day_part_code)]
    if df.empty:
        return pd.DataFrame()

    refs = index_uxsim_links(uxsim_links)
    osm_to_uxsim = build_osm_to_uxsim_index(refs)

    # Spatial fallback dict: zone_name -> uxsim_link_name, for zones
    # whose OSM way id isn't present in the current network.
    zone_to_link: dict[str, str] = {}
    if line_gdf is not None and not line_gdf.empty:
        # Mirror the bridge-OD/congestion call: ``name`` column is the
        # zone label including the OSM id.
        gdf = line_gdf.copy()
        if "name" not in gdf.columns and "zone_name" in gdf.columns:
            gdf = gdf.rename(columns={"zone_name": "name"})
        zone_to_link = spatial_resolve_osm_way_ids(
            gdf, W, name_col="name",
            max_distance_m=spatial_fallback_max_distance_m,
        )

    # Build a {uxsim_link_name: UXsimLinkRef} for quick metadata join
    # on names produced by the spatial fallback.
    ref_by_name = {r.name: r for r in refs}

    records: list[dict] = []
    for _, row in df.iterrows():
        zone_name = row.get("zone_name")
        osm_id = row.get("osm_way_id")
        volume = row.get("zone_volume")
        avg_speed = row.get("avg_trip_speed_mph")
        street_name = row.get("street_name")

        target_refs: list[UXsimLinkRef] = []
        if pd.notna(osm_id) and int(osm_id) in osm_to_uxsim:
            target_refs = osm_to_uxsim[int(osm_id)]
        elif zone_name in zone_to_link:
            ref = ref_by_name.get(zone_to_link[zone_name])
            if ref is not None:
                target_refs = [ref]

        if not target_refs:
            continue

        for r in target_refs:
            records.append({
                "uxsim_link_name": r.name,
                "osm_way_id": int(osm_id) if pd.notna(osm_id) else r.osm_way_id,
                "observed_volume": float(volume) if pd.notna(volume) else None,
                "observed_avg_speed_mph": float(avg_speed) if pd.notna(avg_speed) else None,
                "street_name": street_name,
                "n_streetlight_zones": 1,
                "length_m": r.length,
                "free_flow_speed_ms": r.free_flow_speed,
                "lanes": r.lanes,
                "from_node": r.from_node,
                "to_node": r.to_node,
                "is_reverse": r.is_reverse,
                "source": "za_streets",
            })

    if not records:
        return pd.DataFrame()
    out = pd.DataFrame.from_records(records)
    # Multiple ZA zones can resolve to the same UXsim link (long ways
    # split into multiple zones). Average the volumes — the flow on a
    # conserved-flow link should be the same at every measurement point.
    grouped = out.groupby("uxsim_link_name", as_index=True).agg({
        "osm_way_id": "first",
        "observed_volume": "mean",
        "observed_avg_speed_mph": "mean",
        "street_name": "first",
        "n_streetlight_zones": "sum",
        "length_m": "first",
        "free_flow_speed_ms": "first",
        "lanes": "first",
        "from_node": "first",
        "to_node": "first",
        "is_reverse": "first",
        "source": "first",
    })
    return grouped


__all__ = [
    "UXsimLinkRef",
    "build_link_midpoint_gdf",
    "build_osm_to_uxsim_index",
    "index_uxsim_links",
    "match_segments_to_links",
    "match_za_streets_to_links",
    "spatial_resolve_osm_way_ids",
]
