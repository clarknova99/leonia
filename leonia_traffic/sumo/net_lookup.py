"""Shared OSM ↔ SUMO edge resolution helpers.

These functions are used by both :mod:`scripts.11_export_sumo` (writing
the static SUMO project) and :mod:`leonia_traffic.sumo.runtime` (the
interactive libsumo wrapper). Keeping them here avoids drift between
the two — every consumer follows the same rules for:

* parsing the ``<param key="origId">`` markers netconvert writes onto
  each ``<edge>``,
* recovering WGS84 lon/lat for SUMO edge shapes (so we can spatially
  match StreetLight zones whose OSM ids predate the current OSM
  extract),
* reading the ``leonia.edgedata.meta.csv`` sidecar that pairs each
  ``sumo_edge_id`` with its ``osm_way_id`` and ``street_name``.

Nothing in here imports ``libsumo`` — it's all pure XML / pandas /
geopandas, so it can run in test environments that don't have SUMO.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSM way id → SUMO edge id lookup
# ---------------------------------------------------------------------------


def load_osm_to_sumo_lookup(net_path: Path) -> dict[int, list[str]]:
    """Parse ``leonia.net.xml`` and return ``{osm_way_id: [edge_ids]}``.

    netconvert (with ``--output.original-names``) writes the source
    OSM way id as ``<param key="origId" value="..."/>`` on each edge.
    A single OSM way often maps to 1–3 SUMO edges because netconvert
    splits at junctions; a single edge can also list multiple
    space-separated ids when ways were merged.

    Returns an empty dict if the file is missing or unparseable.
    """
    lookup: dict[int, list[str]] = {}
    if not net_path.exists():
        return lookup
    try:
        tree = ET.parse(net_path)
    except ET.ParseError:
        logger.warning("Could not parse %s as XML; returning empty lookup",
                       net_path)
        return lookup

    for edge in tree.getroot().findall("edge"):
        if edge.get("function") == "internal":
            continue
        edge_id = edge.get("id")
        if not edge_id:
            continue
        # netconvert may emit `origId` as a <param> child *or* as an
        # attribute on the edge itself, *or* on the lane element only.
        orig = edge.get("origId")
        if orig is None:
            param = edge.find("param[@key='origId']")
            if param is not None:
                orig = param.get("value")
        if orig is None:
            lane = edge.find("lane")
            if lane is not None:
                orig = lane.get("origId")
                if orig is None:
                    lane_param = lane.find("param[@key='origId']")
                    if lane_param is not None:
                        orig = lane_param.get("value")
        if not orig:
            continue
        for token in str(orig).split():
            try:
                way_id = int(token)
            except ValueError:
                continue
            lookup.setdefault(way_id, []).append(edge_id)
    return lookup


# ---------------------------------------------------------------------------
# SUMO edge geometries (WGS84)
# ---------------------------------------------------------------------------


def load_sumo_edge_geometries(net_path: Path) -> gpd.GeoDataFrame:
    """Parse SUMO edge shapes back into WGS84 LineStrings.

    netconvert writes edge shapes in the projected CRS (UTM in our
    case, see ``--proj.utm`` in :mod:`scripts.11_export_sumo`). We
    recover lon/lat by undoing the recorded ``netOffset`` and
    transforming through the ``projParameter``.

    Returns a GeoDataFrame with columns ``edge_id`` and ``geometry``
    (LineString in EPSG:4326).
    """
    if not net_path.exists():
        return gpd.GeoDataFrame(
            columns=["edge_id", "geometry"], geometry="geometry",
            crs="EPSG:4326",
        )
    tree = ET.parse(net_path)
    root = tree.getroot()
    location = root.find("location")
    proj_param = location.get("projParameter") if location is not None else ""
    net_offset = location.get("netOffset") if location is not None else None

    transformer: Transformer | None = None
    if proj_param and "+proj=utm" in proj_param:
        try:
            from pyproj import CRS

            target_crs = CRS.from_proj4(proj_param)
            transformer = Transformer.from_crs(
                target_crs, "EPSG:4326", always_xy=True,
            )
        except Exception as exc:
            logger.warning("Could not build inverse transformer (%s); "
                           "falling back to raw projected coordinates", exc)

    offset_x, offset_y = 0.0, 0.0
    if net_offset:
        try:
            offset_x, offset_y = (float(v) for v in net_offset.split(","))
        except ValueError:
            pass

    records: list[dict] = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        edge_id = edge.get("id")
        if not edge_id:
            continue
        lane = edge.find("lane")
        if lane is None or not lane.get("shape"):
            continue
        try:
            pts = []
            for pair in lane.get("shape").split():
                x, y = (float(v) for v in pair.split(","))
                x -= offset_x
                y -= offset_y
                if transformer is not None:
                    lon, lat = transformer.transform(x, y)
                    pts.append((lon, lat))
                else:
                    pts.append((x, y))
            if len(pts) >= 2:
                records.append(
                    {"edge_id": edge_id, "geometry": LineString(pts)}
                )
        except Exception:
            continue
    if not records:
        return gpd.GeoDataFrame(
            columns=["edge_id", "geometry"], geometry="geometry",
            crs="EPSG:4326" if transformer is not None else None,
        )
    return gpd.GeoDataFrame(
        records, geometry="geometry",
        crs="EPSG:4326" if transformer is not None else None,
    )


def _net_inverse_transformer(root) -> tuple[object, float, float]:
    """Return ``(transformer, offset_x, offset_y)`` to undo the net projection.

    Shared by the edge- and junction-geometry readers so both recover
    WGS84 the same way. ``transformer`` is ``None`` when the net has no
    UTM projection (then coordinates are returned raw).
    """
    location = root.find("location")
    proj_param = location.get("projParameter") if location is not None else ""
    net_offset = location.get("netOffset") if location is not None else None

    transformer = None
    if proj_param and "+proj=utm" in proj_param:
        try:
            from pyproj import CRS

            target_crs = CRS.from_proj4(proj_param)
            transformer = Transformer.from_crs(
                target_crs, "EPSG:4326", always_xy=True,
            )
        except Exception as exc:
            logger.warning("Could not build inverse transformer (%s)", exc)

    offset_x, offset_y = 0.0, 0.0
    if net_offset:
        try:
            offset_x, offset_y = (float(v) for v in net_offset.split(","))
        except ValueError:
            pass
    return transformer, offset_x, offset_y


def load_sumo_junction_coords(net_path: Path) -> dict[str, tuple[float, float]]:
    """Return ``{junction_id: (lon, lat)}`` for every real junction.

    netconvert trims each edge's lane shape back from the junction box,
    which leaves a small visual gap at every intersection when the edges
    are drawn directly. Snapping an edge's endpoints to its ``from``/``to``
    junction centre closes those gaps. Internal junctions (``type
    internal``) are skipped — only the real nodes edges reference are
    returned.
    """
    coords: dict[str, tuple[float, float]] = {}
    if not net_path.exists():
        return coords
    try:
        root = ET.parse(net_path).getroot()
    except ET.ParseError:
        return coords
    transformer, offset_x, offset_y = _net_inverse_transformer(root)
    for j in root.findall("junction"):
        if j.get("function") == "internal" or j.get("type") == "internal":
            continue
        jid = j.get("id")
        if not jid or j.get("x") is None or j.get("y") is None:
            continue
        try:
            x = float(j.get("x")) - offset_x
            y = float(j.get("y")) - offset_y
        except (TypeError, ValueError):
            continue
        if transformer is not None:
            lon, lat = transformer.transform(x, y)
            coords[jid] = (lon, lat)
        else:
            coords[jid] = (x, y)
    return coords


def load_sumo_edge_junctions(net_path: Path) -> dict[str, tuple[str, str]]:
    """Return ``{edge_id: (from_junction, to_junction)}`` for real edges."""
    out: dict[str, tuple[str, str]] = {}
    if not net_path.exists():
        return out
    try:
        root = ET.parse(net_path).getroot()
    except ET.ParseError:
        return out
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        eid = e.get("id")
        f, t = e.get("from"), e.get("to")
        if eid and f and t:
            out[eid] = (f, t)
    return out


# ---------------------------------------------------------------------------
# Spatial fallback for stale OSM ids
# ---------------------------------------------------------------------------


def spatial_resolve_zones(
    osm_lookup: dict[int, list[str]],
    zones: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    max_distance_m: float = 300.0,
) -> dict[int, list[str]]:
    """Augment the OSM-id lookup with nearest-edge matches for unmapped zones.

    StreetLight exports occasionally cite OSM way ids from a snapshot
    older than the one ``netconvert`` consumed. For zones that didn't
    resolve via the direct id lookup we fall back to a nearest-edge
    spatial join, keyed on the zone's geometry centroid.

    Parameters
    ----------
    osm_lookup
        Existing ``{osm_way_id: [edge_ids]}`` from
        :func:`load_osm_to_sumo_lookup`.
    zones
        GeoDataFrame with ``osm_way_id`` and a geometry column. Both
        line and polygon geometries are accepted; we use the centroid.
    edges
        GeoDataFrame from :func:`load_sumo_edge_geometries`.
    max_distance_m
        Reject matches further than this distance (metres). Zones in
        the StreetLight exports are tens of metres long, so anything
        further than ~300 m is almost certainly noise.

    Returns
    -------
    dict[int, list[str]]
        New lookup with the spatial matches merged in. The original
        ``osm_lookup`` is *not* mutated.
    """
    if zones.empty or edges.empty or edges.crs is None:
        return dict(osm_lookup)

    if "osm_way_id" not in zones.columns:
        return dict(osm_lookup)

    # Reproject both to a metric CRS for accurate nearest-neighbour
    # distance. EPSG:3857 (Web Mercator) is fine for ~50 km extents.
    zones_m = zones.to_crs(3857)
    edges_m = edges.to_crs(3857)

    augmented = {k: list(v) for k, v in osm_lookup.items()}
    for _, zrow in zones_m.iterrows():
        way = zrow.get("osm_way_id")
        if pd.isna(way):
            continue
        way_int = int(way)
        if augmented.get(way_int):
            continue
        zgeom = zrow.geometry
        if zgeom is None or zgeom.is_empty:
            continue
        target = zgeom.centroid
        dists = edges_m.geometry.distance(target)
        if dists.empty:
            continue
        nearest_idx = dists.idxmin()
        nearest_dist = dists.loc[nearest_idx]
        if nearest_dist > max_distance_m:
            continue
        nearest_id = edges_m.loc[nearest_idx, "edge_id"]
        augmented[way_int] = [nearest_id]
    return augmented


# ---------------------------------------------------------------------------
# meta.csv sidecar (sumo_edge_id ↔ osm_way_id ↔ street_name)
# ---------------------------------------------------------------------------


def load_meta_lookup(meta_csv_path: Path) -> pd.DataFrame:
    """Read ``leonia.edgedata.meta.csv`` and return it as a DataFrame.

    Columns: ``sumo_edge_id``, ``osm_way_id``, ``street_name``,
    ``avg_volume_per_day``, ``avg_speed_mph``, ``speed_limit_mph``.

    Empty DataFrame if the file is missing — callers should treat
    that as "no measured-edge metadata available."
    """
    if not meta_csv_path.exists():
        return pd.DataFrame(
            columns=["sumo_edge_id", "osm_way_id", "street_name",
                     "avg_volume_per_day", "avg_speed_mph",
                     "speed_limit_mph"]
        )
    return pd.read_csv(meta_csv_path)


def edges_for_osm_ways(
    osm_way_ids: Iterable[int],
    osm_lookup: dict[int, list[str]],
) -> list[str]:
    """Flatten a list of OSM way ids to a deduplicated list of SUMO edge ids."""
    seen: dict[str, None] = {}
    for way in osm_way_ids:
        try:
            way_int = int(way)
        except (TypeError, ValueError):
            continue
        for eid in osm_lookup.get(way_int, []):
            if eid not in seen:
                seen[eid] = None
    return list(seen.keys())
