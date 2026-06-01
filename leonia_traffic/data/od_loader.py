"""Generic Origin-Destination loader.

The original stub assumed a generic O-D schema. With the Bridge-destination
export landed (``streetlight/bridge_destination/``), the real schema is
implemented in :mod:`leonia_traffic.data.bridge_od_loader`. This module
keeps the historical public surface (``discover_od_export``,
``load_od_matrix``, ``load_od_zones``, ``classify_zone_category``) as a
thin compatibility shim so existing scripts continue to import from here,
but the heavy lifting now lives in the bridge module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import STREETLIGHT_DIR
from leonia_traffic.data.bridge_od_loader import (
    BRIDGE_OD_DIR,
    discover_bridge_od,
    load_bridge_od,
    load_bridge_zone_shapes,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ODExportPaths:
    folder: Path
    od_csv: Path | None
    zone_shp: Path | None
    filters_txt: Path | None


def discover_od_export(root: Path = BRIDGE_OD_DIR) -> ODExportPaths | None:
    """Compatibility shim. Prefers the bridge export when present.

    The legacy folder ``streetlight/od/`` is also accepted; if both
    exist the bridge export wins because it has the richer schema.
    """
    bridge = discover_bridge_od(BRIDGE_OD_DIR)
    if bridge is not None:
        return ODExportPaths(
            folder=bridge.folder,
            od_csv=bridge.od_all,
            zone_shp=None,
            filters_txt=None,
        )

    if not root.exists():
        return None
    csvs = (
        sorted(root.glob("*od_all*.csv"))
        + sorted(root.glob("*origin_destination*.csv"))
        + sorted(root.glob("*od*.csv"))
    )
    shps = sorted(root.glob("*zone*.shp")) + sorted(root.glob("*.shp"))
    filters = root / "Filters.txt"
    if not csvs:
        return None
    return ODExportPaths(
        folder=root,
        od_csv=csvs[0],
        zone_shp=shps[0] if shps else None,
        filters_txt=filters if filters.exists() else None,
    )


def load_od_matrix(folder: Path | None = None) -> pd.DataFrame:
    """Return the bridge OD matrix in the long format used elsewhere.

    For backward compatibility we also provide simple aliases:
    ``origin_zone``, ``destination_zone``, ``day_type``, ``day_part``,
    ``avg_daily_od_volume``, ``avg_trip_duration_sec``.
    """
    df = load_bridge_od(folder or BRIDGE_OD_DIR)
    if df.empty:
        return df
    aliases = pd.DataFrame({
        "day_type": df["day_type_label"],
        "day_part": df["day_part_label"],
        "avg_daily_od_volume": df["od_volume"],
        "avg_trip_duration_sec": df["avg_travel_time_sec"],
    })
    out = pd.concat([df.reset_index(drop=True), aliases.reset_index(drop=True)], axis=1)
    return out


def load_od_zones(folder: Path | None = None) -> gpd.GeoDataFrame | None:
    """Load origin + destination zone shapefiles concatenated."""
    gdf = load_bridge_zone_shapes(folder or BRIDGE_OD_DIR)
    if gdf is None or gdf.empty:
        return None
    return gdf


def classify_zone_category(zone_name: str) -> str:
    """Best-effort category extraction from an OD zone name.

    The bridge OD export uses human-readable names (e.g.
    ``"Fort Lee Road / 590576"``) rather than category-prefixed strings,
    so this helper is now an informational classifier that returns
    ``"GWB"`` for destination zones containing ``"George Washington
    Bridge"``, the trimmed street name for origin gates, and
    ``"OTHER"`` otherwise.
    """
    if not zone_name:
        return "OTHER"
    s = str(zone_name)
    if "George Washington Bridge" in s:
        return "GWB"
    parts = s.split("_")
    if len(parts) >= 2 and parts[0].upper() in {"DEST", "IN", "ORIG"}:
        return parts[1].upper()
    if " / " in s:
        return s.split(" / ", 1)[0].strip().upper()
    return "OTHER"


__all__ = [
    "ODExportPaths",
    "classify_zone_category",
    "discover_od_export",
    "load_od_matrix",
    "load_od_zones",
]
