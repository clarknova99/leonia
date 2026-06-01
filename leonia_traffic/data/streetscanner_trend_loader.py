"""Loader for the StreetLight Street Scanner **Trend** export.

A "Trend" Street Scanner export has the same folder shape as the daily
``streetscanner`` exports (a ``Filters.txt``, one ``*_streetscanner_*.csv``
and a matching shapefile), but the CSV layout is **wide**: one column per
calendar month plus a final ``Change`` column.

This loader melts the wide layout into a tidy long-format frame keyed by
``(zone_name, day_type, year_month)`` so downstream code can detect
trend acceleration, year-over-year growth, and seasonality on every
monitored Leonia street.

The raw folder lives at ``streetlight/streetscanner_trend/``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import STREETLIGHT_DIR
from leonia_traffic.data.streetlight_loader import (
    parse_zone_name,
)

logger = logging.getLogger(__name__)


STREETSCANNER_TREND_DIR = STREETLIGHT_DIR / "streetscanner_trend"


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreetScannerTrendPaths:
    folder: Path
    csv_path: Path
    shp_path: Path
    filters_path: Path


def discover_streetscanner_trend(
    folder: Path = STREETSCANNER_TREND_DIR,
) -> StreetScannerTrendPaths | None:
    """Locate the trend CSV + shapefile inside *folder*."""
    if not folder.exists():
        return None
    filters = folder / "Filters.txt"
    csvs = sorted(folder.glob("*_streetscanner_*.csv"))
    shps = sorted(folder.glob("*_streetscanner_*.shp"))
    if not filters.exists() or not csvs or not shps:
        return None
    return StreetScannerTrendPaths(
        folder=folder,
        csv_path=csvs[0],
        shp_path=shps[0],
        filters_path=filters,
    )


# ---------------------------------------------------------------------------
# Tidy loader
# ---------------------------------------------------------------------------


_MONTH_HEADER = re.compile(
    r"^(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r",\s*(?P<year>\d{4})\s*$",
    re.IGNORECASE,
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)}


def _parse_month_col(col: str) -> date | None:
    m = _MONTH_HEADER.match(col)
    if not m:
        return None
    return date(int(m.group("year")), _MONTHS[m.group("mon").title()], 1)


def load_streetscanner_trend(
    folder: Path = STREETSCANNER_TREND_DIR,
) -> pd.DataFrame:
    """Return one row per ``(zone_name, day_type, year_month)``.

    Output columns:

    * ``zone_name`` (raw), ``osm_name``, ``osm_way_id`` (Int64),
      ``split_index`` (Int64) — parsed from the StreetLight zone-name
      format ``"<name> / <osm id> / <split #>"``.
    * ``city``, ``county``, ``state`` — split out of the StreetLight
      ``City, County, State`` field.
    * ``road_class``, ``road_name`` — raw passthrough fields.
    * ``zone_direction_deg`` (float), ``zone_bidi`` (bool).
    * ``day_type``, ``day_part_raw`` — passthrough strings.
    * ``year_month`` (date, first of month), ``year`` (Int64),
      ``month`` (Int64).
    * ``avg_volume`` — average daily vehicle volume that month.

    The wide ``Change`` column is dropped — it can be derived trivially
    and conflicts with the long-format design.
    """
    paths = discover_streetscanner_trend(folder)
    if paths is None:
        return pd.DataFrame(columns=[
            "zone_name", "osm_name", "osm_way_id", "split_index",
            "city", "county", "state", "road_class", "road_name",
            "zone_direction_deg", "zone_bidi",
            "day_type", "day_part_raw",
            "year_month", "year", "month", "avg_volume",
        ])

    df = pd.read_csv(paths.csv_path)
    month_cols = [c for c in df.columns if _parse_month_col(c) is not None]
    id_cols = [c for c in df.columns if c not in month_cols and c != "Change"]

    long = df.melt(
        id_vars=id_cols,
        value_vars=month_cols,
        var_name="month_col",
        value_name="avg_volume",
    )
    long["year_month"] = pd.to_datetime(
        long["month_col"].map(_parse_month_col), errors="coerce",
    )
    long["year"] = long["year_month"].dt.year.astype("Int64")
    long["month"] = long["year_month"].dt.month.astype("Int64")
    long["avg_volume"] = pd.to_numeric(long["avg_volume"], errors="coerce")

    # City, County, State split.
    if "City, County, State" in long.columns:
        ccs = long["City, County, State"].astype(str).str.split(",", n=2, expand=True)
        long["city"] = ccs[0].str.strip()
        long["county"] = ccs[1].str.strip() if ccs.shape[1] > 1 else None
        long["state"] = ccs[2].str.strip() if ccs.shape[1] > 2 else None

    # Zone-name parsing (same convention as the daily exports).
    parsed = long["Zone Name"].apply(parse_zone_name)
    long["osm_name"] = [p[0] for p in parsed]
    long["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")
    long["split_index"] = pd.array([p[2] for p in parsed], dtype="Int64")

    # Direction / bi-directionality.
    if "Zone Direction" in long.columns:
        long["zone_direction_deg"] = pd.to_numeric(
            long["Zone Direction"], errors="coerce",
        )
    if "Zone is Bi-Direction" in long.columns:
        long["zone_bidi"] = (
            long["Zone is Bi-Direction"].astype(str).str.lower() == "true"
        )

    renames = {
        "Zone Name": "zone_name",
        "Road Class": "road_class",
        "Road Name": "road_name",
        "Day Type": "day_type",
        "Day Part": "day_part_raw",
    }
    for src, dst in renames.items():
        if src in long.columns:
            long = long.rename(columns={src: dst})

    drop = [c for c in ("Mode of Travel", "City, County, State",
                         "Zone Direction", "Zone is Bi-Direction",
                         "month_col") if c in long.columns]
    long = long.drop(columns=drop)

    return long.reset_index(drop=True)


def load_streetscanner_trend_shapes(
    folder: Path = STREETSCANNER_TREND_DIR,
) -> gpd.GeoDataFrame:
    """Return the line geometries that accompany the trend CSV."""
    paths = discover_streetscanner_trend(folder)
    if paths is None:
        return gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")
    gdf = gpd.read_file(paths.shp_path)
    if "name" not in gdf.columns:
        for alt in ("zone_name", "Zone Name", "Name", "ZONE_NAME"):
            if alt in gdf.columns:
                gdf = gdf.rename(columns={alt: "name"})
                break
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


__all__ = [
    "STREETSCANNER_TREND_DIR",
    "StreetScannerTrendPaths",
    "discover_streetscanner_trend",
    "load_streetscanner_trend",
    "load_streetscanner_trend_shapes",
]
