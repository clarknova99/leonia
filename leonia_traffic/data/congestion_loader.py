"""Loader for the StreetLight Congestion Trends export.

The export lives in ``streetlight/congestion/`` and contains:

* ``*_ct_all.csv`` — 12,232 rows of segment-level congestion metrics
  (136 zones × 3 day types × 30 day parts). Day types are ``All Days``,
  ``Weekday``, ``Weekend Day``. Day parts mix hourly buckets (e.g.
  ``"10: 7am (7am-8am)"``) with aggregates (``"All Day"``, ``"Peak AM"``,
  ``"Peak PM"``, ``"Early AM"``, ``"Mid-Day"``, ``"Late PM"``).
* Per-row metrics: ``Travel Time Index``, ``Buffer Index``,
  ``Planning Time Index``, ``Vehicle Hours of Delay``,
  ``Level of Travel Time Reliability``, ``Reliable Segment?``, 80th/90th
  TTI percentiles, average segment speed, free-flow speed, and 5/10/20/50
  speed percentiles.
* ``Shapefile/*_osm_segment.zip`` — per-segment geometries with
  ``segment_id``, ``name``, and ``segment_ty`` (road class).

Coverage skews towards Motorway / Trunk / Primary / Secondary / Tertiary
plus a few ramps. Residential streets are **not** in this export.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import STREETLIGHT_DIR
from leonia_traffic.data.bridge_od_loader import parse_bridge_zone_name, parse_coded_value

logger = logging.getLogger(__name__)


CONGESTION_DIR = STREETLIGHT_DIR / "congestion"


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CongestionPaths:
    folder: Path
    ct_csv: Path
    shapefile_zip: Path | None


def discover_congestion(folder: Path = CONGESTION_DIR) -> CongestionPaths | None:
    """Find the congestion export. Returns ``None`` if missing."""
    if not folder.exists():
        return None
    csvs = sorted(folder.glob("*_ct_all.csv"))
    if not csvs:
        return None
    shape_dir = folder / "Shapefile"
    shp_zips = sorted(shape_dir.glob("*.zip")) if shape_dir.exists() else []
    return CongestionPaths(
        folder=folder,
        ct_csv=csvs[0],
        shapefile_zip=shp_zips[0] if shp_zips else None,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


_CT_RENAMES = {
    "Data Periods": "data_periods",
    "Mode of Travel": "mode_of_travel",
    "Zone ID": "zone_id",
    "Zone Name": "zone_name",
    "Road Class": "road_class",
    "Line Zone Length (Miles)": "length_mi",
    "Zone Is Pass-Through": "is_pass_through",
    "Zone Direction (degrees)": "direction_deg",
    "Zone is Bi-Direction": "is_bidi",
    "Day Type": "day_type_raw",
    "Day Part": "day_part_raw",
    "Average Daily Segment Traffic (StL Volume)": "avg_daily_volume",
    "Avg Segment Speed (mph)": "avg_speed_mph",
    "Avg Segment Travel Time (sec)": "avg_travel_time_sec",
    "Free Flow Speed (mph)": "free_flow_speed_mph",
    "Vehicle Miles of Travel (StL Volume)": "vmt",
    "Vehicle Hours of Delay (StL Volume)": "vhd",
    "Travel Time Index": "tti",
    "Buffer Index": "buffer_index",
    "Planning Time Index": "planning_time_index",
    "Level of Travel Time Reliability": "reliability_level",
    "Reliable Segment?": "is_reliable",
    "80th Travel Time Index": "tti_80",
    "90th Travel Time Index": "tti_90",
    "5th Speed Percentile": "speed_p05",
    "10th Speed Percentile": "speed_p10",
    "20th Speed Percentile": "speed_p20",
    "50th Speed Percentile": "speed_p50",
}


_NUMERIC_COLS = (
    "length_mi", "direction_deg", "avg_daily_volume", "avg_speed_mph",
    "avg_travel_time_sec", "free_flow_speed_mph", "vmt", "vhd", "tti",
    "buffer_index", "planning_time_index", "reliability_level",
    "tti_80", "tti_90",
    "speed_p05", "speed_p10", "speed_p20", "speed_p50",
)


def classify_reliability(level: float) -> str:
    """Bucket FHWA Level of Travel Time Reliability (LOTTR) into labels.

    Standard thresholds: < 1.5 reliable, 1.5–2.0 moderate, >= 2.0
    unreliable. Returns ``"Unknown"`` for NaN.
    """
    if level is None or pd.isna(level):
        return "Unknown"
    if level < 1.5:
        return "Reliable"
    if level < 2.0:
        return "Moderate"
    return "Unreliable"


def load_congestion(folder: Path = CONGESTION_DIR) -> pd.DataFrame:
    """Load the congestion CSV into a long-format DataFrame.

    Returns an empty DataFrame if the export is missing. Otherwise:

    * Renames CSV columns to snake_case.
    * Coerces numeric columns.
    * Parses ``zone_name`` into ``osm_name`` and ``osm_way_id``.
    * Splits the coded ``Day Type`` / ``Day Part`` strings into
      ``day_type_code`` (0–2), ``day_type_label``, ``day_part_code``
      (0–29), and ``day_part_label``.
    """
    paths = discover_congestion(folder)
    if paths is None:
        return pd.DataFrame()

    df = pd.read_csv(paths.ct_csv)
    df = df.rename(columns=_CT_RENAMES)

    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    parsed = df["zone_name"].apply(parse_bridge_zone_name)
    df["osm_name"] = [p[0] for p in parsed]
    df["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")

    dt = df["day_type_raw"].apply(parse_coded_value)
    df["day_type_code"] = pd.array([p[0] for p in dt], dtype="Int64")
    df["day_type_label"] = [p[1] for p in dt]

    dp = df["day_part_raw"].apply(parse_coded_value)
    df["day_part_code"] = pd.array([p[0] for p in dp], dtype="Int64")
    df["day_part_label"] = [p[1] for p in dp]

    return df


def load_congestion_zones(folder: Path = CONGESTION_DIR) -> gpd.GeoDataFrame:
    """Load the congestion segment shapefile.

    Returns an empty GeoDataFrame if the shapefile zip is missing. The
    returned frame carries the StreetLight ``segment_id`` and parsed
    ``osm_way_id``/``osm_name``.
    """
    paths = discover_congestion(folder)
    if paths is None or paths.shapefile_zip is None:
        return gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")

    gdf = gpd.read_file(f"zip://{paths.shapefile_zip}")
    if "name" in gdf.columns:
        parsed = gdf["name"].apply(parse_bridge_zone_name)
        gdf["osm_name"] = [p[0] for p in parsed]
        gdf["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")
    return gdf


# ---------------------------------------------------------------------------
# Per-link summary
# ---------------------------------------------------------------------------


def summarize_link_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-zone summary of congestion metrics.

    Aggregates over hourly day-parts within Weekday (day_type_code=1):

    * ``worst_tti`` — maximum hourly TTI
    * ``worst_buffer`` — maximum hourly Buffer Index
    * ``total_weekday_vhd`` — sum of Vehicle Hours of Delay across hours
    * ``median_speed_mph`` — median of the per-hour 50th speed percentile
    * ``free_flow_speed_mph`` — observed free-flow speed (constant per zone)
    * ``reliability_level`` — the segment's reliability classification (the
      worst label seen across the included hours, broken by ordinal order
      ``Reliable`` < ``Moderate`` < ``Unreliable``)

    Hourly day parts are codes 8–13, 15–19, 22–28 (i.e. one-hour buckets).
    Aggregate day parts (``Peak AM``, ``Peak PM``, ``All Day``, etc.) are
    skipped to avoid double counting.
    """
    if df.empty:
        return pd.DataFrame()

    hourly = df[(df["day_type_code"] == 1)
                & df["day_part_label"].str.match(r"\d{1,2}[ap]m \(", na=False)]

    if hourly.empty:
        # Fall back to the all-day weekday row.
        hourly = df[(df["day_type_code"] == 1) & (df["day_part_code"] == 0)]

    grouped = hourly.groupby(["zone_name", "osm_way_id", "osm_name", "road_class"], dropna=False).agg(
        worst_tti=("tti", "max"),
        worst_buffer=("buffer_index", "max"),
        total_weekday_vhd=("vhd", "sum"),
        median_speed_mph=("speed_p50", "median"),
        free_flow_speed_mph=("free_flow_speed_mph", "first"),
        worst_lottr=("reliability_level", "max"),
        length_mi=("length_mi", "first"),
    ).reset_index()
    grouped["reliability_class"] = grouped["worst_lottr"].apply(classify_reliability)
    return grouped.sort_values("worst_tti", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Canonical cache loaders
# ---------------------------------------------------------------------------


def _try_canonical(name: str, *, geo: bool = False):
    try:
        from leonia_traffic.data.dataset_io import CANONICAL_DIR, dataset_exists
        if not dataset_exists(CANONICAL_DIR, name):
            return None
        if geo:
            import geopandas as gpd
            return gpd.read_parquet(CANONICAL_DIR / name)
        return pd.read_parquet(CANONICAL_DIR / name)
    except Exception:
        return None


def load_congestion_cached(folder: Path = CONGESTION_DIR) -> pd.DataFrame:
    """Canonical-first loader for the Congestion link table."""
    cached = _try_canonical("congestion_links.parquet")
    return cached if cached is not None else load_congestion(folder)


def load_congestion_zones_cached(folder: Path = CONGESTION_DIR):
    """Canonical-first loader for the Congestion zone geometries."""
    cached = _try_canonical("congestion_zones.parquet", geo=True)
    return cached if cached is not None else load_congestion_zones(folder)


__all__ = [
    "CONGESTION_DIR",
    "CongestionPaths",
    "classify_reliability",
    "discover_congestion",
    "load_congestion",
    "load_congestion_cached",
    "load_congestion_zones",
    "load_congestion_zones_cached",
    "summarize_link_reliability",
]
