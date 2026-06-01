"""Loader for the StreetLight Network Performance export.

The export lives in ``streetlight/2038116_leonia_network/`` (analysis
2038116) and is the project's most granular segment-level traffic
product. Unlike Congestion Trends (arterials only, 3 day types, 30
mixed day-parts) it covers **every selected OSM segment** — arterials,
the GWB approach, *and* residential blocks — at:

* **8 day types** — ``All Days`` plus each weekday Monday..Sunday
  (codes 0..7).
* **25 day parts** — ``All Day`` plus each of the 24 clock hours
  (codes 0..24), so true per-hour peak volumes are available.

Files consumed (paths relative to the export folder):

* ``*_network_performance_seg_metrics.csv`` — the main table: one row
  per ``(zone, day type, day part)`` with average daily segment
  traffic, average / free-flow speed, travel time, VMT, VHD, free-flow
  factor, and 5/15/85/95 speed percentiles.
* ``*_network_performance_seg_prediction_interval.csv`` — the 95%
  prediction range (lower/upper) around the All-Day volume per zone ×
  day type × day part. Small; useful for uncertainty bands.
* ``*_network_performance_seg_monthly_metrics.csv`` — the same metrics
  broken out by ``Year-Month`` (Jan 2025 .. Apr 2026). **Large**
  (~560 MB CSV); the build script can skip it.
* ``*_network_performance_seg_monthly_prediction_interval.csv`` — the
  monthly analogue of the prediction-interval file.
* ``Analysis Details/*_zones.csv`` — the zone roster with StreetLight
  fingerprints.
* ``Shapefile/*_segment_line.zip`` — per-segment LineString geometry
  (plus ``*_start_gate.zip`` / ``*_end_gate.zip`` point geometries).

Zone names use the **3-part** OSM Derivative format
``"<name> / <osm way id> / <split #>"`` (e.g.
``"1st Street / 1007650684 / 1"``), which is why this module ships its
own :func:`parse_network_zone_name` instead of reusing
:func:`leonia_traffic.data.bridge_od_loader.parse_bridge_zone_name`
(that 2-part parser would mistake the trailing split index for the OSM
way ID).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import STREETLIGHT_DIR
from leonia_traffic.data.bridge_od_loader import (
    parse_bridge_zone_name,
    parse_coded_value,
)

logger = logging.getLogger(__name__)


NETWORK_PERF_DIR = STREETLIGHT_DIR / "2038116_leonia_network"


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkPerformancePaths:
    folder: Path
    seg_metrics: Path
    seg_prediction: Path | None
    seg_monthly: Path | None
    seg_monthly_prediction: Path | None
    zones: Path | None
    shapefile_segment_line: Path | None
    shapefile_start_gate: Path | None
    shapefile_end_gate: Path | None


def _first(folder: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]
    return None


def discover_network_performance(
    folder: Path = NETWORK_PERF_DIR,
) -> NetworkPerformancePaths | None:
    """Find the Network Performance export under ``folder``.

    Returns ``None`` if no ``*_seg_metrics.csv`` is present. Optional
    files (prediction intervals, monthly metrics, zone roster,
    shapefiles) are ``None`` when missing.
    """
    if not folder.exists():
        return None
    seg_metrics = _first(folder, "*_network_performance_seg_metrics.csv")
    if seg_metrics is None:
        return None

    details_dir = folder / "Analysis Details"
    shp_dir = folder / "Shapefile"

    return NetworkPerformancePaths(
        folder=folder,
        seg_metrics=seg_metrics,
        seg_prediction=_first(
            folder, "*_network_performance_seg_prediction_interval.csv"
        ),
        seg_monthly=_first(
            folder, "*_network_performance_seg_monthly_metrics.csv"
        ),
        seg_monthly_prediction=_first(
            folder, "*_network_performance_seg_monthly_prediction_interval.csv"
        ),
        zones=_first(details_dir, "*_zones.csv") if details_dir.exists() else None,
        shapefile_segment_line=(
            _first(shp_dir, "*_segment_line.zip") if shp_dir.exists() else None
        ),
        shapefile_start_gate=(
            _first(shp_dir, "*_start_gate.zip") if shp_dir.exists() else None
        ),
        shapefile_end_gate=(
            _first(shp_dir, "*_end_gate.zip") if shp_dir.exists() else None
        ),
    )


# ---------------------------------------------------------------------------
# Zone-name parsing (3-part OSM Derivative format)
# ---------------------------------------------------------------------------


# "<name> / <osm way id> / <split #>" — capture name, way id, split.
_NETWORK_ZONE_RE = re.compile(r"^(.*?)\s*/\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_network_zone_name(zone_name: str) -> tuple[str, int | None, int | None]:
    """Return ``(street_name, osm_way_id, split_num)`` from a zone name.

    Network Performance zones use the OSM Derivative 3-part format
    ``"<name> / <osm way id> / <split #>"``. Falls back to the 2-part
    parser (``"<name> / <osm way id>"``) when the trailing split index
    is absent, returning ``split_num=None`` in that case.
    """
    if zone_name is None:
        return ("", None, None)
    s = str(zone_name)
    m = _NETWORK_ZONE_RE.match(s)
    if m:
        return (m.group(1).strip(), int(m.group(2)), int(m.group(3)))
    # 2-part fallback ("<name> / <osm id>").
    label, osm_way_id = parse_bridge_zone_name(s)
    return (label, osm_way_id, None)


def _parse_zone_columns(df: pd.DataFrame) -> None:
    """Add ``street_name`` / ``osm_way_id`` / ``split_num`` from ``zone_name``."""
    parsed = df["zone_name"].apply(parse_network_zone_name)
    df["street_name"] = [p[0] for p in parsed]
    df["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")
    df["split_num"] = pd.array([p[2] for p in parsed], dtype="Int64")


def _parse_day_columns(df: pd.DataFrame) -> None:
    if "day_type_raw" in df.columns:
        dt = df["day_type_raw"].apply(parse_coded_value)
        df["day_type_code"] = pd.array([p[0] for p in dt], dtype="Int64")
        df["day_type_label"] = [p[1] for p in dt]
    if "day_part_raw" in df.columns:
        dp = df["day_part_raw"].apply(parse_coded_value)
        df["day_part_code"] = pd.array([p[0] for p in dp], dtype="Int64")
        df["day_part_label"] = [p[1] for p in dp]


def _coerce_numeric(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# ---------------------------------------------------------------------------
# Main segment-metrics loader
# ---------------------------------------------------------------------------


_SEG_RENAMES = {
    "Data Periods": "data_periods",
    "Mode of Travel": "mode_of_travel",
    "Vehicle Weight": "vehicle_weight",
    "Zone ID": "zone_id",
    "Zone Name": "zone_name",
    "Line Zone Length (Miles)": "length_mi",
    "Zone Is Pass-Through": "is_pass_through",
    "Zone Direction (degrees)": "direction_deg",
    "Zone is Bi-Direction": "is_bidi",
    "Year-Month": "year_month",
    "Day Type": "day_type_raw",
    "Day Part": "day_part_raw",
    "Inferred Volume": "inferred_volume",
    "Average Daily Segment Traffic (StL Volume)": "avg_daily_volume",
    "Avg Segment Speed (mph)": "avg_speed_mph",
    "Avg Segment Travel Time (sec)": "avg_travel_time_sec",
    "Free Flow Speed (mph)": "free_flow_speed_mph",
    "Free Flow Factor": "free_flow_factor",
    "Vehicle Miles of Travel (StL Volume)": "vmt",
    "Vehicle Hours of Delay (StL Volume)": "vhd",
    "5th Speed Percentile": "speed_p05",
    "15th Speed Percentile": "speed_p15",
    "85th Speed Percentile": "speed_p85",
    "95th Speed Percentile": "speed_p95",
}


_SEG_NUMERIC = (
    "length_mi", "direction_deg",
    "avg_daily_volume", "avg_speed_mph", "avg_travel_time_sec",
    "free_flow_speed_mph", "free_flow_factor", "vmt", "vhd",
    "speed_p05", "speed_p15", "speed_p85", "speed_p95",
)


def _load_seg_like(path: Path) -> pd.DataFrame:
    """Shared loader for the seg-metrics / monthly-metrics CSVs."""
    df = pd.read_csv(path)
    df = df.rename(columns=_SEG_RENAMES)
    _coerce_numeric(df, _SEG_NUMERIC)
    # ``Free Flow Factor`` is the StreetLight free-flow ratio; congestion
    # is its complement. Derive it so callers can map directly.
    if "free_flow_factor" in df.columns:
        df["congestion"] = 1.0 - df["free_flow_factor"]
    _parse_zone_columns(df)
    _parse_day_columns(df)
    return df


def load_network_performance(folder: Path = NETWORK_PERF_DIR) -> pd.DataFrame:
    """Load the main Network Performance segment-metrics table.

    Returns a long-format DataFrame keyed by ``(zone_name, day_type_code,
    day_part_code)`` with parsed ``street_name`` / ``osm_way_id`` /
    ``split_num`` and snake_case metric columns. Empty frame if the
    export is missing.
    """
    paths = discover_network_performance(folder)
    if paths is None:
        return pd.DataFrame()
    return _load_seg_like(paths.seg_metrics)


def load_network_performance_monthly(
    folder: Path = NETWORK_PERF_DIR,
) -> pd.DataFrame:
    """Load the per-``Year-Month`` segment-metrics table.

    This is the large file (~560 MB CSV). Same schema as
    :func:`load_network_performance` plus a ``year_month`` column.
    """
    paths = discover_network_performance(folder)
    if paths is None or paths.seg_monthly is None:
        return pd.DataFrame()
    return _load_seg_like(paths.seg_monthly)


# ---------------------------------------------------------------------------
# Prediction-interval loaders
# ---------------------------------------------------------------------------


_PRED_RENAMES = {
    "Data Periods": "data_periods",
    "Mode of Travel": "mode_of_travel",
    "Zone ID": "zone_id",
    "Zone Name": "zone_name",
    "Zone Is Pass-Through": "is_pass_through",
    "Zone Direction (degrees)": "direction_deg",
    "Zone is Bi-Direction": "is_bidi",
    "Year-Month": "year_month",
    "Day Type": "day_type_raw",
    "Day Part": "day_part_raw",
    "Average Daily Zone Traffic (StL Volume)": "avg_daily_volume",
    "Lower 95 Prediction Range": "pred_lower_95",
    "Upper 95 Prediction Range": "pred_upper_95",
}

_PRED_NUMERIC = ("direction_deg", "avg_daily_volume", "pred_lower_95", "pred_upper_95")


def _load_pred_like(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=_PRED_RENAMES)
    _coerce_numeric(df, _PRED_NUMERIC)
    _parse_zone_columns(df)
    _parse_day_columns(df)
    return df


def load_network_performance_prediction(
    folder: Path = NETWORK_PERF_DIR,
) -> pd.DataFrame:
    """Load the 95% prediction-interval table (per zone × day type × day part)."""
    paths = discover_network_performance(folder)
    if paths is None or paths.seg_prediction is None:
        return pd.DataFrame()
    return _load_pred_like(paths.seg_prediction)


def load_network_performance_monthly_prediction(
    folder: Path = NETWORK_PERF_DIR,
) -> pd.DataFrame:
    """Load the monthly 95% prediction-interval table (adds ``year_month``)."""
    paths = discover_network_performance(folder)
    if paths is None or paths.seg_monthly_prediction is None:
        return pd.DataFrame()
    return _load_pred_like(paths.seg_monthly_prediction)


# ---------------------------------------------------------------------------
# Zone roster
# ---------------------------------------------------------------------------


_ZONES_RENAMES = {
    "Data Periods": "data_periods",
    "Zone Type": "zone_type",
    "Zone ID": "zone_id",
    "Zone Name": "zone_name",
    "Line Zone Length (Miles)": "length_mi",
    "Zone Is Pass-Through": "is_pass_through",
    "Zone Direction (degrees)": "direction_deg",
    "Zone is Bi-Direction": "is_bidi",
    "Fingerprint1": "fingerprint1",
    "Fingerprint2": "fingerprint2",
}


def load_network_performance_zones(folder: Path = NETWORK_PERF_DIR) -> pd.DataFrame:
    """Load the zone roster (one row per zone, with StreetLight fingerprints)."""
    paths = discover_network_performance(folder)
    if paths is None or paths.zones is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.zones)
    df = df.rename(columns=_ZONES_RENAMES)
    _coerce_numeric(df, ("length_mi", "direction_deg"))
    _parse_zone_columns(df)
    return df


# ---------------------------------------------------------------------------
# Shapefile loaders
# ---------------------------------------------------------------------------


def _empty_geo() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["name", "geometry"], geometry="geometry", crs="EPSG:4326")


def _load_shape(zip_path: Path | None) -> gpd.GeoDataFrame:
    if zip_path is None:
        return _empty_geo()
    gdf = gpd.read_file(f"zip://{zip_path}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if "name" in gdf.columns:
        parsed = gdf["name"].apply(parse_network_zone_name)
        gdf["street_name"] = [p[0] for p in parsed]
        gdf["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")
        gdf["split_num"] = pd.array([p[2] for p in parsed], dtype="Int64")
    return gdf


def load_network_performance_shapes(
    folder: Path = NETWORK_PERF_DIR,
) -> gpd.GeoDataFrame:
    """Load the per-segment LineString geometries.

    Returns a GeoDataFrame with ``name``, parsed ``street_name`` /
    ``osm_way_id`` / ``split_num``, ``road_type``, ``direction``, and
    ``geometry``. Empty GeoDataFrame if the shapefile is missing.
    """
    paths = discover_network_performance(folder)
    return _load_shape(paths.shapefile_segment_line if paths else None)


def load_network_performance_gate_shapes(
    folder: Path = NETWORK_PERF_DIR, *, kind: str = "start",
) -> gpd.GeoDataFrame:
    """Load the start- or end-gate point geometries (``kind="start"|"end"``)."""
    paths = discover_network_performance(folder)
    if paths is None:
        return _empty_geo()
    zip_path = (
        paths.shapefile_start_gate if kind == "start" else paths.shapefile_end_gate
    )
    return _load_shape(zip_path)


# ---------------------------------------------------------------------------
# Per-segment summary
# ---------------------------------------------------------------------------


def peak_hour_volume(
    df: pd.DataFrame, *, day_type_code: int = 0,
) -> pd.DataFrame:
    """One-row-per-segment peak-hour summary from the long metrics table.

    Aggregates over the 24 clock-hour day parts (codes 1..24) for the
    requested ``day_type_code`` (default ``0`` = All Days):

    * ``peak_volume`` / ``peak_hour_code`` — the maximum hourly volume
      and the day-part code at which it occurs.
    * ``peak_am_volume`` / ``peak_pm_volume`` — max hourly volume in the
      7-10am (codes 8..10) and 4-7pm (codes 17..19) windows.
    * ``all_day_volume`` — the All-Day (code 0) average daily volume.
    * ``free_flow_speed_mph`` — observed free-flow speed (per segment).
    * ``min_speed_p15`` — minimum 15th-percentile speed across hours, a
      proxy for the worst-case congested speed.
    """
    if df.empty:
        return pd.DataFrame()

    keys = ["zone_name", "osm_way_id", "street_name"]
    sub = df[df["day_type_code"] == day_type_code]
    hourly = sub[(sub["day_part_code"] >= 1) & (sub["day_part_code"] <= 24)].copy()
    if hourly.empty:
        return pd.DataFrame()

    def _peak_row(g: pd.DataFrame) -> pd.Series:
        vols = g["avg_daily_volume"]
        if vols.notna().any():
            idx = vols.idxmax()
            peak_volume = g.loc[idx, "avg_daily_volume"]
            peak_hour_code = g.loc[idx, "day_part_code"]
        else:
            peak_volume = float("nan")
            peak_hour_code = pd.NA
        am = g[g["day_part_code"].between(8, 10)]["avg_daily_volume"]
        pm = g[g["day_part_code"].between(17, 19)]["avg_daily_volume"]
        return pd.Series({
            "peak_volume": peak_volume,
            "peak_hour_code": peak_hour_code,
            "peak_am_volume": am.max() if not am.empty else float("nan"),
            "peak_pm_volume": pm.max() if not pm.empty else float("nan"),
            "free_flow_speed_mph": g["free_flow_speed_mph"].dropna().max(),
            "min_speed_p15": g["speed_p15"].min(),
            "length_mi": g["length_mi"].iloc[0],
        })

    summary = hourly.groupby(keys, dropna=False).apply(
        _peak_row, include_groups=False
    ).reset_index()

    all_day = (
        sub[sub["day_part_code"] == 0]
        .groupby(keys, dropna=False)["avg_daily_volume"]
        .first()
        .reset_index()
        .rename(columns={"avg_daily_volume": "all_day_volume"})
    )
    summary = summary.merge(all_day, on=keys, how="left")
    return summary.sort_values("peak_volume", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Canonical cache loaders
# ---------------------------------------------------------------------------


def _try_canonical(name: str, *, geo: bool = False):
    try:
        from leonia_traffic.data.dataset_io import CANONICAL_DIR, dataset_exists
        if not dataset_exists(CANONICAL_DIR, name):
            return None
        if geo:
            return gpd.read_parquet(CANONICAL_DIR / name)
        return pd.read_parquet(CANONICAL_DIR / name)
    except Exception:
        return None


def load_network_performance_cached(
    folder: Path = NETWORK_PERF_DIR,
) -> pd.DataFrame:
    """Canonical-first loader for the main Network Performance table."""
    cached = _try_canonical("network_performance_segments.parquet")
    return cached if cached is not None else load_network_performance(folder)


def load_network_performance_prediction_cached(
    folder: Path = NETWORK_PERF_DIR,
) -> pd.DataFrame:
    cached = _try_canonical("network_performance_prediction.parquet")
    return cached if cached is not None else load_network_performance_prediction(folder)


def load_network_performance_monthly_cached(
    folder: Path = NETWORK_PERF_DIR,
) -> pd.DataFrame:
    """Canonical-first loader for the large monthly table."""
    cached = _try_canonical("network_performance_monthly.parquet")
    return cached if cached is not None else load_network_performance_monthly(folder)


def load_network_performance_shapes_cached(
    folder: Path = NETWORK_PERF_DIR,
) -> gpd.GeoDataFrame:
    cached = _try_canonical("network_performance_shapes.parquet", geo=True)
    return cached if cached is not None else load_network_performance_shapes(folder)


__all__ = [
    "NETWORK_PERF_DIR",
    "NetworkPerformancePaths",
    "discover_network_performance",
    "parse_network_zone_name",
    "load_network_performance",
    "load_network_performance_cached",
    "load_network_performance_monthly",
    "load_network_performance_monthly_cached",
    "load_network_performance_prediction",
    "load_network_performance_prediction_cached",
    "load_network_performance_monthly_prediction",
    "load_network_performance_zones",
    "load_network_performance_shapes",
    "load_network_performance_shapes_cached",
    "load_network_performance_gate_shapes",
    "peak_hour_volume",
]
