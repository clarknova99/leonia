"""Loader for the StreetLight Zone Activity export covering Leonia streets.

The export lives in ``streetlight/2034227_leonia_streets/`` and is the
project's first granular pass-through measurement on residential blocks
(OSM Tertiary Segments, Jan 2025). It includes:

* ``*_za_all.csv`` — main Zone Activity table: one row per
  ``(zone, day type, day part, home/work filter)`` with average daily
  pass-through volume and average travel time / trip length.
* ``*_zone_trip_all.csv`` — full per-zone trip-attribute distributions
  (travel time bins, trip length bins, speed bins, circuity bins).
* ``Home Work/`` folder containing several home- and work-location
  cross-tabulations:
    * ``*_home_distance_all.csv``
    * ``*_home_zip_codes_top_all.csv``
    * ``*_home_state_all.csv``
    * ``*_tourist_summary_all.csv``
    * (plus the larger block-group / zip / metro CSVs which we expose
      via the same loader pattern but are too big to merge by default)
* ``Shapefile/*_zone_activity_line.zip`` — line geometries (with
  ``road_type``, ``gate_lat``, ``gate_lon``).

Per the plan we treat **Visitor** rows as the canonical pass-through
signal and the optional **Resident** rows as a comparison baseline.
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


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


ZA_STREETS_DIR = STREETLIGHT_DIR / "2034227_leonia_streets"


@dataclass(frozen=True)
class ZAStreetsPaths:
    folder: Path
    za_all: Path
    zone_trip_all: Path | None
    home_work_dir: Path | None
    home_distance: Path | None
    home_zips_top: Path | None
    home_state: Path | None
    home_metro_area: Path | None
    work_distance: Path | None
    work_block_groups: Path | None
    tourist_summary: Path | None
    shapefile_line_zip: Path | None
    shapefile_poly_zip: Path | None


def _first(folder: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]
    return None


def discover_za_streets(folder: Path = ZA_STREETS_DIR) -> ZAStreetsPaths | None:
    """Find the Leonia-streets ZA export under ``folder``.

    Returns ``None`` if no ``*_za_all.csv`` is present. Otherwise
    returns a populated ``ZAStreetsPaths`` with ``None`` for any
    optional files that are missing.
    """
    if not folder.exists():
        return None
    za_all = _first(folder, "*_za_all.csv")
    if za_all is None:
        return None

    hw_dir = folder / "Home Work"
    if not hw_dir.exists():
        hw_dir = None

    shp_dir = folder / "Shapefile"
    shp_line = _first(shp_dir, "*_zone_activity_line.zip") if shp_dir.exists() else None
    shp_poly = _first(shp_dir, "*_zone_activity.zip") if shp_dir.exists() else None
    # The polygon variant pattern also matches the line zip; explicitly
    # exclude any path containing "_line".
    if shp_poly is not None and "_line" in shp_poly.name:
        shp_poly = None

    return ZAStreetsPaths(
        folder=folder,
        za_all=za_all,
        zone_trip_all=_first(folder, "*_zone_trip_all.csv"),
        home_work_dir=hw_dir,
        home_distance=_first(hw_dir, "*_home_distance_all.csv") if hw_dir else None,
        home_zips_top=_first(hw_dir, "*_home_zip_codes_top_all.csv") if hw_dir else None,
        home_state=_first(hw_dir, "*_home_state_all.csv") if hw_dir else None,
        home_metro_area=_first(hw_dir, "*_home_metro_area_all.csv") if hw_dir else None,
        work_distance=_first(hw_dir, "*_work_distance_all.csv") if hw_dir else None,
        work_block_groups=_first(hw_dir, "*_work_block_groups_all.csv") if hw_dir else None,
        tourist_summary=_first(hw_dir, "*_tourist_summary_all.csv") if hw_dir else None,
        shapefile_line_zip=shp_line,
        shapefile_poly_zip=shp_poly,
    )


# ---------------------------------------------------------------------------
# Column renames + parsing helpers
# ---------------------------------------------------------------------------


_ZA_RENAMES = {
    "Data Periods": "data_periods",
    "Mode of Travel": "mode_of_travel",
    "Home and Work Filter": "filter",
    "Intersection Type": "intersection_type",
    "Zone ID": "zone_id",
    "Zone Name": "zone_name",
    "Zone Is Pass-Through": "pass_through",
    "Zone Direction (degrees)": "direction_deg",
    "Zone is Bi-Direction": "bidi",
    "Day Type": "day_type_raw",
    "Day Part": "day_part_raw",
    "Average Daily Zone Traffic (StL Volume)": "zone_volume",
    "Avg Travel Time (sec)": "avg_travel_time_sec",
    "Avg All Travel Time (sec)": "avg_all_travel_time_sec",
    "Avg Trip Length (mi)": "avg_trip_length_mi",
    "Avg All Trip Length (mi)": "avg_all_trip_length_mi",
    "Avg Trip Speed (mph)": "avg_trip_speed_mph",
    "Avg All Trip Speed (mph)": "avg_all_trip_speed_mph",
}


def _coerce_numeric(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _parse_zone_columns(df: pd.DataFrame) -> None:
    """Add ``street_name`` / ``osm_way_id`` from the ``zone_name`` column."""
    parsed = df["zone_name"].apply(parse_bridge_zone_name)
    df["street_name"] = [p[0] for p in parsed]
    df["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")


def _parse_day_columns(df: pd.DataFrame) -> None:
    """Add ``day_type_code``/``day_type_label`` and parts from the raw columns."""
    if "day_type_raw" in df.columns:
        dt = df["day_type_raw"].apply(parse_coded_value)
        df["day_type_code"] = pd.array([p[0] for p in dt], dtype="Int64")
        df["day_type_label"] = [p[1] for p in dt]
    if "day_part_raw" in df.columns:
        dp = df["day_part_raw"].apply(parse_coded_value)
        df["day_part_code"] = pd.array([p[0] for p in dp], dtype="Int64")
        df["day_part_label"] = [p[1] for p in dp]


# ---------------------------------------------------------------------------
# Main ZA loader
# ---------------------------------------------------------------------------


def _empty_za_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "zone_name", "street_name", "osm_way_id",
        "filter", "intersection_type",
        "day_type_code", "day_type_label",
        "day_part_code", "day_part_label",
        "zone_volume", "avg_travel_time_sec", "avg_trip_length_mi",
    ])


def load_za_main(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load the main Zone Activity table.

    Returns a long-format DataFrame keyed by ``(zone_name, filter,
    day_type_code, day_part_code)`` with parsed street name + OSM way ID
    and the canonical volume / time / length columns. Returns an empty
    frame with the expected columns if the export is missing.
    """
    paths = discover_za_streets(folder)
    if paths is None:
        return _empty_za_frame()
    df = pd.read_csv(paths.za_all)
    df = df.rename(columns=_ZA_RENAMES)
    _coerce_numeric(df, (
        "direction_deg",
        "zone_volume",
        "avg_travel_time_sec",
        "avg_all_travel_time_sec",
        "avg_trip_length_mi",
        "avg_all_trip_length_mi",
    ))
    _parse_zone_columns(df)
    _parse_day_columns(df)
    return df


# ---------------------------------------------------------------------------
# Trip-attribute loader
# ---------------------------------------------------------------------------


# Regexes for the bin columns in the trip CSV. Each bin is given as a
# percentage of trips in that range.
_TT_BIN_RE = re.compile(r"^Travel Time (\d+)-(\d+) min \(percent\)$")
_TT_TAIL_RE = re.compile(r"^Travel Time (\d+)\+ min \(percent\)$")
_TL_BIN_RE = re.compile(r"^Trip Length (\d+)-(\d+) mi \(percent\)$")
_TL_TAIL_RE = re.compile(r"^Trip Length (\d+)\+ mi \(percent\)$")
_SPD_BIN_RE = re.compile(r"^Trip Speed (\d+)-(\d+) mph \(percent\)$")
_SPD_TAIL_RE = re.compile(r"^Trip Speed (\d+)\+ mph \(percent\)$")
_CIR_BIN_RE = re.compile(r"^Circuity (\d+)-(\d+) \(percent\)$")
_CIR_TAIL_RE = re.compile(r"^Circuity (\d+)\+ \(percent\)$")


def _short_bin_name(col: str) -> str | None:
    """Translate a verbose bin column name into a compact identifier."""
    for rx, fam in (
        (_TT_BIN_RE, "tt_min"),
        (_TL_BIN_RE, "len_mi"),
        (_SPD_BIN_RE, "spd_mph"),
        (_CIR_BIN_RE, "circuity"),
    ):
        m = rx.match(col)
        if m:
            return f"{fam}_{m.group(1)}_{m.group(2)}"
    for rx, fam in (
        (_TT_TAIL_RE, "tt_min"),
        (_TL_TAIL_RE, "len_mi"),
        (_SPD_TAIL_RE, "spd_mph"),
        (_CIR_TAIL_RE, "circuity"),
    ):
        m = rx.match(col)
        if m:
            return f"{fam}_{m.group(1)}_plus"
    return None


def load_za_trip(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load the per-zone trip-attribute distributions.

    Bin columns are renamed to short ids: ``tt_min_0_10``,
    ``len_mi_10_20``, ``spd_mph_30_40``, ``circuity_2_3``, ``len_mi_100_plus``,
    etc., so downstream callers don't have to handle the verbose
    StreetLight column names.
    """
    paths = discover_za_streets(folder)
    if paths is None or paths.zone_trip_all is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.zone_trip_all)
    df = df.rename(columns=_ZA_RENAMES)

    # Short-rename bin columns. Anything else stays as-is.
    bin_rename: dict[str, str] = {}
    for c in df.columns:
        short = _short_bin_name(c)
        if short is not None:
            bin_rename[c] = short
    if bin_rename:
        df = df.rename(columns=bin_rename)

    numeric = list(bin_rename.values()) + [
        "zone_volume",
        "avg_travel_time_sec",
        "avg_all_travel_time_sec",
        "avg_trip_length_mi",
        "avg_all_trip_length_mi",
        "avg_trip_speed_mph",
        "avg_all_trip_speed_mph",
    ]
    _coerce_numeric(df, tuple(numeric))

    _parse_zone_columns(df)
    _parse_day_columns(df)
    return df


# ---------------------------------------------------------------------------
# Home/Work loaders
# ---------------------------------------------------------------------------


_HW_RENAMES = dict(_ZA_RENAMES)
_HW_RENAMES.update({
    "Percent by Home Location": "pct_home_location",
    "Percent by Work Location": "pct_work_location",
    "Block Group ID": "block_group_id",
    "Zip Code": "zip_code",
    "Primary State": "zip_primary_state",
    "Primary Metro Area": "zip_primary_metro",
    "Name of Metro Area": "metro_area",
    "State Name": "state_name",
    "Rank": "rank",
})


def _load_hw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=_HW_RENAMES)
    _coerce_numeric(df, (
        "zone_volume",
        "pct_home_location",
        "pct_work_location",
        "rank",
    ))
    _parse_zone_columns(df)
    _parse_day_columns(df)
    return df


def load_za_home_distance(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load the per-zone home-distance distribution.

    Distance bin columns are kept verbose for clarity (e.g.
    ``Percent Home less than 1 mi``) since there are only 8 of them.
    """
    paths = discover_za_streets(folder)
    if paths is None or paths.home_distance is None:
        return pd.DataFrame()
    df = _load_hw_csv(paths.home_distance)
    distance_cols = [c for c in df.columns if c.startswith("Percent Home")]
    _coerce_numeric(df, tuple(distance_cols))
    return df


def load_za_home_zips_top(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load the top-ranked home ZIP codes per zone."""
    paths = discover_za_streets(folder)
    if paths is None or paths.home_zips_top is None:
        return pd.DataFrame()
    return _load_hw_csv(paths.home_zips_top)


def load_za_tourist_summary(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load the in/out-of-state tourist summary."""
    paths = discover_za_streets(folder)
    if paths is None or paths.tourist_summary is None:
        return pd.DataFrame()
    df = _load_hw_csv(paths.tourist_summary)
    pct_cols = [c for c in df.columns if c.startswith("Percent ")]
    _coerce_numeric(df, tuple(pct_cols))
    return df


def load_za_home_state(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load the per-zone home state distribution."""
    paths = discover_za_streets(folder)
    if paths is None or paths.home_state is None:
        return pd.DataFrame()
    return _load_hw_csv(paths.home_state)


def load_za_work_distance(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load the per-zone work-distance distribution (Visitor destinations).

    This is the destination analogue to ``load_za_home_distance``. For
    pass-through Visitor trips it characterises where the driver's
    *workplace* sits relative to the zone — a strong proxy for
    "where is this cut-through commute going".
    """
    paths = discover_za_streets(folder)
    if paths is None or paths.work_distance is None:
        return pd.DataFrame()
    df = _load_hw_csv(paths.work_distance)
    distance_cols = [c for c in df.columns if c.startswith("Percent Work")]
    _coerce_numeric(df, tuple(distance_cols))
    return df


def load_za_work_block_groups(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Load per-zone Visitor work block-group destinations.

    Returns one row per ``(zone, day type, day part, block_group_id)``
    with ``pct_work_location`` (StreetLight's share of Visitor trips
    whose driver's workplace falls in that Census block group) and
    derived ``state_fips`` (2 chars), ``county_fips`` (5 chars), and
    ``tract`` (11 chars) helpers for aggregation. Block group IDs are
    stripped of the wrapping single-quotes that StreetLight adds to
    preserve them as strings in CSV.
    """
    paths = discover_za_streets(folder)
    if paths is None or paths.work_block_groups is None:
        return pd.DataFrame()
    df = _load_hw_csv(paths.work_block_groups)
    if "block_group_id" in df.columns:
        clean = df["block_group_id"].astype(str).str.strip("'").str.strip()
        df["block_group_id"] = clean
        df["state_fips"] = clean.str[:2]
        df["county_fips"] = clean.str[:5]
        df["tract"] = clean.str[:11]
    return df


# ---------------------------------------------------------------------------
# Shapefile loader
# ---------------------------------------------------------------------------


def load_za_line_shapes(folder: Path = ZA_STREETS_DIR) -> gpd.GeoDataFrame:
    """Load the line shapefile for the ZA zones.

    Returns a GeoDataFrame with columns ``id``, ``name``,
    ``street_name``, ``osm_way_id``, ``direction``, ``road_type``,
    ``geometry``. Empty GeoDataFrame if the shapefile is missing.
    """
    paths = discover_za_streets(folder)
    if paths is None or paths.shapefile_line_zip is None:
        return gpd.GeoDataFrame(columns=["name", "geometry"], geometry="geometry", crs="EPSG:4326")
    gdf = gpd.read_file(f"zip://{paths.shapefile_line_zip}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    parsed = gdf["name"].apply(parse_bridge_zone_name)
    gdf["street_name"] = [p[0] for p in parsed]
    gdf["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")
    return gdf


def load_za_polygon_shapes(folder: Path = ZA_STREETS_DIR) -> gpd.GeoDataFrame:
    """Load the polygon shapefile for the ZA zones."""
    paths = discover_za_streets(folder)
    if paths is None or paths.shapefile_poly_zip is None:
        return gpd.GeoDataFrame(columns=["name", "geometry"], geometry="geometry", crs="EPSG:4326")
    gdf = gpd.read_file(f"zip://{paths.shapefile_poly_zip}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    parsed = gdf["name"].apply(parse_bridge_zone_name)
    gdf["street_name"] = [p[0] for p in parsed]
    gdf["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")
    return gdf


# ---------------------------------------------------------------------------
# Convenience: filter to "Visitors" only
# ---------------------------------------------------------------------------


def visitors_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where the ``filter`` column equals ``Visitors``.

    The ZA export contains both Visitor and Resident rows; almost every
    cut-through analysis uses Visitors only. This helper keeps the
    downstream code readable.
    """
    if df is None or df.empty or "filter" not in df.columns:
        return df
    return df[df["filter"] == "Visitors"].copy()


# ---------------------------------------------------------------------------
# Canonical cache loaders
# ---------------------------------------------------------------------------
# These prefer the parquets produced by ``scripts/00_build_datasets.py``
# (``data/processed/streetlight/za_*.parquet``) and fall back to
# re-parsing the raw CSVs if the canonical lake hasn't been built yet.
# The work-block-groups table in particular is ~2.8M rows; reading the
# parquet is ~50x faster than re-parsing the source CSV.


def _try_canonical(name: str):
    try:
        from leonia_traffic.data.dataset_io import CANONICAL_DIR, dataset_exists
        if not dataset_exists(CANONICAL_DIR, name):
            return None
        if name.endswith("_shapes.parquet"):
            import geopandas as gpd
            return gpd.read_parquet(CANONICAL_DIR / name)
        return pd.read_parquet(CANONICAL_DIR / name)
    except Exception:
        return None


def load_za_main_cached(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Canonical-first loader for the main ZA volume table."""
    cached = _try_canonical("za_volume.parquet")
    return cached if cached is not None else load_za_main(folder)


def load_za_trip_cached(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    cached = _try_canonical("za_trips.parquet")
    return cached if cached is not None else load_za_trip(folder)


def load_za_home_distance_cached(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    cached = _try_canonical("za_home_distance.parquet")
    return cached if cached is not None else load_za_home_distance(folder)


def load_za_home_zips_top_cached(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    cached = _try_canonical("za_home_zips_top.parquet")
    return cached if cached is not None else load_za_home_zips_top(folder)


def load_za_work_block_groups_cached(folder: Path = ZA_STREETS_DIR) -> pd.DataFrame:
    """Canonical-first loader for the 2.8M-row work-block-groups table.

    The raw CSV takes ~6 seconds to parse; the parquet is ~0.1s. Prefer
    this in any code path that runs more than once per session.
    """
    cached = _try_canonical("za_work_block_groups.parquet")
    return cached if cached is not None else load_za_work_block_groups(folder)


def load_za_line_shapes_cached(folder: Path = ZA_STREETS_DIR) -> gpd.GeoDataFrame:
    cached = _try_canonical("za_line_shapes.parquet")
    return cached if cached is not None else load_za_line_shapes(folder)


__all__ = [
    "ZA_STREETS_DIR",
    "ZAStreetsPaths",
    "discover_za_streets",
    "load_za_main",
    "load_za_main_cached",
    "load_za_trip",
    "load_za_trip_cached",
    "load_za_home_distance",
    "load_za_home_distance_cached",
    "load_za_home_zips_top",
    "load_za_home_zips_top_cached",
    "load_za_home_state",
    "load_za_tourist_summary",
    "load_za_work_distance",
    "load_za_work_block_groups",
    "load_za_work_block_groups_cached",
    "load_za_line_shapes",
    "load_za_line_shapes_cached",
    "load_za_polygon_shapes",
    "visitors_only",
]
