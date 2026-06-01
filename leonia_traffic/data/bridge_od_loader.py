"""Loader for the StreetLight Bridge-destination O-D Analysis export.

The export lives in ``streetlight/bridge_destination/`` and contains:

* ``*_od_all.csv`` — the core OD matrix (469 rows): origin gate × destination
  gate × Day Type × Day Part, with ``Average Daily O-D Traffic (StL Volume)``
  as the volume metric and ``Avg Travel Time (sec)`` as a per-pair quality
  measure.
* Six **attribute CSVs** sharing the same join keys
  ``(Origin Zone Name, Destination Zone Name, Day Type, Day Part)``:
    * ``*_od_traveler_trip_purpose_all.csv`` — Home-to-Work / Home-to-Other /
      Non-Home-Based shares.
    * ``*_od_traveler_equity_all.csv`` — race, ethnicity, foreign-born,
      English proficiency, disability shares.
    * ``*_od_traveler_household_all.csv`` — kids, tenure, vehicles, unit
      structure shares.
    * ``*_od_traveler_education_income_all.csv`` — 16 income brackets and
      7 education levels.
    * ``*_od_traveler_employment_all.csv`` — industry and worker class shares.
    * ``*_od_trip_all.csv`` — circuity, trip length, travel time, speed
      distributions plus speed and travel-time percentiles.
* Shapefile zips in ``Shapefile/`` describing the origin and destination
  zone geometries (line and polygon variants).

Per the plan we keep **all** attribute columns by default so the
downstream report can render equity narratives next to operational
metrics.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import STREETLIGHT_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


# Switched from the legacy 5-window export
# (``streetlight/bridge_destination/``, analysis 2034043) to the new
# 24-hour-daypart export (analysis 2036064). The schema is identical
# — same column names, same Day Type values, same per-row attribute
# CSVs — but ``Day Part`` is now hourly (``01: 12am``, …, ``24: 11pm``)
# instead of the legacy buckets (``1: Early AM``, …, ``5: Late PM``).
# Downstream code that previously hardcoded the 5 window codes 1–5
# now iterates over hourly codes 1–24 (see
# :data:`leonia_traffic.sumo.demand_builder.BRIDGE_OD_WINDOWS`).
BRIDGE_OD_DIR = STREETLIGHT_DIR / "2036064_Destinations"

_OD_JOIN_KEYS = (
    "Origin Zone Name",
    "Destination Zone Name",
    "Day Type",
    "Day Part",
)


@dataclass(frozen=True)
class BridgeODPaths:
    folder: Path
    od_all: Path
    trip_purpose: Path | None
    equity: Path | None
    household: Path | None
    education_income: Path | None
    employment: Path | None
    trip_stats: Path | None
    shapefile_dir: Path | None

    @property
    def attribute_files(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if self.trip_purpose is not None:
            out["trip_purpose"] = self.trip_purpose
        if self.equity is not None:
            out["equity"] = self.equity
        if self.household is not None:
            out["household"] = self.household
        if self.education_income is not None:
            out["education_income"] = self.education_income
        if self.employment is not None:
            out["employment"] = self.employment
        if self.trip_stats is not None:
            out["trip_stats"] = self.trip_stats
        return out


def _first(folder: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]
    return None


def discover_bridge_od(folder: Path = BRIDGE_OD_DIR) -> BridgeODPaths | None:
    """Find the bridge OD export under ``folder``.

    Returns ``None`` if no ``*_od_all.csv`` is present (the export hasn't
    been downloaded yet). Otherwise returns a populated ``BridgeODPaths``
    where any missing attribute files are ``None``.
    """
    if not folder.exists():
        return None
    od_all = _first(folder, "*_od_all.csv")
    if od_all is None:
        return None
    return BridgeODPaths(
        folder=folder,
        od_all=od_all,
        trip_purpose=_first(folder, "*_od_traveler_trip_purpose_all.csv"),
        equity=_first(folder, "*_od_traveler_equity_all.csv"),
        household=_first(folder, "*_od_traveler_household_all.csv"),
        education_income=_first(folder, "*_od_traveler_education_income_all.csv"),
        employment=_first(folder, "*_od_traveler_employment_all.csv"),
        trip_stats=_first(folder, "*_od_trip_all.csv"),
        shapefile_dir=folder / "Shapefile" if (folder / "Shapefile").exists() else None,
    )


# ---------------------------------------------------------------------------
# Zone Name parsing
# ---------------------------------------------------------------------------


_ZONE_NAME_TAIL = re.compile(r"\s*/\s*(\d+)\s*$")


def parse_bridge_zone_name(zone_name: str) -> tuple[str, int | None]:
    """Return ``(zone_label, osm_way_id)`` from an OD zone name.

    Bridge OD zone names follow the format ``"<street/place> / <osm id>"``
    (no Split # suffix because each zone is a single OSM way segment).
    Examples:
        ``"Fort Lee Road / 590576"``
        ``"George Washington Bridge (lower level) / 590410"``
    """
    if zone_name is None:
        return ("", None)
    s = str(zone_name)
    m = _ZONE_NAME_TAIL.search(s)
    if not m:
        return (s.strip(), None)
    osm_way_id = int(m.group(1))
    label = s[: m.start()].strip()
    return (label, osm_way_id)


# ---------------------------------------------------------------------------
# Day Type / Day Part code parsing
# ---------------------------------------------------------------------------


_CODE_PATTERN = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")


def parse_coded_value(value: str) -> tuple[int | None, str | None]:
    """Return ``(code, label)`` from a ``"<code>: <label>"`` string."""
    if value is None:
        return (None, None)
    m = _CODE_PATTERN.match(str(value))
    if not m:
        return (None, str(value).strip())
    return (int(m.group(1)), m.group(2).strip())


# ---------------------------------------------------------------------------
# OD matrix loader
# ---------------------------------------------------------------------------


_OD_RENAMES = {
    "Data Periods": "data_periods",
    "Mode of Travel": "mode_of_travel",
    "Origin Zone ID": "origin_zone_id",
    "Origin Zone Name": "origin_zone",
    "Origin Zone Is Pass-Through": "origin_pass_through",
    "Origin Zone Direction (degrees)": "origin_direction_deg",
    "Origin Zone is Bi-Direction": "origin_bidi",
    "Destination Zone ID": "destination_zone_id",
    "Destination Zone Name": "destination_zone",
    "Destination Zone Is Pass-Through": "destination_pass_through",
    "Destination Zone Direction (degrees)": "destination_direction_deg",
    "Destination Zone is Bi-Direction": "destination_bidi",
    "Day Type": "day_type_raw",
    "Day Part": "day_part_raw",
    "Average Daily O-D Traffic (StL Volume)": "od_volume",
    "Average Daily Origin Zone Traffic (StL Volume)": "origin_total_volume",
    "Average Daily Destination Zone Traffic (StL Volume)": "destination_total_volume",
    "Avg Travel Time (sec)": "avg_travel_time_sec",
}


def load_bridge_od(folder: Path = BRIDGE_OD_DIR) -> pd.DataFrame:
    """Load the core bridge OD matrix.

    Returns a long-format DataFrame with one row per
    ``(origin_zone, destination_zone, day_type_code, day_part_code)`` and
    the following key fields:

    * ``origin_zone`` / ``destination_zone`` — raw zone names (used as
      the canonical join key for attribute CSVs).
    * ``origin_osm_way_id`` / ``destination_osm_way_id`` — parsed OSM
      way IDs (Int64, nullable).
    * ``origin_label`` / ``destination_label`` — clean street/place
      names without the OSM ID suffix.
    * ``day_type_code`` (0–7), ``day_type_label`` (e.g. ``"Monday (M-M)"``).
    * ``day_part_code`` (0–24), ``day_part_label`` (e.g. ``"7am (7am-8am)"``).
      Code 0 is the All-Day total; codes 1–24 are hourly windows
      ``[code-1, code)`` (e.g. code 8 = 7am–8am).
    * ``od_volume`` — average daily OD trips (StreetLight Volume).
    * ``avg_travel_time_sec`` — average per-pair travel time.

    Returns an empty DataFrame if the export has not been downloaded.
    """
    paths = discover_bridge_od(folder)
    if paths is None:
        return pd.DataFrame(columns=[
            "origin_zone", "destination_zone",
            "day_type_code", "day_type_label",
            "day_part_code", "day_part_label",
            "od_volume", "avg_travel_time_sec",
            "origin_osm_way_id", "destination_osm_way_id",
            "origin_label", "destination_label",
        ])

    df = pd.read_csv(paths.od_all)
    df = df.rename(columns=_OD_RENAMES)

    for c in (
        "od_volume",
        "origin_total_volume",
        "destination_total_volume",
        "avg_travel_time_sec",
        "origin_direction_deg",
        "destination_direction_deg",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    parsed_origin = df["origin_zone"].apply(parse_bridge_zone_name)
    df["origin_label"] = [p[0] for p in parsed_origin]
    df["origin_osm_way_id"] = pd.array([p[1] for p in parsed_origin], dtype="Int64")

    parsed_dest = df["destination_zone"].apply(parse_bridge_zone_name)
    df["destination_label"] = [p[0] for p in parsed_dest]
    df["destination_osm_way_id"] = pd.array([p[1] for p in parsed_dest], dtype="Int64")

    dt = df["day_type_raw"].apply(parse_coded_value)
    df["day_type_code"] = pd.array([p[0] for p in dt], dtype="Int64")
    df["day_type_label"] = [p[1] for p in dt]

    dp = df["day_part_raw"].apply(parse_coded_value)
    df["day_part_code"] = pd.array([p[0] for p in dp], dtype="Int64")
    df["day_part_label"] = [p[1] for p in dp]

    return df


# ---------------------------------------------------------------------------
# Joined attribute frame
# ---------------------------------------------------------------------------


# Columns that appear in every attribute CSV and would create duplicates if
# we did a naive merge. We keep them from the OD-all frame only.
_DUPE_ATTR_COLS = {
    "Data Periods", "Mode of Travel",
    "Origin Zone ID", "Origin Zone Is Pass-Through",
    "Origin Zone Direction (degrees)", "Origin Zone is Bi-Direction",
    "Destination Zone ID", "Destination Zone Is Pass-Through",
    "Destination Zone Direction (degrees)", "Destination Zone is Bi-Direction",
    "Average Daily O-D Traffic (StL Volume)",
    "Average Daily Origin Zone Traffic (StL Volume)",
    "Average Daily Destination Zone Traffic (StL Volume)",
    "Avg Travel Time (sec)",
}


def _prefix_attribute_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix non-join non-duplicate columns so attribute frames don't collide."""
    keep = list(_OD_JOIN_KEYS)
    rename: dict[str, str] = {}
    for col in df.columns:
        if col in keep:
            continue
        if col in _DUPE_ATTR_COLS:
            continue
        rename[col] = f"{prefix}::{col}"
    return df[keep + list(rename)].rename(columns=rename)


def load_bridge_attributes(
    folder: Path = BRIDGE_OD_DIR,
    *,
    include: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load the bridge OD matrix joined with all attribute CSVs.

    Each attribute column is prefixed with its source bucket (e.g.
    ``trip_purpose::``, ``equity::``, ``household::``, ``income::``,
    ``employment::``, ``trip_stats::``) so the resulting wide frame can
    be filtered downstream by bucket without column-name collisions.

    Parameters
    ----------
    folder
        Bridge OD export folder.
    include
        Optional whitelist of attribute bucket names to merge. Defaults
        to all six. The OD-all frame is always loaded.
    """
    paths = discover_bridge_od(folder)
    if paths is None:
        return pd.DataFrame()

    od = load_bridge_od(folder)
    if od.empty:
        return od

    # Re-attach the raw join keys (we renamed them above).
    od_raw = pd.read_csv(paths.od_all)
    od_keys = od_raw[list(_OD_JOIN_KEYS)].reset_index(drop=True)
    od_joined = pd.concat([od_keys, od.reset_index(drop=True)], axis=1)

    bucket_map = {
        "trip_purpose": paths.trip_purpose,
        "equity": paths.equity,
        "household": paths.household,
        "income": paths.education_income,
        "employment": paths.employment,
        "trip_stats": paths.trip_stats,
    }

    wanted = set(include) if include is not None else set(bucket_map.keys())

    merged = od_joined
    for bucket, p in bucket_map.items():
        if bucket not in wanted or p is None or not p.exists():
            continue
        attr = pd.read_csv(p)
        prefixed = _prefix_attribute_columns(attr, bucket)
        merged = merged.merge(prefixed, on=list(_OD_JOIN_KEYS), how="left")

    return merged


# ---------------------------------------------------------------------------
# Zone shapefiles
# ---------------------------------------------------------------------------


def _read_shapefile_zip(zip_path: Path) -> gpd.GeoDataFrame:
    """Read an unpacked or zipped shapefile into a GeoDataFrame."""
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if zip_path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{zip_path}")
    return gpd.read_file(zip_path)


def load_bridge_zone_shapes(
    folder: Path = BRIDGE_OD_DIR,
    *,
    kind: str = "line",
) -> gpd.GeoDataFrame:
    """Load origin + destination zone shapefiles, concatenated.

    Parameters
    ----------
    folder
        Bridge OD export folder.
    kind
        ``"line"`` (default) returns the line-zone geometries.
        ``"polygon"`` returns the polygon variants. The shapefile zip
        layout from StreetLight uses suffixes ``_origin`` /
        ``_destination`` for line zones and ``*_polygon*`` for polygons,
        so the function falls back to substring matching.

    The returned frame has an extra ``zone_role`` column
    (``"origin"`` or ``"destination"``).
    """
    paths = discover_bridge_od(folder)
    if paths is None or paths.shapefile_dir is None:
        return gpd.GeoDataFrame(columns=["name", "geometry", "zone_role"], crs="EPSG:4326")

    zips = sorted(paths.shapefile_dir.glob("*.zip"))
    if not zips:
        return gpd.GeoDataFrame(columns=["name", "geometry", "zone_role"], crs="EPSG:4326")

    if kind == "polygon":
        candidates = [z for z in zips if "polygon" in z.stem.lower()]
    else:
        candidates = [z for z in zips if "polygon" not in z.stem.lower()]

    frames: list[gpd.GeoDataFrame] = []
    for z in candidates:
        gdf = _read_shapefile_zip(z)
        if "origin" in z.stem.lower():
            role = "origin"
        elif "destination" in z.stem.lower():
            role = "destination"
        else:
            role = "unknown"
        gdf = gdf.copy()
        gdf["zone_role"] = role
        # Normalize name column so caller can join on a single attribute.
        if "name" not in gdf.columns:
            # Some StreetLight shapefile schemas use different name fields.
            for alt in ("zone_name", "Name", "ZONE_NAME"):
                if alt in gdf.columns:
                    gdf = gdf.rename(columns={alt: "name"})
                    break
        frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(columns=["name", "geometry", "zone_role"], crs="EPSG:4326")

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return out


# ---------------------------------------------------------------------------
# Day-type / day-part code helpers (stable lookups for downstream code)
# ---------------------------------------------------------------------------


DAY_TYPE_CODES = {
    0: "All Days (M-Su)",
    1: "Monday (M-M)",
    2: "Tuesday (Tu-Tu)",
    3: "Wednesday (W-W)",
    4: "Thursday (Th-Th)",
    5: "Friday (F-F)",
    6: "Saturday (Sa-Sa)",
    7: "Sunday (Su-Su)",
}

DAY_PART_CODES = {
    0: "All Day (12am-12am)",
    1: "Early AM (12am-6am)",
    2: "Peak AM (6am-10am)",
    3: "Mid-Day (10am-3pm)",
    4: "Peak PM (3pm-7pm)",
    5: "Late PM (7pm-12am)",
}

WEEKDAY_CODES = (1, 2, 3, 4, 5)  # Mon–Fri
WEEKEND_CODES = (6, 7)            # Sat–Sun


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


def load_bridge_od_cached(folder: Path = BRIDGE_OD_DIR) -> pd.DataFrame:
    """Canonical-first loader for the Bridge OD core table."""
    cached = _try_canonical("bridge_od.parquet")
    return cached if cached is not None else load_bridge_od(folder)


def load_bridge_attributes_cached(
    folder: Path = BRIDGE_OD_DIR,
    *,
    include: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Canonical-first loader for the Bridge OD attributes table.

    Note: ``include`` is honoured by re-filtering the canonical parquet
    (cheap) rather than re-parsing the underlying CSVs.
    """
    cached = _try_canonical("bridge_attributes.parquet")
    if cached is None:
        return load_bridge_attributes(folder, include=include)
    if include is not None and "attribute_kind" in cached.columns:
        cached = cached[cached["attribute_kind"].isin(include)].copy()
    return cached


def load_bridge_zone_shapes_cached(folder: Path = BRIDGE_OD_DIR):
    """Canonical-first loader for the Bridge OD zone geometries."""
    cached = _try_canonical("bridge_od_zones.parquet", geo=True)
    return cached if cached is not None else load_bridge_zone_shapes(folder)


__all__ = [
    "BRIDGE_OD_DIR",
    "BridgeODPaths",
    "DAY_PART_CODES",
    "DAY_TYPE_CODES",
    "WEEKDAY_CODES",
    "WEEKEND_CODES",
    "discover_bridge_od",
    "load_bridge_attributes",
    "load_bridge_attributes_cached",
    "load_bridge_od",
    "load_bridge_od_cached",
    "load_bridge_zone_shapes",
    "load_bridge_zone_shapes_cached",
    "parse_bridge_zone_name",
    "parse_coded_value",
]
