"""Loader for the StreetLight **cut-through O-D + Middle-Filter** export.

This is an "O-D Analysis with Middle Filter" run: every observation is a
triple ``(origin, middle, destination)`` where the **middle** zone is a
Leonia tertiary street segment. For each triple we get the average daily
volume, travel time, and (in the trip CSV) the full distribution of
travel time, trip length, speed, and circuity.

The export lives at ``streetlight/2034993_cut_through/`` and follows the
standard StreetLight folder layout:

* ``2034993_cut_through_mf_all.csv`` — main O-M-D volume table.
* ``2034993_cut_through_mf_trip_all.csv`` — per-triple trip-attribute
  distributions (circuity / trip length / speed / travel-time buckets).
* ``Zone Activity/``
    * ``2034993_cut_through_zone_mf_all.csv`` — total daily volume per
      zone (origin, middle, destination).
    * ``2034993_cut_through_zone_trip_all.csv`` — per-zone trip-attribute
      distributions.
* ``Analysis Details/``
    * ``2034993_cut_through_zones.csv`` — zone roster + bearing / pass-
      through flags.
    * ``Analysis.txt`` — boilerplate.
* ``Shapefile/`` — one ``.zip`` per zone role (origin, middle_filter,
  destination), each with a polygon and a line variant.
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
from leonia_traffic.data.bridge_od_loader import (
    parse_bridge_zone_name,
    parse_coded_value,
)

logger = logging.getLogger(__name__)


CUTTHROUGH_OMD_DIR = STREETLIGHT_DIR / "2034993_cut_through"


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CutthroughOMDPaths:
    folder: Path
    mf_all: Path
    mf_trip_all: Path
    zone_mf_all: Path | None
    zone_trip_all: Path | None
    zones_roster: Path | None
    shapefile_dir: Path | None


def discover_cutthrough_omd(
    folder: Path = CUTTHROUGH_OMD_DIR,
) -> CutthroughOMDPaths | None:
    if not folder.exists():
        return None

    mf = list(folder.glob("*_mf_all.csv"))
    mft = list(folder.glob("*_mf_trip_all.csv"))
    if not mf or not mft:
        return None

    za_dir = folder / "Zone Activity"
    ad_dir = folder / "Analysis Details"
    sh_dir = folder / "Shapefile"

    zone_mf = next(iter(za_dir.glob("*_zone_mf_all.csv")), None) if za_dir.is_dir() else None
    zone_trip = next(iter(za_dir.glob("*_zone_trip_all.csv")), None) if za_dir.is_dir() else None
    roster = next(iter(ad_dir.glob("*_zones.csv")), None) if ad_dir.is_dir() else None

    return CutthroughOMDPaths(
        folder=folder,
        mf_all=mf[0],
        mf_trip_all=mft[0],
        zone_mf_all=zone_mf,
        zone_trip_all=zone_trip,
        zones_roster=roster,
        shapefile_dir=sh_dir if sh_dir.is_dir() else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attach_zone_parses(df: pd.DataFrame, role: str, raw_col: str) -> pd.DataFrame:
    """Add ``{role}_label`` and ``{role}_osm_way_id`` columns from raw_col."""
    parsed = df[raw_col].apply(parse_bridge_zone_name)
    df[f"{role}_label"] = [p[0] for p in parsed]
    df[f"{role}_osm_way_id"] = pd.array(
        [p[1] for p in parsed], dtype="Int64",
    )
    return df


def _attach_day_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "day_type_raw" in df.columns:
        dt = df["day_type_raw"].apply(parse_coded_value)
        df["day_type_code"] = pd.array([p[0] for p in dt], dtype="Int64")
        df["day_type_label"] = [p[1] for p in dt]
    if "day_part_raw" in df.columns:
        dp = df["day_part_raw"].apply(parse_coded_value)
        df["day_part_code"] = pd.array([p[0] for p in dp], dtype="Int64")
        df["day_part_label"] = [p[1] for p in dp]
    return df


_MF_RENAMES = {
    "Mode of Travel": "mode_of_travel",
    "Origin Zone ID": "origin_zone_id",
    "Origin Zone Name": "origin_zone",
    "Origin Zone Is Pass-Through": "origin_pass_through",
    "Origin Zone Direction (degrees)": "origin_direction_deg",
    "Origin Zone is Bi-Direction": "origin_bidi",
    "Middle Filter Zone ID": "middle_zone_id",
    "Middle Filter Zone Name": "middle_zone",
    "Middle Filter Zone Direction (degrees)": "middle_direction_deg",
    "Middle Filter Zone is Bi-Direction": "middle_bidi",
    "Destination Zone ID": "destination_zone_id",
    "Destination Zone Name": "destination_zone",
    "Destination Zone Is Pass-Through": "destination_pass_through",
    "Destination Zone Direction (degrees)": "destination_direction_deg",
    "Destination Zone is Bi-Direction": "destination_bidi",
    "Day Type": "day_type_raw",
    "Day Part": "day_part_raw",
    "Average Daily O-M-D Traffic (StL Volume)": "omd_volume",
    "Average Daily Origin Zone Traffic (StL Volume)": "origin_total_volume",
    "Average Daily Middle Filter Zone Traffic (StL Volume)": "middle_total_volume",
    "Average Daily Destination Zone Traffic (StL Volume)": "destination_total_volume",
    "Avg Travel Time (sec)": "avg_travel_time_sec",
}


def load_cutthrough_omd(folder: Path = CUTTHROUGH_OMD_DIR) -> pd.DataFrame:
    """Load the O-M-D volume matrix (one row per origin × middle × dest × day-type × day-part).

    Returns an empty DataFrame if the export folder is missing.
    """
    paths = discover_cutthrough_omd(folder)
    if paths is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.mf_all)
    df = df.rename(columns=_MF_RENAMES)
    for c in ("omd_volume", "origin_total_volume", "middle_total_volume",
              "destination_total_volume", "avg_travel_time_sec",
              "origin_direction_deg", "middle_direction_deg",
              "destination_direction_deg"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _attach_zone_parses(df, "origin", "origin_zone")
    df = _attach_zone_parses(df, "middle", "middle_zone")
    df = _attach_zone_parses(df, "destination", "destination_zone")
    df = _attach_day_codes(df)
    return df


# ---------------------------------------------------------------------------
# Trip distributions: keep all columns but tag them by family.
# ---------------------------------------------------------------------------


_TRIP_RENAMES = {
    **_MF_RENAMES,
    "Average Daily O-M-D Traffic (StL Volume)": "omd_volume",
    "Avg Travel Time (sec)": "avg_travel_time_sec",
    "Avg All Travel Time (sec)": "avg_all_travel_time_sec",
    "Avg Trip Length (mi)": "avg_trip_length_mi",
    "Avg All Trip Length (mi)": "avg_all_trip_length_mi",
    "Avg Trip Speed (mph)": "avg_trip_speed_mph",
    "Avg All Trip Speed (mph)": "avg_all_trip_speed_mph",
    "5th Speed Percentile": "speed_pct_05",
    "15th Speed Percentile": "speed_pct_15",
    "85th Speed Percentile": "speed_pct_85",
    "95th Speed Percentile": "speed_pct_95",
    "5th Travel Time Percentile": "tt_pct_05",
    "50th Travel Time Percentile": "tt_pct_50",
    "80th Travel Time Percentile": "tt_pct_80",
    "90th Travel Time Percentile": "tt_pct_90",
    "95th Travel Time Percentile": "tt_pct_95",
}


_CIRCUITY_BUCKETS = ["Circuity 1-2 (percent)", "Circuity 2-3 (percent)",
                     "Circuity 3-4 (percent)", "Circuity 4-5 (percent)",
                     "Circuity 5-6 (percent)", "Circuity 6+ (percent)"]


def load_cutthrough_omd_trips(folder: Path = CUTTHROUGH_OMD_DIR) -> pd.DataFrame:
    """Load the per-triple trip-attribute distributions.

    All StreetLight bucket columns ("Travel Time 0-10 min (percent)", ...)
    are preserved as-is. We add lightweight summary columns that
    downstream code uses heavily:

    * ``share_circuity_ge_3`` — share of trips with circuity ≥ 3
      (strong cut-through signal).
    * ``share_trip_le_2mi`` — share of trips under 2 miles
      (likely local users vs through-traffic).
    * ``share_trip_ge_5mi`` — share of trips over 5 miles
      (long-distance cut-through).
    * ``share_speed_ge_30`` — share of trips averaging ≥ 30 mph
      (likely speeding on residential streets).
    """
    paths = discover_cutthrough_omd(folder)
    if paths is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.mf_trip_all, low_memory=False)
    df = df.rename(columns=_TRIP_RENAMES)
    # Coerce numerics.
    num_cols = [c for c in df.columns
                if c in {"omd_volume", "avg_travel_time_sec",
                         "avg_all_travel_time_sec",
                         "avg_trip_length_mi", "avg_all_trip_length_mi",
                         "avg_trip_speed_mph", "avg_all_trip_speed_mph"}
                or c.startswith(("speed_pct_", "tt_pct_"))
                or "(percent)" in c]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _attach_zone_parses(df, "origin", "origin_zone")
    df = _attach_zone_parses(df, "middle", "middle_zone")
    df = _attach_zone_parses(df, "destination", "destination_zone")
    df = _attach_day_codes(df)

    # Summary shares — guard against the all-N/A rows the export contains
    # when a triple has zero observed trips.
    def _sum_cols(cols):
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.Series(pd.NA, index=df.index, dtype="Float64")
        return df[existing].sum(axis=1, min_count=1)

    df["share_circuity_ge_3"] = _sum_cols([
        "Circuity 3-4 (percent)", "Circuity 4-5 (percent)",
        "Circuity 5-6 (percent)", "Circuity 6+ (percent)",
    ])
    df["share_trip_le_2mi"] = _sum_cols([
        "Trip Length 0-1 mi (percent)", "Trip Length 1-2 mi (percent)",
    ])
    df["share_trip_ge_5mi"] = _sum_cols([
        "Trip Length 5-10 mi (percent)", "Trip Length 10-20 mi (percent)",
        "Trip Length 20-30 mi (percent)", "Trip Length 30-40 mi (percent)",
        "Trip Length 40-50 mi (percent)", "Trip Length 50-60 mi (percent)",
        "Trip Length 60-70 mi (percent)", "Trip Length 70-80 mi (percent)",
        "Trip Length 80-90 mi (percent)", "Trip Length 90-100 mi (percent)",
        "Trip Length 100+ mi (percent)",
    ])
    df["share_speed_ge_30"] = _sum_cols([
        "Trip Speed 30-40 mph (percent)", "Trip Speed 40-50 mph (percent)",
        "Trip Speed 50-60 mph (percent)", "Trip Speed 60-70 mph (percent)",
        "Trip Speed 70+ mph (percent)",
    ])
    return df


# ---------------------------------------------------------------------------
# Zone activity + roster
# ---------------------------------------------------------------------------


def load_cutthrough_omd_zone_activity(
    folder: Path = CUTTHROUGH_OMD_DIR,
) -> pd.DataFrame:
    """Return per-zone daily volumes (origin / middle / destination)."""
    paths = discover_cutthrough_omd(folder)
    if paths is None or paths.zone_mf_all is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.zone_mf_all)
    df = df.rename(columns={
        "Mode of Travel": "mode_of_travel",
        "Zone Type": "zone_role",
        "Zone ID": "zone_id",
        "Zone Name": "zone",
        "Zone Is Pass-Through": "pass_through",
        "Zone Direction (degrees)": "direction_deg",
        "Zone is Bi-Direction": "bidi",
        "Day Type": "day_type_raw",
        "Day Part": "day_part_raw",
        "Average Daily Zone Traffic (StL Volume)": "zone_volume",
    })
    df["zone_volume"] = pd.to_numeric(df["zone_volume"], errors="coerce")
    df = _attach_zone_parses(df, "zone", "zone")
    df = _attach_day_codes(df)
    df["zone_role"] = df["zone_role"].astype(str).str.lower().str.replace(
        "middle filter", "middle", regex=False,
    )
    return df


def load_cutthrough_omd_roster(
    folder: Path = CUTTHROUGH_OMD_DIR,
) -> pd.DataFrame:
    """Load the per-zone roster (one row per origin/middle/destination zone)."""
    paths = discover_cutthrough_omd(folder)
    if paths is None or paths.zones_roster is None:
        return pd.DataFrame()
    df = pd.read_csv(paths.zones_roster)
    df = df.rename(columns={
        "Zone Type": "zone_role",
        "Zone ID": "zone_id",
        "Zone Name": "zone",
        "Zone Is Pass-Through": "pass_through",
        "Zone Direction (degrees)": "direction_deg",
        "Zone is Bi-Direction": "bidi",
        "Fingerprint1": "fingerprint1",
        "Fingerprint2": "fingerprint2",
    })
    df = _attach_zone_parses(df, "zone", "zone")
    df["zone_role"] = df["zone_role"].astype(str).str.lower().str.replace(
        "middle filter", "middle", regex=False,
    )
    return df


# ---------------------------------------------------------------------------
# Shapefiles
# ---------------------------------------------------------------------------


_ROLE_PATTERN = re.compile(r"(origin|destination|middle_filter)", re.IGNORECASE)


def _read_shapefile_zip(zip_path: Path) -> gpd.GeoDataFrame:
    """Read a zipped shapefile in place via the GDAL ``zip+file://`` URI."""
    return gpd.read_file(f"zip://{zip_path}")


def load_cutthrough_omd_shapes(
    folder: Path = CUTTHROUGH_OMD_DIR,
    *,
    kind: str = "line",
) -> gpd.GeoDataFrame:
    """Load origin + middle + destination shapes concatenated.

    Adds a ``zone_role`` column (``origin`` / ``middle`` / ``destination``).
    Set ``kind="polygon"`` to load the polygon variants; default is line.
    """
    paths = discover_cutthrough_omd(folder)
    if paths is None or paths.shapefile_dir is None:
        return gpd.GeoDataFrame(
            columns=["name", "geometry", "zone_role"], crs="EPSG:4326",
        )
    want_line = kind == "line"
    zips = sorted(paths.shapefile_dir.glob("*.zip"))
    cand = [
        z for z in zips
        if ("_line" in z.stem.lower()) == want_line
    ]
    frames: list[gpd.GeoDataFrame] = []
    for z in cand:
        m = _ROLE_PATTERN.search(z.stem.lower())
        if not m:
            continue
        role = m.group(1).replace("middle_filter", "middle")
        gdf = _read_shapefile_zip(z).copy()
        if "name" not in gdf.columns:
            for alt in ("zone_name", "Name", "ZONE_NAME"):
                if alt in gdf.columns:
                    gdf = gdf.rename(columns={alt: "name"})
                    break
        gdf["zone_role"] = role
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame(
            columns=["name", "geometry", "zone_role"], crs="EPSG:4326",
        )
    out = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=frames[0].crs,
    )
    return out


__all__ = [
    "CUTTHROUGH_OMD_DIR",
    "CutthroughOMDPaths",
    "discover_cutthrough_omd",
    "load_cutthrough_omd",
    "load_cutthrough_omd_trips",
    "load_cutthrough_omd_zone_activity",
    "load_cutthrough_omd_roster",
    "load_cutthrough_omd_shapes",
]
