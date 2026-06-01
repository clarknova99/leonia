"""Canonical data-lake IO helpers.

The Leonia framework processes four StreetLight products plus several
derived analytics tables. This module is the single place that knows
where those canonical parquets live, how to write them with
consistent metadata, and how to update the per-folder manifest.

Directory layout (all paths relative to ``data/processed/``):

* ``streetlight/`` — canonical raw-ish tables, one parquet per
  StreetLight CSV family (segments, OD, congestion, ZA volume/trip/
  home/work/shapes). Geometry columns are persisted as **GeoParquet**
  (WGS84, EPSG:4326).
* ``derived/`` — downstream analytics tables computed from the
  canonical layer (composite cut-through index, hourly profiles,
  peak-hour intensity, etc).

Each folder has a ``_manifest.json`` recording per-file build
timestamps, row counts, and the raw source paths so callers can tell
at a glance whether a parquet is up to date.

This module exposes only IO helpers. The actual "build everything"
orchestration lives in ``scripts/00_build_datasets.py``; the schema
of every output file is documented in ``docs/DATA.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from leonia_traffic.config import DATA_PROCESSED_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CANONICAL_DIR = DATA_PROCESSED_DIR / "streetlight"
DERIVED_DIR = DATA_PROCESSED_DIR / "derived"
CRASHES_DIR = DATA_PROCESSED_DIR / "crashes"

CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
CRASHES_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_NAME = "_manifest.json"
CANONICAL_MANIFEST = CANONICAL_DIR / MANIFEST_NAME
DERIVED_MANIFEST = DERIVED_DIR / MANIFEST_NAME
CRASHES_MANIFEST = CRASHES_DIR / MANIFEST_NAME


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    """One row in a folder's manifest."""

    name: str                          # e.g. "streetscanner_segments.parquet"
    rows: int
    columns: int
    bytes: int
    sources: list[str]                 # list of raw CSV/SHP paths consumed
    has_geometry: bool
    built_at: str                      # ISO-8601 UTC


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"entries": {}}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"entries": {}}


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    data.setdefault("entries", {})
    data["updated_at"] = _now_iso()
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def update_manifest(folder: Path, entry: ManifestEntry) -> None:
    """Idempotently upsert a manifest entry for ``folder``."""
    manifest_path = folder / MANIFEST_NAME
    data = _read_manifest(manifest_path)
    data["entries"][entry.name] = {
        "rows": entry.rows,
        "columns": entry.columns,
        "bytes": entry.bytes,
        "sources": entry.sources,
        "has_geometry": entry.has_geometry,
        "built_at": entry.built_at,
    }
    _write_manifest(manifest_path, data)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_dataframe(
    df: pd.DataFrame,
    *,
    folder: Path,
    name: str,
    sources: list[Path] | list[str],
) -> Path:
    """Write a non-spatial DataFrame as parquet and update the manifest.

    Returns the absolute path written. Empty frames are still written
    (zero-row parquet) so downstream code can rely on the file
    existing.
    """
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / name
    df.to_parquet(out, index=False)
    update_manifest(folder, ManifestEntry(
        name=name,
        rows=int(len(df)),
        columns=int(df.shape[1]),
        bytes=int(out.stat().st_size),
        sources=[str(s) for s in sources],
        has_geometry=False,
        built_at=_now_iso(),
    ))
    return out


def write_geodataframe(
    gdf,
    *,
    folder: Path,
    name: str,
    sources: list[Path] | list[str],
    crs: int = 4326,
) -> Path:
    """Write a GeoDataFrame as GeoParquet (WGS84 by default).

    GeoParquet is readable by geopandas, QGIS (>=3.32), DuckDB-spatial,
    and most modern GIS tooling. The geometry column is preserved with
    proper CRS metadata, no WKB byte-blob conversion required.
    """
    folder.mkdir(parents=True, exist_ok=True)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif gdf.crs.to_epsg() != crs:
        gdf = gdf.to_crs(crs)
    out = folder / name
    gdf.to_parquet(out, index=False)
    update_manifest(folder, ManifestEntry(
        name=name,
        rows=int(len(gdf)),
        columns=int(gdf.shape[1]),
        bytes=int(out.stat().st_size),
        sources=[str(s) for s in sources],
        has_geometry=True,
        built_at=_now_iso(),
    ))
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_dataset(folder: Path, name: str) -> pd.DataFrame:
    """Load a canonical or derived parquet by name. Raises if missing."""
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical dataset not built: {path}. "
            "Run `venv/bin/python scripts/00_build_datasets.py` first."
        )
    return pd.read_parquet(path)


def read_geodataset(folder: Path, name: str):
    """Load a GeoParquet (returns a GeoDataFrame)."""
    import geopandas as gpd

    path = folder / name
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical geodataset not built: {path}. "
            "Run `venv/bin/python scripts/00_build_datasets.py` first."
        )
    return gpd.read_parquet(path)


def dataset_exists(folder: Path, name: str) -> bool:
    return (folder / name).exists()


# ---------------------------------------------------------------------------
# Canonical filenames (single source of truth)
# ---------------------------------------------------------------------------


class CanonicalFiles:
    """Filenames under ``data/processed/streetlight/``.

    Centralising the names here means a typo in one consumer can't
    silently look at a different file than the writer produced.
    """

    # Street Scanner — daily speed/volume on tertiary segments
    streetscanner_segments = "streetscanner_segments.parquet"

    # Bridge Origin-Destination — to/from GWB approach
    bridge_od = "bridge_od.parquet"
    bridge_od_zones = "bridge_od_zones.parquet"
    bridge_attributes = "bridge_attributes.parquet"

    # Congestion Trends — per-link reliability and delay
    congestion_links = "congestion_links.parquet"
    congestion_zones = "congestion_zones.parquet"

    # Network Performance — segment-level volume/speed/VMT/VHD on every
    # selected OSM segment (arterials + GWB approach + residential), at
    # hourly day-parts and per-day-of-week day types.
    network_performance_segments = "network_performance_segments.parquet"
    network_performance_prediction = "network_performance_prediction.parquet"
    network_performance_monthly = "network_performance_monthly.parquet"
    network_performance_zones = "network_performance_zones.parquet"
    network_performance_shapes = "network_performance_shapes.parquet"

    # Street Scanner Trend — monthly volume Jan 2023 → present
    streetscanner_trend = "streetscanner_trend.parquet"
    streetscanner_trend_shapes = "streetscanner_trend_shapes.parquet"

    # O-D + Middle-Filter cut-through analysis
    cutthrough_omd = "cutthrough_omd.parquet"
    cutthrough_omd_trips = "cutthrough_omd_trips.parquet"
    cutthrough_omd_zone_activity = "cutthrough_omd_zone_activity.parquet"
    cutthrough_omd_roster = "cutthrough_omd_roster.parquet"
    cutthrough_omd_shapes = "cutthrough_omd_shapes.parquet"

    # Zone Activity on Leonia tertiary streets
    za_volume = "za_volume.parquet"
    za_trips = "za_trips.parquet"
    za_home_distance = "za_home_distance.parquet"
    za_home_zips_top = "za_home_zips_top.parquet"
    za_home_state = "za_home_state.parquet"
    za_work_distance = "za_work_distance.parquet"
    za_work_block_groups = "za_work_block_groups.parquet"
    za_tourist_summary = "za_tourist_summary.parquet"
    za_line_shapes = "za_line_shapes.parquet"
    za_polygon_shapes = "za_polygon_shapes.parquet"


class DerivedFiles:
    """Filenames under ``data/processed/derived/``."""

    cutthrough_index = "cutthrough_index.parquet"
    hourly_profiles = "hourly_profiles.parquet"
    peak_intensity_am = "peak_intensity_am.parquet"
    peak_intensity_pm = "peak_intensity_pm.parquet"
    street_trend = "street_trend.parquet"
    cutthrough_attribution = "cutthrough_attribution.parquet"
    od_bypass_pairs = "od_bypass_pairs.parquet"


class CrashFiles:
    """Filenames under ``data/processed/crashes/``.

    See :mod:`leonia_traffic.data.njdot_crash_loader` for the schema
    of each file and :mod:`scripts.14_build_crash_overlay` for the
    build pipeline.
    """

    # One row per crash, geocoded to lat/lon and joined to its
    # nearest OSM way. Source: NJDOT raw crash zips 2017+.
    crashes = "njdot_crashes.parquet"
    # One row per OSM way × 5/10-yr window with crash counts and
    # EPDO weighting (the standard NJDOT severity-weighted index).
    crashes_by_segment = "crashes_by_segment.parquet"
    # One row per pedestrian-involved crash with age/sex (joined
    # on the case number). Useful for the demographic overlay.
    pedestrian_crashes = "njdot_pedestrian_crashes.parquet"


__all__ = [
    "CANONICAL_DIR",
    "DERIVED_DIR",
    "CRASHES_DIR",
    "CANONICAL_MANIFEST",
    "DERIVED_MANIFEST",
    "CRASHES_MANIFEST",
    "ManifestEntry",
    "CanonicalFiles",
    "DerivedFiles",
    "CrashFiles",
    "update_manifest",
    "write_dataframe",
    "write_geodataframe",
    "read_dataset",
    "read_geodataset",
    "dataset_exists",
]
