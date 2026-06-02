"""Loader for StreetLight Street Scanner exports.

A StreetLight Street Scanner export is a folder that contains:
  - ``Filters.txt`` (free-form metadata about what was queried)
  - ``*_streetscanner_*.csv`` (the metric rows, one per zone)
  - ``*_streetscanner_*.shp`` + companions (zone geometries)
  - ``README_StreetScanner.txt`` (boilerplate)

This module auto-discovers every such folder under ``streetlight/`` (the
root folder counts too), parses the filters, joins the CSV to the
shapefile, and returns one unified long-format GeoDataFrame.

Important behaviors documented in the plan:

* The Weekday/Weekend exports collapse multiple selected Day Parts into a
  single row whose ``Day Part`` value is the semicolon-joined list of
  parts. We expose the raw string in ``day_part_raw`` and a list in
  ``day_parts`` for downstream code.
* ``Zone Name`` has the format ``[OSM Name] / [OSM Way ID] / [Split #]``.
  We parse this and expose ``osm_way_id`` (int) and ``split_index`` (int)
  for joining to OSMnx.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from leonia_traffic.config import (
    DATA_STAGE1_DIR,
    STREETLIGHT_DIR,
    STREETLIGHT_FOLDER_TO_LABEL,
    STUDY_AREA_CITIES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreetLightSource:
    """A single Street Scanner export folder on disk."""

    label: str
    folder: Path
    csv_path: Path
    shp_path: Path
    filters_path: Path

    @property
    def name(self) -> str:
        return self.label


def discover_sources(root: Path = STREETLIGHT_DIR) -> list[StreetLightSource]:
    """Find every Street Scanner export under ``root``.

    A folder qualifies if it contains both a ``Filters.txt`` and exactly
    one ``*_streetscanner_*.csv`` with a matching ``.shp``.
    """
    if not root.exists():
        raise FileNotFoundError(f"StreetLight root does not exist: {root}")

    candidates: list[Path] = [root]
    candidates.extend(p for p in root.iterdir() if p.is_dir())

    sources: list[StreetLightSource] = []
    for folder in candidates:
        filters = folder / "Filters.txt"
        if not filters.exists():
            continue
        csvs = sorted(folder.glob("*_streetscanner_*.csv"))
        shps = sorted(folder.glob("*_streetscanner_*.shp"))
        if not csvs or not shps:
            continue
        rel = folder.relative_to(root).as_posix()
        if rel in ("", "."):
            rel = ""
        label = STREETLIGHT_FOLDER_TO_LABEL.get(rel, rel or "all_days")
        sources.append(
            StreetLightSource(
                label=label,
                folder=folder,
                csv_path=csvs[0],
                shp_path=shps[0],
                filters_path=filters,
            )
        )

    sources.sort(key=lambda s: s.label)
    return sources


# ---------------------------------------------------------------------------
# Filters.txt parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterMetadata:
    subscription_id: str | None
    organization: str | None
    mode_of_travel: str | None
    zone_library_type: str | None
    output_type: str | None
    data_periods: list[str]
    day_types: list[str]
    day_parts: list[str]
    road_classes: list[str]
    metrics: list[str]


_KV_PATTERN = re.compile(r"^([^:]+):\s*(.*)$")


def parse_filters(path: Path) -> FilterMetadata:
    """Parse a ``Filters.txt`` file into structured metadata.

    The file mixes top-level key/value lines with section-style blocks
    where the section header is a key and the values follow as
    ``  - item`` lines.
    """
    text = path.read_text(encoding="utf-8")

    kv: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            current_section = None
            continue

        stripped = line.lstrip()
        if stripped.startswith("- "):
            if current_section is None:
                continue
            sections.setdefault(current_section, []).append(stripped[2:].strip())
            continue

        m = _KV_PATTERN.match(stripped)
        if not m:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()

        if value == "":
            current_section = key
            sections.setdefault(current_section, [])
        else:
            kv[key] = value
            current_section = None

    return FilterMetadata(
        subscription_id=kv.get("Subscription ID"),
        organization=kv.get("Organization"),
        mode_of_travel=kv.get("Mode of Travel"),
        zone_library_type=kv.get("Zone Library Type"),
        output_type=kv.get("Output Type"),
        data_periods=sections.get("Data Periods", []),
        day_types=sections.get("Day Types", []),
        day_parts=sections.get("Day Parts", []),
        road_classes=sections.get("Road Classes", []),
        metrics=sections.get("Metrics", []),
    )


# ---------------------------------------------------------------------------
# Zone Name parsing
# ---------------------------------------------------------------------------


# Format: "[OSM Name] / [OSM Way ID] / [Split #]"
# Examples:
#   "Oleri Terrace / 11586338 / 7"
#   "I 95 / 12345678 / 12"
# The OSM Name itself can contain spaces and slashes occasionally;
# we anchor on the *last two* slash-separated tokens.
_ZONE_NAME_TAIL = re.compile(r"\s*/\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_zone_name(zone_name: str) -> tuple[str, int | None, int | None]:
    """Return ``(osm_name, osm_way_id, split_index)`` from a Zone Name.

    Returns ``(zone_name, None, None)`` if the trailing pattern does not
    match (so callers can still rely on the first element).
    """
    if zone_name is None:
        return ("", None, None)
    s = str(zone_name)
    m = _ZONE_NAME_TAIL.search(s)
    if not m:
        return (s.strip(), None, None)
    osm_way_id = int(m.group(1))
    split_index = int(m.group(2))
    osm_name = s[: m.start()].strip()
    return (osm_name, osm_way_id, split_index)


# ---------------------------------------------------------------------------
# Per-source loader
# ---------------------------------------------------------------------------


_CSV_NUMERIC_COLS = ("Speed Limit (mph)", "Average Speed (mph)", "Average Volume")


def load_single_source(src: StreetLightSource) -> gpd.GeoDataFrame:
    """Load one StreetLight export folder into a GeoDataFrame.

    The returned frame has one row per StreetLight zone (segment) and
    carries:
      - the original CSV columns (renamed to snake_case)
      - parsed ``osm_name``, ``osm_way_id``, ``split_index``
      - ``day_part_raw`` and ``day_parts`` (list[str])
      - ``source`` (the source label, e.g. ``"weekdays"``)
      - geometry from the shapefile (EPSG:4326)
      - shapefile-side ``direction_deg`` and ``is_bidi``
    """
    meta = parse_filters(src.filters_path)

    df = pd.read_csv(src.csv_path)
    gdf_shp = gpd.read_file(src.shp_path)

    rename = {
        "Data Periods": "data_periods",
        "Mode of Travel": "mode_of_travel",
        "City, County, State": "city_county_state",
        "Road Class": "road_class",
        "Road Name": "road_name",
        "Zone Name": "zone_name",
        "Zone Direction": "zone_direction_deg",
        "Zone is Bi-Direction": "is_bidi",
        "Day Type": "day_type",
        "Day Part": "day_part_raw",
        "Speed Limit (mph)": "speed_limit_mph",
        "Average Speed (mph)": "avg_speed_mph",
        "Average Volume": "avg_volume",
    }
    df = df.rename(columns=rename)

    for col in ("speed_limit_mph", "avg_speed_mph", "avg_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed = df["zone_name"].apply(parse_zone_name)
    df["osm_name"] = [p[0] for p in parsed]
    df["osm_way_id"] = pd.array([p[1] for p in parsed], dtype="Int64")
    df["split_index"] = pd.array([p[2] for p in parsed], dtype="Int64")

    df["day_parts"] = df["day_part_raw"].fillna("").apply(
        lambda s: [p.strip() for p in str(s).split(";") if p.strip()]
    )

    df["source"] = src.label
    df["source_folder"] = str(src.folder)
    df["filter_data_periods"] = "; ".join(meta.data_periods)
    df["filter_day_types"] = "; ".join(meta.day_types)
    df["filter_day_parts"] = "; ".join(meta.day_parts)

    shp = gdf_shp.rename(
        columns={
            "name": "zone_name",
            "road_class": "shp_road_class",
            "direction": "direction_deg",
            "is_bidi": "shp_is_bidi",
        }
    )[["zone_name", "shp_road_class", "direction_deg", "shp_is_bidi", "geometry"]]

    merged = shp.merge(df, on="zone_name", how="inner", validate="one_to_one")

    if len(merged) != len(df):
        missing = set(df["zone_name"]) - set(shp["zone_name"])
        logger.warning(
            "%s: %d CSV rows had no matching shapefile zone (e.g. %s)",
            src.label,
            len(missing),
            sorted(missing)[:3],
        )

    if merged.crs is None:
        merged = merged.set_crs("EPSG:4326")

    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all_sources(
    root: Path = STREETLIGHT_DIR,
    sources: Iterable[StreetLightSource] | None = None,
) -> gpd.GeoDataFrame:
    """Load every Street Scanner export and stack them into one frame."""
    src_list = list(sources) if sources is not None else discover_sources(root)
    if not src_list:
        raise FileNotFoundError(f"No StreetLight sources found under {root}")

    frames = [load_single_source(s) for s in src_list]
    out = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    return out


def restrict_to_study_area(
    gdf: gpd.GeoDataFrame,
    cities: Iterable[str] = STUDY_AREA_CITIES,
) -> gpd.GeoDataFrame:
    """Keep only rows whose City/County/State is in the study area."""
    mask = gdf["city_county_state"].isin(list(cities))
    return gdf.loc[mask].copy()


def pivot_by_source(
    gdf: gpd.GeoDataFrame,
    value_col: str = "avg_volume",
) -> gpd.GeoDataFrame:
    """Pivot a long-format frame to wide on ``source``.

    Returns one row per ``zone_name`` with columns
    ``<value_col>__all_days``, ``<value_col>__weekdays``,
    ``<value_col>__weekend`` (and any other sources present), plus
    geometry and the stable per-zone attributes.
    """
    keep_cols = [
        "zone_name",
        "osm_name",
        "osm_way_id",
        "split_index",
        "road_class",
        "shp_road_class",
        "road_name",
        "city_county_state",
        "direction_deg",
        "is_bidi",
        "speed_limit_mph",
        "geometry",
    ]
    static = (
        gdf[keep_cols]
        .drop_duplicates(subset="zone_name")
        .set_index("zone_name")
    )

    pivoted = gdf.pivot_table(
        index="zone_name",
        columns="source",
        values=value_col,
        aggfunc="mean",
    )
    pivoted.columns = [f"{value_col}__{c}" for c in pivoted.columns]

    out = static.join(pivoted, how="left").reset_index()
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


# ---------------------------------------------------------------------------
# Parquet cache
# ---------------------------------------------------------------------------


_CACHE_PATH = DATA_STAGE1_DIR / "streetlight_segments.parquet"


def load_cached(
    root: Path = STREETLIGHT_DIR,
    *,
    rebuild: bool = False,
    cache_path: Path = _CACHE_PATH,
) -> gpd.GeoDataFrame:
    """Load all sources, caching to a Parquet file.

    The list[str] ``day_parts`` column is serialized as a
    semicolon-joined string in the cache (Parquet handles lists, but
    GeoPandas' to_parquet path is finicky with object-dtype lists across
    versions; the cached form is simpler and stable).
    """
    if cache_path.exists() and not rebuild:
        try:
            gdf = gpd.read_parquet(cache_path)
            gdf["day_parts"] = gdf["day_part_raw"].fillna("").apply(
                lambda s: [p.strip() for p in str(s).split(";") if p.strip()]
            )
            return gdf
        except Exception as exc:  # pragma: no cover - cache corruption fallback
            logger.warning("Cache read failed (%s); rebuilding", exc)

    gdf = load_all_sources(root)
    to_save = gdf.drop(columns=["day_parts"], errors="ignore")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    to_save.to_parquet(cache_path)
    return gdf


__all__ = [
    "FilterMetadata",
    "StreetLightSource",
    "discover_sources",
    "load_all_sources",
    "load_cached",
    "load_single_source",
    "parse_filters",
    "parse_zone_name",
    "pivot_by_source",
    "restrict_to_study_area",
]
