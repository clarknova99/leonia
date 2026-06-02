"""Build the canonical Leonia data lake from raw StreetLight exports.

This is the **one** orchestrator that turns every raw export under
``streetlight/`` into analysis-ready parquet files under
``data/processed/``. It also rebuilds the small set of derived
analytics tables that downstream reports consume.

Usage:

    venv/bin/python scripts/00_build_datasets.py

    venv/bin/python scripts/00_build_datasets.py --skip-derived
    venv/bin/python scripts/00_build_datasets.py --only streetscanner za
    venv/bin/python scripts/00_build_datasets.py --skip work_block_groups

Outputs (paths relative to repo root):

    data/processed/streetlight/
        streetscanner_segments.parquet            (Street Scanner)
        bridge_od.parquet, bridge_od_zones.parquet,
            bridge_attributes.parquet             (Bridge OD)
        congestion_links.parquet,
            congestion_zones.parquet              (Congestion Trends)
        network_performance_segments.parquet,
            network_performance_prediction.parquet,
            network_performance_zones.parquet,
            network_performance_monthly.parquet,
            network_performance_shapes.parquet    (Network Performance)
        za_volume.parquet, za_trips.parquet,
            za_home_distance.parquet,
            za_home_zips_top.parquet,
            za_home_state.parquet,
            za_work_distance.parquet,
            za_work_block_groups.parquet,
            za_tourist_summary.parquet,
            za_line_shapes.parquet,
            za_polygon_shapes.parquet             (ZA on Leonia streets)
        za_volume_history.parquet,
            za_trips_history.parquet,
            za_line_shapes_history.parquet,
            za_polygon_shapes_history.parquet     (ZA historical baseline 2038018)
        _manifest.json
    data/processed/derived/
        cutthrough_index.parquet
        hourly_profiles.parquet
        peak_intensity_am.parquet
        peak_intensity_pm.parquet
        _manifest.json

See ``docs/DATA.md`` for the full column-level schema of every output.

Design notes:

* Products are **explicit**: adding a new StreetLight product means
  adding a function in this script. We deliberately avoid auto-
  discovery so unexpected folders never silently slip into the lake.
* The script is **idempotent** and safe to rerun. Each builder
  overwrites its parquet(s) and updates the manifest.
* Geometry columns are persisted as **GeoParquet** (WGS84). Open in
  geopandas via ``gpd.read_parquet(...)`` or in QGIS >=3.32 directly.
* Builds skip cleanly when the raw export folder is missing — the
  manifest just doesn't gain that entry. This lets users build a
  partial lake (e.g. before all StreetLight exports are downloaded).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from leonia_traffic.data import (
    bridge_od_loader as bridge,
    congestion_loader as cong,
    cutthrough_omd_loader as omd,
    network_performance_loader as netperf,
    streetlight_loader as scanner,
    streetscanner_trend_loader as scanner_trend,
    za_streets_loader as za,
)
from leonia_traffic.data.dataset_io import (
    CANONICAL_DIR,
    DERIVED_DIR,
    CanonicalFiles,
    DerivedFiles,
    write_dataframe,
    write_geodataframe,
)

# Wide column display when this script prints a row count summary.
pd.set_option("display.width", 160)


# ---------------------------------------------------------------------------
# Per-product builders
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _ok(name: str, n_rows: int) -> None:
    print(f"  wrote {name}  ({n_rows:,} rows)")


def _skip(name: str, reason: str) -> None:
    print(f"  skip {name}: {reason}")


def build_streetscanner(force_rebuild: bool = True) -> dict[str, int]:
    """Street Scanner: tertiary-segment daily speed/volume.

    Reads ``streetlight/<root>``, ``streetlight/weekdays/``, and
    ``streetlight/weekend/``. Output is one tidy long-format
    GeoParquet keyed by ``(zone_name, source, day_type, day_part_raw)``.

    The existing ``leonia_traffic.data.streetlight_loader`` already
    builds a tidy frame; we rebuild it here, persist via the dataset
    IO helper (so the manifest gains an entry), and store it under
    the new canonical folder. The legacy
    ``data/processed/streetlight_segments.parquet`` is left untouched
    for backward compatibility with any external notebook references.
    """
    _section("Street Scanner")
    try:
        gdf = scanner.load_cached(rebuild=force_rebuild)
    except Exception as exc:  # pragma: no cover - degraded path
        _skip(CanonicalFiles.streetscanner_segments, f"load failed ({exc})")
        return {}
    if gdf is None or gdf.empty:
        _skip(CanonicalFiles.streetscanner_segments, "no exports found")
        return {}
    # Drop the list-typed ``day_parts`` column; we keep the
    # canonical ``day_part_raw`` string and a couple of split-out
    # ints (filter_day_parts) per the loader contract.
    gdf = gdf.drop(columns=["day_parts"], errors="ignore")
    write_geodataframe(
        gdf,
        folder=CANONICAL_DIR,
        name=CanonicalFiles.streetscanner_segments,
        sources=[scanner.STREETLIGHT_DIR],
    )
    _ok(CanonicalFiles.streetscanner_segments, len(gdf))
    return {CanonicalFiles.streetscanner_segments: len(gdf)}


def build_bridge_od() -> dict[str, int]:
    """Bridge OD: origin/destination analysis to/from the GWB approach.

    Three outputs:

    * ``bridge_od.parquet`` — tidy OD volumes per origin × destination
      × day-type × day-part.
    * ``bridge_od_zones.parquet`` — zone geometries (GeoParquet).
    * ``bridge_attributes.parquet`` — driver attributes (home distance,
      trip purpose, etc.) per zone, with attribute kind in a single
      column for tidy filtering.
    """
    _section("Bridge OD")
    paths = bridge.discover_bridge_od()
    if paths is None:
        _skip("bridge_*", "streetlight/bridge_destination/ not found")
        return {}
    counts: dict[str, int] = {}
    od = bridge.load_bridge_od()
    write_dataframe(
        od, folder=CANONICAL_DIR, name=CanonicalFiles.bridge_od,
        sources=[bridge.BRIDGE_OD_DIR],
    )
    _ok(CanonicalFiles.bridge_od, len(od))
    counts[CanonicalFiles.bridge_od] = len(od)

    try:
        zones = bridge.load_bridge_zone_shapes(kind="line")
        write_geodataframe(
            zones, folder=CANONICAL_DIR, name=CanonicalFiles.bridge_od_zones,
            sources=[bridge.BRIDGE_OD_DIR],
        )
        _ok(CanonicalFiles.bridge_od_zones, len(zones))
        counts[CanonicalFiles.bridge_od_zones] = len(zones)
    except Exception as exc:  # pragma: no cover
        _skip(CanonicalFiles.bridge_od_zones, f"{exc}")

    attrs = bridge.load_bridge_attributes()
    if attrs is not None and not attrs.empty:
        write_dataframe(
            attrs, folder=CANONICAL_DIR, name=CanonicalFiles.bridge_attributes,
            sources=[bridge.BRIDGE_OD_DIR],
        )
        _ok(CanonicalFiles.bridge_attributes, len(attrs))
        counts[CanonicalFiles.bridge_attributes] = len(attrs)
    return counts


def build_streetscanner_trend() -> dict[str, int]:
    """Street Scanner Trend: monthly volume Jan 2023 → present.

    Long-format parquet (one row per zone × month) plus a GeoParquet of
    the underlying line geometries. Downstream code uses this to detect
    accelerating cut-through and seasonal patterns.
    """
    _section("Street Scanner Trend")
    paths = scanner_trend.discover_streetscanner_trend()
    if paths is None:
        _skip(CanonicalFiles.streetscanner_trend,
              "streetlight/streetscanner_trend/ not found")
        return {}

    counts: dict[str, int] = {}
    df = scanner_trend.load_streetscanner_trend()
    write_dataframe(
        df, folder=CANONICAL_DIR, name=CanonicalFiles.streetscanner_trend,
        sources=[paths.folder],
    )
    _ok(CanonicalFiles.streetscanner_trend, len(df))
    counts[CanonicalFiles.streetscanner_trend] = len(df)

    try:
        gdf = scanner_trend.load_streetscanner_trend_shapes()
        if gdf is not None and not gdf.empty:
            write_geodataframe(
                gdf, folder=CANONICAL_DIR,
                name=CanonicalFiles.streetscanner_trend_shapes,
                sources=[paths.folder],
            )
            _ok(CanonicalFiles.streetscanner_trend_shapes, len(gdf))
            counts[CanonicalFiles.streetscanner_trend_shapes] = len(gdf)
    except Exception as exc:  # pragma: no cover
        _skip(CanonicalFiles.streetscanner_trend_shapes, f"{exc}")
    return counts


def build_cutthrough_omd() -> dict[str, int]:
    """O-D + Middle-Filter cut-through analysis.

    Emits five parquets: OMD volumes, per-triple trip-attribute
    distributions, per-zone activity totals, the zone roster, and the
    union of origin / middle / destination line shapes.
    """
    _section("Cut-through O-D + Middle Filter")
    paths = omd.discover_cutthrough_omd()
    if paths is None:
        _skip("cutthrough_omd_*",
              "streetlight/2034993_cut_through/ not found")
        return {}

    counts: dict[str, int] = {}
    src = [paths.folder]

    df = omd.load_cutthrough_omd()
    write_dataframe(
        df, folder=CANONICAL_DIR, name=CanonicalFiles.cutthrough_omd,
        sources=src,
    )
    _ok(CanonicalFiles.cutthrough_omd, len(df))
    counts[CanonicalFiles.cutthrough_omd] = len(df)

    tdf = omd.load_cutthrough_omd_trips()
    write_dataframe(
        tdf, folder=CANONICAL_DIR, name=CanonicalFiles.cutthrough_omd_trips,
        sources=src,
    )
    _ok(CanonicalFiles.cutthrough_omd_trips, len(tdf))
    counts[CanonicalFiles.cutthrough_omd_trips] = len(tdf)

    zdf = omd.load_cutthrough_omd_zone_activity()
    if zdf is not None and not zdf.empty:
        write_dataframe(
            zdf, folder=CANONICAL_DIR,
            name=CanonicalFiles.cutthrough_omd_zone_activity, sources=src,
        )
        _ok(CanonicalFiles.cutthrough_omd_zone_activity, len(zdf))
        counts[CanonicalFiles.cutthrough_omd_zone_activity] = len(zdf)

    rdf = omd.load_cutthrough_omd_roster()
    if rdf is not None and not rdf.empty:
        write_dataframe(
            rdf, folder=CANONICAL_DIR,
            name=CanonicalFiles.cutthrough_omd_roster, sources=src,
        )
        _ok(CanonicalFiles.cutthrough_omd_roster, len(rdf))
        counts[CanonicalFiles.cutthrough_omd_roster] = len(rdf)

    try:
        sgdf = omd.load_cutthrough_omd_shapes(kind="line")
        if sgdf is not None and not sgdf.empty:
            write_geodataframe(
                sgdf, folder=CANONICAL_DIR,
                name=CanonicalFiles.cutthrough_omd_shapes, sources=src,
            )
            _ok(CanonicalFiles.cutthrough_omd_shapes, len(sgdf))
            counts[CanonicalFiles.cutthrough_omd_shapes] = len(sgdf)
    except Exception as exc:  # pragma: no cover
        _skip(CanonicalFiles.cutthrough_omd_shapes, f"{exc}")

    return counts


def build_congestion() -> dict[str, int]:
    """Congestion Trends: per-link reliability and delay."""
    _section("Congestion Trends")
    paths = cong.discover_congestion()
    if paths is None:
        _skip("congestion_*", "streetlight/congestion/ not found")
        return {}
    counts: dict[str, int] = {}
    links = cong.load_congestion()
    write_dataframe(
        links, folder=CANONICAL_DIR, name=CanonicalFiles.congestion_links,
        sources=[cong.CONGESTION_DIR],
    )
    _ok(CanonicalFiles.congestion_links, len(links))
    counts[CanonicalFiles.congestion_links] = len(links)

    try:
        zones = cong.load_congestion_zones()
        write_geodataframe(
            zones, folder=CANONICAL_DIR, name=CanonicalFiles.congestion_zones,
            sources=[cong.CONGESTION_DIR],
        )
        _ok(CanonicalFiles.congestion_zones, len(zones))
        counts[CanonicalFiles.congestion_zones] = len(zones)
    except Exception as exc:  # pragma: no cover
        _skip(CanonicalFiles.congestion_zones, f"{exc}")
    return counts


def build_network_performance(skip_files: set[str] | None = None) -> dict[str, int]:
    """Network Performance: segment-level volume/speed/VMT/VHD.

    Emits five parquets: the main hourly metrics table, the 95%
    prediction-interval table, the zone roster, the per-segment line
    geometry (GeoParquet), and — unless skipped — the large monthly
    metrics table.

    The monthly CSV is ~560 MB; pass its canonical filename to
    ``skip_files`` (CLI: ``--skip network_performance_monthly.parquet``)
    to bypass it when iterating.
    """
    _section("Network Performance")
    skip_files = skip_files or set()
    paths = netperf.discover_network_performance()
    if paths is None:
        _skip("network_performance_*",
              "streetlight/2038116_leonia_network/ not found")
        return {}

    counts: dict[str, int] = {}
    src = [netperf.NETWORK_PERF_DIR]

    def _emit(df, name: str) -> None:
        if name in skip_files:
            _skip(name, "user requested skip")
            return
        if df is None or df.empty:
            _skip(name, "empty in raw export")
            return
        write_dataframe(df, folder=CANONICAL_DIR, name=name, sources=src)
        _ok(name, len(df))
        counts[name] = len(df)

    _emit(netperf.load_network_performance(),
          CanonicalFiles.network_performance_segments)
    _emit(netperf.load_network_performance_prediction(),
          CanonicalFiles.network_performance_prediction)
    _emit(netperf.load_network_performance_zones(),
          CanonicalFiles.network_performance_zones)

    if CanonicalFiles.network_performance_monthly in skip_files:
        _skip(CanonicalFiles.network_performance_monthly, "user requested skip")
    else:
        monthly = netperf.load_network_performance_monthly()
        _emit(monthly, CanonicalFiles.network_performance_monthly)

    try:
        gdf = netperf.load_network_performance_shapes()
        if gdf is not None and not gdf.empty:
            write_geodataframe(
                gdf, folder=CANONICAL_DIR,
                name=CanonicalFiles.network_performance_shapes, sources=src,
            )
            _ok(CanonicalFiles.network_performance_shapes, len(gdf))
            counts[CanonicalFiles.network_performance_shapes] = len(gdf)
    except Exception as exc:  # pragma: no cover
        _skip(CanonicalFiles.network_performance_shapes, f"{exc}")
    return counts


def build_za_streets(skip_files: set[str] | None = None) -> dict[str, int]:
    """Zone Activity on Leonia tertiary streets.

    Emits ten parquets: main volume, per-trip distributions, four
    home-side tables, two work-side tables, the tourist summary, and
    two GeoParquets for the line / polygon shapes.

    Pass ``skip_files`` to bypass particularly slow tables (the
    ~2.8M-row work-block-groups file takes several seconds on the
    first build).
    """
    _section("ZA — Leonia streets")
    skip_files = skip_files or set()
    if za.discover_za_streets() is None:
        _skip("za_*", "streetlight/2034227_leonia_streets/ not found")
        return {}

    counts: dict[str, int] = {}
    src = [za.ZA_STREETS_DIR]

    def _emit(df, name: str) -> None:
        if name in skip_files:
            _skip(name, "user requested skip")
            return
        if df is None or df.empty:
            _skip(name, "empty in raw export")
            return
        write_dataframe(df, folder=CANONICAL_DIR, name=name, sources=src)
        _ok(name, len(df))
        counts[name] = len(df)

    def _emit_geo(gdf, name: str) -> None:
        if name in skip_files:
            _skip(name, "user requested skip")
            return
        if gdf is None or gdf.empty:
            _skip(name, "empty in raw export")
            return
        write_geodataframe(gdf, folder=CANONICAL_DIR, name=name, sources=src)
        _ok(name, len(gdf))
        counts[name] = len(gdf)

    _emit(za.load_za_main(),              CanonicalFiles.za_volume)
    _emit(za.load_za_trip(),              CanonicalFiles.za_trips)
    _emit(za.load_za_home_distance(),     CanonicalFiles.za_home_distance)
    _emit(za.load_za_home_zips_top(),     CanonicalFiles.za_home_zips_top)
    _emit(za.load_za_home_state(),        CanonicalFiles.za_home_state)
    _emit(za.load_za_work_distance(),     CanonicalFiles.za_work_distance)
    _emit(za.load_za_work_block_groups(), CanonicalFiles.za_work_block_groups)
    _emit(za.load_za_tourist_summary(),   CanonicalFiles.za_tourist_summary)
    _emit_geo(za.load_za_line_shapes(),   CanonicalFiles.za_line_shapes)
    _emit_geo(za.load_za_polygon_shapes(), CanonicalFiles.za_polygon_shapes)
    return counts


def build_za_history(skip_files: set[str] | None = None) -> dict[str, int]:
    """Historical ZA baseline on Leonia tertiary streets (analysis 2038018).

    A single multi-year aggregate covering
    ``Jan 01, 2022 - May 31, 2023; Jan 01, 2024 - Apr 30, 2026`` for all 375
    OSM tertiary segments. Unlike the recent-year ``build_za_streets`` export
    it has **no Visitor/Resident split and no Home/Work folder**, so only the
    main volume, per-trip distributions, and the two shapefiles are emitted.
    Kept separate from the ``za_*`` tables so the Visitor-filtered analyses
    stay comparable.
    """
    _section("ZA — historical baseline (2038018)")
    skip_files = skip_files or set()
    folder = za.ZA_STREETS_HISTORY_DIR
    if za.discover_za_streets(folder) is None:
        _skip("za_*_history", f"{folder} not found")
        return {}

    counts: dict[str, int] = {}
    src = [folder]

    def _emit(df, name: str) -> None:
        if name in skip_files:
            _skip(name, "user requested skip")
            return
        if df is None or df.empty:
            _skip(name, "empty in raw export")
            return
        write_dataframe(df, folder=CANONICAL_DIR, name=name, sources=src)
        _ok(name, len(df))
        counts[name] = len(df)

    def _emit_geo(gdf, name: str) -> None:
        if name in skip_files:
            _skip(name, "user requested skip")
            return
        if gdf is None or gdf.empty:
            _skip(name, "empty in raw export")
            return
        write_geodataframe(gdf, folder=CANONICAL_DIR, name=name, sources=src)
        _ok(name, len(gdf))
        counts[name] = len(gdf)

    _emit(za.load_za_main(folder),              CanonicalFiles.za_volume_history)
    _emit(za.load_za_trip(folder),              CanonicalFiles.za_trips_history)
    _emit_geo(za.load_za_line_shapes(folder),   CanonicalFiles.za_line_shapes_history)
    _emit_geo(za.load_za_polygon_shapes(folder), CanonicalFiles.za_polygon_shapes_history)
    return counts


# ---------------------------------------------------------------------------
# Derived analytics builders
# ---------------------------------------------------------------------------


def build_derived() -> dict[str, int]:
    """Recompute the derived analytics tables from canonical data."""
    _section("Derived analytics")

    if not (CANONICAL_DIR / CanonicalFiles.za_volume).exists():
        _skip("derived/*", "ZA canonical not built; rerun without --skip za")
        return {}

    # Local imports keep cold start cheap when --skip-derived is set.
    from leonia_traffic.analysis import cutthrough_streets as cs
    from leonia_traffic.analysis.jurisdiction import filter_segments_to_leonia
    from leonia_traffic.data.dataset_io import read_dataset, read_geodataset

    counts: dict[str, int] = {}

    za_main = read_dataset(CANONICAL_DIR, CanonicalFiles.za_volume)
    za_trip = read_dataset(CANONICAL_DIR, CanonicalFiles.za_trips)
    za_home = read_dataset(CANONICAL_DIR, CanonicalFiles.za_home_distance)
    line_gdf = read_geodataset(CANONICAL_DIR, CanonicalFiles.za_line_shapes)

    # Composite cut-through index, scoped to in-borough municipal streets.
    EXCLUDE_NAME_TAGS = {
        "motorway_link", "trunk_link", "primary_link",
        "secondary_link", "tertiary_link",
        "service", "track", "unclassified",
        "tertiary",
    }
    imbalance = cs.weekday_weekend_imbalance(za_main)
    weekday_vol = cs.weekday_all_day_volume(za_main)
    long_trip = cs.long_trip_share(za_trip)
    speeding = cs.speeding_share(za_trip)
    home_share = cs.non_local_home_share(za_home)
    for df in (imbalance, weekday_vol, long_trip, speeding, home_share):
        if df is not None and not df.empty and "street_name" in df.columns:
            df.drop(df.index[df["street_name"].isin(EXCLUDE_NAME_TAGS)],
                    inplace=True)

    index_df = cs.composite_cutthrough_index(
        imbalance_df=imbalance,
        weekday_volume_df=weekday_vol,
        long_trip_df=long_trip,
        speeding_df=speeding,
        home_dist_df=home_share,
    )
    if not index_df.empty and line_gdf is not None and not line_gdf.empty:
        index_df = filter_segments_to_leonia(
            index_df.rename(columns={"street_name": "osm_name"}),
            line_gdf.rename(columns={"name": "zone_name"}),
        ).rename(columns={"osm_name": "street_name"})
        index_df = index_df.sort_values("cutthrough_index",
                                        ascending=False).reset_index(drop=True)
        index_df["rank"] = index_df.index + 1

    write_dataframe(
        index_df, folder=DERIVED_DIR, name=DerivedFiles.cutthrough_index,
        sources=[
            CANONICAL_DIR / CanonicalFiles.za_volume,
            CANONICAL_DIR / CanonicalFiles.za_trips,
            CANONICAL_DIR / CanonicalFiles.za_home_distance,
            CANONICAL_DIR / CanonicalFiles.za_line_shapes,
        ],
    )
    _ok(DerivedFiles.cutthrough_index, len(index_df))
    counts[DerivedFiles.cutthrough_index] = len(index_df)

    # Hourly profile (All-Days day-type for broadest coverage).
    hourly = cs.weekday_hourly_profile(za_main, day_types=(cs.ALL_DAYS_TYPE,))
    write_dataframe(
        hourly, folder=DERIVED_DIR, name=DerivedFiles.hourly_profiles,
        sources=[CANONICAL_DIR / CanonicalFiles.za_volume],
    )
    _ok(DerivedFiles.hourly_profiles, len(hourly))
    counts[DerivedFiles.hourly_profiles] = len(hourly)

    # Peak-vs-midday intensity (AM and PM).
    intensity_am = cs.peak_hour_intensity(
        za_main, peak_hours=cs.PEAK_AM_HOURS, day_types=(cs.ALL_DAYS_TYPE,),
    )
    write_dataframe(
        intensity_am, folder=DERIVED_DIR, name=DerivedFiles.peak_intensity_am,
        sources=[CANONICAL_DIR / CanonicalFiles.za_volume],
    )
    _ok(DerivedFiles.peak_intensity_am, len(intensity_am))
    counts[DerivedFiles.peak_intensity_am] = len(intensity_am)

    intensity_pm = cs.peak_hour_intensity(
        za_main, peak_hours=cs.PEAK_PM_HOURS, day_types=(cs.ALL_DAYS_TYPE,),
    )
    write_dataframe(
        intensity_pm, folder=DERIVED_DIR, name=DerivedFiles.peak_intensity_pm,
        sources=[CANONICAL_DIR / CanonicalFiles.za_volume],
    )
    _ok(DerivedFiles.peak_intensity_pm, len(intensity_pm))
    counts[DerivedFiles.peak_intensity_pm] = len(intensity_pm)

    # Street trend metrics (only if the trend canonical exists).
    trend_path = CANONICAL_DIR / CanonicalFiles.streetscanner_trend
    if trend_path.exists():
        from leonia_traffic.analysis import street_trend as st
        trend_df = read_dataset(CANONICAL_DIR, CanonicalFiles.streetscanner_trend)
        st_df = st.street_trend_metrics(trend_df)
        write_dataframe(
            st_df, folder=DERIVED_DIR, name=DerivedFiles.street_trend,
            sources=[trend_path],
        )
        _ok(DerivedFiles.street_trend, len(st_df))
        counts[DerivedFiles.street_trend] = len(st_df)
    else:
        _skip(DerivedFiles.street_trend, "streetscanner_trend canonical missing")

    # Cut-through attribution + OD bypass pairs (require OMD canonical).
    omd_path = CANONICAL_DIR / CanonicalFiles.cutthrough_omd
    if omd_path.exists():
        from leonia_traffic.analysis import cutthrough_attribution as cta
        omd_df = read_dataset(CANONICAL_DIR, CanonicalFiles.cutthrough_omd)
        omd_trip_path = CANONICAL_DIR / CanonicalFiles.cutthrough_omd_trips
        trips_df = (read_dataset(CANONICAL_DIR, CanonicalFiles.cutthrough_omd_trips)
                    if omd_trip_path.exists() else None)
        attr_df = cta.per_street_attribution(omd_df, trips_df)
        write_dataframe(
            attr_df, folder=DERIVED_DIR,
            name=DerivedFiles.cutthrough_attribution,
            sources=[omd_path] + ([omd_trip_path] if omd_trip_path.exists() else []),
        )
        _ok(DerivedFiles.cutthrough_attribution, len(attr_df))
        counts[DerivedFiles.cutthrough_attribution] = len(attr_df)

        bypass_df = cta.top_od_bypass_pairs(omd_df)
        write_dataframe(
            bypass_df, folder=DERIVED_DIR,
            name=DerivedFiles.od_bypass_pairs, sources=[omd_path],
        )
        _ok(DerivedFiles.od_bypass_pairs, len(bypass_df))
        counts[DerivedFiles.od_bypass_pairs] = len(bypass_df)
    else:
        _skip(DerivedFiles.cutthrough_attribution, "cutthrough_omd canonical missing")

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


KNOWN_PRODUCTS = {
    "streetscanner": build_streetscanner,
    "streetscanner_trend": build_streetscanner_trend,
    "bridge": build_bridge_od,
    "congestion": build_congestion,
    "network_performance": build_network_performance,
    "cutthrough_omd": build_cutthrough_omd,
    "za": build_za_streets,
    "za_history": build_za_history,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical Leonia data lake from raw "
                    "StreetLight exports."
    )
    parser.add_argument(
        "--only", nargs="+", choices=list(KNOWN_PRODUCTS.keys()),
        help="Build only the named product(s). Default: build all.",
    )
    parser.add_argument(
        "--skip-derived", action="store_true",
        help="Skip the derived/ analytics build. Use when you only want "
             "to refresh canonical raw tables.",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Skip specific canonical files by name for the ZA and "
             "Network Performance products (e.g. "
             "'za_work_block_groups.parquet' or "
             "'network_performance_monthly.parquet'). Useful when "
             "iterating; the work-block-groups CSV is 2.8M rows and the "
             "Network Performance monthly CSV is ~560 MB.",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    targets = list(args.only) if args.only else list(KNOWN_PRODUCTS.keys())
    summary: dict[str, int] = {}

    for product in targets:
        builder = KNOWN_PRODUCTS[product]
        kw = {}
        if product in ("za", "network_performance"):
            kw["skip_files"] = set(args.skip)
        summary.update(builder(**kw))

    if not args.skip_derived:
        summary.update(build_derived())

    print(f"\nBuilt {len(summary)} dataset(s) in {time.time() - t0:.1f}s")
    print(f"Canonical: {CANONICAL_DIR}")
    print(f"Derived:   {DERIVED_DIR}")
    print("See docs/DATA.md for the column-level schema of every file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
