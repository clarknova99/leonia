"""Build the NJDOT crash overlay datasets.

Two data sources are supported, both producing the same output
schema (``data/processed/crashes/njdot_crashes.parquet`` +
``crashes_by_segment.parquet``):

1. **NJDOT Crash Data Dashboard JSON** (``--source dashboard``,
   default). Pulls 2019-2026 crash records from the public dashboard
   API at ``https://njdot.aashtowaresafety.net/api/dashboards/...``,
   with KABCO severity ratings, ~92% geocoded points, NJDOT-snapped
   ``Geopoint Calculated`` fallback, pre-computed per-segment AADT /
   CRC, and SHSP emphasis-area tags. Cached under
   ``data/raw/njdot_dashboard/leonia_dashboard.json``.

2. **Legacy NJDOT raw zips** (``--source zip``). Pulls 2017-2022
   per-county fixed-width text from
   ``https://www.nj.gov/transportation/refdata/accident/<YEAR>/...``.
   Lower geocoding rate (~30% raw + ~35% name-based fallback) and
   F/I/P severity scale, but no auth needed and richer per-row text
   fields. Useful for historic time-series comparisons or when the
   dashboard auth token expires.

The dashboard endpoint is anonymous-friendly *if* you supply the
``entityToken`` query parameter that the SPA bakes into its URLs.
The token is opaque, signed by Numetric, and identifies a specific
metric/dashboard-tab combination — it does not encode any PII or
session cookie. We've checked it in to a default value pulled from
the public dashboard, but the user can pass ``--token ...`` to
override if it ever rotates.

Usage
-----

::

    # Default: pull from the dashboard (2019-2026, ~1,700 rows).
    venv/bin/python scripts/14_build_crash_overlay.py

    # Force a re-fetch from the dashboard (cache busts).
    venv/bin/python scripts/14_build_crash_overlay.py --refresh

    # Restrict to Leonia local streets (drops Interstate/State Hwy).
    venv/bin/python scripts/14_build_crash_overlay.py --local-only

    # Use the legacy zip pipeline instead.
    venv/bin/python scripts/14_build_crash_overlay.py --source zip
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from leonia_traffic.config import DATA_PROCESSED_DIR  # noqa: E402
from leonia_traffic.data.dataset_io import (  # noqa: E402
    CRASHES_DIR,
    CrashFiles,
    write_dataframe,
)
from leonia_traffic.data.njdot_crash_loader import (  # noqa: E402
    aggregate_by_segment,
    assign_to_osm_way,
    geocode_by_street_name,
    parse_crash_table,
    parse_dashboard_json,
    parse_pedestrian_table,
)
from leonia_traffic.sumo.net_lookup import (  # noqa: E402
    load_meta_lookup,
    load_sumo_edge_geometries,
)

logger = logging.getLogger("build_crash_overlay")


RAW_ROOT = REPO_ROOT / "data" / "raw" / "njdot_crashes"
DASHBOARD_DIR = REPO_ROOT / "data" / "raw" / "njdot_dashboard"
SUMO_DIR = DATA_PROCESSED_DIR / "sumo"
NET_PATH = SUMO_DIR / "leonia.net.xml"
META_PATH = SUMO_DIR / "leonia.edgedata.meta.csv"

# Bergen county + Leonia muni code (zip pipeline only).
DEFAULT_COUNTY = "Bergen"
DEFAULT_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
LEONIA_MUNI_CODES = ["29"]

# Per-county zip URL pattern (zip pipeline).
ZIP_URL_PATTERN = (
    "https://www.nj.gov/transportation/refdata/accident/{year}/"
    "{county}{year}{table}.zip"
)


# ---------------------------------------------------------------------------
# NJDOT Crash Data Dashboard (Numetric / AASHTOWare Safety) fetcher
# ---------------------------------------------------------------------------

# These IDs are baked into the public dashboard URL. They never
# change unless NJDOT rebuilds the dashboard, in which case the
# user can override with --dashboard-id / --metric-id / --token.
DEFAULT_DASHBOARD_ID = "7769c554-2a58-4914-b727-6b2d4b178ffc"
DEFAULT_METRIC_ID = "8f8f404e-93ea-46e4-bdf9-6de5078bd33a"
DEFAULT_DATASET_ID = "4008b3e5-a75c-4efe-ac1f-4dd7c7e26a65"
DEFAULT_ENTITY_TOKEN = (
    "alP8W0VPOKNH3wEp7FUATHDwB1VhTOxZyE6%2Fr8wGirHbTRRwhZgqyCCkzexZ1"
    "RQvrnAvTzRfsCD2yxo9xEKvLFJEVbsspbeT%2B646OgcItGOL8dxjsUsi0dpl5"
    "9y8AGsxHLmfXxNaOuAQbU1gs6SinRxMG8Zy9MnQGArltQSrSD8%3D"
)
PAGE_SIZE = 500          # capped server-side; dashboard always returns 500
MAX_PAGES = 40           # safety stop for cursor-paged fetches

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 Chrome/124.0.0.0")


def _muni_filter(muni_value: str = "Leonia Boro",
                 dataset_id: str = DEFAULT_DATASET_ID) -> dict:
    """The dashboard's ``Municipality`` group filter shape."""
    inner = {
        "id": "7ba32315-d504-49c6-ac72-4ee35fa61081",
        "datasetId": dataset_id,
        "key": "Municipality", "field": "Municipality",
        "displayName": "Municipality",
        "value": muni_value,
        "filter": "term", "type": "term",
        "must": True, "joinOperation": None,
    }
    return {
        **inner,
        "value": [muni_value],
        "filter": "group",
        "filters": [inner],
    }


def fetch_dashboard_rows(
    *,
    muni_value: str = "Leonia Boro",
    dashboard_id: str = DEFAULT_DASHBOARD_ID,
    metric_id: str = DEFAULT_METRIC_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    entity_token: str = DEFAULT_ENTITY_TOKEN,
) -> list[dict]:
    """Cursor-paginate the NJDOT dashboard ``search`` endpoint.

    The endpoint always returns 500 rows per call regardless of any
    ``size``/``limit`` parameters (its schema is strict — only
    ``filters`` + ``sorting`` are accepted). To page past the cap,
    we sort by ``id_cr DESC`` and add a ``range filter`` on
    ``id_cr < <last_seen>`` for each subsequent call. Stops when
    a page returns no fresh rows.
    """
    url = (
        "https://njdot.aashtowaresafety.net/api/dashboards/"
        f"{dashboard_id}/metrics/{metric_id}/search"
        f"?entityToken={entity_token}"
    )
    sorting = {"direction": "desc", "sortBy": "id_cr"}
    base_filters = [_muni_filter(muni_value, dataset_id)]

    all_rows: list[dict] = []
    seen_ids: set[int] = set()
    cursor: int | None = None
    total_expected: int | None = None

    for page in range(1, MAX_PAGES + 1):
        filters = list(base_filters)
        if cursor is not None:
            filters.append({
                "id": "page-window",
                "datasetId": dataset_id,
                "key": "id_cr", "field": "id_cr",
                "displayName": "Crash ID",
                "filter": "range", "type": "number",
                "lt": cursor, "must": True,
            })

        body = {"filters": filters, "sorting": sorting}
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "User-Agent": UA,
                "Content-Type": "application/json",
                "Origin": "https://njdot.aashtowaresafety.net",
                "Referer": ("https://njdot.aashtowaresafety.net/"
                            "njdot-crash-data-dashboard"),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
        if not payload.get("success"):
            raise RuntimeError(f"dashboard error: {payload}")

        rows = payload["data"]["rows"]
        if total_expected is None:
            total_expected = payload["data"]["totalRows"]
            logger.info("dashboard reports %d total rows", total_expected)

        fresh = [r for r in rows if r["id_cr"] not in seen_ids]
        for r in fresh:
            seen_ids.add(r["id_cr"])
        all_rows.extend(fresh)
        logger.info("page %d: fetched %d (fresh %d) — running total %d",
                    page, len(rows), len(fresh), len(all_rows))

        if not fresh:
            break
        cursor = rows[-1]["id_cr"]
        if total_expected is not None and len(all_rows) >= total_expected:
            break

    return all_rows


def _ensure_dashboard_cached(
    *,
    refresh: bool,
    cache_path: Path,
    muni_value: str,
    entity_token: str,
) -> Path:
    """Fetch the dashboard JSON if missing or ``--refresh`` is set."""
    if cache_path.exists() and not refresh:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = fetch_dashboard_rows(
        muni_value=muni_value, entity_token=entity_token,
    )
    payload = {
        "success": True,
        "data": {"rows": rows, "totalRows": len(rows)},
        "_source": ("NJDOT Crash Data Dashboard "
                    "(Numetric/AASHTOWare Safety)"),
        "_filter": f"Municipality = {muni_value}",
        "_endpoint": ("https://njdot.aashtowaresafety.net/api/"
                      "dashboards/.../metrics/.../search"),
    }
    with cache_path.open("w") as fh:
        json.dump(payload, fh)
    logger.info("wrote %d rows to %s (%d bytes)",
                len(rows), cache_path, cache_path.stat().st_size)
    return cache_path


# ---------------------------------------------------------------------------
# Legacy NJDOT zip downloader
# ---------------------------------------------------------------------------


def _download(url: str, out_path: Path, *, force: bool = False) -> Path:
    if out_path.exists() and not force:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    logger.info("downloading %s", url)
    with urllib.request.urlopen(req, timeout=60) as resp, \
            open(out_path, "wb") as fh:
        fh.write(resp.read())
    return out_path


def _ensure_unzipped(zip_path: Path) -> Path:
    import zipfile

    out_dir = zip_path.parent
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not members:
            raise RuntimeError(f"no .txt member in {zip_path}")
        target = out_dir / members[0]
        if not target.exists():
            zf.extract(members[0], out_dir)
        return target


def fetch_and_parse_year(
    year: int,
    *,
    county: str = DEFAULT_COUNTY,
    muni_codes: list[str] | None = None,
    raw_root: Path = RAW_ROOT,
):
    """Legacy zip-based fetch — see :func:`fetch_dashboard_rows` for new path."""
    import pandas as pd

    crash_zip = raw_root / f"{county}{year}Accidents.zip"
    ped_zip = raw_root / f"{county}{year}Pedestrians.zip"

    _download(ZIP_URL_PATTERN.format(year=year, county=county,
                                     table="Accidents"), crash_zip)
    crash_txt = _ensure_unzipped(crash_zip)
    crashes = parse_crash_table(crash_txt, muni_codes=muni_codes)

    peds = pd.DataFrame()
    try:
        _download(ZIP_URL_PATTERN.format(year=year, county=county,
                                         table="Pedestrians"), ped_zip)
        ped_txt = _ensure_unzipped(ped_zip)
        peds = parse_pedestrian_table(ped_txt, muni_codes=muni_codes)
    except Exception:
        logger.exception("could not load pedestrians for %s", year)

    return crashes, peds


# ---------------------------------------------------------------------------
# Geocoding context (SUMO edges + meta CSV → name-keyed gdf)
# ---------------------------------------------------------------------------


def _build_osm_ways_gdf():
    if not NET_PATH.exists() or not META_PATH.exists():
        logger.warning(
            "SUMO net (%s) or meta CSV (%s) missing; the geocoder will "
            "only honour rows that already have lat/lon.",
            NET_PATH.exists(), META_PATH.exists(),
        )
        return None
    edges = load_sumo_edge_geometries(NET_PATH)
    meta = load_meta_lookup(META_PATH)
    if edges.empty or meta.empty:
        return None
    return edges.merge(
        meta[["sumo_edge_id", "osm_way_id", "street_name"]],
        left_on="edge_id", right_on="sumo_edge_id", how="left",
    )


def _leonia_polygon():
    try:
        from leonia_traffic.config import load_leonia_polygon
        return load_leonia_polygon()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _main_dashboard(args, osm_ways, leonia_poly, meta_df):
    """Load + assemble the dashboard-sourced crash overlay."""
    import pandas as pd

    cache = DASHBOARD_DIR / "leonia_dashboard.json"
    _ensure_dashboard_cached(
        refresh=args.refresh, cache_path=cache,
        muni_value=args.muni_name, entity_token=args.token,
    )
    crashes = parse_dashboard_json(cache, drop_state_system=args.local_only)
    logger.info("parsed %d dashboard rows (local_only=%s)",
                len(crashes), args.local_only)
    if crashes.empty:
        logger.error("no crash data was loaded — aborting")
        return None, None, pd.DataFrame()

    # Rows that lack any geocoding can fall through the name-based
    # geocoder. The dashboard already surfaces ~92% with `Geopoint
    # Calculated`, so this only fires on ~8% of rows.
    if osm_ways is not None and "geocoded_method" in crashes.columns:
        needs_name = crashes["geocoded_method"] == "none"
        if needs_name.any():
            sub = crashes[needs_name].copy()
            sub = geocode_by_street_name(
                sub, osm_ways, muni_polygon=leonia_poly,
            )
            for col in ("geocoded_lat", "geocoded_lon",
                        "geocoded_osm_way_id", "geocoded_method"):
                if col in sub.columns:
                    crashes.loc[needs_name, col] = sub[col].values

    if osm_ways is not None:
        crashes = assign_to_osm_way(
            crashes, osm_ways, max_distance_m=args.max_snap_m,
        )

    by_segment = aggregate_by_segment(
        crashes, osm_meta=meta_df,
    )
    return crashes, by_segment, pd.DataFrame()  # no peds frame for dashboard


def _main_zip(args, osm_ways, leonia_poly, meta_df):
    """Load + assemble the legacy zip-sourced crash overlay."""
    import pandas as pd

    logger.info("fetching %d zip year(s) for %s, muni_codes=%s",
                len(args.years), args.county, args.muni_codes)
    all_crashes: list[pd.DataFrame] = []
    all_peds: list[pd.DataFrame] = []
    for year in args.years:
        try:
            crashes_y, peds_y = fetch_and_parse_year(
                year, county=args.county,
                muni_codes=args.muni_codes,
                raw_root=RAW_ROOT,
            )
        except Exception:
            logger.exception("year %s failed", year)
            continue
        logger.info("year %s: crashes=%d, peds=%d",
                    year, len(crashes_y), len(peds_y))
        if not crashes_y.empty:
            all_crashes.append(crashes_y)
        if not args.no_pedestrians and not peds_y.empty:
            all_peds.append(peds_y)

    if not all_crashes:
        logger.error("no crash data was loaded — aborting")
        return None, None, pd.DataFrame()

    crashes = pd.concat(all_crashes, ignore_index=True)
    peds = pd.concat(all_peds, ignore_index=True) if all_peds else pd.DataFrame()
    logger.info("zip totals: %d crashes (raw lat/lon: %d)",
                len(crashes), int(crashes["latitude"].notna().sum()))

    if osm_ways is not None:
        crashes = geocode_by_street_name(
            crashes, osm_ways, muni_polygon=leonia_poly,
        )
        crashes = assign_to_osm_way(
            crashes, osm_ways, max_distance_m=args.max_snap_m,
        )
    else:
        crashes["geocoded_lat"] = crashes["latitude"]
        crashes["geocoded_lon"] = crashes["longitude"]
        crashes["geocoded_osm_way_id"] = pd.NA
        crashes["geocoded_method"] = pd.NA

    by_segment = aggregate_by_segment(
        crashes, osm_meta=meta_df, years=args.years,
    )
    return crashes, by_segment, peds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the NJDOT crash overlay datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", choices=["dashboard", "zip"],
                        default="dashboard",
                        help="Which NJDOT source to use (default: dashboard).")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch from the dashboard even if cached.")
    parser.add_argument("--local-only", action="store_true",
                        help="Drop Interstate / NJ Turnpike / State Hwy "
                             "crashes (dashboard source only).")
    parser.add_argument("--muni-name", default="Leonia Boro",
                        help="Dashboard `Municipality` filter value "
                             "(dashboard source only).")
    parser.add_argument("--token", default=DEFAULT_ENTITY_TOKEN,
                        help="Override the dashboard entityToken.")
    parser.add_argument("--years", type=int, nargs="*",
                        default=DEFAULT_YEARS,
                        help="Years to fetch (zip source only).")
    parser.add_argument("--county", default=DEFAULT_COUNTY)
    parser.add_argument("--muni-codes", nargs="*",
                        default=LEONIA_MUNI_CODES,
                        help="Filter to these NJDOT muni codes "
                             "(zip source only; default: Leonia=29).")
    parser.add_argument("--no-pedestrians", action="store_true",
                        help="Skip the Pedestrian-table download "
                             "(zip source only).")
    parser.add_argument("--max-snap-m", type=float, default=50.0,
                        help="Max distance (m) for crash → OSM way snap.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import pandas as pd

    osm_ways = _build_osm_ways_gdf()
    leonia_poly = _leonia_polygon()
    meta_df = (
        load_meta_lookup(META_PATH) if META_PATH.exists() else pd.DataFrame()
    )

    if args.source == "dashboard":
        crashes, by_segment, peds = _main_dashboard(
            args, osm_ways, leonia_poly, meta_df,
        )
    else:
        crashes, by_segment, peds = _main_zip(
            args, osm_ways, leonia_poly, meta_df,
        )

    if crashes is None:
        return 1

    logger.info(
        "geocoded methods: %s",
        crashes["geocoded_method"].value_counts(dropna=False).to_dict(),
    )
    logger.info(
        "segments with ≥1 crash: %d (top EPDO=%.0f)",
        len(by_segment),
        float(by_segment["epdo_total"].iloc[0]) if len(by_segment) else 0.0,
    )

    # ``geocoded_osm_way_id`` may be a mix of Python ints and pd.NA;
    # coerce to nullable Int64 so parquet round-trips cleanly.
    if "geocoded_osm_way_id" in crashes.columns:
        crashes["geocoded_osm_way_id"] = pd.to_numeric(
            crashes["geocoded_osm_way_id"], errors="coerce"
        ).astype("Int64")

    sources_meta = (
        ["njdot_dashboard:" + args.muni_name] if args.source == "dashboard"
        else [ZIP_URL_PATTERN.format(year=y, county=args.county,
                                      table="Accidents")
              for y in args.years]
    )
    crashes_path = write_dataframe(
        crashes, folder=CRASHES_DIR, name=CrashFiles.crashes,
        sources=sources_meta,
    )
    by_segment_path = write_dataframe(
        by_segment, folder=CRASHES_DIR,
        name=CrashFiles.crashes_by_segment,
        sources=[CrashFiles.crashes],
    )
    if args.source == "zip" and not peds.empty:
        write_dataframe(
            peds, folder=CRASHES_DIR, name=CrashFiles.pedestrian_crashes,
            sources=[ZIP_URL_PATTERN.format(year=y, county=args.county,
                                            table="Pedestrians")
                     for y in args.years],
        )

    print()
    print(f"Wrote {len(crashes):,} crashes → {crashes_path}")
    print(f"Wrote {len(by_segment):,} segments → {by_segment_path}")
    if args.source == "zip" and not peds.empty:
        print(f"Wrote {len(peds):,} pedestrian rows → "
              f"{CRASHES_DIR / CrashFiles.pedestrian_crashes}")
    print()
    if not by_segment.empty:
        print("Top 10 segments by EPDO:")
        cols = [c for c in ("street_name", "osm_way_id", "n_crashes",
                            "n_fatal", "n_injury", "n_ped", "epdo_total")
                if c in by_segment.columns]
        print(by_segment.head(10)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
