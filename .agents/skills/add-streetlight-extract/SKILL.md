---
name: add-streetlight-extract
description: Ingest a new StreetLight (or NJDOT) data export into the canonical lake. Use when a new export folder is dropped in and needs to flow into stage-1/stage-2 and the docs.
---

# Add a StreetLight extract

Two cases: a **new export of an existing product** (just refresh) versus a
**new product family** (small code change, by design).

## Case A — new export of an existing product

Drop the export into the matching `data/raw/streetlight/<product>/`
folder and rebuild — the loaders auto-discover files by glob.

```bash
venv/bin/python scripts/00_build_datasets.py --only <product>
# then rerun the consumers that matter (reports / SUMO export / webapp)
```

Confirm the new date range / row counts in the manifest and `docs/DATA.md`
match expectations (watch for granularity changes, e.g. 5 vs 24 day-part
windows). If the schema or date coverage changed, integrate it as a
parallel baseline rather than overwriting an existing table (see how the
`za_*_history` tables sit beside the recent-year `za_*` tables).

## Case B — new product family

The orchestrator uses an **explicit** product list so unexpected folders
never silently enter the lake. Steps:

1. Add a loader module under `leonia_traffic/data/` (mirror
   `bridge_od_loader.py`): parse raw CSV/shapefile → tidy DataFrame,
   parse `zone_name` → `street_name` / `osm_way_id`, parse day-type /
   day-part codes.
2. Add a filename constant to `CanonicalFiles` (or `DerivedFiles`) in
   `leonia_traffic/data/dataset_io.py`.
3. Add a `build_<product>()` function in `scripts/00_build_datasets.py`
   and register it in `KNOWN_PRODUCTS`.
4. Document the new file(s) in a new subsection of `docs/DATA.md`
   (source path, grain, used-by, full column dictionary).
5. Add/extend a loader test under `tests/`.

## Conventions

- Output paths come from `dataset_io.py` constants — never hardcode.
- Every geospatial parquet is written as GeoParquet in EPSG:4326.
- Day-type codes: 0=All, 1-4=Mon-Thu, 5=Sat, 6=Sun (no Friday).
- Rerun `venv/bin/python -m pytest -q` after the change.
