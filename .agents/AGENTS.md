# Agent guide — Leonia traffic analysis & mitigation

Single source of truth for AI agents working in this repo. Root
`AGENTS.md` and `CLAUDE.md` are symlinks to this file. Tool configs
should point here, never duplicate this content.

## What this repo is

A Python framework that fuses **five StreetLight** data products with an
**Eclipse SUMO** microsimulation of Leonia, NJ and its George Washington
Bridge approaches, plus a read-only **deck.gl web app** for stakeholders.

- Full overview, quick start, ingest order, SUMO layer, and webapp
  deploy: [`README.md`](../README.md).
- Column-level schema for every parquet: [`docs/DATA.md`](../docs/DATA.md).
- Webapp build/serve/Docker detail: [`webapp/README.md`](../webapp/README.md).

> **SUMO is the only simulation engine.** The legacy UXsim path was
> removed. The only surviving engine-neutral pieces are the scenario DSL
> (`leonia_traffic/scenarios.py`) and GEH scoring
> (`leonia_traffic/analysis/scoring.py`). Do not reintroduce UXsim.

## Data tree (all under `data/`, git-ignored & regenerable)

```
data/
  raw/        upstream source (StreetLight in raw/streetlight/, NJDOT crash)
  stage-1/    canonical parquet from raw: streetlight/ + crash/
  stage-2/    analytics derived from stage-1 (cut-through index, profiles, …)
  network/    OSM/sim network cache + overrides.yaml + leonia_borough.geojson
  sumo/
    base/             base SUMO inputs (leonia.net.xml, .sumocfg, routes, …)
    runs/             analyst run outputs (ts-named)
    precache_build/   heavy webapp build tree (~17 GB)
  webapp/     THE published serve set — Git LFS + baked into the image
```

**Exceptions to git-ignore:** `data/webapp/**` (Git LFS) and
`data/network/leonia_borough.geojson` are tracked; everything else under
`data/` is local-only and not recoverable from git.

## Path conventions — never hardcode

All paths are single-sourced. Import constants; do not write
`REPO_ROOT / "data" / ...` literals.

- `leonia_traffic/config.py`: `DATA_RAW_DIR`, `STREETLIGHT_DIR`,
  `DATA_STAGE1_DIR`, `DATA_STAGE2_DIR`, `DATA_NETWORK_DIR`,
  `SUMO_BASE_DIR`, `SUMO_RUNS_DIR`, `SUMO_PRECACHE_DIR`,
  `WEBAPP_PUBLISH_DIR`. Root override via `LEONIA_DATA_DIR`.
- `leonia_traffic/data/dataset_io.py`: `CANONICAL_DIR`, `DERIVED_DIR`,
  `CRASHES_DIR` + the `CanonicalFiles` / `DerivedFiles` filename tables.

## Pipeline order

`00_build_datasets` → `09_leonia_streets_report` (caches the cut-through
ranking) → `07_bridge_od_report` → `11_export_sumo` →
`14_build_crash_overlay` → `12_sumo_baseline` / `13_sumo_scenarios` /
`15_sumo_weekday_vs_sunday` / `16_sumo_signal_control` → webapp build.
Per-script descriptions live in [`README.md`](../README.md#ingest-pipeline).

## Conventions & gotchas

- **Python**: 3.13, run via `venv/bin/python`. Tests: `venv/bin/python -m pytest -q`.
- **libsumo vs pyarrow**: `import libsumo` permanently breaks `pyarrow` in
  the same process. The SUMO orchestrators spawn a worker subprocess for
  the simulation half (plain CSV) and do parquet/scoring/HTML in the
  parent. Keep that split — don't import libsumo in the parent process.
- **Webapp is read-only**: every map is precomputed. `build_precache.py`
  writes the heavy build tree (`data/sumo/precache_build/`) then publishes
  the slim subset to `data/webapp/`. Re-publish with
  `make publish-webapp-data`.
- **Git LFS**: `data/webapp/**` is LFS-tracked. A clone/CI without git-lfs
  gets pointer stubs. CI checks out with `lfs: true`.
- **CRS**: every geospatial parquet is EPSG:4326 (WGS84 lon/lat).
- **Day-type codes** (StreetLight): 0=All, 1-4=Mon-Thu, 5=Sat, 6=Sun
  (no Friday — see the Friday caveat in `docs/DATA.md`).

## Skills

Repeatable workflows live in `.agents/skills/<topic>/SKILL.md`:

- `rebuild-data` — rebuild the stage-1/stage-2 lake from raw exports.
- `rebuild-and-deploy-webapp` — rebuild + publish the webapp serve set and
  ship it via Git LFS + image bake.
- `add-streetlight-extract` — ingest a new StreetLight export.
