# Leonia, NJ — traffic analysis and mitigation framework

A reusable Python framework that combines **five StreetLight** data
products with an **Eclipse SUMO** microscopic traffic simulation of
Leonia, NJ and its approaches to the George Washington Bridge (GWB):

- **Street Scanner** — segment-level speed/volume averages for
  inferring cut-through patterns from observed data alone.
- **Bridge-destination O-D Analysis** — per-day-part trip volumes from
  the 5 Leonia perimeter gates to the GWB, joined with rich traveler
  attributes (trip purpose, equity, household, income, education,
  employment, trip distribution stats).
- **Congestion Trends** — per-segment hourly Travel Time Index, Buffer
  Index, Vehicle Hours of Delay, and reliability classification on
  arterial / motorway approaches.
- **Leonia Streets Zone Activity** — per-zone pass-through volume,
  travel time, trip length, speed, and home-ZIP distributions on **OSM
  tertiary segments inside Leonia**. A recent-year Visitor-filtered
  export (`2034227`, Apr 2025 – Mar 2026) plus a long-run all-trips
  historical baseline (`2038018`, Jan 2022 – Apr 2026).
- **Network Performance** — segment-level volume, speed, VMT, VHD,
  free-flow/congestion, and 5/15/85/95 speed percentiles on **every
  selected OSM segment** (arterials, the GWB approach, *and*
  residential blocks), at hourly day-parts and per-day-of-week day
  types.

The framework supports:

- a **canonical Parquet data lake** built from the raw StreetLight and
  NJDOT exports (`scripts/00_build_datasets.py`),
- **evidence-first stakeholder reporting** from the OD, congestion, and
  demographic exports (`reports/07_bridge_od.md`),
- **per-residential-street cut-through evidence** with composite-index
  ranking and home-ZIP origin breakdown (`reports/09_leonia_streets.md`),
- a real **SUMO simulation** driven through `libsumo` with selectable
  demand, runtime scenario controls, GEH scoring, and adaptive-signal
  control,
- declarative mitigation scenarios (closures, traffic calming, one-way
  conversions, lane reductions),
- a **deck.gl stakeholder web app** (`webapp/`) with a *Static Maps* tab
  (measured StreetLight traffic + NJDOT crashes) and a *Simulation* tab
  that animates precomputed per-scenario SUMO flows over a token-free
  MapLibre basemap. See [`webapp/README.md`](webapp/README.md).

The framework is the Borough of Leonia's Safe Streets for All (SS4A)
data subscription product turned into a reproducible analysis pipeline.

> **Simulation engine.** SUMO is the sole simulation engine. An earlier
> UXsim mesoscopic path was removed once SUMO (a superset) covered every
> use case; the only surviving shared pieces are the engine-neutral
> scenario DSL (`leonia_traffic/scenarios.py`) and GEH scoring
> (`leonia_traffic/analysis/scoring.py`).

---

## Data layout

All data lives under `data/` and is **git-ignored and regenerable**,
with one exception: the published webapp serve set (`data/webapp/`) and
the small borough polygon are tracked (the former via **Git LFS**) so
they can be baked into the container image.

```
data/
  raw/                     # upstream source data (not regenerable here)
    streetlight/           # all StreetLight exports (Street Scanner, OD,
                           #   congestion, ZA, network performance, …)
    njdot_crashes/         # NJDOT fixed-width crash tables
    njdot_dashboard/       # NJDOT Crash Data Dashboard JSON cache
  stage-1/                 # canonical parquet built directly from raw
    streetlight/           #   one file per StreetLight CSV family
    crash/                 #   NJDOT crash overlay (geocoded to OSM ways)
  stage-2/                 # analytics derived from stage-1
                           #   (cut-through index, hourly profiles, peak
                           #    intensity, trends, OMD attribution, …)
  network/                 # OSM/sim network cache + overrides.yaml +
                           #   leonia_borough.geojson + speed-override caches
  sumo/
    base/                  # base SUMO inputs (leonia.net.xml, .sumocfg,
                           #   routes, osm, poly, edgedata, README_SUMO.md)
    runs/                  # analyst run outputs (regenerable; ts-named)
    precache_build/        # heavy webapp build tree (~17 GB): per-scenario
                           #   edge_history/edge_summary parquet + _nets/
  webapp/                  # the ONE published serve set (Git LFS + baked
                           #   into the image): catalog.json + per-scenario
                           #   flow.json + _static/ + _overlays/
```

Paths are single-sourced in
[`leonia_traffic/config.py`](leonia_traffic/config.py)
(`DATA_RAW_DIR`, `DATA_STAGE1_DIR`, `DATA_STAGE2_DIR`, `DATA_NETWORK_DIR`,
`SUMO_BASE_DIR`, `SUMO_RUNS_DIR`, `SUMO_PRECACHE_DIR`,
`WEBAPP_PUBLISH_DIR`) and
[`leonia_traffic/data/dataset_io.py`](leonia_traffic/data/dataset_io.py)
(`CANONICAL_DIR`, `DERIVED_DIR`, `CRASHES_DIR`). Override the data root
with `LEONIA_DATA_DIR`. See [`docs/DATA.md`](docs/DATA.md) for the
column-level schema of every parquet.

---

## Quick start

```bash
# 1. Activate the Python 3.13 virtualenv and install deps.
source venv/bin/activate
pip install -r requirements.txt

# 2. Build the canonical + derived data lake from the raw exports.
#    Writes data/stage-1/{streetlight,crash}/ and data/stage-2/.
python scripts/00_build_datasets.py

# 3. Stakeholder evidence report (Bridge OD + Congestion Trends).
python scripts/09_leonia_streets_report.py   # caches the cut-through ranking
python scripts/07_bridge_od_report.py         # residential rules pick it up
open reports/07_bridge_od.md

# 4. Export the SUMO project (network + routes + config) to data/sumo/base/.
python scripts/11_export_sumo.py
sumo-gui -c data/sumo/base/leonia.sumocfg     # optional: open in the GUI

# 5. NJDOT crash overlay (adds the safety layer downstream consumers detect).
python scripts/14_build_crash_overlay.py

# 6. Run a SUMO baseline / scenarios via libsumo.
python scripts/12_sumo_baseline.py --demand peak_am_slice
python scripts/13_sumo_scenarios.py --demand bridge_od_weekday_24h

# 7. Build + serve the web app (see Web app section + webapp/README.md).
make build-webapp-data                        # overlays + static + publish
python webapp/scripts/build_precache.py --parallel 8   # full scenario build
venv/bin/uvicorn webapp.app.main:app --reload --port 8000
```

---

## Ingest pipeline

The pipeline runs in dependency order; each script reads the canonical
lake (or the prior stage) and writes the outputs noted below.

| Step | Script | Reads | Writes |
| --- | --- | --- | --- |
| 0 | `00_build_datasets.py` | `data/raw/streetlight/` | `data/stage-1/{streetlight,crash}/`, `data/stage-2/` (canonical + derived parquet) |
| 1 | `01_explore_data.py` | Street Scanner raw | `reports/01_exploration.md` (exploratory) |
| 2 | `07_bridge_od_report.py` | bridge OD + congestion canonical, `data/stage-2/` | `reports/07_bridge_od.md`, `data/network/speed_overrides_*.parquet` |
| 3 | `09_leonia_streets_report.py` | ZA canonical, `data/stage-2/cutthrough_index.parquet` | `reports/09_leonia_streets.md`, `data/stage-2/leonia_streets_cutthrough_index.parquet` |
| 4 | `11_export_sumo.py` | canonical lake | `data/sumo/base/leonia.{net,osm,poly,flows,edgedata,sumocfg}` + `README_SUMO.md` |
| 5 | `14_build_crash_overlay.py` | `data/raw/njdot_*`, SUMO net | `data/stage-1/crash/*.parquet` |
| 6 | `12_sumo_baseline.py` | `data/sumo/base/`, canonical demand | `data/sumo/runs/<ts>_<demand>/`, `reports/12_sumo_baseline.md` |
| 7 | `13_sumo_scenarios.py` | `data/sumo/base/`, scenario specs | `data/sumo/runs/<ts>_*`, `reports/13_sumo_scenarios.md`, `reports/scenarios_sumo/` |
| 8 | `15_sumo_weekday_vs_sunday.py` | `data/sumo/base/` | `data/sumo/runs/<ts>_weekday_vs_sunday/` |
| 9 | `16_sumo_signal_control.py` | `data/sumo/base/` | `data/sumo/runs/<ts>_*_adaptive/` (adaptive-signal run + compare) |
| 10 | webapp build (below) | canonical + `data/sumo/precache_build/` | `data/webapp/` |

Surviving scripts at a glance:

- **`00_build_datasets.py`** — build the canonical (stage-1) + derived
  (stage-2) Parquet lake from the raw StreetLight/NJDOT exports.
- **`01_explore_data.py`** — exploratory Street Scanner volume/speed
  notebook-style report.
- **`07_bridge_od_report.py`** — Pass-A stakeholder evidence report
  (OD + congestion + demographics + recommendations) and the per-link
  speed-override cache.
- **`09_leonia_streets_report.py`** — per-residential-street cut-through
  report; reuses the derived cut-through ranking and caches the per-street
  index the precache builder scopes to.
- **`11_export_sumo.py`** — export the canonical lake to a SUMO-ready
  project under `data/sumo/base/`.
- **`12_sumo_baseline.py`** — single-demand SUMO baseline via libsumo +
  stakeholder bundle.
- **`13_sumo_scenarios.py`** — SUMO baseline + the mitigation scenarios
  with dual-compare maps.
- **`14_build_crash_overlay.py`** — NJDOT crash overlay (geocoded to OSM
  ways) feeding the safety layer.
- **`15_sumo_weekday_vs_sunday.py`** — paired weekday-vs-Sunday SUMO run.
- **`16_sumo_signal_control.py`** — SUMO run under an adaptive
  max-pressure signal controller (+ before/after compare).

---

## SUMO simulation (libsumo)

`leonia_traffic/sumo/` is the runtime layer:

- **`runtime.py`** — `SumoRuntime`, an interactive libsumo wrapper
  (`start / step / run_until / run_to_end / apply_closure / restore /
  set_speed / set_traffic_light / close`).
- **`demand_builder.py`** — `DemandSource` enum + routes.xml emitter
  (`bridge_od_weekday_24h`, `bridge_od_sunday_24h`, `peak_am_slice`, …).
- **`scenarios_sumo.py`** — adapter from the engine-neutral
  `leonia_traffic.scenarios` DSL to libsumo edge controls.
- **`scoring.py`** — GEH scoring of a run vs. Street Scanner, using the
  shared primitives in `leonia_traffic/analysis/scoring.py`.
- **`signal_control.py`** — adaptive max-pressure signal controller.
- **`comparison.py` / `trip_metrics.py` / `visualizations.py`** —
  before/after KPIs, trip-time metrics, and flow/animation rendering.

> **Process model.** `libsumo`'s C++ binding registers a competing
> pyarrow filesystem-scheme handler at import time, which permanently
> breaks `pyarrow` in the same Python process. The orchestrators
> therefore spawn a worker subprocess for the simulation half (it writes
> plain CSV) and post-process (scoring, parquet output, HTML rendering)
> in the parent.

---

## Web app — stakeholder map explorer (`webapp/`)

A read-only FastAPI service that turns the simulation and measurement
layers into an interactive deck.gl map. Every artefact is precomputed
offline; the server only maps UI selections to cached JSON. Two tabs:

- **Static Maps** — measured StreetLight volumes (Day Type × Day Part)
  and NJDOT crashes (by year) over a MapLibre basemap.
- **Simulation** — animated per-scenario SUMO flow: pick a street, a
  change (close / speed-hump), and a demand cohort (weekday / Sunday).

### Architecture

- **Build tree** (`data/sumo/precache_build/`, ~17 GB, git-ignored) —
  full per-scenario SUMO outputs incl. heavy `edge_history.parquet` /
  `edge_summary.parquet` and the netconvert `_nets/` scratch.
- **Published serve set** (`data/webapp/`, ~0.4 GB) — the slim subset the
  webapp actually serves: `catalog.json` + per-scenario `flow.json` /
  small JSON / HTML + `_static/` + `_overlays/`. Tracked via **Git LFS**
  and **baked into the container image**.

`webapp/scripts/build_precache.py` writes the build tree and then
publishes the slim subset into `data/webapp/` (excluding the heavy
parquet/_nets); `webapp/app/config.py` resolves `WEBAPP_PUBLISH_DIR`
(`data/webapp/`) as the served directory.

### Build

```bash
# Full scenario build (heavy; writes the build tree, then publishes data/webapp).
python webapp/scripts/build_precache.py --parallel 8

# Lightweight refresh of overlays + static maps, then re-publish data/webapp.
make build-webapp-data
```

### Serve

```bash
venv/bin/uvicorn webapp.app.main:app --reload --port 8000
# health: GET /healthz   catalog: GET /api/catalog.json
```

### Deploy

The published serve set in `data/webapp/` is committed via Git LFS and
baked into the image (`webapp/Dockerfile`, default `bundled` target). CI
([`.github/workflows/webapp.yaml`](.github/workflows/webapp.yaml)) checks
out with `lfs: true` and builds the `bundled` target, so every published
image is self-contained — no runtime data mount.

> **Git LFS.** `data/webapp/**` is tracked in `.gitattributes`. Install
> once per machine: `brew install git-lfs && git lfs install`. Commits
> work normally; `git push` first uploads changed LFS objects. Clones/CI
> need git-lfs installed (and `lfs: true`) or they get pointer stubs
> instead of the real data. GitHub's free LFS tier is 1 GB storage +
> 1 GB bandwidth/month; a $5 data pack is recommended if the webapp data
> is rebuilt more than a few times a month.
>
> **External step (one-time):** remove the now-unused `runs_precache`
> volume mount from the pi-talos leonia HelmRelease — the data ships in
> the image.

See [`webapp/README.md`](webapp/README.md) for the full build/serve/Docker
guide and [`docs/DATA.md`](docs/DATA.md) for the artefact schemas.

---

## Refreshing StreetLight data

Drop a new export into `data/raw/streetlight/<product>/` and rerun
`scripts/00_build_datasets.py` followed by the relevant report/SUMO
scripts; the loaders auto-discover new files.

---

## Adding a mitigation scenario

```python
from leonia_traffic.scenarios import (
    Closure, OneWayConversion, SpeedHumpCalming, LaneReduction,
)
from leonia_traffic.sumo import SumoRuntime, DemandSource
from leonia_traffic.sumo.scenarios_sumo import apply_scenarios

sc = SpeedHumpCalming(
    name="park_ave_calming",
    osm_way_ids=[11580528, 11580538, 11580547],
    free_flow_speed_factor=0.5,
)
# Build a SUMO run, then apply the scenario at runtime via apply_scenarios.
```

---

## Running tests

```bash
venv/bin/python -m pytest -q
```

The suite covers the StreetLight / Bridge-OD / Congestion / ZA /
Network-Performance loaders, OD and per-residential-street cut-through
analytics, congestion analysis, the jurisdiction filter, the GEH
statistic, the SUMO scenario adapter / demand / runtime / lookup, and
the recommendation engine. No live network or simulation calls.
