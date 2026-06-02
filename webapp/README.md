# Leonia stakeholder webapp

Interactive scenario explorer for traffic impact studies on Leonia local
streets. The user picks a street, a change to apply (close or speed-hump),
and a demand cohort (average weekday or average Sunday); the page swaps to
a precomputed SUMO simulation comparing the baseline vs. the chosen
scenario.

The webapp itself is **read-only**: every scenario is computed offline by
`build_precache.py` and the FastAPI service simply maps dropdown
selections to the cached JSON/HTML artefact.

## Data layout: build tree vs. published serve set

- **Build tree** — `data/sumo/precache_build/` (~17 GB, git-ignored). The
  full per-scenario SUMO output incl. the heavy `edge_history.parquet` /
  `edge_summary.parquet` and the netconvert `_nets/` scratch.
- **Published serve set** — `data/webapp/` (~0.4 GB). The slim subset the
  webapp actually serves (`catalog.json` + per-scenario `flow.json` /
  small JSON / HTML + `_static/` + `_overlays/`). Tracked via **Git LFS**
  and **baked into the container image** (no runtime volume mount).

`build_precache.py` writes the build tree, then publishes the slim subset
into `data/webapp/`. Re-publish without rebuilding via
`make publish-webapp-data` (= `build_precache.py --publish-only`).

## Layout

```
webapp/
├── app/
│   ├── main.py              FastAPI routes
│   └── config.py            env-var-driven path resolution
├── templates/
│   └── stakeholder.html     dropdowns + deck.gl map container
├── static/
│   ├── styles.css
│   ├── deckgl_flow.js       reusable deck.gl/MapLibre flow renderer
│   └── scenario_picker.js   client-side catalog -> flow.json -> deck.gl
├── scripts/
│   └── build_precache.py    offline scenario builder + serve-set publisher
├── Dockerfile
├── requirements.txt
└── README.md
```

## How it works

1. **Precache build** (offline, ~2 h with `--parallel 8`)
   `webapp/scripts/build_precache.py` enumerates the streets in
   `data/stage-2/leonia_streets_cutthrough_index.parquet`, crosses
   them with the change types (`closure`, `speed_hump`) and two demand
   cohorts, and dispatches a SUMO worker subprocess per run. Each run
   writes:
   - `edge_history.parquet` / `edge_summary.parquet`
   - `flow.json` — compact per-link, per-15-min vph dataset the
     stakeholder page renders with deck.gl (the primary map artefact)
   - `animated.html` / `animated_dual.html` / `compare.html` — legacy
     folium maps, kept as fallbacks
   - `manifest.json`
   to `data/sumo/precache_build/<scenario_key>/`. A top-level
   `catalog.json` is rebuilt from disk on every invocation. The slim
   serve set is then published to `data/webapp/`.

   > **Note:** `flow.json` was added after the original precache. To
   > backfill it for every scenario, re-run the full build with
   > `--force` (it regenerates each run's `edge_history` and emits
   > `flow.json`). Scenarios without `flow.json` show a "rebuild the
   > precache" status on the page instead of a map.

2. **Webapp serve**
   FastAPI loads `catalog.json` once per process and exposes:
   - `GET /` → stakeholder page (dropdowns + deck.gl map)
   - `GET /api/catalog.json` → raw catalog (for the JS to populate
     dropdowns and resolve the cache key)
   - `GET /precache/{path}` → static-serves files under the published
     serve set (`data/webapp/`)
   - `GET /healthz` → catalog readiness probe

   No SUMO is invoked at request time.

3. **Frontend**
   Vanilla JS (`scenario_picker.js`) fetches the catalog once on page
   load, populates the street dropdown, builds a key like
   `willow_tree_road__speed_hump__weekday` on every dropdown change,
   fetches `/precache/<key>/flow.json`, and feeds it to the deck.gl
   renderer (`deckgl_flow.js`), which animates per-link vehicles/hour
   over a token-free MapLibre basemap. The selected street is outlined
   in white; hovering a road shows live vph. If a combination isn't in
   the precache (or predates `flow.json`), it surfaces a status banner
   instead. deck.gl + MapLibre load from CDN.

## Building the precache

The precache build runs SUMO out-of-process, so you need SUMO + the
project's data tree on the build host. From the repo root:

```bash
# Smoke test (one street, one change, one demand)
venv/bin/python webapp/scripts/build_precache.py \
    --streets willow_tree_road \
    --change-types closure \
    --demands bridge_od_weekday_24h \
    --parallel 1

# Full build (~2 hours wallclock with parallel 8)
venv/bin/python webapp/scripts/build_precache.py --parallel 8
```

The script is **idempotent and resumable**: any scenario whose
`manifest.json` already exists is skipped, so you can `Ctrl-C` and
resume at will. The catalog is regenerated from on-disk state on every
invocation.

Tunables you'll likely care about:

- `--parallel N` — workers in flight (default 8). Each worker peaks
  around 1 GB RAM, so trim down on smaller hosts.
- `--top-n N` — only build the top-N streets by cutthrough rank
  (useful for council-meeting scope cuts).
- `--streets foo bar` — limit to specific street slugs.
- `--change-types closure speed_hump` — limit to subset.
- `--demands bridge_od_weekday_24h` — limit to subset.
- `--force` — rebuild every scenario, even if cached.

## Running locally (no Docker)

```bash
# From the repo root, with the project venv active and the
# precache already built (see above):
venv/bin/uvicorn webapp.app.main:app --reload --port 8000
# -> http://127.0.0.1:8000
```

Health check:

```bash
curl -sS http://127.0.0.1:8000/healthz | jq
```

## Building the Docker image

The Dockerfile `COPY`s the published serve set (`data/webapp/`) into the
image, so a `docker run` without volume mounts is self-contained. SUMO is
*not* installed in the image because all SUMO work happens during
`build_precache.py`, out-of-band from the image build.

```bash
# Build + publish the serve set first (out-of-band)
venv/bin/python webapp/scripts/build_precache.py --parallel 8

# Then build the image (default `bundled` target bakes data/webapp/)
docker build -t leonia-webapp -f webapp/Dockerfile .

# Run with the baked-in serve set
docker run --rm -p 8000:8000 leonia-webapp
# -> http://127.0.0.1:8000
```

> **Git LFS.** `data/webapp/` is tracked with Git LFS. Any host that
> builds the image must have `git-lfs` installed and the real files
> checked out (`git lfs pull`); otherwise the `COPY` bakes pointer stubs
> and the webapp serves nothing. CI checks out with `lfs: true` for this
> reason. See the repo `README.md` for the Git LFS setup/quota notes.

### Mounting the serve set instead of baking it

For dev iteration you can skip the baked data and point the container at
a host directory via `LEONIA_PRECACHE_DIR` (or re-root everything with
`LEONIA_DATA_DIR`):

```bash
docker run --rm -p 8000:8000 \
    -e LEONIA_PRECACHE_DIR=/served \
    -v $(pwd)/data/webapp:/served \
    leonia-webapp
```

`webapp/app/config.py` honors `LEONIA_PRECACHE_DIR` and
`leonia_traffic.config` honors `LEONIA_DATA_DIR`.

### Image size budget

Approximate split for the default `bundled` build:

| layer                       | size   |
| --------------------------- | ------ |
| python:3.13-slim base       | ~140 MB |
| pip deps (fastapi + pandas) | ~250 MB |
| leonia_traffic + webapp     | ~5 MB |
| serve set (`data/webapp/`)  | ~0.4 GB |

The serve set is the dominant data cost; the heavy build tree
(`data/sumo/precache_build/`, ~17 GB) is **not** shipped.

## Operating notes

- **Catalog is cached in memory** after the first request. Restart the
  process to pick up a freshly-rebuilt catalog (or call
  `reset_catalog_cache()` in code).
- **Build-tree `_nets/` subdir** (under `data/sumo/precache_build/`)
  holds the per-street netconvert rebuilds. It's a build-time artefact,
  excluded from the published serve set, and can be deleted to reclaim
  disk if you don't need to re-run any scenarios.
- **Tests**: the webapp itself doesn't have a dedicated test suite;
  routes are thin enough that the local smoke (`uvicorn` + `curl`) is
  the recommended verification. The precache builder reuses the
  existing `tests/test_sumo_runtime.py` and `test_sumo_demand.py`
  test surfaces.
- **Out-of-scope for v1**: arbitrary scenario submission, multi-street
  combinations, persistent run database, auth.
