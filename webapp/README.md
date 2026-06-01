# Leonia stakeholder webapp

Interactive scenario explorer for traffic impact studies on Leonia local
streets. The user picks a street, a change to apply (close, speed-hump,
or convert to one-way), and a demand cohort (average weekday or average
Sunday); the page swaps to a precomputed SUMO simulation comparing the
baseline vs. the chosen scenario.

The webapp itself is **read-only**: every scenario is computed offline by
`build_precache.py` and the FastAPI service simply maps dropdown
selections to the cached HTML artefact.

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
│   └── build_precache.py    540-run offline scenario builder
├── Dockerfile
├── requirements.txt
└── README.md
```

## How it works

1. **Precache build** (offline, ~2 h with `--parallel 8`)
   `webapp/scripts/build_precache.py` enumerates the 90 streets in
   `data/processed/leonia_streets_cutthrough_index.parquet`, crosses
   them with three change types and two demand cohorts (= 540 runs),
   and dispatches a SUMO worker subprocess per run. Each run writes:
   - `edge_history.parquet` / `edge_summary.parquet`
   - `flow.json` — compact per-link, per-15-min vph dataset the
     stakeholder page renders with deck.gl (the primary map artefact)
   - `animated.html` / `animated_dual.html` / `compare.html` — legacy
     folium maps, kept as fallbacks
   - `manifest.json`
   to `data/processed/sumo/runs_precache/<scenario_key>/`. A top-level
   `catalog.json` is rebuilt from disk on every invocation.

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
   - `GET /precache/{path}` → static-serves files under
     `runs_precache/`
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

# Full 540-run build (~2 hours wallclock with parallel 8)
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

The Dockerfile bakes the precache into the image so a `docker run`
without volume mounts is self-contained. SUMO is *not* installed in
the image because all SUMO work happens during `build_precache.py`,
out-of-band from the image build.

```bash
# Build the precache first (out-of-band)
venv/bin/python webapp/scripts/build_precache.py --parallel 8

# Then build the image
docker build -t leonia-webapp -f webapp/Dockerfile .

# Run with the baked-in precache
docker run --rm -p 8000:8000 leonia-webapp
# -> http://127.0.0.1:8000
```

### Mounting the precache instead of baking it

For dev iteration, skip the COPY of `runs_precache/` and mount the
host's data dir at runtime via `LEONIA_DATA_DIR`:

```bash
docker run --rm -p 8000:8000 \
    -e LEONIA_DATA_DIR=/data \
    -v $(pwd)/data:/data \
    leonia-webapp
```

`leonia_traffic.config` honors `LEONIA_DATA_DIR` and re-roots all
data paths under it (4-line patch in `leonia_traffic/config.py`).

### Image size budget

Approximate split for the default build (with 540 scenarios cached):

| layer                       | size   |
| --------------------------- | ------ |
| python:3.13-slim base       | ~140 MB |
| pip deps (fastapi + pandas) | ~250 MB |
| leonia_traffic + webapp     | ~5 MB |
| precache (`runs_precache/`) | ~10 GB |

The precache is the dominant cost. If you're shipping over the
network, consider hosting it as a separate volume / S3 mount and
running with `LEONIA_DATA_DIR=/mnt/precache`.

## Operating notes

- **Catalog is cached in memory** after the first request. Restart the
  process to pick up a freshly-rebuilt catalog (or call
  `reset_catalog_cache()` in code).
- **Precache `_nets/` subdir** holds the per-street one-way
  netconvert rebuilds. It's a build-time artefact and can be deleted
  to reclaim disk if you don't need to re-run any one-way scenarios.
- **Tests**: the webapp itself doesn't have a dedicated test suite;
  routes are thin enough that the local smoke (`uvicorn` + `curl`) is
  the recommended verification. The precache builder reuses the
  existing `tests/test_sumo_runtime.py` and `test_sumo_demand.py`
  test surfaces.
- **Out-of-scope for v1**: arbitrary scenario submission, multi-street
  combinations, persistent run database, auth.
