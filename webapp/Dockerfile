# syntax=docker/dockerfile:1.6
#
# Leonia stakeholder webapp — runtime image.
#
# This image bundles:
#   * the FastAPI service (webapp/app)
#   * the leonia_traffic library (so leonia_traffic.config still
#     resolves cleanly, even though no SUMO is invoked at runtime)
#   * the precomputed scenario cache under
#     /app/data/processed/sumo/runs_precache/
#   * a thin Jinja2 + uvicorn web layer
#
# What it does NOT bundle:
#   * SUMO (netconvert / sumo / sumo-gui) — the precache is built
#     out-of-band before the image is built. See webapp/README.md.
#   * The full data/ tree — only the artefacts the webapp actually
#     reads at request time.
#
# Build targets:
#   * `base`    — code-only image. No data baked in; the precache is
#                 expected to be mounted at runtime (k8s volume / -v).
#                 This is what CI builds and what the Talos deploy runs.
#   * `bundled` — extends `base` by baking the local data tree (precache
#                 + borough polygon) for a self-contained local run.
#                 This is the DEFAULT target (last stage) so the local
#                 build command below keeps working unchanged.
#
# Build (self-contained, bakes local precache — DEFAULT target):
#   docker build -t leonia-webapp -f webapp/Dockerfile .
#
# Build (code-only, data mounted at runtime — CI / k8s):
#   docker build -t leonia-webapp --target base -f webapp/Dockerfile .
#
# Run (in-image precache, `bundled`):
#   docker run --rm -p 8000:8000 leonia-webapp
#
# Run (mounted data dir, dev/iter or `base`):
#   docker run --rm -p 8000:8000 \
#       -e LEONIA_DATA_DIR=/data \
#       -v $PWD/data:/data \
#       leonia-webapp

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LEONIA_DATA_DIR=/app/data

# ---------------------------------------------------------------------------
# System deps: just enough to import geopandas/shapely/pyproj wheels.
# We do NOT install SUMO here — the precache must be prebuilt.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python deps. Webapp requirements first (small + cacheable), then the
# leonia_traffic deps (large pandas/geopandas wheels) — separating these
# layers keeps rebuilds fast when only the webapp code changes.
# ---------------------------------------------------------------------------
COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install -r /app/webapp/requirements.txt

# leonia_traffic transitively pulls pandas + geopandas + shapely.
# Pin a minimal set so the image stays slim; if the rest of the
# library starts importing more, add to this list.
RUN pip install \
        "pandas>=2.2" \
        "geopandas>=1.0" \
        "shapely>=2.0" \
        "pyproj>=3.6" \
        "numpy>=1.26"

# ---------------------------------------------------------------------------
# Application code.
# ---------------------------------------------------------------------------
COPY leonia_traffic/ /app/leonia_traffic/
COPY webapp/ /app/webapp/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "webapp.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers"]

# ---------------------------------------------------------------------------
# `bundled` (default) — bakes the local data tree for a self-contained image.
#
# These paths live under data/ which is .gitignored, so this stage only
# builds when the data tree is present in the build context. CI builds the
# `base` target instead (--target base) and mounts the precache at runtime.
# ---------------------------------------------------------------------------
FROM base AS bundled

# Network polygon — leonia_traffic.config.load_leonia_polygon reads it.
COPY data/network/leonia_borough.geojson /app/data/network/leonia_borough.geojson

# Precomputed scenario cache. This is the big payload.
COPY data/processed/sumo/runs_precache/ /app/data/processed/sumo/runs_precache/
