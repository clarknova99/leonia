"""Webapp configuration.

All paths are env-var-driven so the same image works in three modes:

* In-repo development (no env vars set) — paths resolve to the repo
  checkout via :mod:`leonia_traffic.config`.
* Docker with baked-in precache (env vars unset, ``/app/data``
  baked into the image at build time).
* Docker with mounted data volume (``LEONIA_DATA_DIR=/data``,
  ``-v /host/path:/data``).
"""

from __future__ import annotations

import os
from pathlib import Path

from leonia_traffic.config import DATA_PROCESSED_DIR

PRECACHE_SUBDIR = os.environ.get(
    "LEONIA_PRECACHE_SUBDIR", "sumo/runs_precache",
)

PRECACHE_DIR: Path = DATA_PROCESSED_DIR / PRECACHE_SUBDIR
CATALOG_PATH: Path = PRECACHE_DIR / "catalog.json"

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = WEBAPP_ROOT / "templates"
STATIC_DIR = WEBAPP_ROOT / "static"
