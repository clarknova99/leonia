"""Webapp configuration.

The served dataset is the single published set under
:data:`leonia_traffic.config.WEBAPP_PUBLISH_DIR` (``data/webapp/``), which
is git-LFS tracked and baked into the container image. The same image
therefore works in two modes:

* In-repo development (no env vars set) — paths resolve to the repo
  checkout via :mod:`leonia_traffic.config`.
* Docker with baked-in data (``LEONIA_DATA_DIR=/app/data``, ``data/webapp/``
  copied into the image at build time).

``LEONIA_PRECACHE_DIR`` can override the served directory outright (e.g.
to point at a mounted volume for ad-hoc data swaps).
"""

from __future__ import annotations

import os
from pathlib import Path

from leonia_traffic.config import WEBAPP_PUBLISH_DIR

PRECACHE_DIR: Path = Path(
    os.environ.get("LEONIA_PRECACHE_DIR", WEBAPP_PUBLISH_DIR)
)
CATALOG_PATH: Path = PRECACHE_DIR / "catalog.json"

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = WEBAPP_ROOT / "templates"
STATIC_DIR = WEBAPP_ROOT / "static"
