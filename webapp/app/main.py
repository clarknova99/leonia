"""FastAPI service for the Leonia stakeholder webapp.

Routes
------

* ``GET /`` — the dropdown-driven stakeholder page.
* ``GET /api/catalog.json`` — the precache catalog the page reads
  to populate dropdowns and resolve scenario keys.
* ``GET /precache/{path:path}`` — static-serves the precomputed
  HTML / JSON / parquet artefacts under ``runs_precache/``.
* ``GET /healthz`` — liveness/readiness probe with catalog +
  precache status.

The application performs *no* simulation at request time. Every
scenario is precomputed via ``webapp/scripts/build_precache.py``;
this server is just a static + small JSON shell.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import CATALOG_PATH, PRECACHE_DIR, STATIC_DIR, TEMPLATES_DIR

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Leonia stakeholder webapp",
    description=(
        "Interactive scenario picker for traffic impact studies. "
        "Backed by a precomputed SUMO simulation cache."
    ),
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Catalog loader (cached at process start)
# ---------------------------------------------------------------------------


_CATALOG_CACHE: dict[str, Any] | None = None
_CATALOG_CACHE_MTIME: float | None = None


def _load_catalog() -> dict[str, Any]:
    """Read catalog.json from disk, invalidating on mtime change.

    The catalog is small (<1 MB even at 540 scenarios) so we keep
    a parsed copy in memory. We compare the on-disk mtime on every
    call; a precache rebuild updates ``catalog.json``'s mtime,
    which transparently invalidates the cache without needing a
    server restart.
    """
    global _CATALOG_CACHE, _CATALOG_CACHE_MTIME
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. Run "
            "webapp/scripts/build_precache.py first."
        )
    mtime = CATALOG_PATH.stat().st_mtime
    if _CATALOG_CACHE is not None and _CATALOG_CACHE_MTIME == mtime:
        return _CATALOG_CACHE
    _CATALOG_CACHE = json.loads(CATALOG_PATH.read_text())
    _CATALOG_CACHE_MTIME = mtime
    logger.info(
        "Loaded catalog: %d scenarios, %d streets",
        len(_CATALOG_CACHE.get("scenarios", {})),
        len(_CATALOG_CACHE.get("streets", [])),
    )
    return _CATALOG_CACHE


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Stakeholder page with three dropdowns + iframe."""
    try:
        catalog = _load_catalog()
        n_scenarios = len(catalog.get("scenarios", {}))
        catalog_ready = True
    except FileNotFoundError:
        n_scenarios = 0
        catalog_ready = False
    return templates.TemplateResponse(
        request, "stakeholder.html",
        {
            "request": request,
            "catalog_ready": catalog_ready,
            "n_scenarios": n_scenarios,
        },
    )


@app.get("/api/catalog.json")
def catalog_json() -> JSONResponse:
    """Serve catalog.json for client-side dropdown population."""
    try:
        return JSONResponse(_load_catalog())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/precache/{path:path}")
def serve_precache(path: str) -> FileResponse:
    """Static-serve files under the precache directory.

    Implemented as an explicit handler (rather than a
    ``StaticFiles`` mount) because the precache directory may not
    exist at app boot (the catalog endpoint is the authoritative
    "ready?" signal), and we want a 404 with a JSON-friendly path
    rather than a confusing 500 from StaticFiles.
    """
    full_path = (PRECACHE_DIR / path).resolve()
    try:
        full_path.relative_to(PRECACHE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal denied.")
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    return FileResponse(full_path)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness + readiness probe.

    Reports catalog presence and counts so an ops dashboard can
    distinguish "boot in progress" from "precache missing".
    """
    catalog_status: dict[str, Any] = {
        "path": str(CATALOG_PATH),
        "exists": CATALOG_PATH.exists(),
    }
    if catalog_status["exists"]:
        try:
            cat = _load_catalog()
            catalog_status["n_scenarios"] = len(cat.get("scenarios", {}))
            catalog_status["n_streets"] = len(cat.get("streets", []))
            catalog_status["built_at"] = cat.get("built_at")
        except Exception as exc:
            catalog_status["error"] = str(exc)
    return {
        "status": "ok" if catalog_status["exists"] else "degraded",
        "catalog": catalog_status,
        "precache_dir": {
            "path": str(PRECACHE_DIR),
            "exists": PRECACHE_DIR.exists(),
        },
    }


def reset_catalog_cache() -> None:
    """Clear the in-memory catalog cache (useful after a rebuild).

    Normally ``_load_catalog`` invalidates automatically via mtime,
    so this is mostly retained for tests and edge cases (e.g.
    catalog written with the same mtime as the cached copy).
    """
    global _CATALOG_CACHE, _CATALOG_CACHE_MTIME
    _CATALOG_CACHE = None
    _CATALOG_CACHE_MTIME = None
