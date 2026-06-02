---
name: rebuild-data
description: Rebuild the canonical (stage-1) + derived (stage-2) Parquet lake from the raw StreetLight/NJDOT exports. Use when raw data changes, a loader changes, or the lake is stale/missing.
---

# Rebuild the data lake

Builds `data/stage-1/{streetlight,crash}/` and `data/stage-2/` from the
raw exports under `data/raw/`. Idempotent — reruns overwrite parquets and
refresh `_manifest.json`. ~20 s total (95% is the 2.8M-row work-block-
groups CSV; reruns that skip it finish in ~2 s).

## Steps

```bash
# Full rebuild (canonical + derived)
venv/bin/python scripts/00_build_datasets.py

# Only one product family
venv/bin/python scripts/00_build_datasets.py --only za

# Skip the slow file while iterating
venv/bin/python scripts/00_build_datasets.py --skip za_work_block_groups.parquet

# Canonical only (skip the stage-2 derived layer)
venv/bin/python scripts/00_build_datasets.py --skip-derived
```

The crash overlay (`data/stage-1/crash/`) is a separate downstream step:

```bash
venv/bin/python scripts/14_build_crash_overlay.py            # cached source
venv/bin/python scripts/14_build_crash_overlay.py --refresh  # re-pull NJDOT
```

## Verify

- `data/stage-1/streetlight/_manifest.json` and `data/stage-2/_manifest.json`
  have fresh `built_at` timestamps and expected row counts.
- `venv/bin/python -m pytest -q` (loader/analytics tests) passes.

## Notes

- The orchestrator uses an **explicit** product list; adding a new
  product needs a code change (loader + `CanonicalFiles` constant +
  `build_<product>()` registered in `KNOWN_PRODUCTS`). See the
  `add-streetlight-extract` skill.
- Never hardcode output paths — they come from
  `leonia_traffic/data/dataset_io.py`.
