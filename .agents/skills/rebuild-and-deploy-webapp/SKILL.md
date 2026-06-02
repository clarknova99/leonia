---
name: rebuild-and-deploy-webapp
description: Rebuild and/or re-publish the webapp serve set and ship it via Git LFS + image bake. Use when SUMO scenarios, overlays, static maps, or the served data/webapp/ set need refreshing or deploying.
---

# Rebuild & deploy the web app

The webapp is read-only. Everything is precomputed into a heavy **build
tree** (`data/sumo/precache_build/`, ~17 GB, git-ignored) and a slim
**published serve set** (`data/webapp/`, ~0.4 GB) that is Git-LFS-tracked
and baked into the container image. There is no runtime data mount.

## Build options

```bash
# Full scenario rebuild (heavy; writes the build tree, then publishes data/webapp/).
venv/bin/python webapp/scripts/build_precache.py --parallel 8

# Lightweight: refresh StreetLight overlays + static maps, then re-publish.
make build-webapp-data

# Re-publish data/webapp/ from the existing build tree (no rebuild).
make publish-webapp-data        # = build_precache.py --publish-only
```

`build_precache.py` is idempotent/resumable (skips scenarios whose
`manifest.json` exists; `--force` to rebuild). Useful flags: `--streets`,
`--change-types closure speed_hump`, `--demands`, `--top-n`, `--parallel`.

## Serve locally to verify

```bash
venv/bin/uvicorn webapp.app.main:app --reload --port 8000
curl -sS http://127.0.0.1:8000/healthz | jq
# catalog: GET /api/catalog.json
```

## Deploy (Git LFS + CI image bake)

1. One-time per machine: `brew install git-lfs && git lfs install`.
2. Commit the refreshed serve set: `git add data/webapp && git commit`.
   `data/webapp/**` is LFS-tracked via `.gitattributes`, so the bytes go
   to the LFS store and only pointers land in the pack.
3. `git push` — uploads changed LFS objects first, then the commit.
4. CI (`.github/workflows/webapp.yaml`) checks out with `lfs: true` and
   builds the Dockerfile `bundled` target, which `COPY`s `data/webapp/`
   into a self-contained image.

## Gotchas

- A clone/CI **without git-lfs** gets pointer stubs → the image bakes tiny
  text files and the webapp serves nothing. Always `lfs: true` / `git lfs pull`.
- GitHub free LFS tier is 1 GB storage + 1 GB bandwidth/month; budget a
  $5 data pack if rebuilding more than a few times a month.
- libsumo work happens in `build_precache.py` (out-of-band); SUMO is not
  installed in the webapp image.
