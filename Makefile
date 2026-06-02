.PHONY: help build-webapp-data publish-webapp-data

PY ?= venv/bin/python

help:
	@echo "Targets:"
	@echo "  build-webapp-data    Rebuild StreetLight overlays + static maps into the"
	@echo "                       SUMO build tree, then publish the slim serve set to"
	@echo "                       data/webapp/ (the Git-LFS-tracked, image-baked set)."
	@echo "  publish-webapp-data  Re-publish data/webapp/ from the existing build tree"
	@echo "                       (no rebuild). Run after a manual precache build."

# Rebuild the lightweight overlay + static-map artefacts into the SUMO build
# tree (data/sumo/precache_build) and then mirror the slim served subset into
# data/webapp/. data/webapp/ is committed via Git LFS and baked into the
# container image, so a commit + CI build is all that is needed to deploy.
# (A full scenario rebuild is the heavier `webapp/scripts/build_precache.py`.)
build-webapp-data:
	$(PY) webapp/scripts/build_streetlight_overlay.py
	$(PY) webapp/scripts/build_static_maps.py
	$(PY) webapp/scripts/build_precache.py --publish-only

publish-webapp-data:
	$(PY) webapp/scripts/build_precache.py --publish-only
