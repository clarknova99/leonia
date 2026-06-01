.PHONY: help sync-webapp-data build-webapp-data

help:
	@echo "Targets:"
	@echo "  build-webapp-data  Rebuild _overlays/ + _static/ artefacts (no upload)"
	@echo "  sync-webapp-data   Rebuild artefacts and push the deployable subset to the NAS"

# Rebuild the StreetLight overlays + static maps and push them to the NAS share
# the Kubernetes pod mounts. Implemented in a .gitignored script because it
# holds NAS credentials (see scripts/sync_webapp_data.sh).
sync-webapp-data:
	@scripts/sync_webapp_data.sh

build-webapp-data:
	@scripts/sync_webapp_data.sh --build-only
