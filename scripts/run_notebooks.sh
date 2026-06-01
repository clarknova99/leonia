#!/usr/bin/env bash
# Headless render of the stakeholder notebooks to HTML.
#
# Usage (from repo root):
#   scripts/run_notebooks.sh           # render stakeholder + dev
#   scripts/run_notebooks.sh stake     # just stakeholder
#   scripts/run_notebooks.sh dev       # just dev
#
# Outputs land in reports/notebooks/.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="${PY:-$REPO/venv/bin/python}"
JUP="${JUP:-$REPO/venv/bin/jupyter}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python interpreter not found at $PY" >&2
  echo "set PY=... or create the venv: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

mode="${1:-all}"

OUT_DIR="$REPO/reports/notebooks"
mkdir -p "$OUT_DIR"

render() {
  local src="$1"
  local subdir="$2"
  local hide="$3"   # "hide" | "show"
  local dst_dir="$OUT_DIR/$subdir"
  mkdir -p "$dst_dir"

  echo "→ executing $src"
  "$JUP" nbconvert --to notebook --execute --inplace "$src" \
    --ExecutePreprocessor.timeout=900

  echo "→ rendering $src → HTML"
  local args=(--to html --output-dir "$dst_dir")
  if [[ "$hide" == "hide" ]]; then
    args+=(--no-input --TagRemovePreprocessor.remove_input_tags='{"remove-input"}')
  fi
  "$JUP" nbconvert "${args[@]}" "$src"
}

if [[ "$mode" == "stake" || "$mode" == "all" ]]; then
  for nb in notebooks/stakeholder/*.ipynb; do
    render "$nb" "stakeholder" "hide"
  done
fi

if [[ "$mode" == "dev" || "$mode" == "all" ]]; then
  for nb in notebooks/dev/*.ipynb; do
    render "$nb" "dev" "show"
  done
fi

echo ""
echo "Done. Rendered HTML under: $OUT_DIR"
ls -1 "$OUT_DIR"/*/*.html 2>/dev/null | sed "s|$REPO/||"
