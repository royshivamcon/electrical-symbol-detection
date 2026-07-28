#!/usr/bin/env bash
# Export quill_local bbox crops at zoom=4x for train_rids.txt (+ val RIDs).
#
#   bash symbol_embed/run_export_quill.sh
#   bash symbol_embed/run_export_quill.sh --force
#   bash symbol_embed/run_export_quill.sh --max-sheets 2   # smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/symbol_matcher_app${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m symbol_embed.scripts.export_quill_crops "$@"
