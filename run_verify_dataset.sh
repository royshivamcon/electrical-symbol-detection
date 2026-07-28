#!/usr/bin/env bash
# Build/refresh pooled_10rids_quill RID split and write verify_report.md
#
#   bash symbol_embed/run_export_quill.sh
#   bash symbol_embed/run_verify_dataset.sh
#   bash symbol_embed/run_verify_dataset.sh --rebuild-pool
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/symbol_matcher_app${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m symbol_embed.scripts.verify_dataset "$@"
