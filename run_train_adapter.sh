#!/usr/bin/env bash
# Train DINOv2-Reg adapter on quill 10-RID pool (RID holdout).
#
#   bash symbol_embed/run_export_quill.sh
#   bash symbol_embed/run_verify_dataset.sh --rebuild-pool
#   bash symbol_embed/run_train_adapter.sh --pool-quill --arm both --layers 0,11
#
# Legacy FastSAM pool (omit --pool-quill):
#   bash symbol_embed/run_train_adapter.sh --arm both --layers 0,11
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/symbol_matcher_app${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m symbol_embed.scripts.train_adapter "$@"
