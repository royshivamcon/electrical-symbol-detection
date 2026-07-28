#!/usr/bin/env bash
# Full DINOv2-Reg fine-tune on quill RID-holdout pool.
#
#   bash symbol_embed/run_verify_dataset.sh
#   bash symbol_embed/run_train_dino_full.sh --arm both
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/symbol_matcher_app${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m symbol_embed.scripts.train_dino_full "$@"
