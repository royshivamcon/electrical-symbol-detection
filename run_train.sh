#!/usr/bin/env bash
# Train DINOv2 embedding arms on a POOL of RIDs (default 10).
#
# Usage:
#   bash symbol_embed/run_train.sh                         # 10 rids with crops
#   bash symbol_embed/run_train.sh --n-rids 10 --arm all
#   bash symbol_embed/run_train.sh --rids id1,id2,id3
#   bash symbol_embed/run_train.sh --rebuild-pool --wandb-mode offline
#
# Per-RID eval / t-SNE (pass a single --rid):
#   bash symbol_embed/run_eval.sh --rid <RID>
#   bash symbol_embed/run_visualise.sh --rid <RID>
#
# Env: PYTHON, WANDB_API_KEY, WANDB_ENTITY

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" -m symbol_embed.scripts.train "$@"
