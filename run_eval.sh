#!/usr/bin/env bash
# Evaluate embedding checkpoints on ONE request (RID).
#
# Usage:
#   bash symbol_embed/run_eval.sh --rid <RID>
#   bash symbol_embed/run_eval.sh --rid <RID> --split all
#   bash symbol_embed/run_eval.sh --rid <RID> \
#        --ckpt contrastive=symbol_embed/runs/contrastive/.../best.pt
#
# Writes symbol_embed/eval/<rid>/… metrics + embeddings.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" -m symbol_embed.scripts.eval_rid "$@"
