#!/usr/bin/env bash
# Visualise embedding t-SNE for ONE request (RID) — required --rid.
# Uses latest multi-RID training checkpoints under runs/ by default.
#
# Usage:
#   bash symbol_embed/run_visualise.sh --rid <RID>
#   bash symbol_embed/run_visualise.sh --rid <RID> --split all
#   bash symbol_embed/run_visualise.sh --rid <RID> \
#        --ckpt pretrained=symbol_embed/runs/pretrained/.../best.pt \
#        --ckpt contrastive=symbol_embed/runs/contrastive/.../best.pt \
#        --ckpt arcface=symbol_embed/runs/arcface/.../best.pt
#
# Writes under symbol_embed/viz/<rid>/ :
#   {arm}_tsne.png
#   {arm}_tsne_hover.html
#   compare_tsne_hover.html

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" -m symbol_embed.scripts.plot_tsne "$@"
