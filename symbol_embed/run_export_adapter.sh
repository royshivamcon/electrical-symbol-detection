#!/usr/bin/env bash
# Export FastSAM×SAM2 crops for the locked adapter 5-RID pool (NVIDIA L40S).
#
# Uses symbol_embed/adapter_rids.txt:
#   0ead8522-b3a9-4bd6-a430-dd0b4dbec6ad
#   c9a5099a-37cb-4693-b6e0-a2b5592b50b2
#   aaaf192e-0925-40a1-a19a-09efecd8b3b9
#   1aaf9456-be5b-4bbf-a821-1c14b6551e7e
#   8b9aa9a4-1df5-4b65-a9b5-b2f2bfe79e72
#
# L40S + 16-vCPU defaults: --workers 16 (CPU postproc). Extra args are forwarded
# (e.g. --force, --max-sheets 1 for a smoke sheet).
#
#   bash symbol_embed/run_export_adapter.sh
#   bash symbol_embed/run_export_adapter.sh --force
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/symbol_matcher_app${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

RIDS_FILE="${ROOT}/symbol_embed/adapter_rids.txt"
exec "$PYTHON" -m symbol_embed.scripts.export_crops \
  --rids-file "$RIDS_FILE" \
  --workers 16 \
  "$@"
