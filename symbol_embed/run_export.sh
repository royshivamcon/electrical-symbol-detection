#!/usr/bin/env bash
# Export fastsamx_sam2 convex-hull crops.
#
# Usage:
#   bash symbol_embed/run_export.sh                    # default: 10 RIDs
#   bash symbol_embed/run_export.sh --n-rids 10
#   bash symbol_embed/run_export.sh --rid <RID>
#   bash symbol_embed/run_export.sh --rid <RID1> --rid <RID2> --force
# Adapter pool (locked 5 RIDs, L40S-tuned workers):
#   bash symbol_embed/run_export_adapter.sh
#   bash symbol_embed/run_export.sh --rids-file symbol_embed/adapter_rids.txt --workers 16
#
# Requires the symbol_matcher FastSAM/SAM2 stack.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/symbol_matcher_app${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" -m symbol_embed.scripts.export_crops "$@"
