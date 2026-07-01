# Electrical Symbol Detection

Tools and experiments for detecting and highlighting electrical symbols on
construction worksheet drawings.

## Contents

- **`symbol_matcher_app/`** — a FastAPI + OpenCV web app to browse worksheets,
  drag-select a symbol and highlight every similar patch (multi-scale
  `matchTemplate` + NMS), plus SAM-based box finding (FastSAM / HQ-SAM / SAM 2.1)
  from the worksheet's ground-truth reference points, with optional glow
  pseudo-coloring and sharpening preprocessing. See
  [`symbol_matcher_app/README.md`](symbol_matcher_app/README.md) and
  [`symbol_matcher_app/PIPELINE.md`](symbol_matcher_app/PIPELINE.md).
- **`eda/`** — exploratory notebooks and plots (data inspection, geometry
  overlays, symbol clustering, template/label matching).
- **`table_processing/`** — legend/table extraction experiments.

## Getting started

The app runs against locally cached worksheet metadata/geometries and downloads
worksheet rasters on demand:

```bash
cd symbol_matcher_app
../.envs/vsam/bin/uvicorn main:app --reload --port 8000
# open http://127.0.0.1:8000/
```

## Not included in this repo

Large or sensitive artifacts are intentionally git-ignored (see `.gitignore`):
worksheet data (`data/`), model weights (`models/`), run outputs (`outputs/`),
virtual environments (`.envs/`), logs, third-party clones (`srcs/`), image
caches, oversized notebooks, and any credential files (`.env`, `config.yaml`).
