# Symbol Matcher

A small FastAPI + OpenCV web app for electrical worksheets.

Pick a **request (rid)** and a **worksheet (wid)**, the app downloads that
worksheet's raster from its `image_url`, then you **drag a box around one
symbol** and every visually-similar patch on the sheet is highlighted.

Matching uses normalized cross-correlation (`cv2.matchTemplate`,
`TM_CCOEFF_NORMED`) evaluated over several template scales, followed by
non-maximum suppression to remove overlapping detections.

There are also two **SAM** modes that segment symbols using the worksheet's
annotated **reference points** (the `geometry_type == 1` Point features from
`worksheet_geometries`), one box per point:

- **FastSAM boxes** (cyan): runs FastSAM on a small crop around each point and
  keeps the tightest mask covering it. Default SAM mode; fastest.
- **HQ-SAM boxes** (magenta): runs Light HQ-SAM (`vit_tiny`) with three
  dense-region tricks — (1) an **adaptive crop** shrunk toward the nearest
  neighbouring reference point so a crowded symbol fills its patch and neighbours
  are excluded, (2) neighbouring reference points fed as **negative** prompts, and
  (3) a **refinement pass** that feeds the first mask's logits back with the
  negatives to sharpen the boundary. Slower but crisper on thin line-drawing
  symbols and in dense grids.
- **SAM2.1 boxes** (amber): runs **SAM 2.1** (Hiera-small, via Ultralytics) with
  the same adaptive-crop + neighbour-negative recipe. This is the SAM-family
  model that runs on Apple **MPS**, so it's the Mac-friendly stand-in for SAM 3.
  Heaviest of the three (re-encodes each crop) but a useful cross-check.

Both share a connected-component fallback and `min_symbol_px`/`max_symbol_px`
size clamps. MobileSAM is still available via `?model=mobilesam`. The worksheet
dropdown marks sheets that have geometries (●) and shows page numbers, sorted
geometry-first.

**Pseudo-color** (the "Pseudo-color" checkbox / `?pseudocolor=1`, default off):
SAM-family models are trained on natural photos and struggle with razor-thin,
1-px monochrome CAD lines. When enabled, each crop fed to FastSAM / HQ-SAM /
SAM 2.1 is first run through a multi-scale per-channel Gaussian pyramid
("glow" mode, `pseudocolor.py`): the red channel gets a tight kernel (3×3),
green a medium kernel (7×7) and blue a wide kernel (13×13), producing a coloured
chromatic-aberration gradient around every line that SAM perceives like natural
depth/lighting. The kernels are fixed at `(3, 7, 13)`. Box coordinates and the
connected-component fallback still use the original (un-blurred) image, so only
what the model *sees* changes.

**Sharpen** (the "Sharpen" checkbox / `?sharpen=1`, default off): an alternative
preprocessing that unsharp-masks the crop fed to the SAM model with a
deliberately aggressive amount (`pseudocolor.sharpen`, `amount=1.5`) so
razor-thin strokes pop. Like pseudo-color it only changes the model input;
coordinates and the fallback stay on the original image. It can be combined with
pseudo-color (sharpen is applied first, then the glow).

Weights: `models/FastSAM-s.pt` and `models/sam2.1_s.pt` (auto-downloaded by
ultralytics), `models/sam_hq_vit_tiny.pth` (Light HQ-SAM, from
`huggingface.co/lkeab/hq-sam`), and optionally `models/mobile_sam.pt`.

### SAM 3 vs SAM 2.1 on Mac

SAM 3 (`srcs/sam3`, `sam3.build_sam3_image_model`) is a text/exemplar concept
segmenter but is **not usable on this Mac**, so we use **SAM 2.1** (`model=sam2`,
amber) as the Mac-friendly SAM-family stand-in — it runs on Apple MPS through
Ultralytics and auto-downloads its weights. SAM 3 blockers found while evaluating:

1. **MPS abort** — even with `PYTORCH_ENABLE_MPS_FALLBACK=1`, a forward pass
   aborts with `MPSNDArrayMatrixMultiplication ... cannot have different datatype`
   (SIGABRT). It cannot run on the Apple GPU; CPU-only would be very slow.
2. **Weights** — the official `facebook/sam3` checkpoint is gated (needs an HF
   token), and the local `models/weights/best_stg1(1).pth` uses a
   `backbone.dinov3.*` naming that does not match the installed builder
   (`backbone.vision_backbone.trunk.*`), so the vision backbone won't load.

To use SAM 3 itself, run it on a CUDA machine with the proper `facebook/sam3`
weights following `srcs/sam3/examples/sam3_for_sam1_task_example.ipynb` (SAM
1-style point/box prompting) or `sam3_image_predictor_example.ipynb` (text/
exemplar concept prompting, which fits "find all instances of this symbol").

## Layout

```
symbol_matcher_app/
  main.py              FastAPI app + routes
  worksheet_loader.py  rid/wid discovery + image_url download (disk-cached)
  matcher.py           multi-scale matchTemplate + NMS
  sam_boxes.py         reference-point loader + MobileSAM box-finding
  fastsam_boxes.py     FastSAM crop-wise box-finding (default SAM mode)
  hqsam_boxes.py       Light HQ-SAM crop-wise box-finding (model=hqsam)
  sam2_boxes.py        SAM 2.1 crop-wise box-finding, Mac-friendly (model=sam2)
  pseudocolor.py       glow pseudo-coloring + unsharp-mask sharpening for SAM input
  static/              index.html, app.js, style.css  (drag-select UI)
  requirements.txt
```

Worksheet metadata is read from `../data/requests/<rid>/worksheets_metadata.json`.
Downloaded images are cached under `symbol_matcher_app/.image_cache/`.

## Run

The repo already has a `vsam` virtualenv with OpenCV; FastAPI/uvicorn were
installed into it. From inside this folder:

```bash
cd symbol_matcher_app
../.envs/vsam/bin/uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000/

> Loading worksheet images requires internet access (the rasters live on
> Google Cloud Storage). The first load of a worksheet downloads and caches it.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/requests` | List request ids that have cached metadata |
| GET | `/api/requests/{rid}/worksheets` | List worksheets (wid, name, title, size) |
| GET | `/api/worksheet/{rid}/{wid}/image` | Worksheet PNG |
| GET | `/api/worksheet/{rid}/{wid}/meta` | `{width, height}` of the raster |
| POST | `/api/worksheet/{rid}/{wid}/match` | Body `{x,y,w,h,threshold,scales?}` → matching boxes |
| GET | `/api/worksheet/{rid}/{wid}/sam_points?limit=N&model=fastsam&pseudocolor=0&sharpen=0` | SAM boxes from reference points (`model`=fastsam\|hqsam\|sam2\|mobilesam, `limit` caps points, `pseudocolor=1` glow- and/or `sharpen=1` unsharp-preprocesses the SAM input) |

Coordinates in `/match` are in **original image pixels**.
