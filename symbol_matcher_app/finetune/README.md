# FastSAM mask-confidence head (whole-sheet detector)

Trains a small network that **re-scores each FastSAM mask's confidence** for electrical
symbols. FastSAM stays **frozen**; we only learn a better confidence than the raw
YOLOv8-seg objectness (`res.boxes.conf`) that the pipeline uses today
(`seg_models.py:132`).

## Why

FastSAM has no dedicated confidence estimator — the `SamBox.score` is YOLO objectness,
which is poorly calibrated for tiny CAD glyphs. Earlier we labeled by "covers the centered
reference point + size gate", which forced every training crop to be point-centered and
made training diverge from `visualise.py` (segment-everything). Now we **target the app's
own bounding boxes** and score on segment-everything masks, so training == the
detection-time distribution.

## Label rule (IoU vs app GT boxes)

1. **GT boxes**: run the app finder (`boxes_from_points`, model `PrepCfg.gt_model`,
   default `fastsam`) at the sheet's **electrical** reference points → per-symbol GT boxes,
   kept in rendered space (same zoom branch as `main.run_sam`).
2. **Candidates**: run frozen FastSAM **segment-everything** across the whole sheet in
   overlapping tiles (same scan as `visualise.py`) → every mask FastSAM proposes, after a
   size gate (`cand_min_px ≤ min(w,h)`, `bbox_area ≤ cand_max_frac*tile_area`).
3. **Label** each candidate by max IoU with a GT box:

| condition | label |
|-----------|-------|
| `IoU ≥ pos_iou` (default **0.5**) | **positive** |
| `IoU < neg_iou` (default **0.1**) | negative (sampled) |
| in between | dropped (ambiguous band) |

Per sheet we keep all positives + `neg_per_pos` negatives per positive (half near-miss by
IoU, half random). Each kept mask is cached on a **centroid-centered** window (the same
window `visualise._head_scores` scores on), so training and inference match.

Both `prep_dataset.py` (training labels) and `eval.py` use this IoU rule. `eval.py`
recomputes the label from the manifest's stored `iou` via `dataset.labels_from_iou(pos_iou)`
and reports how many labels flipped versus what was baked at prep time.

> **Caveat:** GT only exists at annotated reference points, so an un-annotated real symbol
> becomes a negative (label noise). The `neg_iou` ignore band + near-miss sampling limit
> the impact; switching `gt_model` to `mix`/`hqsam` is a one-line config change.

## Confidence head

The score is a small CNN over the `[gray, mask, point-heatmap]` crop, made
mask-specific and calibrated:

- **Masked pooling**: features are pooled *inside the mask* (plus a global-average branch
  for context) so two overlapping candidates sharing a crop get distinct embeddings —
  the score is about *this mask*, not the neighbourhood.
- **Scalar-feature fusion** (`features.build_feats`): the pooled embedding is concatenated
  with `[FastSAM conf, bbox area frac, aspect, extent]` before the FC, so the head reuses
  FastSAM's objectness and cheap geometry instead of throwing them away.
- **Temperature calibration**: after training, a single temperature `T` is fit on the val
  split (min NLL) and stored in the checkpoint; inference reports `sigmoid(logit / T)`, a
  calibrated probability. `T` doesn't change ranking, so AUC is unchanged.

## Loss

Plain `BCEWithLogitsLoss` on the binary IoU label. A positive is a mask that overlaps a
GT symbol box (`IoU ≥ pos_iou`); everything else is a negative, so BCE alone already
pushes junk/oversized masks toward a low score. Class imbalance is handled by the
`WeightedRandomSampler` in `train.py`, not the loss.

## Memory

Sheets can be ~28000×21000 at 4x. We **never** load the full sheet: GT boxes and
segment-everything candidates are both produced by rendering tiles on demand via
`worksheet_loader.pdf_tile_renderer` (one PDF open per sheet). FastSAM runs under
`torch.no_grad()` in tile batches with periodic `gc` / `mps.empty_cache`. Only the kept
masks' centroid crops + masks are cached to `dataset/crops/*.npz`, so training/eval never
re-render or re-run FastSAM.

## Run (from `Electrical/symbol_matcher_app/`, vsam env)

```bash
PY=../.envs/vsam/bin/python

# 1) DATASET — app GT boxes + segment-everything candidates + IoU labels, centroid crops.
#    Build over 5 sheets (drop --limit-sheets for all; --limit-points caps GT points/sheet):
$PY finetune/prep_dataset.py --limit-sheets 5
#    optional: pin to one request's sheets
#    $PY finetune/prep_dataset.py --rid 0ead8522-b3a9-4bd6-a430-dd0b4dbec6ad --limit-sheets 5

# 2) TRAIN the head (plain BCE; writes finetune/checkpoints/mask_conf.pt):
$PY finetune/train.py --epochs 20

# 3) EVAL — head AUC vs baseline FastSAM-conf AUC on held-out sheets, using the IoU rule.
#    --pos-iou defaults to config; override to re-threshold without regenerating:
$PY finetune/eval.py
$PY finetune/eval.py --pos-iou 0.5
#    restrict to one sheet (evaluates all its candidates): --rid <id> --wid <id>

# 4) VISUALISE — segment-everything sheet overlay re-scored by the trained head; each
#    mask coloured green (high head conf) -> red (low). Writes per-tile PNGs + one
#    stitched full-sheet composite under dataset/viz_sheets/<rid8>_<wid8>/.
$PY finetune/visualise.py --rid 0ead8522-b3a9-4bd6-a430-dd0b4dbec6ad \
                          --wid 516d8877-f013-42d1-a160-a4c77a6d180e
$PY finetune/visualise.py --limit-tiles 12          # quick dry-run on random tiles
$PY finetune/visualise.py --rid <id> --wid <id> --min-score 0.5   # keep only confident masks
$PY finetune/visualise.py --rid <id> --wid <id> --gt-boxes        # overlay app GT boxes (blue)
```

Full run (follow-up): drop `--limit-sheets` in step 1 and keep `--epochs` high in step 2.
`visualise.py` needs the **full** rid/wid (they are passed straight to the PDF renderer);
`eval.py` accepts id **prefixes**.

## Files

| file | role |
|------|------|
| `config.py` | `PrepCfg` (tiling + IoU label knobs) + `TrainCfg` |
| `prep_dataset.py` | app GT boxes + segment-everything candidates, IoU labels, centroid-crop cache |
| `features.py` | `build_input` ([3,H,W] gray+mask+heatmap) + `build_feats` (conf+geometry) |
| `dataset.py` | `MaskConfDataset` over the manifest (`labels_from_iou` for eval) |
| `model.py` | `MaskConfidenceNet` (masked-pool CNN + scalar-feat fusion → logit) |
| `train.py` | BCE + weighted-sampler loop, temperature calibration, best-by-val-AUC ckpt |
| `infer.py` | `MaskConfidenceScorer` — load ckpt, score masks (batched, calibrated) |
| `eval.py` | head vs baseline-conf AUC, P/R/F1, calibration (IoU rule) |
| `visualise.py` | segment-everything sheet overlay re-scored by the head; optional `--gt-boxes` overlay; stitched full-sheet PNG |
| `metrics.py` | dependency-free AUC / P-R-F1 / calibration |
| `integrate.md` | opt-in patch to plug the head into `seg_models._select_from_covering` |

Outputs (`dataset/`, `checkpoints/`) are git-ignored.
