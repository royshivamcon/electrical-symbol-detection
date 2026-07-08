# Plugging the head into the live pipeline (opt-in)

**Not applied yet.** Flip this on only after `eval.py` shows the head beats baseline conf.
It is gated by an env var, so with the var unset the app behaves exactly as today.

> **Note — the head is now a whole-sheet detector.** It is trained on **centroid-centered**
> segment-everything masks, labeled by IoU against the app's own GT boxes (see `README.md`).
> Its native use is re-ranking segment-everything masks, exactly what `visualise.py` does
> (re-center the crop on each mask's centroid, pass the centroid as the prompt point). The
> point-prompted hook below still works as an approximation, but for best match pass the
> selected mask's **centroid** (not the reference point) as `(cx, cy)`.

## Where

The confidence a mask carries today is set in `FastAdapter.encode_predict`
(`seg_models.py:195`), whose `score` comes from `_select_from_covering` →
`score = max(cf ...)` (`seg_models.py:132`). That method has the crop (`model_crop`),
the point (`cx, cy`) and the covering masks — everything the head needs — so it is the
natural hook.

## Patch (seg_models.py)

Add a lazy loader near the other helpers:

```python
_MASK_CONF_SCORER = None
_MASK_CONF_TRIED = False

def _mask_conf_scorer():
    """Lazily load the trained mask-confidence head if MASK_CONF_CKPT is set."""
    global _MASK_CONF_SCORER, _MASK_CONF_TRIED
    if _MASK_CONF_TRIED:
        return _MASK_CONF_SCORER
    _MASK_CONF_TRIED = True
    import os
    ckpt = os.environ.get("MASK_CONF_CKPT")
    if ckpt:
        import sys
        sys.path.insert(0, str(APP_DIR / "finetune"))
        from infer import MaskConfidenceScorer
        _MASK_CONF_SCORER = MaskConfidenceScorer.load(ckpt)
    return _MASK_CONF_SCORER
```

Then in `FastAdapter.encode_predict`, force the union mask when the head is on and
replace the score:

```python
    def encode_predict(self, model_crop, cx, cy, negatives, *, want_mask, cfg: Cfg):
        model = self.get_model()
        ch, cw = model_crop.shape[:2]
        res = model(model_crop, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                    retina_masks=True, verbose=False)[0]
        covering = _covering_masks(res, cx, cy, ch, cw, ch * cw, cfg.max_box_frac)
        if not covering:
            return None
        scorer = _mask_conf_scorer()
        ux0, uy0, ux1, uy1, score, mask = _select_from_covering(
            covering, cfg.size_ratio, cfg.max_symbol_px, ch, cw,
            want_mask or scorer is not None)          # need the mask to score it
        if scorer is not None and mask is not None:
            # match training: score at the mask centroid (not the reference point), and
            # pass FastSAM objectness (current `score`) as the fused conf feature
            ys, xs = np.where(mask)
            mcx, mcy = float(xs.mean()), float(ys.mean())
            score = scorer.score(model_crop, mask, mcx, mcy, conf=score)  # learned confidence
        clipped = ux0 <= 1 or uy0 <= 1 or ux1 >= cw - 2 or uy1 >= ch - 2
        return Pred(int(round(ux0)), int(round(uy0)), int(round(ux1)), int(round(uy1)),
                    float(score), clipped, mask if want_mask else None)
```

## Enable

```bash
export MASK_CONF_CKPT="$PWD/finetune/checkpoints/mask_conf.pt"
../.envs/vsam/bin/uvicorn main:app --reload --port 8000
```

Unset the var to revert. The `score` field then carries the learned confidence
everywhere it already flows (`SamBox.score`, the `/sam_points` JSON, the eval overlay).

## Later options (not needed for v1)

- **Selection, not just scoring:** use the head to *pick* which covering mask(s) to
  union (replace the size-median heuristic in `_select_from_covering`).
- **Rejection gate:** drop boxes whose head score < threshold to raise precision — tune
  the threshold on the `evaluate` endpoint.
