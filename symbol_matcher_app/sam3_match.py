"""One-shot concept matching via SAM 3 (ultralytics ``SAM3SemanticPredictor``).

A third "template match" backend for the ``/match`` endpoint, alongside the
classical NCC matcher (:mod:`matcher`) and the SAM-HQ feature matchers
(:mod:`feat_match`). Where those *slide* a template/kernel across the sheet, this
hands the user's selected rectangle to SAM 3 as a **visual exemplar** and lets its
promptable concept segmentation return every instance of that symbol in one pass.

Unlike :mod:`feat_match`, this is **not tiled**. SAM 3 concept segmentation is
inherently whole-image: the exemplar box points at one instance *inside the encoded
image* and the model finds the rest, so the exemplar only exists in one tile and
tiling would blind every other tile to it. The whole sheet is therefore letterboxed
to a single square ``imgsz`` -- raise ``imgsz`` to recover small symbols on large
sheets (at a memory/time cost), lower it if inference is slow or OOMs.

Weights: drop ``sam3.pt`` in the repo ``models/`` dir (same place as the FastSAM /
HQ-SAM checkpoints). If it is absent we fall back to the bare name ``"sam3.pt"`` so
ultralytics can attempt its auto-download, matching :mod:`seg_models`' convention.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import numpy as np

from matcher import Match

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SAM3_CKPT = PROJECT_ROOT / "models" / "sam3.pt"

# ultralytics' predictor keeps per-call state (``self.batch`` / ``self.features``),
# so it is not reentrant; serialize match calls behind one lock (the model is heavy
# enough that one-at-a-time is the right policy anyway).
_PREDICTORS: dict[int, object] = {}
_LOCK = threading.Lock()


def _preflight() -> None:
    """Fail fast with an actionable message if SAM 3's runtime deps are missing.

    Called before the predictor is built so a live ``/match`` request returns a clear
    error instead of:

    - ultralytics attempting a ``pip install`` of the CLIP fork *inside the request*
      when ``clip`` is absent (``build_sam3_image_model`` runs ``check_requirements``);
    - a confusing auto-download failure when the weights aren't on disk (``sam3.pt`` is
      not published as an ultralytics asset, so it cannot be fetched automatically).

    Raises ``FileNotFoundError`` (missing weights) or ``RuntimeError`` (missing CLIP);
    the ``/match`` endpoint maps both to an HTTP 503 with this message.
    """
    if not SAM3_CKPT.exists():
        raise FileNotFoundError(
            f"SAM 3 weights not found at {SAM3_CKPT}. Download the SAM 3 image "
            "checkpoint and place it there — it is not auto-downloadable via ultralytics."
        )
    if importlib.util.find_spec("clip") is None:
        raise RuntimeError(
            "SAM 3 needs the CLIP package (ultralytics fork). Install it into this "
            "environment first: pip install 'git+https://github.com/ultralytics/CLIP.git'"
        )


def _device() -> str:
    """ultralytics device string: 'mps' on Apple, '0' for the first CUDA GPU, else 'cpu'."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _get_predictor(imgsz: int):
    """Return a cached ``SAM3SemanticPredictor`` for ``imgsz`` (built lazily).

    ``imgsz`` fixes the square letterbox size, so it is baked into the model setup
    and cached per value; ``conf`` / ``iou`` are cheap post-process knobs and are set
    per call instead (see :func:`match_template`).
    """
    pred = _PREDICTORS.get(imgsz)
    if pred is not None:
        return pred
    with _LOCK:
        if imgsz not in _PREDICTORS:
            from ultralytics.models.sam import SAM3SemanticPredictor

            model_arg = str(SAM3_CKPT) if SAM3_CKPT.exists() else "sam3.pt"
            _PREDICTORS[imgsz] = SAM3SemanticPredictor(
                overrides=dict(
                    model=model_arg, imgsz=imgsz, device=_device(),
                    save=False, verbose=False, mode="predict",
                )
            )
    return _PREDICTORS[imgsz]


def match_template(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    threshold: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 2048,
    min_symbol_px: int = 3,
    max_symbol_px: int | None = None,
    pad: int = 0,
    max_matches: int = 4000,
) -> list[Match]:
    """Find every instance of the exemplar ``box`` via SAM 3 concept segmentation.

    Parameters
    ----------
    box : ``(x, y, w, h)`` exemplar rectangle in original image pixels.
    threshold : SAM 3 detection confidence (``conf``) to keep a match. Note this is a
        detector score, not the NCC/cosine similarity used by the other backends --
        it usually wants a *lower* value (~0.2-0.4) than ``classical``/``tmr``.
    iou : IoU for SAM 3's built-in NMS over the returned instances.
    imgsz : square size the whole sheet is letterboxed to before encoding; larger
        recovers smaller symbols at higher memory/time cost.
    min_symbol_px / max_symbol_px : drop returned boxes whose longer side is outside
        this range (``max_symbol_px=None`` disables the upper bound).
    pad : pixels added around each returned box.
    """
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if w < 3 or h < 3:
        return []
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = x, y, min(x + w, W), min(y + h, H)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return []

    _preflight()  # clear error if weights / CLIP are missing (before ultralytics tries)
    predictor = _get_predictor(int(imgsz))

    with _LOCK:
        # conf/iou are read fresh by ``postprocess`` each call, so setting them here
        # honours the per-request threshold without rebuilding the model.
        predictor.args.conf = float(threshold)
        predictor.args.iou = float(iou)
        results = predictor(
            source=image_bgr,
            bboxes=[[x1, y1, x2, y2]],
            labels=[1],
        )

    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    out: list[Match] = []
    for (bx0, by0, bx1, by1), sc in zip(xyxy, conf):
        bw, bh = float(bx1 - bx0), float(by1 - by0)
        if bw < 1 or bh < 1:
            continue
        longest = max(bw, bh)
        if longest < min_symbol_px:
            continue
        if max_symbol_px is not None and longest > max_symbol_px:
            continue
        mx = max(0, int(round(bx0)) - pad)
        my = max(0, int(round(by0)) - pad)
        mw = min(W - mx, int(round(bw)) + 2 * pad)
        mh = min(H - my, int(round(bh)) + 2 * pad)
        if mw > 0 and mh > 0:
            out.append(Match(mx, my, mw, mh, float(sc)))

    if len(out) > max_matches:
        out = sorted(out, key=lambda m: m.score, reverse=True)[:max_matches]
    return out
