"""Config for the FastSAM mask-confidence head fine-tuning.

Detection knobs mirror the live pipeline (``main.py`` API defaults + ``seg_models.Cfg``)
so the dataset the head trains on matches what it will score at inference. Base-space
size gates are scaled by ``zoom`` at render time (see ``rendered_gates``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
APP_DIR = FT_DIR.parent  # Electrical/symbol_matcher_app
DATASET_DIR = FT_DIR / "dataset"
CKPT_DIR = FT_DIR / "checkpoints"
MANIFEST = DATASET_DIR / "manifest.jsonl"
CROPS_DIR = DATASET_DIR / "crops"
CKPT_PATH = CKPT_DIR / "mask_conf.pt"


@dataclass
class PrepCfg:
    """Dataset-generation knobs for the whole-sheet, app-box-targeted detector.

    Ground-truth symbol boxes come from the app itself (``boxes_from_points``); FastSAM
    segment-everything masks across the whole sheet are the candidates, labeled by IoU
    against those GT boxes. Base-space sizes are scaled by ``zoom`` (see ``rendered_gates``).
    """

    zoom: float = 4.0            # PDF render zoom (main.py api default)
    crop: int = 90               # base-space half-window (GT crop + centroid scoring window)
    min_symbol_px: int = 16      # base-space floor for the GT box call (api default)
    max_symbol_px: int = 50      # base-space ceiling for the GT box call (api default)
    pad: int = 3                 # GT box fit margin, rendered space (api default)
    imgsz: int = 1024            # FastSAM input size (seg_models.Cfg)
    conf: float = 0.25           # FastSAM confidence (seg_models.Cfg)
    iou: float = 0.9             # FastSAM NMS IoU (seg_models.Cfg)
    remove_text: bool = True     # render text-free PDF (api default)

    # Whole-sheet segment-everything candidates
    tile: int = 1024             # tile size for the segment-everything scan
    overlap: int = 128           # tile overlap
    cand_min_px: int = 12        # drop candidate masks whose min(w,h) < this (rendered px)
    cand_max_frac: float = 0.1   # drop candidate masks whose bbox area > this frac of the tile

    # IoU labeling vs the app GT boxes
    pos_iou: float = 0.5         # positive when max IoU with a GT box >= this
    neg_iou: float = 0.1         # negative when max IoU < this (ignore the band between)
    neg_per_pos: int = 4         # negatives kept per positive (half near-miss, half random)

    gt_model: str = "fastsam"    # app model used to produce GT boxes
    electrical_only: bool = True  # keep only electrical-named reference points for GT

    input_size: int = 128        # saved-crop-independent; head input HxW
    val_frac: float = 0.2        # sheet-level held-out fraction
    seed: int = 0

    def rendered_gates(self) -> tuple[int, int, int]:
        """(zcrop, zmax, zmin) in rendered pixels for the configured zoom.

        Matches ``run_sam``'s scaling: ``crop``/``max_symbol_px``/``min_symbol_px``
        are multiplied by ``zoom`` so they mean the same physical size at zoom.
        """
        z = float(self.zoom) if self.zoom and self.zoom > 1.0 else 1.0
        zcrop = max(1, int(round(self.crop * z)))
        zmax = max(1, int(round(self.max_symbol_px * z)))
        zmin = max(1, int(round(self.min_symbol_px * z)))
        return zcrop, zmax, zmin


@dataclass
class TrainCfg:
    input_size: int = 128
    in_ch: int = 3
    width: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    val_frac: float = 0.2   # only used when splits fall back to point-level
    seed: int = 0
    patience: int = 5       # early-stop on val AUC
    num_workers: int = 0    # 0 avoids MPS/fork issues
