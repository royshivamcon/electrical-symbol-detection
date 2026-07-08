"""Load the trained mask-confidence head and score masks.

Shared by ``eval.py`` and the (opt-in) integration in ``integrate.md``. ``score_many``
scores every candidate mask for one point-centered crop in a single batched forward.
"""

from __future__ import annotations

import sys
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
APP_DIR = FT_DIR.parent
for _p in (str(APP_DIR), str(FT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import CKPT_PATH  # noqa: E402
from features import build_feats, build_input  # noqa: E402
from model import MaskConfidenceNet  # noqa: E402


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MaskConfidenceScorer:
    def __init__(self, model: MaskConfidenceNet, device: str, input_size: int,
                 temperature: float = 1.0):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.temperature = float(temperature) if temperature else 1.0

    @classmethod
    def load(cls, ckpt: str | Path = CKPT_PATH, device: str | None = None) -> "MaskConfidenceScorer":
        ckpt = Path(ckpt)
        # weights_only=False: this is our own trusted checkpoint and it bundles
        # non-tensor metadata (cfg dict, temperature, best_val_auc) with the state_dict.
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        c = blob.get("cfg", {})
        model = MaskConfidenceNet(in_ch=c.get("in_ch", 3), width=c.get("width", 32),
                                  n_feats=c.get("n_feats", 4))
        model.load_state_dict(blob["state_dict"])
        dev = device or _device()
        model.to(dev).eval()
        return cls(model, dev, int(c.get("input_size", 128)),
                   temperature=blob.get("temperature", 1.0))

    @torch.no_grad()
    def score_many(self, gray_u8: np.ndarray, masks: list[np.ndarray],
                   cx: float, cy: float, confs: list[float] | None = None) -> np.ndarray:
        """Calibrated confidence in [0,1] for each mask (all sharing one grayscale crop).

        ``confs`` are the FastSAM objectness scores for each mask (one scalar feature the
        head fuses); if omitted they default to 0.
        """
        if not masks:
            return np.zeros((0,), dtype=np.float32)
        if confs is None:
            confs = [0.0] * len(masks)
        batch = np.stack([build_input(gray_u8, m, cx, cy, self.input_size) for m in masks], 0)
        feats = np.stack([build_feats(m, cf) for m, cf in zip(masks, confs)], 0)
        t = torch.from_numpy(batch).to(self.device)
        ft = torch.from_numpy(feats).to(self.device)
        logits = self.model(t, ft) / self.temperature
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    def score(self, crop_bgr: np.ndarray, mask_bool: np.ndarray, cx: float, cy: float,
              conf: float = 0.0) -> float:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        return float(self.score_many(gray, [mask_bool], cx, cy, [conf])[0])
