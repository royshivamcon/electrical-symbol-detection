"""Dataset over the cached mask-confidence candidates.

Reads ``manifest.jsonl`` (one row per candidate) and rebuilds the ``[3,size,size]``
input from the point's cached ``.npz`` (grayscale crop + packed masks + point). A small
LRU cache keeps recently-touched ``.npz`` files in memory since many candidates share
one crop.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features import build_feats, build_input


@lru_cache(maxsize=256)
def _load_npz(path: str):
    d = np.load(path)
    shape = tuple(int(v) for v in d["mask_shape"])  # (K, ch, cw)
    k, ch, cw = shape
    masks = np.unpackbits(d["masks"])[: k * ch * cw].reshape(shape).astype(bool)
    cx, cy = (int(v) for v in d["cxcy"])
    return d["gray"], masks, cx, cy


class MaskConfDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, input_size: int,
                 rid: str | None = None, wid: str | None = None):
        manifest_path = Path(manifest_path)
        self.crops_dir = manifest_path.parent / "crops"
        self.size = input_size
        with open(manifest_path) as fh:
            rows = [json.loads(line) for line in fh]

        def _keep(r: dict) -> bool:
            if split not in (None, "all") and r.get("split") != split:
                return False
            if rid and not r["rid"].startswith(rid):
                return False
            if wid and not r["wid"].startswith(wid):
                return False
            return True

        self.rows = [r for r in rows if _keep(r)]

    def labels(self) -> np.ndarray:
        return np.array([r["label"] for r in self.rows], dtype=np.int64)

    def labels_from_iou(self, pos_iou: float) -> np.ndarray:
        """Recompute positive labels from the stored IoU-vs-app-box (``iou``), so eval
        can re-threshold without regenerating the dataset: positive iff ``iou >= pos_iou``."""
        return np.array([1 if r["iou"] >= pos_iou else 0 for r in self.rows], dtype=np.int64)

    def confs(self) -> np.ndarray:
        return np.array([r["conf"] for r in self.rows], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        gray, masks, cx, cy = _load_npz(str(self.crops_dir / r["npz"]))
        mask = masks[r["k"]]
        x = build_input(gray, mask, cx, cy, self.size)
        f = build_feats(mask, float(r["conf"]))
        return (torch.from_numpy(x),
                torch.from_numpy(f),
                torch.tensor(float(r["label"]), dtype=torch.float32))
