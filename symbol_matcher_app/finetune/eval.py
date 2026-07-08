"""Evaluate the trained mask-confidence head on the held-out split.

Headline question: does the learned head separate *correct symbol mask* from *wrong
mask* better than FastSAM's raw objectness (``res.boxes.conf``)? We report both AUCs on
the same val candidates, plus precision/recall/F1 and a calibration table for the head.

    ../.envs/vsam/bin/python finetune/eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
APP_DIR = FT_DIR.parent
for _p in (str(APP_DIR), str(FT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import CKPT_PATH, MANIFEST, PrepCfg, TrainCfg  # noqa: E402
from dataset import MaskConfDataset  # noqa: E402
from infer import MaskConfidenceScorer  # noqa: E402
from metrics import best_f1, calibration, prf_at, roc_auc  # noqa: E402


@torch.no_grad()
def _head_scores(scorer: MaskConfidenceScorer, ds: MaskConfDataset, batch: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    out = []
    for x, f, _ in loader:
        logits = scorer.model(x.to(scorer.device), f.to(scorer.device)) / scorer.temperature
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(MANIFEST))
    ap.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    ap.add_argument("--split", type=str, default=None,
                    help="train/val/all; defaults to 'val', or 'all' when --rid/--wid is set")
    ap.add_argument("--rid", type=str, default=None, help="restrict to one request id (prefix ok)")
    ap.add_argument("--wid", type=str, default=None, help="restrict to one sheet id (prefix ok)")
    _pc = PrepCfg()
    ap.add_argument("--pos-iou", type=float, default=_pc.pos_iou,
                    help="positive when mask IoU with the app GT box >= this")
    args = ap.parse_args()

    # When a specific sheet is requested, evaluate all of its candidates (both splits)
    # unless the user overrode --split explicitly.
    split = args.split or ("all" if (args.rid or args.wid) else "val")

    cfg = TrainCfg()
    ds = MaskConfDataset(args.manifest, split, cfg.input_size, rid=args.rid, wid=args.wid)
    if len(ds) == 0:
        print(f"[eval] no candidates for split='{split}' rid={args.rid} wid={args.wid} in {args.manifest}")
        return

    scorer = MaskConfidenceScorer.load(args.ckpt)
    # Positive labels are recomputed from the stored IoU-vs-app-box, so eval can
    # re-threshold independently of the baked label.
    baked = ds.labels()
    y = ds.labels_from_iou(args.pos_iou)
    conf = ds.confs()                      # FastSAM raw objectness (baseline)
    head = _head_scores(scorer, ds, cfg.batch_size)

    flips = int((y != baked).sum())
    scope = f"split='{split}'" + (f" wid={args.wid}" if args.wid else "") + (f" rid={args.rid}" if args.rid else "")
    print(f"[eval] rule: positive when IoU(mask, app_box) >= {args.pos_iou}")
    print(f"[eval] {scope} n={len(ds)} pos={int(y.sum())} neg={int((y==0).sum())} "
          f"(relabeled {flips}/{len(ds)} vs baked) device={scorer.device}")
    print(f"[eval] AUC  head={roc_auc(y, head):.4f}   baseline(conf)={roc_auc(y, conf):.4f}")
    hp = prf_at(y, head, 0.5)
    print(f"[eval] head@0.5  P={hp['precision']} R={hp['recall']} F1={hp['f1']} "
          f"(tp={hp['tp']} fp={hp['fp']} fn={hp['fn']})")
    bf = best_f1(y, head)
    print(f"[eval] head@best thr={bf['thr']}  P={bf['precision']} R={bf['recall']} F1={bf['f1']} "
          f"(tp={bf['tp']} fp={bf['fp']} fn={bf['fn']})")
    print("[eval] head calibration (mean_score vs frac_pos):")
    for b in calibration(y, head, bins=10):
        print(f"        {b['bin']} n={b['n']:5d} mean={b['mean_score']:.3f} frac_pos={b['frac_pos']:.3f}")


if __name__ == "__main__":
    main()
