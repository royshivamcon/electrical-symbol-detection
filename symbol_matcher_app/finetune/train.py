"""Train the mask-confidence head on the cached dataset.

FastSAM is NOT touched here — we only train the small ``MaskConfidenceNet`` on the
pre-extracted (crop, mask, point) -> label examples. Class imbalance is handled with a
weighted sampler; the best checkpoint (by val AUC) is written to ``checkpoints/mask_conf.pt``.

    ../.envs/vsam/bin/python finetune/train.py --epochs 5
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
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from config import CKPT_PATH, MANIFEST, TrainCfg  # noqa: E402
from dataset import MaskConfDataset  # noqa: E402
from features import N_FEATS  # noqa: E402
from metrics import prf_at, roc_auc  # noqa: E402
from model import MaskConfidenceNet  # noqa: E402


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    per_class = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = per_class[labels]
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


@torch.no_grad()
def _collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, raw logits) over a loader — logits, not probs, so the same pass
    can drive both AUC/PRF and temperature fitting."""
    model.eval()
    ys, zs = [], []
    for x, f, y in loader:
        logits = model(x.to(device), f.to(device))
        zs.append(logits.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(zs)


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter temperature scaling: find T>0 minimizing val NLL of
    ``sigmoid(logit / T)``. Makes the output a calibrated probability without changing
    the ranking (so AUC is unchanged)."""
    z = torch.from_numpy(logits.astype(np.float32))
    t = torch.from_numpy(y.astype(np.float32))
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) keeps T > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(MANIFEST))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", type=str, default=str(CKPT_PATH))
    args = ap.parse_args()

    cfg = TrainCfg()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.lr is not None:
        cfg.lr = args.lr

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _device()

    train_ds = MaskConfDataset(args.manifest, "train", cfg.input_size)
    val_ds = MaskConfDataset(args.manifest, "val", cfg.input_size)
    if len(train_ds) == 0:
        print("[train] empty train split — run prep_dataset.py first")
        return
    print(f"[train] device={device} train={len(train_ds)} val={len(val_ds)} "
          f"epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr}")

    tr_labels = train_ds.labels()
    print(f"[train] train class balance: pos={int(tr_labels.sum())} neg={int((tr_labels==0).sum())}")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=_sampler(tr_labels), num_workers=cfg.num_workers)
    val_loader = (DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers) if len(val_ds) else None)

    model = MaskConfidenceNet(in_ch=cfg.in_ch, width=cfg.width, n_feats=N_FEATS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, since_best = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss = n = 0
        for x, f, y in train_loader:
            x, f, y = x.to(device), f.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, f), y)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * len(y); n += len(y)
        msg = f"[train] epoch {epoch:02d} loss={run_loss/max(1,n):.4f}"
        if val_loader is not None:
            yv, zv = _collect_logits(model, val_loader, device)
            pv = 1.0 / (1.0 + np.exp(-zv))
            auc = roc_auc(yv, pv)
            prf = prf_at(yv, pv, 0.5)
            msg += f" val_auc={auc:.4f} P={prf['precision']} R={prf['recall']} F1={prf['f1']}"
            score = auc if not np.isnan(auc) else -1.0
            if score > best_auc:
                best_auc = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                since_best = 0
            else:
                since_best += 1
        print(msg)
        if val_loader is not None and since_best >= cfg.patience:
            print(f"[train] early stop (no val AUC gain for {cfg.patience} epochs)")
            break

    state = best_state if best_state is not None else model.state_dict()

    # Temperature-scale on the val split (best model) so the saved score is calibrated.
    temperature = 1.0
    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        yv, zv = _collect_logits(model, val_loader, device)
        if 0 < int(yv.sum()) < len(yv):  # need both classes for a meaningful fit
            temperature = _fit_temperature(zv, yv)
        print(f"[train] fitted temperature={temperature:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state,
                "cfg": {"in_ch": cfg.in_ch, "width": cfg.width,
                        "input_size": cfg.input_size, "n_feats": N_FEATS},
                "temperature": float(temperature),
                "best_val_auc": float(best_auc)}, out)
    print(f"[train] saved {out} (best_val_auc={best_auc:.4f} temperature={temperature:.4f})")


if __name__ == "__main__":
    main()
