"""Train / eval loops for pretrained, contrastive, and ArcFace arms."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from symbol_embed.config import ARMS, Paths, TrainCfg, WANDB_PROJECT
from symbol_embed.dataset import (
    PositivePairDataset,
    SymbolCropDataset,
    eval_transform,
    make_splits,
    train_transform,
)
from symbol_embed.losses import ArcFaceLoss, PositivePairLoss
from symbol_embed.model import EmbedModel


def _device(cfg: TrainCfg) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _knn_recall(emb: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out 1-NN accuracy on L2 embeddings."""
    if len(labels) < 2:
        return 0.0
    x = emb.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    pred = labels[sim.argmax(axis=1)]
    return float((pred == labels).mean())


@torch.no_grad()
def extract_embeddings(model: EmbedModel, loader: DataLoader, device: torch.device):
    model.eval()
    zs, ys, paths = [], [], []
    for batch in loader:
        x, y, p = batch
        z = model.embed(x.to(device))
        zs.append(z.cpu().numpy())
        ys.append(y.numpy())
        paths.extend(p)
    return np.concatenate(zs), np.concatenate(ys), paths


def setup_wandb(arm: str, run_name: str, cfg: TrainCfg, crops_dir: Path):
    mode = (cfg.wandb_mode or "online").lower()
    if mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        print("wandb: disabled")
        return None

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_MODE"] = mode
    os.environ["WANDB_INIT_TIMEOUT"] = str(cfg.wandb_init_timeout)

    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "arm": arm,
            "crops_dir": str(crops_dir),
            **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        },
        reinit=True,
        settings=wandb.Settings(init_timeout=cfg.wandb_init_timeout),
    )
    return run


def train_arm(
    arm: str,
    crops_dir: Path,
    cfg: TrainCfg | None = None,
    *,
    paths: Paths | None = None,
) -> Path:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    cfg = cfg or TrainCfg()
    paths = paths or Paths()
    crops_dir = Path(crops_dir)
    device = _device(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    splits = make_splits(
        crops_dir,
        min_per_class=cfg.min_per_class,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )
    n_classes = int(splits["n_classes"])
    print(
        f"[train] arm={arm} classes={n_classes} "
        f"train={splits['n_train']} val={splits['n_val']} device={device}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{arm}_{stamp}"
    out_dir = paths.arm_run_dir(arm, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.wandb_fresh:
        import shutil

        wb = out_dir / "wandb"
        if wb.exists():
            shutil.rmtree(wb)

    model = EmbedModel(
        arm,
        n_classes,
        proj_dim=cfg.proj_dim,
        arcface_s=cfg.arcface_s,
        arcface_m=cfg.arcface_m,
    ).to(device)

    tf_train = train_transform(cfg.img_size)
    tf_eval = eval_transform(cfg.img_size)
    val_ds = SymbolCropDataset(crops_dir, "val", transform=tf_eval, splits=splits)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers
    )

    wb = setup_wandb(arm, run_name, cfg, crops_dir)

    # Pretrained: just dump checkpoint + val metric
    if arm == "pretrained":
        emb, lab, _ = extract_embeddings(model, val_loader, device)
        knn = _knn_recall(emb, lab)
        ckpt = {
            "arm": arm,
            "n_classes": n_classes,
            "name_to_label": splits["name_to_label"],
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "val_knn": knn,
        }
        torch.save(ckpt, out_dir / "best.pt")
        (out_dir / "metrics.json").write_text(json.dumps({"val_knn": knn}, indent=2))
        print(f"[train] pretrained val_knn={knn:.4f} → {out_dir}")
        if wb is not None:
            import wandb

            wandb.log({"val/knn": knn})
            wandb.finish()
        return out_dir

    if arm == "contrastive":
        train_ds = PositivePairDataset(
            crops_dir, transform=tf_train, splits=splits, seed=cfg.seed
        )
        criterion = PositivePairLoss()
    else:
        train_ds = SymbolCropDataset(
            crops_dir, "train", transform=tf_train, splits=splits
        )
        criterion = ArcFaceLoss()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = []
    if model.proj is not None:
        head_params += list(model.proj.parameters())
    if model.arcface is not None:
        head_params += list(model.arcface.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_knn = -1.0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            if arm == "contrastive":
                xa, xp, _y = batch
                za = model(xa.to(device))
                zp = model(xp.to(device))
                loss = criterion(za, zp)
            else:
                x, y, _p = batch
                y = y.to(device)
                logits, _emb = model(x.to(device), y)
                loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        emb, lab, _ = extract_embeddings(model, val_loader, device)
        knn = _knn_recall(emb, lab)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        row = {"epoch": epoch, "loss": mean_loss, "val_knn": knn, "sec": time.time() - t0}
        history.append(row)
        print(
            f"[train] {arm} epoch {epoch}/{cfg.epochs} "
            f"loss={mean_loss:.4f} val_knn={knn:.4f}"
        )
        if wb is not None:
            import wandb

            wandb.log(
                {
                    "train/loss": mean_loss,
                    "val/knn": knn,
                    "lr/backbone": opt.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )

        if knn >= best_knn:
            best_knn = knn
            torch.save(
                {
                    "arm": arm,
                    "n_classes": n_classes,
                    "name_to_label": splits["name_to_label"],
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_knn": knn,
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(
        json.dumps({"best_val_knn": best_knn}, indent=2)
    )
    # also keep last
    torch.save(
        {
            "arm": arm,
            "n_classes": n_classes,
            "name_to_label": splits["name_to_label"],
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "val_knn": history[-1]["val_knn"] if history else None,
            "epoch": cfg.epochs,
        },
        out_dir / "last.pt",
    )
    if wb is not None:
        import wandb

        wandb.finish()
    print(f"[train] done {arm} best_knn={best_knn:.4f} → {out_dir}")
    return out_dir


def load_model_from_ckpt(ckpt_path: Path, device: torch.device | None = None) -> EmbedModel:
    device = device or torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arm = ckpt["arm"]
    n_classes = int(ckpt["n_classes"])
    cfg = ckpt.get("cfg") or {}
    model = EmbedModel(
        arm,
        n_classes,
        proj_dim=int(cfg.get("proj_dim", 128)),
        arcface_s=float(cfg.get("arcface_s", 30.0)),
        arcface_m=float(cfg.get("arcface_m", 0.5)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
