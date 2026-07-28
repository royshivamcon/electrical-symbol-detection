#!/usr/bin/env python3
"""Train ResNet50 from ImageNet (SupCon / ArcFace) on quill RID-holdout pool.

  bash symbol_embed/run_verify_dataset.sh
  python -m symbol_embed.scripts.train_resnet --arm both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbol_embed.config import ADAPTER_ARMS, Paths
from symbol_embed.dataset import build_pooled_quill_dataset
from symbol_embed.resnet_model import ResNetEmbedModel
from symbol_embed.train_backbone import BackboneTrainCfg, train_backbone_arm
from symbol_embed.verify_gate import require_verify_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=(*ADAPTER_ARMS, "both"), default="both")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--embed-batch", type=int, default=512)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--proto-max-per-class", type=int, default=64)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    ap.add_argument("--wandb-init-timeout", type=int, default=180)
    args = ap.parse_args()

    paths = Paths()
    pool_dir = Path(args.pool) if args.pool else paths.pooled_quill_dir()
    require_verify_ok(pool_dir, skip=args.skip_verify, paths=paths)
    if args.rebuild_pool or not (pool_dir / "splits.json").exists():
        build_pooled_quill_dataset(
            out_dir=pool_dir,
            min_per_class=args.min_per_class,
            seed=args.seed,
            paths=paths,
        )

    cfg = BackboneTrainCfg(
        epochs=args.epochs,
        batch=args.batch,
        embed_batch=args.embed_batch,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        num_workers=args.num_workers,
        min_per_class=args.min_per_class,
        seed=args.seed,
        amp=args.amp,
        device=args.device,
        proto_max_per_class=args.proto_max_per_class,
        wandb_mode=args.wandb_mode,
        wandb_init_timeout=args.wandb_init_timeout,
    )
    arms = list(ADAPTER_ARMS) if args.arm == "both" else [args.arm]
    for arm in arms:
        train_backbone_arm(
            arm,
            pool_dir,
            cfg,
            model_factory=ResNetEmbedModel,
            run_family="resnet50",
            paths=paths,
        )


if __name__ == "__main__":
    main()
