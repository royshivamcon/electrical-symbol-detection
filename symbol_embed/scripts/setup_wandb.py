#!/usr/bin/env python3
"""Log in to W&B and bootstrap project symbol-dino-embed."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbol_embed.config import WANDB_PROJECT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", default=os.environ.get("WANDB_API_KEY"))
    ap.add_argument("--project", default=WANDB_PROJECT)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    args = ap.parse_args()

    import wandb

    if args.key:
        os.environ["WANDB_API_KEY"] = args.key
        wandb.login(key=args.key, relogin=True)
    else:
        ok = wandb.login()
        if not ok:
            raise SystemExit("wandb login failed; set WANDB_API_KEY or run wandb login")

    os.environ["WANDB_PROJECT"] = args.project
    os.environ["WANDB_MODE"] = "online"
    os.environ.pop("WANDB_DISABLED", None)

    init_kwargs = dict(
        project=args.project,
        name="_project_bootstrap",
        job_type="setup",
        reinit=True,
        settings=wandb.Settings(init_timeout=180),
    )
    if args.entity:
        init_kwargs["entity"] = args.entity

    run = wandb.init(**init_kwargs)
    print(f"W&B ready: https://wandb.ai/{run.entity}/{run.project}")
    run.finish()


if __name__ == "__main__":
    main()
