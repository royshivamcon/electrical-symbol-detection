"""Paths and training defaults for quill_local classifier / DINOv2 embedding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
MATCHER_APP = PROJECT_ROOT / "symbol_matcher_app"
REQUESTS_DIR = PROJECT_ROOT / "data" / "requests"
QUILL_LOCAL_DIR = PROJECT_ROOT / "data" / "quill_local"
WANDB_PROJECT = "symbol-classifier-quill"

ARMS = ("pretrained", "contrastive", "arcface")
ADAPTER_ARMS = ("supcon", "arcface")
DEFAULT_N_RIDS = 10
DEFAULT_ZOOM = 4.0
POOL_NAME_QUILL = "pooled_10rids_quill"

# Locked 8/2 RID split (also in train_rids.txt / val_rids.txt).
VAL_RIDS: tuple[str, ...] = (
    "061c9552-1984-4ed1-b325-20ca0940cc54",
    "fd904995-8db7-402b-84a4-3ce2898a374e",
)


@dataclass(frozen=True)
class Paths:
    root: Path = PACKAGE_ROOT
    requests: Path = REQUESTS_DIR
    quill_local: Path = QUILL_LOCAL_DIR

    def adapter_rids_file(self) -> Path:
        return self.root / "adapter_rids.txt"

    def train_rids_file(self) -> Path:
        return self.root / "train_rids.txt"

    def val_rids_file(self) -> Path:
        return self.root / "val_rids.txt"

    def crops_dir(self, rid: str) -> Path:
        """Legacy FastSAM detection crops."""
        return self.requests / rid / "symbol_crops"

    def crops_quill_dir(self, rid: str) -> Path:
        """Quill_local bbox crops at zoom=4x."""
        return self.requests / rid / "symbol_crops_quill"

    def pooled_dir(self, n_rids: int = DEFAULT_N_RIDS) -> Path:
        """Combined train set under symbol_embed/data/pooled_<n>rids/."""
        return self.root / "data" / f"pooled_{n_rids}rids"

    def pooled_quill_dir(self) -> Path:
        return self.root / "data" / POOL_NAME_QUILL

    def verify_report_dir(self) -> Path:
        return self.pooled_quill_dir()

    def runs_dir(self) -> Path:
        return self.root / "runs"

    def arm_run_dir(self, arm: str, run_name: str) -> Path:
        return self.runs_dir() / arm / run_name

    def viz_dir(self, rid: str) -> Path:
        return self.root / "viz" / rid[:12]

    def eval_dir(self, rid: str) -> Path:
        return self.root / "eval" / rid[:12]


@dataclass
class TrainCfg:
    epochs: int = 30
    batch: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    seed: int = 0
    num_workers: int = 4
    img_size: int = 224
    min_per_class: int = 5
    val_frac: float = 0.2
    embed_dim: int = 384
    proj_dim: int = 128
    arcface_s: float = 30.0
    arcface_m: float = 0.5
    wandb_mode: str = "online"
    wandb_init_timeout: int = 180
    wandb_fresh: bool = False
    device: str | None = None
