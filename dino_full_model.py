"""Unfrozen DINOv2-Reg backbone + 256-d proj (+ optional ArcFace)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from symbol_embed.model import ArcFaceHead

DINO_DIM = 384
PROJ_DIM = 256


def load_dinov2_reg() -> nn.Module:
    return torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14_reg",
        trust_repo=True,
    )


class DinoFullModel(nn.Module):
    """Full-network fine-tune from DINO-Reg weights."""

    def __init__(
        self,
        *,
        n_classes: int = 0,
        with_arcface: bool = False,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
        proj_dim: int = PROJ_DIM,
    ) -> None:
        super().__init__()
        self.backbone = load_dinov2_reg()
        self.proj = nn.Sequential(
            nn.Linear(DINO_DIM, DINO_DIM),
            nn.GELU(),
            nn.Linear(DINO_DIM, proj_dim),
        )
        self.arcface: ArcFaceHead | None = None
        if with_arcface:
            if n_classes < 2:
                raise ValueError("ArcFace needs n_classes >= 2")
            self.arcface = ArcFaceHead(proj_dim, n_classes, s=arcface_s, m=arcface_m)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:
            out = out[:, 0]
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self._encode(x)), dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        emb = self.embed(x)
        if self.arcface is not None:
            if labels is None:
                raise ValueError("labels required for ArcFace")
            return self.arcface(emb, labels), emb
        return emb

    def param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        head = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": list(self.backbone.parameters()), "lr": lr_backbone},
            {"params": head, "lr": lr_head},
        ]
