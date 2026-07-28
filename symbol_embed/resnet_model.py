"""Trainable ResNet50 (ImageNet) + 256-d proj (+ optional ArcFace)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from symbol_embed.model import ArcFaceHead

RESNET_DIM = 2048
PROJ_DIM = 256


class ResNetEmbedModel(nn.Module):
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
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_dim = RESNET_DIM
        self.proj = nn.Sequential(
            nn.Linear(RESNET_DIM, RESNET_DIM // 2),
            nn.GELU(),
            nn.Linear(RESNET_DIM // 2, proj_dim),
        )
        self.arcface: ArcFaceHead | None = None
        if with_arcface:
            if n_classes < 2:
                raise ValueError("ArcFace needs n_classes >= 2")
            self.arcface = ArcFaceHead(proj_dim, n_classes, s=arcface_s, m=arcface_m)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return f.flatten(1)

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
