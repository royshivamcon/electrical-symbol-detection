"""DINOv2-S encoder with optional projection / ArcFace head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_dinov2_small() -> tuple[nn.Module, int]:
    """Return (backbone, embed_dim). Prefer torch.hub; fall back to timm."""
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        return backbone, 384
    except Exception:
        import timm

        backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        return backbone, int(getattr(backbone, "num_features", 384))


class DinoEncoder(nn.Module):
    def __init__(self, *, freeze: bool = False) -> None:
        super().__init__()
        self.backbone, self.embed_dim = _load_dinov2_small()
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:
            out = out[:, 0]  # CLS if sequence returned
        return out


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ArcFaceHead(nn.Module):
    """Additive angular margin classifier on L2-normalized features."""

    def __init__(self, in_dim: int, n_classes: int, s: float = 30.0, m: float = 0.5) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Stable ArcFace: avoid acos (NaNs under AMP when cos≈±1).
        emb = F.normalize(emb, dim=-1)
        W = F.normalize(self.weight, dim=-1)
        cosine = F.linear(emb, W).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(1e-7))
        # cos(θ+m) = cosθ·cosm − sinθ·sinm
        target_logit = cosine * math.cos(self.m) - sine * math.sin(self.m)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).to(dtype=cosine.dtype)
        logits = cosine * (1.0 - one_hot) + target_logit * one_hot
        return logits * self.s


class EmbedModel(nn.Module):
    """Shared wrapper: backbone (+ optional proj / arcface)."""

    def __init__(
        self,
        arm: str,
        n_classes: int,
        *,
        proj_dim: int = 128,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
    ) -> None:
        super().__init__()
        if arm not in ("pretrained", "contrastive", "arcface"):
            raise ValueError(arm)
        self.arm = arm
        self.encoder = DinoEncoder(freeze=(arm == "pretrained"))
        self.proj = None
        self.arcface = None
        if arm == "contrastive":
            self.proj = ProjectionHead(self.encoder.embed_dim, proj_dim)
        elif arm == "arcface":
            self.arcface = ArcFaceHead(
                self.encoder.embed_dim, n_classes, s=arcface_s, m=arcface_m
            )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized embedding used for retrieval / t-SNE."""
        z = self.encoder(x)
        if self.proj is not None:
            return self.proj(z)
        return F.normalize(z, dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        z = self.encoder(x)
        if self.arm == "contrastive":
            assert self.proj is not None
            return self.proj(z)
        if self.arm == "arcface":
            assert self.arcface is not None and labels is not None
            emb = F.normalize(z, dim=-1)
            return self.arcface(emb, labels), emb
        return F.normalize(z, dim=-1)
