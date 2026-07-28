"""Frozen DINOv2-Reg + trainable Transformer adapter → 256-d unit sphere.

Backbone: ``dinov2_vits14_reg`` (261 tokens = CLS + 4 registers + 256 patches).

Modes:
- ``layers="last"`` (default legacy): final-block prenorm sequence (261 tokens).
- ``layers=[0, 11]``: early + late **patch** tokens, concat-fuse → 384-d sequence
  (256 patches), then one TransformerEncoderLayer → mean-pool → 256-d.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from symbol_embed.model import ArcFaceHead

DINO_DIM = 384
N_TOKENS = 261  # CLS + 4 reg + 256 patches
N_PATCHES = 256
N_BLOCKS = 12
PROJ_DIM = 256


def load_dinov2_reg() -> nn.Module:
    return torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14_reg",
        trust_repo=True,
    )


def parse_adapter_layers(layers: str | list[int] | None) -> list[int] | None:
    """Return sorted unique block indices, or ``None`` for legacy last-prenorm mode.

    ``None`` / ``\"last\"`` / ``\"\"`` → legacy prenorm path.
    ``\"0,11\"`` / ``[0, 11]`` → intermediate patch layers.
    """
    if layers is None or layers == "" or layers == []:
        return None
    if isinstance(layers, str):
        s = layers.strip().lower()
        if s in ("last", "prenorm"):
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        idxs = [int(p) for p in parts]
    else:
        idxs = [int(x) for x in layers]
    out = sorted({i for i in idxs if 0 <= i < N_BLOCKS})
    if not out:
        raise ValueError(f"no valid DINOv2 layers in {layers!r} (need 0..{N_BLOCKS - 1})")
    return out


class DinoAdapter(nn.Module):
    """Frozen DINO-Reg trunk + one trainable Transformer layer + 256-d proj."""

    def __init__(
        self,
        *,
        n_classes: int = 0,
        with_arcface: bool = False,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
        nhead: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        layers: str | list[int] | None = "0,11",
    ) -> None:
        super().__init__()
        self.layers = parse_adapter_layers(layers)
        self.backbone = load_dinov2_reg()
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.fuse: nn.Module | None = None
        if self.layers is not None and len(self.layers) > 1:
            self.fuse = nn.Sequential(
                nn.Linear(DINO_DIM * len(self.layers), DINO_DIM),
                nn.GELU(),
            )

        self.transformer = nn.TransformerEncoderLayer(
            d_model=DINO_DIM,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.proj = nn.Linear(DINO_DIM, PROJ_DIM)
        self.arcface: ArcFaceHead | None = None
        if with_arcface:
            if n_classes < 2:
                raise ValueError("ArcFace needs n_classes >= 2")
            self.arcface = ArcFaceHead(PROJ_DIM, n_classes, s=arcface_s, m=arcface_m)

    @torch.no_grad()
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return frozen token sequence ``(B, T, 384)`` for the adapter."""
        if self.layers is None:
            feats = self.backbone.forward_features(x)
            tokens = feats["x_prenorm"]
            if tokens.shape[1] != N_TOKENS:
                raise RuntimeError(
                    f"expected {N_TOKENS} tokens, got {tokens.shape[1]} "
                    f"(use dinov2_vits14_reg)"
                )
            return tokens

        outs = self.backbone.get_intermediate_layers(
            x,
            n=self.layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        # Each out: (patches (B, N, D), cls (B, D)); N should be 256 for reg models
        # (registers stripped by hub helper).
        patch_list = []
        for patches, _cls in outs:
            if patches.shape[1] != N_PATCHES:
                raise RuntimeError(
                    f"expected {N_PATCHES} patch tokens, got {patches.shape[1]}"
                )
            patch_list.append(patches)
        if len(patch_list) == 1:
            return patch_list[0]
        return torch.cat(patch_list, dim=-1)  # (B, 256, D * L)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized ``(B, 256)`` embedding (trainable head only)."""
        tokens = self.extract_tokens(x)
        # Gradients flow through fuse / transformer / proj only.
        tokens = tokens.detach()
        if self.fuse is not None:
            tokens = self.fuse(tokens)
        h = self.transformer(tokens)
        pooled = h.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        if self.arcface is None:
            return emb
        if labels is None:
            raise ValueError("labels required when ArcFace head is enabled")
        return self.arcface(emb, labels), emb

    def trainable_parameters(self):
        """Parameters that should receive gradients (backbone excluded)."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p
