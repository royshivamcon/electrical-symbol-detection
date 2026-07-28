"""Positive-pair cosine attraction + ArcFace CE wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositivePairLoss(nn.Module):
    """Attract same-class (or twin-view) embeddings: L = 1 - cos(a, p).

    No negative-pair / InfoNCE term.
    """

    def forward(self, z_a: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        z_a = F.normalize(z_a, dim=-1)
        z_p = F.normalize(z_p, dim=-1)
        return (1.0 - (z_a * z_p).sum(dim=-1)).mean()


class ArcFaceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)
