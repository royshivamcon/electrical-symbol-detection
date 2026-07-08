"""Small CNN that re-scores a FastSAM mask's confidence.

Deliberately tiny (a few conv blocks -> pool -> FC) so it trains fast on MPS/CPU and
adds negligible latency when plugged into ``seg_models._select_from_covering``.

Two ideas make the score about *this specific mask* rather than the whole crop:

* **masked pooling** — features are pooled *inside the mask* (weighted by the mask
  channel), concatenated with a plain global average for context. Two overlapping
  candidates that share a crop now get distinct embeddings.
* **scalar-feature fusion** — the pooled embedding is concatenated with ``build_feats``
  (FastSAM objectness + mask geometry) before the FC, so the head combines its visual
  judgement with the signal it would otherwise ignore.

Outputs a single logit; apply sigmoid (optionally with a calibration temperature) for a
confidence in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MaskConfidenceNet(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 32, n_feats: int = 4):
        super().__init__()
        self.n_feats = n_feats
        self.features = nn.Sequential(
            _block(in_ch, width),        # 128 -> 64
            _block(width, width * 2),    # 64 -> 32
            _block(width * 2, width * 4),  # 32 -> 16
            _block(width * 4, width * 4),  # 16 -> 8
        )
        c = width * 4
        self.head = nn.Sequential(
            nn.Linear(2 * c + n_feats, width * 2),  # masked-pool + global-pool + scalars
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(width * 2, 1),
        )

    def forward(self, x: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        f = self.features(x)                                  # [B, C, 8, 8]
        # mask channel (index 1) downsampled to the feature-map grid -> per-cell mask frac
        m = F.adaptive_avg_pool2d(x[:, 1:2], f.shape[-2:])    # [B, 1, 8, 8]
        masked = (f * m).sum(dim=(2, 3)) / (m.sum(dim=(2, 3)) + 1e-6)  # [B, C]
        glob = f.mean(dim=(2, 3))                             # [B, C]
        if feats is None:
            feats = x.new_zeros((x.shape[0], self.n_feats))
        z = torch.cat([masked, glob, feats], dim=1)
        return self.head(z).squeeze(-1)                       # [B] logits
