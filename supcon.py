"""Supervised Contrastive (SupCon) loss — Khosla et al."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """In-batch supervised contrastive loss on L2-normalized embeddings.

    Expects ``z`` of shape ``(B, D)`` or ``(2B, D)`` when two views are
    concatenated, and integer ``labels`` of length ``B`` (duplicated for views).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        device = z.device
        n = z.shape[0]
        if n < 2:
            return z.new_zeros(())

        labels = labels.view(-1)
        if labels.numel() != n:
            raise ValueError(f"labels length {labels.numel()} != embeddings {n}")

        sim = (z @ z.T) / self.temperature
        # Mask self-comparisons.
        logits_mask = torch.ones((n, n), dtype=torch.bool, device=device)
        logits_mask.fill_diagonal_(False)
        # Positives: same class, not self.
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = label_eq & logits_mask

        # For numerical stability.
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

        # Mean log-prob over positives per anchor (skip anchors with no positives).
        pos_counts = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_counts
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return z.new_zeros(())
        return -(mean_log_prob_pos[has_pos]).mean()
