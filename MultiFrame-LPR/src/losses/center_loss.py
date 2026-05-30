"""Center loss for character embeddings (Wen et al. 2016).

Pulls the per-position feature vector toward the centre of its predicted class
so easily confused glyph pairs (``0/O``, ``1/I``, ``8/B``) separate better.
We use the **predicted** class (argmax of CTC logits) per timestep as the
class assignment — this is the standard CTC-friendly approximation (we do
not have explicit per-position labels).

Blank predictions are masked out from the loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Maintains learnable class centres and pulls features towards them.

    Args:
        num_classes: vocabulary size including blank.
        feat_dim: feature/token dim D.
        blank_index: index of the blank class to mask out (default 0).
    """

    def __init__(self, num_classes: int, feat_dim: int, blank_index: int = 0):
        super().__init__()
        self.blank_index = blank_index
        self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))
        nn.init.normal_(self.centers, mean=0.0, std=0.05)

    def forward(self, features: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """Args:
            features: [B, T, D] — the token features feeding the CTC head.
            log_probs: [B, T, C] — log-softmax over classes.
        Returns:
            scalar loss.
        """
        with torch.no_grad():
            preds = log_probs.argmax(dim=-1)              # [B, T]
            mask = (preds != self.blank_index)            # [B, T]
        if mask.sum() == 0:
            return torch.zeros((), device=features.device, dtype=features.dtype)

        # Gather centres for predicted classes
        flat_preds = preds.reshape(-1)                    # [B*T]
        flat_feats = features.reshape(-1, features.size(-1))  # [B*T, D]
        flat_mask = mask.reshape(-1)                      # [B*T]

        centres = self.centers[flat_preds]                # [B*T, D]
        diff = (flat_feats - centres).pow(2).sum(dim=1)   # [B*T]
        loss = diff[flat_mask].mean()
        return loss
