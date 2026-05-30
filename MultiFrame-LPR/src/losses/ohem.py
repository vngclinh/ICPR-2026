"""Online Hard Example Mining for CTC.

Computes the per-sample CTC loss without reduction, sorts by magnitude, and
back-props only on the hardest ``top_k`` fraction. Following Shrivastava et
al. 2016, ``top_k=0.7`` is a reasonable starting point; tune per variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def ohem_ctc_loss(
    log_probs: torch.Tensor,         # [B, T, C]
    targets: torch.Tensor,           # concatenated 1-D
    target_lengths: torch.Tensor,    # [B]
    top_k: float = 0.7,
    blank: int = 0,
) -> torch.Tensor:
    """Return the mean CTC loss over the hardest ``top_k`` fraction of samples."""
    b, t, _ = log_probs.shape
    input_lengths = torch.full((b,), t, dtype=torch.long, device=log_probs.device)
    per_sample = nn.functional.ctc_loss(
        log_probs.permute(1, 0, 2),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=True,
    )
    k = max(1, int(round(b * top_k)))
    hard, _ = torch.topk(per_sample, k=k, largest=True, sorted=False)
    return hard.mean()
