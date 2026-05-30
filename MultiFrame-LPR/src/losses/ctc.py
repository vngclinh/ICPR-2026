"""Main and auxiliary CTC loss wrappers.

The model emits ``[B, T, C]`` log-softmax. CTC expects ``[T, B, C]``, so the
wrappers handle the permute internally to keep the trainer concise.

``MainCTC`` is the primary head loss. ``AuxCTC`` is the auxiliary loss taken
from an intermediate encoder layer (Section 5 of the design doc) — it shares
the structure but uses a separate small Linear head (provided by the caller)
so the two heads can disagree mildly and regularize each other.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MainCTC(nn.Module):
    """Primary CTC loss applied to model log-softmax outputs."""

    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    def forward(
        self,
        log_probs: torch.Tensor,        # [B, T, C]
        targets: torch.Tensor,          # concatenated 1-D
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = log_probs.shape
        input_lengths = torch.full((b,), t, dtype=torch.long, device=log_probs.device)
        return self.ctc(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)


class AuxCTC(nn.Module):
    """Auxiliary CTC loss applied at an intermediate encoder layer.

    The caller supplies the auxiliary linear head (a small ``nn.Linear`` from
    encoder dim to ``num_classes``). This module owns just the loss + log-
    softmax to avoid double-counting parameters in the variant module.
    """

    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    def forward(
        self,
        aux_features: torch.Tensor,     # [B, T, D]
        head: nn.Linear,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        logits = head(aux_features).log_softmax(dim=-1)  # [B, T, C]
        b, t, _ = logits.shape
        input_lengths = torch.full((b,), t, dtype=torch.long, device=logits.device)
        return self.ctc(logits.permute(1, 0, 2), targets, input_lengths, target_lengths)
