"""Length-penalty loss: encourage greedy-decoded length to match the plate.

Brazilian (old and Mercosur) plates are exactly **7 characters**. CTC can
slip into over- or under-segmentation; this regulariser penalises greedy
decodes whose length deviates from the target.

Implementation is non-differentiable through the argmax — we treat the
expected non-blank count as a soft per-timestep probability (``1 - p_blank``)
and use Huber loss against the target length. This gives smooth gradients.
"""
from __future__ import annotations

import torch


def length_penalty_loss(
    log_probs: torch.Tensor,       # [B, T, C]
    target_length: int = 7,
    blank: int = 0,
    delta: float = 1.0,
) -> torch.Tensor:
    """Smooth penalty on predicted character count.

    Approximates the expected number of non-blank emissions as the sum of
    ``1 - softmax(blank)`` over time, then applies a Huber penalty against
    ``target_length``.
    """
    probs = log_probs.exp()                              # [B, T, C]
    p_non_blank = 1.0 - probs[..., blank]                # [B, T]
    expected_len = p_non_blank.sum(dim=1)                # [B]
    diff = expected_len - float(target_length)

    abs_diff = diff.abs()
    quad = 0.5 * diff.pow(2)
    lin = delta * (abs_diff - 0.5 * delta)
    huber = torch.where(abs_diff <= delta, quad, lin)
    return huber.mean()
