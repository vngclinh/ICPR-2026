"""STN regularization to keep the predicted transform near identity / well-formed.

For affine theta we penalise deviation from the identity 2x3 matrix.
For TPS control points we penalise deviation from the canonical (identity)
control-point layout.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def stn_regularization_loss(
    params: torch.Tensor,
    mode: str = "tps",
    identity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regularize STN parameters towards identity.

    Args:
        params: affine theta [B, 2, 3] or TPS control points [B, K, 2].
        mode: ``"affine"`` or ``"tps"``.
        identity: optional pre-computed identity tensor. For affine mode the
            canonical 2x3 identity is used. For TPS the caller should pass the
            canonical control-point grid ``[K, 2]`` (or ``None`` to skip).

    Returns:
        scalar L2 deviation.
    """
    if mode == "affine":
        if identity is None:
            identity = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=params.dtype,
                device=params.device,
            )
        return F.mse_loss(params, identity.expand_as(params))

    if mode == "tps":
        if identity is None:
            return torch.zeros((), dtype=params.dtype, device=params.device)
        identity = identity.to(params.device, dtype=params.dtype)
        return F.mse_loss(params, identity.unsqueeze(0).expand_as(params))

    raise ValueError(f"Unknown STN mode: {mode!r}")
