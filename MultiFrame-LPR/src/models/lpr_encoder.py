"""Configurable Transformer encoder used by all four LPR variants.

Wraps ``nn.TransformerEncoder`` with a sinusoidal positional encoding and
optional auxiliary-CTC tap: when ``aux_tap_layer`` is set, the encoder also
returns the hidden state at that intermediate layer so the trainer can attach
an auxiliary CTC head (see Section 5 of the design doc).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class LPREncoder(nn.Module):
    """Transformer encoder with optional intermediate tap for aux CTC.

    Args:
        d_model: token dim.
        nhead: attention heads.
        num_layers: encoder layers (varies per variant for V2 / V3 / V4).
        ff_dim: feed-forward hidden dim.
        dropout: dropout rate.
        aux_tap_layer: if not None, return the hidden state after this layer
            (1-indexed) in addition to the final output.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        aux_tap_layer: int | None = None,
    ):
        super().__init__()
        self.pe = SinusoidalPE(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        if aux_tap_layer is not None and not (1 <= aux_tap_layer <= num_layers):
            raise ValueError(
                f"aux_tap_layer must be in [1, {num_layers}], got {aux_tap_layer}"
            )
        self.aux_tap_layer = aux_tap_layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args: x [B, T, D]. Returns (out [B, T, D], aux [B, T, D] | None)."""
        x = self.pe(x)
        aux = None
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if self.aux_tap_layer == i:
                aux = x
        x = self.norm(x)
        return x, aux
