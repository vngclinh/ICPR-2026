"""Transformer-style decoders for the LPR pipeline.

V2/V3 use ``CTCDecoder``: a thin self-attention stack that refines the fused
sequence — output is fed to a CTC head exactly like the encoder.

V4 uses ``CrossAttnCTCDecoder``: the decoder cross-attends into the full
5-frame token memory (``[B, F*T, D]``) so it can pick the strongest frame at
each position without a hard fusion step. Output is still aligned to a length
matching ``num_queries`` and fed to a CTC head — that means the decoder is
non-autoregressive, which simplifies training and is sufficient for plates
since the layout is fixed (7 characters).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.lpr_encoder import SinusoidalPE


class CTCDecoder(nn.Module):
    """Refining decoder for V2/V3: a few self-attention layers + LN.

    Input/output shape: ``[B, T, D]``. The trainer adds a CTC head on top.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pe = SinusoidalPE(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        x = self.layers(x)
        return self.norm(x)


class CrossAttnCTCDecoder(nn.Module):
    """V4 decoder: learnable queries cross-attend to multi-frame memory.

    Args:
        d_model: token dim.
        nhead: attention heads.
        num_layers: decoder layers.
        ff_dim: feed-forward hidden dim.
        dropout: dropout rate.
        num_queries: output sequence length T'. We choose T' equal to the
            CTC-compatible sequence length (same as the per-frame token count
            from the backbone) so the head sees a length that comfortably
            covers the 7-character plate.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 16,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Args: memory [B, F*T, D]. Returns [B, num_queries, D]."""
        b = memory.size(0)
        q = self.queries.expand(b, -1, -1)
        out = self.layers(tgt=q, memory=memory)
        return self.norm(out)
