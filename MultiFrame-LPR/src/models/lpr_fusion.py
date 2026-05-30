"""Multi-frame quality-attention fusion for the ICPR 2026 LPR pipeline.

Two modes:
* ``QualityFusionMap``  — operates on feature MAPS ``[B*F, C, H, W]`` and
  returns ``[B, C, H, W]`` (used by V1 early-fusion variants).
* ``QualityFusionSeq``  — operates on token SEQUENCES ``[B*F, T, D]`` and
  returns ``[B, T, D]`` (used by V2 / V3 late-fusion variants).

For V4 we expose ``stack_frames_as_memory`` which returns ``[B, F*T, D]``
without fusing — the cross-attention decoder attends over the whole stack
and effectively learns its own per-token frame attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reshape_frames(x: torch.Tensor, num_frames: int) -> tuple[torch.Tensor, int]:
    """Reshape [B*F, ...] -> [B, F, ...] (returns the inferred batch size)."""
    bf = x.shape[0]
    if bf % num_frames != 0:
        raise ValueError(f"Tensor first dim {bf} is not divisible by num_frames={num_frames}")
    b = bf // num_frames
    return x.view(b, num_frames, *x.shape[1:]), b


class QualityFusionMap(nn.Module):
    """Per-frame quality attention on feature maps (CNN-domain fusion).

    Compatible drop-in for ``src.models.components.AttentionFusion``.

    Args:
        channels: feature channels.
        per_position: if True, predict a quality map ``[B, F, 1, H, W]`` (so
            each position picks its own frame); otherwise a scalar per frame.
        num_frames: fixed burst size (5).
    """

    def __init__(self, channels: int, per_position: bool = True, num_frames: int = 5):
        super().__init__()
        self.num_frames = num_frames
        self.per_position = per_position
        hidden = max(channels // 8, 8)

        if per_position:
            self.score_net = nn.Sequential(
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )
        else:
            self.score_net = nn.Sequential(
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B*F, C, H, W]. Returns fused [B, C, H, W]."""
        x_view, b = _reshape_frames(x, self.num_frames)  # [B, F, C, H, W]
        _, f, c, h, w = x_view.shape

        if self.per_position:
            scores = self.score_net(x).view(b, f, 1, h, w)
        else:
            scores = self.score_net(x).view(b, f, 1, 1, 1)

        weights = F.softmax(scores, dim=1)
        return torch.sum(x_view * weights, dim=1)


class QualityFusionSeq(nn.Module):
    """Per-frame quality attention on token sequences (encoder-domain fusion).

    Args:
        dim: token dimension ``D``.
        per_position: if True, predict a quality vector ``[B, F, T]`` (each
            timestep picks its own frame); otherwise a scalar per frame.
        num_frames: fixed burst size (5).
    """

    def __init__(self, dim: int, per_position: bool = True, num_frames: int = 5):
        super().__init__()
        self.num_frames = num_frames
        self.per_position = per_position
        hidden = max(dim // 4, 32)
        self.score_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B*F, T, D]. Returns fused [B, T, D]."""
        x_view, b = _reshape_frames(x, self.num_frames)  # [B, F, T, D]
        _, f, t, d = x_view.shape

        if self.per_position:
            scores = self.score_net(x_view).squeeze(-1)  # [B, F, T]
        else:
            pooled = x_view.mean(dim=2)  # [B, F, D]
            scores = self.score_net(pooled).squeeze(-1)  # [B, F]
            scores = scores.unsqueeze(-1).expand(b, f, t)

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, F, T, 1]
        return torch.sum(x_view * weights, dim=1)


def stack_frames_as_memory(x: torch.Tensor, num_frames: int = 5) -> torch.Tensor:
    """Reshape per-frame token sequences into one memory tensor for cross-attn.

    Args:
        x: [B*F, T, D].
    Returns:
        memory: [B, F*T, D] — concatenated along the sequence dim.
    """
    x_view, b = _reshape_frames(x, num_frames)
    _, f, t, d = x_view.shape
    return x_view.reshape(b, f * t, d)


class FactorizedTemporalAttention(nn.Module):
    """Factorized temporal attention.

    Operates on feature maps ``[B*F, C, 1, W]``: applies a Transformer encoder
    along the **frame axis only**, independently at each spatial position, then
    mean-pools across frames. This "factorisation" treats each width slot as a
    separate temporal sequence (5 frames per position) — much cheaper than a
    full spatio-temporal attention and avoids blur-averaging across frames.

    Args:
        channels: feature channels (token dim).
        num_frames: burst size (5).
        num_heads, num_layers, ff_dim, dropout: transformer hyperparams.
    """

    def __init__(
        self,
        channels: int,
        num_frames: int = 5,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels

        self.frame_pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, 1, channels) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B*F, C, 1, W]. Returns fused [B, C, 1, W]."""
        total, c, h, w = x.size()
        assert h == 1, f"FactorizedTemporalAttention expects H=1 (got {h})"
        B = total // self.num_frames
        F_ = self.num_frames

        # [B, F, W, C]
        x = x.squeeze(2).view(B, F_, c, w).permute(0, 1, 3, 2)
        x = x + self.frame_pos_embedding

        # [B*W, F, C] — transformer along frame axis at each spatial position
        x_time = x.permute(0, 2, 1, 3).contiguous().view(B * w, F_, c)
        x_time = self.transformer(x_time)
        x_time = self.norm(x_time)

        # Back to [B, F, W, C], mean-pool over frames
        x_fused = x_time.view(B, w, F_, c).permute(0, 2, 1, 3).mean(dim=1)

        # [B, C, 1, W]
        return x_fused.permute(0, 2, 1).unsqueeze(2)
