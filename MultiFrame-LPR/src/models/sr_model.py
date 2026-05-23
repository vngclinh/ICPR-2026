"""Lightweight RRDBNet super-resolution backbone (ESRGAN-style).

The implementation follows the original ESRGAN generator but is sized for
license-plate inputs: 8 RRDB blocks, 32 base channels, 16 growth channels.
Total parameters stay under ~1.6M for scale=2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def _conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)


class ResidualDenseBlock(nn.Module):
    """5-layer dense block with residual scaling (beta=0.2).

    inplace=False on LeakyReLU is required for gradient checkpointing
    compatibility — inplace ops on saved tensors break the recompute pass.
    """

    def __init__(self, nf: int = 32, gc: int = 16):
        super().__init__()
        self.conv1 = _conv3x3(nf, gc)
        self.conv2 = _conv3x3(nf + gc, gc)
        self.conv3 = _conv3x3(nf + 2 * gc, gc)
        self.conv4 = _conv3x3(nf + 3 * gc, gc)
        self.conv5 = _conv3x3(nf + 4 * gc, nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: three dense blocks + residual scaling."""

    def __init__(self, nf: int = 32, gc: int = 16):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # allow per-RDB checkpointing when called from a checkpointed RRDB
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

    def forward_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Same as forward but each RDB is individually checkpointed.

        empty_cache() between checkpoints returns freed intermediates to CUDA,
        preventing accumulation of fragmented blocks in the PyTorch cache that
        would cause OOM on Windows/WDDM despite ample total free GPU memory.
        """
        out = grad_checkpoint(self.rdb1, x, use_reentrant=False)
        torch.cuda.empty_cache()
        out = grad_checkpoint(self.rdb2, out, use_reentrant=False)
        torch.cuda.empty_cache()
        out = grad_checkpoint(self.rdb3, out, use_reentrant=False)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Lightweight RRDB-based super-resolution network.

    Pipeline:
        Conv -> N x RRDB -> trunk conv (+skip) -> upsample(s) -> HR conv -> out conv

    When ``use_checkpoint=True``, each RRDB block uses gradient checkpointing
    to trade ~8x activation memory for ~2x extra compute, preventing OOM
    when the SR is trained jointly with a large OCR backbone.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        nf: int = 32,
        gc: int = 16,
        num_blocks: int = 8,
        scale: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {scale}")
        self.scale = scale
        self.use_checkpoint = use_checkpoint

        self.conv_first = _conv3x3(in_channels, nf)
        # ModuleList so we can iterate and apply per-block checkpointing
        self.body = nn.ModuleList([RRDB(nf=nf, gc=gc) for _ in range(num_blocks)])
        self.trunk_conv = _conv3x3(nf, nf)

        # Each upsample step doubles spatial resolution
        self.upconv1 = _conv3x3(nf, nf)
        self.upconv2 = _conv3x3(nf, nf) if scale == 4 else None

        self.hr_conv = _conv3x3(nf, nf)
        self.conv_last = _conv3x3(nf, out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        # Zero-init the final conv so SR(x) == bicubic(x) at the start of training.
        # This keeps the downstream OCR backbone close to its pretrained behavior
        # during the SR-freeze warm-up epochs.
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [N, C, H, W]. Returns: [N, C, scale*H, scale*W]."""
        feat = self.conv_first(x)
        shortcut = feat  # saved for skip connection after the body

        for rrdb in self.body:
            if self.use_checkpoint and feat.requires_grad:
                feat = rrdb.forward_checkpoint(feat)
                torch.cuda.empty_cache()
            else:
                feat = rrdb(feat)

        trunk = self.trunk_conv(feat)
        feat = shortcut + trunk  # residual skip: matches the original Sequential design

        feat = self.lrelu(self.upconv1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.upconv2 is not None:
            feat = self.lrelu(self.upconv2(F.interpolate(feat, scale_factor=2, mode="nearest")))

        feat = self.lrelu(self.hr_conv(feat))
        out = self.conv_last(feat)

        # Residual from bicubic-upsampled input: stabilizes early training
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return out + base

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
