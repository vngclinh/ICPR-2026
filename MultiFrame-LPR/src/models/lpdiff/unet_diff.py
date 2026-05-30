"""Conditional U-Net denoiser used inside LP-Diff.

Adapted from https://github.com/haoyGONG/LP-Diff/blob/main/model/LPDiff_modules/unet.py
(itself based on SR3 / ResDiff). The conditional input (MTA output) is
concatenated with the noisy tensor along the channel dimension, so
``in_channel`` defaults to 6 (3 cond + 3 noisy).
"""
from __future__ import annotations

import math
from inspect import isfunction
from typing import List

import torch
import torch.nn as nn


def _exists(x) -> bool:
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if isfunction(d) else d


class PositionalEncoding(nn.Module):
    """Sinusoidal embedding of the continuous noise level (sqrt alpha cumprod)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        return torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)


class FeatureWiseAffine(nn.Module):
    """Add a noise-conditioned bias (and optional scale) to a feature map."""

    def __init__(self, in_channels: int, out_channels: int, use_affine_level: bool = False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Linear(in_channels, out_channels * (1 + int(use_affine_level)))

    def forward(self, x: torch.Tensor, noise_embed: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            return (1 + gamma) * x + beta
        return x + self.noise_func(noise_embed).view(batch, -1, 1, 1)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class _Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 32, dropout: float = 0.0):
        super().__init__()
        layers = [nn.GroupNorm(groups, dim), Swish()]
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv2d(dim, dim_out, 3, padding=1))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResnetBlock(nn.Module):
    def __init__(
        self, dim: int, dim_out: int, noise_level_emb_dim: int | None = None,
        dropout: float = 0.0, use_affine_level: bool = False, norm_groups: int = 32,
    ):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = _Block(dim, dim_out, groups=norm_groups)
        self.block2 = _Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class _SelfAttention(nn.Module):
    """O(N^2) self-attention; only used at the bottleneck resolution."""

    def __init__(self, in_channel: int, n_head: int = 1, norm_groups: int = 32):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(x)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + x


class _ResnetBlockWithAttn(nn.Module):
    def __init__(
        self, dim: int, dim_out: int, *, noise_level_emb_dim: int | None = None,
        norm_groups: int = 32, dropout: float = 0.0, with_attn: bool = False,
    ):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = _ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout,
        )
        if with_attn:
            self.attn = _SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


class DiffusionUNet(nn.Module):
    """SR3-style conditional U-Net.

    Args:
        in_channel: Channels of the concatenated (cond, noisy) input. Default 6.
        out_channel: Channels of the predicted noise. Default 3.
        inner_channel: Base channel count. Halved from 64 -> 32 vs upstream to
            keep the model trainable on mid-range GPUs.
        norm_groups: GroupNorm group count; must divide every channel multiplier.
        channel_mults: Channel multipliers per stage (length determines depth).
            Upstream uses 5 stages [1,2,4,8,8]; we default to 4 stages
            [1,2,4,4] so a 64xH input bottoms out at 8x(W/8) instead of 4x.
        attn_res: Spatial resolutions at which to insert self-attention.
        res_blocks: Number of ResNet blocks per stage.
        image_size: Reference height of the input; used together with ``attn_res``
            to decide where attention layers are inserted.
    """

    def __init__(
        self,
        in_channel: int = 6,
        out_channel: int = 3,
        inner_channel: int = 32,
        norm_groups: int = 32,
        channel_mults: List[int] = (1, 2, 4, 4),
        attn_res: List[int] = (16,),
        res_blocks: int = 2,
        dropout: float = 0.0,
        with_noise_level_emb: bool = True,
        image_size: int = 64,
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel),
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size

        downs: list[nn.Module] = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(_ResnetBlockWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,
                ))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(_Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            _ResnetBlockWithAttn(
                pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                norm_groups=norm_groups, dropout=dropout, with_attn=False,
            )
        ])

        ups: list[nn.Module] = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                ups.append(_ResnetBlockWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,
                ))
                pre_channel = channel_mult
            if not is_last:
                ups.append(_Upsample(pre_channel))
                now_res = now_res * 2
        self.ups = nn.ModuleList(ups)

        self.final_conv = _Block(pre_channel, _default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t = self.noise_level_mlp(time) if _exists(self.noise_level_mlp) else None
        feats: list[torch.Tensor] = []
        for layer in self.downs:
            if isinstance(layer, _ResnetBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, _ResnetBlockWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        for layer in self.ups:
            if isinstance(layer, _ResnetBlockWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        return self.final_conv(x)
