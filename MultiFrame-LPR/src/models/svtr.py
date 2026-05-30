"""SVTR backbone (Du et al. 2022) — light vision-transformer designed for OCR.

- Patch embed downsamples by 4 (two stride-2 3x3 convs with BN+GELU).
- Three stages with dims (64, 128, 256) and depths (3, 6, 3).
- Mixed Local (window 7x11) + Global attention per layer.
- Sub-sample layers shrink height only (stride (2, 1)) to preserve width
  for the OCR sequence dimension.
- Final adaptive pool to height 1, then 1x1 conv to ``out_channels`` (192).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor()
    return x / keep * mask


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.p, self.training)


class SVTRPatchEmbed(nn.Module):
    """Two stride-2 3x3 conv-BN-GELU layers → 1/4 spatial size."""

    def __init__(self, img_size=(32, 128), in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.num_patches = (img_size[0] // 4) * (img_size[1] // 4)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).permute(0, 2, 1)


class SVTRMlp(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, drop: float = 0.0):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SVTRAttention(nn.Module):
    """Mixed Local/Global multi-head self-attention.

    ``mixer='Local'`` applies a window mask of size ``local_k`` around each
    token's spatial position; ``mixer='Global'`` is vanilla MHSA.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mixer: str = "Global",
        HW: tuple[int, int] | None = None,
        local_k: tuple[int, int] = (7, 11),
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mixer = mixer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if mixer == "Local" and HW is not None:
            H, W = HW
            hk, wk = local_k
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1)
            for h in range(H):
                for w in range(W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.0
            mask = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask_inf = torch.full((H * W, H * W), float("-inf"))
            self.register_buffer(
                "local_mask",
                torch.where(mask < 1, torch.zeros_like(mask_inf), mask_inf)
                .unsqueeze(0)
                .unsqueeze(0),
            )
        else:
            self.local_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.local_mask is not None:
            attn = attn + self.local_mask
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class SVTRBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str = "Global",
        HW: tuple[int, int] | None = None,
        local_k: tuple[int, int] = (7, 11),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        dp: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SVTRAttention(dim, num_heads, mixer, HW, local_k, qkv_bias, attn_drop, drop)
        self.dp = DropPath(dp) if dp > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SVTRMlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dp(self.attn(x)))
        x = self.norm2(x + self.dp(self.mlp(x)))
        return x


class SVTRSubSample(nn.Module):
    def __init__(self, in_c: int, out_c: int, HW: tuple[int, int], stride=(2, 1)):
        super().__init__()
        self.HW = HW
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.HW
        B, N, C = x.shape
        x = self.conv(x.permute(0, 2, 1).reshape(B, C, H, W))
        return self.norm(x.flatten(2).permute(0, 2, 1))


class SVTRBackbone(nn.Module):
    """Full SVTR backbone with three stages, mixed local+global attention.

    Args:
        img_size: input (H, W). Default (32, 128).
        in_channels: input channels.
        embed_dim: per-stage embed dims.
        depth: per-stage block counts. Total 12 blocks default.
        num_heads: per-stage attention heads.
        mixer: per-block mixer kind, length=sum(depth). Default: 6 Local + 6 Global.
        local_mixer: window sizes for Local blocks per stage.
        out_channels: final projection channels (192 in the ref paper).
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (32, 128),
        in_channels: int = 3,
        embed_dim: tuple[int, int, int] = (64, 128, 256),
        depth: tuple[int, int, int] = (3, 6, 3),
        num_heads: tuple[int, int, int] = (2, 4, 8),
        mixer: tuple[str, ...] = ("Local",) * 6 + ("Global",) * 6,
        local_mixer: tuple = ((7, 11), (7, 11), (7, 11)),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_channels: int = 192,
        last_drop: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        self.patch_embed = SVTRPatchEmbed(img_size, in_channels, embed_dim[0])
        HW0 = (img_size[0] // 4, img_size[1] // 4)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim[0]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = np.linspace(0, drop_path_rate, sum(depth)).tolist()
        HW1 = HW0
        self.blocks1 = nn.ModuleList([
            SVTRBlock(embed_dim[0], num_heads[0], mixer[i], HW1, local_mixer[0],
                      mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth[0])
        ])
        self.sub_sample1 = SVTRSubSample(embed_dim[0], embed_dim[1], HW1)
        HW2 = (HW1[0] // 2, HW1[1])
        self.blocks2 = nn.ModuleList([
            SVTRBlock(embed_dim[1], num_heads[1], mixer[depth[0] + i], HW2, local_mixer[1],
                      mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[depth[0] + i])
            for i in range(depth[1])
        ])
        self.sub_sample2 = SVTRSubSample(embed_dim[1], embed_dim[2], HW2)
        HW3 = (HW2[0] // 2, HW2[1])
        self.HW3 = HW3
        self.blocks3 = nn.ModuleList([
            SVTRBlock(embed_dim[2], num_heads[2], mixer[depth[0] + depth[1] + i], HW3,
                      local_mixer[2], mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                      dpr[depth[0] + depth[1] + i])
            for i in range(depth[2])
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.last_conv = nn.Conv2d(embed_dim[2], out_channels, 1, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(last_drop)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 3, H, W] -> [B, out_channels, 1, W']."""
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        for b in self.blocks1:
            x = b(x)
        x = self.sub_sample1(x)
        for b in self.blocks2:
            x = b(x)
        x = self.sub_sample2(x)
        for b in self.blocks3:
            x = b(x)
        B, N, C = x.shape
        H3, W3 = self.HW3
        x = x.permute(0, 2, 1).reshape(B, C, H3, W3)
        return self.dropout(self.hardswish(self.last_conv(self.avg_pool(x))))
