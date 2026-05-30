"""Multi-frame Temporal Aggregation (MTA) module from LP-Diff.

Takes three LR frames (already upscaled to HR spatial size in the dataloader)
and emits a 3-channel coarse HR estimate. The diffusion model then learns the
residual HR - MTA(LR1,LR2,LR3).

Adapted from https://github.com/haoyGONG/LP-Diff (CVPR 2025); simplified to the
modules actually used by ``MTA.forward``: Encoder, Decoder, CrossAttentionLayer,
GradientCurvatureAttention, IntraframeAtt (= channel-attention + spatial-attention
fusion). Components that are defined but unused in the upstream forward pass
(Sobel module, directional conv, ResNetWithUpsample, FeatureFusionModule, etc.)
are dropped.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _kernel_size(in_channel: int) -> int:
    """ECA-style adaptive kernel size for 1D conv on pooled channel descriptors."""
    k = int((math.log2(in_channel) + 1) // 2)
    return k + 1 if k % 2 == 0 else k


class Encoder(nn.Module):
    """3 -> 64 channels with 8x spatial downsampling."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """64 -> 3 channels with 8x spatial upsampling."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class FeatureFusion(nn.Module):
    """Concat two feature maps and project back to ``out_channel`` with 2 conv layers."""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor, fuse: torch.Tensor) -> torch.Tensor:
        out = torch.cat((x, fuse), dim=1)
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        return self.batchnorm(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Upstream uses the same MaxPool here and calls it "median" — kept for parity.
        self.median_pool = nn.AdaptiveMaxPool2d(1)
        self.k = _kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(6, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(6, 1, kernel_size=self.k, padding=self.k // 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t1_avg = self.avg_pool(t1)
        t2_avg = self.avg_pool(t2)
        t1_max = self.max_pool(t1)
        t2_max = self.max_pool(t2)
        t1_med = self.median_pool(t1)
        t2_med = self.median_pool(t2)
        pool = torch.cat(
            [t1_avg, t1_max, t1_med, t2_avg, t2_max, t2_med], dim=2,
        ).squeeze(-1).transpose(1, 2)
        a1 = self.channel_conv1(pool)
        a2 = self.channel_conv2(pool)
        stack = torch.stack([a1, a2], dim=0)
        return self.softmax(stack).transpose(-1, -2).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_conv1 = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.spatial_conv2 = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t1_avg = torch.mean(t1, dim=1, keepdim=True)
        t2_avg = torch.mean(t2, dim=1, keepdim=True)
        t1_max = torch.max(t1, dim=1, keepdim=True)[0]
        t2_max = torch.max(t2, dim=1, keepdim=True)[0]
        t1_med = torch.median(t1, dim=1, keepdim=True)[0]
        t2_med = torch.median(t2, dim=1, keepdim=True)[0]
        pool = torch.cat([t1_avg, t1_max, t1_med, t2_avg, t2_max, t2_med], dim=1)
        a1 = self.spatial_conv1(pool)
        a2 = self.spatial_conv2(pool)
        return self.softmax(torch.stack([a1, a2], dim=0))


class IntraframeAtt(nn.Module):
    """Channel+spatial gated fusion of two feature maps."""

    def __init__(self, in_channel: int):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = SpatialAttention()
        self.feature_fusion = FeatureFusion(in_channel * 2, in_channel)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        channel_stack = self.channel_attention(t1, t2)
        spatial_stack = self.spatial_attention(t1, t2)
        stack = channel_stack + spatial_stack + 1
        return stack[0] * t1 + stack[1] * t2


class CrossAttentionLayer(nn.Module):
    """Standard transformer cross-attention block (pre-norm style).

    Adapted to be reusable across frame pairs — the upstream layer hard-codes
    d_model=64; we keep the same shape but expose ``d_model``/``num_heads``.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 8, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = f1.size()
        f1 = f1.view(batch_size, self.d_model, -1).transpose(1, 2)  # [B, N, D]
        f2 = f2.view(batch_size, self.d_model, -1).transpose(1, 2)

        q = self.q_linear(f2).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(f1).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(f1).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        output = self.norm1(output + self.q_linear(f2))

        ff_output = self.fc2(F.relu(self.fc1(output)))
        output = self.norm2(ff_output + output)
        return output.transpose(1, 2).view(batch_size, self.d_model, height, width)


class GradientCurvatureAttention(nn.Module):
    """Edge/curvature-driven attention; kernels are precomputed once per call.

    Upstream re-creates the kernel tensors per forward pass; we keep them as
    buffers for efficiency.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        sobel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        kxx = torch.tensor([[1.0, -2.0, 1.0], [2.0, -4.0, 2.0], [1.0, -2.0, 1.0]])
        kyy = torch.tensor([[1.0, 2.0, 1.0], [-2.0, -4.0, -2.0], [1.0, 2.0, 1.0]])
        kxy = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
        for name, kernel in {
            "sobel_x": sobel_x, "sobel_y": sobel_y,
            "kxx": kxx, "kyy": kyy, "kxy": kxy,
        }.items():
            self.register_buffer(name, kernel.view(1, 1, 3, 3), persistent=False)
        self.softmax = nn.Softmax(dim=1)

    def _conv_per_channel(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        return F.conv2d(x, kernel.repeat(c, 1, 1, 1), padding=1, groups=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self._conv_per_channel(x, self.sobel_x)
        gy = self._conv_per_channel(x, self.sobel_y)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        ixx = self._conv_per_channel(x, self.kxx)
        iyy = self._conv_per_channel(x, self.kyy)
        ixy = self._conv_per_channel(x, self.kxy)
        eps = 1e-6
        gx2 = gx * gx
        gy2 = gy * gy
        numerator = gx2 * iyy - 2 * gx * gy * ixy + gy2 * ixx
        denominator = (gx2 + gy2 + eps) ** 1.5
        curvature = numerator / denominator

        attn = self.softmax(grad_mag + curvature) + 1.0
        return attn * x


class MTA(nn.Module):
    """Multi-frame Temporal Aggregation module.

    Pipeline:
        for i in {1,2,3}: f_i = Encoder(LR_i)        # [B, 64, H/8, W/8]
        c1 = CrossAttn(f1, f2) + f2
        c2 = CrossAttn(f2, f3) + f3
        g1 = GCA(c1); g2 = GCA(c2)
        fused = IntraframeAtt(g1, g2)
        out = Decoder(fused)                          # [B, 3, H, W]
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        inner_channel = 3
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.GCA = GradientCurvatureAttention()
        self.IntraframeAtt = IntraframeAtt(inner_channel)
        self.crossatt = CrossAttentionLayer(
            d_model=embed_dim, num_heads=num_heads, dim_feedforward=embed_dim * 4,
        )
        self._kaiming_init(self.encoder)
        self._kaiming_init(self.decoder)
        self._kaiming_init(self.IntraframeAtt)
        self._kaiming_init(self.crossatt)

    @staticmethod
    def _kaiming_init(module: nn.Module) -> None:
        for layer in module.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias, 0)

    def forward(
        self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.encoder(f1)
        e2 = self.encoder(f2)
        e3 = self.encoder(f3)
        c1 = self.crossatt(e1, e2) + e2
        c2 = self.crossatt(e2, e3) + e3
        g1 = self.GCA(c1)
        g2 = self.GCA(c2)
        fused = self.IntraframeAtt(g1, g2)
        return self.decoder(fused)
