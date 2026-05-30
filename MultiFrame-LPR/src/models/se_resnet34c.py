"""SE-ResNet34-C backbone for the ICPR 2026 LPR pipeline.

ResNet34 + Squeeze-and-Excitation (Hu et al. 2018) channel attention, with the
"-C" stem variant from Bag-of-Tricks (He et al. 2019): the standard 7x7 stem
conv is replaced by three 3x3 convs.

Strides in layer3/layer4 are modified to (2, 1) so the feature map keeps its
horizontal resolution (sequence length) while shrinking vertically. Final
adaptive pool collapses the height to 1 — output shape per frame is
[B, 512, 1, W'].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class SEBasicBlock(nn.Module):
    """ResNet-34 basic block with an SE module on the residual path."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride=(1, 1),
        downsample: nn.Module | None = None,
        reduction: int = 16,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


def _make_stem_c(in_channels: int = 3, stem_out: int = 64) -> nn.Sequential:
    """ResNet-C stem: three stacked 3x3 convs instead of a single 7x7."""
    mid = stem_out // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, mid, 3, 2, 1, bias=False),
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid, stem_out, 3, 1, 1, bias=False),
        nn.BatchNorm2d(stem_out),
        nn.ReLU(inplace=True),
    )


class SEResNet34C(nn.Module):
    """SE-ResNet-34 with "-C" stem, customised for OCR (preserve width)."""

    def __init__(self, in_channels: int = 3, reduction: int = 16):
        super().__init__()
        self.stem = _make_stem_c(in_channels, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer configs: (planes, blocks, first-stride)
        # Layer 1: keep spatial; layer 2: stride 2 both dims; layer 3/4: stride
        # (2, 1) to keep width for the sequence dimension.
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=(1, 1), reduction=reduction)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=(2, 2), reduction=reduction)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=(2, 1), reduction=reduction)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=(2, 1), reduction=reduction)

        self.out_channels = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        in_planes: int,
        planes: int,
        blocks: int,
        stride,
        reduction: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != (1, 1) or in_planes != planes * SEBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * SEBasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * SEBasicBlock.expansion),
            )

        layers = [SEBasicBlock(in_planes, planes, stride, downsample, reduction)]
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(planes, planes, (1, 1), None, reduction))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, 3, H, W]. Returns features [B, 512, 1, W']."""
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, None))
        return x
