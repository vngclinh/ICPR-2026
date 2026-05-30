"""Spatial Transformer Network for the ICPR 2026 LPR pipeline.

Supports two rectification modes:
* ``affine``  — 2x3 affine matrix (legacy compatible with ``components.STNBlock``).
* ``tps``     — Thin-Plate-Spline rectification with K control points (Baek et al.
  2019 use this for scene text / plates that are bent or perspective-skewed).

Both modes are initialised to the identity transform so training stays stable
during the first epochs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LocalizationCNN(nn.Module):
    """Small CNN shared between affine and TPS modes."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class AffineSTN(nn.Module):
    """Legacy affine STN: predicts a 2x3 matrix, identity-initialised."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.loc = _LocalizationCNN(in_channels)
        self.head = nn.Linear(256, 6)
        # Identity init
        self.head.weight.data.zero_()
        self.head.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.head(self.loc(x)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False), theta


def _build_base_grid(out_h: int, out_w: int) -> torch.Tensor:
    """Build the canonical output grid in normalized [-1, 1] coords."""
    ys = torch.linspace(-1.0, 1.0, out_h)
    xs = torch.linspace(-1.0, 1.0, out_w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]


def _build_control_points(num_x: int, num_y: int) -> torch.Tensor:
    """Anchor control points on the boundary of the canonical rectangle.

    For OCR/plates ``num_y=2`` (top + bottom edges) is common.
    """
    xs = torch.linspace(-1.0, 1.0, num_x)
    ys = torch.linspace(-1.0, 1.0, num_y)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [K, 2]


def _tps_solver(c_src: torch.Tensor, c_dst: torch.Tensor) -> torch.Tensor:
    """Solve the TPS coefficients (T = [W; A]) given matched control points.

    Always runs in float32 with autocast disabled — ``torch.linalg.solve`` on
    CUDA can hit illegal-memory-access under fp16 AMP. We also add Tikhonov
    regularisation so the (K+3)x(K+3) system stays well-conditioned even when
    early-training control points are nearly degenerate.

    c_src: [B, K, 2] source control points (output canonical grid).
    c_dst: [B, K, 2] predicted destination control points (input image).
    Returns T: [B, K+3, 2] in the original dtype of ``c_src``.
    """
    orig_dtype = c_src.dtype
    with torch.amp.autocast(device_type=c_src.device.type, enabled=False):
        c_src_f = c_src.float()
        c_dst_f = c_dst.float()
        b, k, _ = c_src_f.shape
        d2 = torch.cdist(c_src_f, c_src_f, p=2.0) ** 2
        U = d2 * torch.log(d2 + 1e-8)

        ones = torch.ones(b, k, 1, dtype=c_src_f.dtype, device=c_src_f.device)
        P = torch.cat([ones, c_src_f], dim=2)

        zeros = torch.zeros(b, 3, 3, dtype=c_src_f.dtype, device=c_src_f.device)
        L_top = torch.cat([U, P], dim=2)
        L_bot = torch.cat([P.transpose(1, 2), zeros], dim=2)
        L = torch.cat([L_top, L_bot], dim=1)  # [B, K+3, K+3]
        # Tikhonov regularisation for numerical stability.
        eye = torch.eye(L.size(-1), dtype=L.dtype, device=L.device).unsqueeze(0)
        L = L + 1e-4 * eye

        Y = torch.cat(
            [c_dst_f, torch.zeros(b, 3, 2, dtype=c_src_f.dtype, device=c_src_f.device)],
            dim=1,
        )
        T = torch.linalg.solve(L, Y)
    return T.to(orig_dtype)


def _tps_grid(T: torch.Tensor, c_src: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Sample a TPS grid of shape [B, out_h, out_w, 2] from coefficients T.

    Computed in float32 with autocast disabled to match ``_tps_solver``.
    """
    orig_dtype = T.dtype
    with torch.amp.autocast(device_type=T.device.type, enabled=False):
        T_f = T.float()
        c_src_f = c_src.float()
        b = T_f.shape[0]
        base = _build_base_grid(out_h, out_w).to(T_f.device, dtype=T_f.dtype)
        pts = base.view(-1, 2).unsqueeze(0).expand(b, -1, -1)

        d2 = torch.cdist(pts, c_src_f, p=2.0) ** 2
        U = d2 * torch.log(d2 + 1e-8)
        ones = torch.ones(b, pts.shape[1], 1, dtype=T_f.dtype, device=T_f.device)
        Q = torch.cat([U, ones, pts], dim=2)
        sampled = torch.bmm(Q, T_f)
    return sampled.view(b, out_h, out_w, 2).to(orig_dtype)


class TPSSTN(nn.Module):
    """Thin-Plate-Spline rectifier with identity initialisation.

    Args:
        in_channels: input channel count.
        num_control_points: K, total points (must be ``num_x * num_y``).
        num_x, num_y: control-point grid layout (defaults 10x2 = 20 points).
        out_h, out_w: rectified output size (defaults to input size if None).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_x: int = 10,
        num_y: int = 2,
        out_h: int | None = None,
        out_w: int | None = None,
    ):
        super().__init__()
        self.num_x = num_x
        self.num_y = num_y
        k = num_x * num_y
        self.k = k
        self.out_h = out_h
        self.out_w = out_w

        self.loc = _LocalizationCNN(in_channels)
        self.head = nn.Linear(256, k * 2)
        # Identity init: predict the canonical control points themselves.
        ctrl = _build_control_points(num_x, num_y)  # [K, 2]
        self.register_buffer("ctrl_src", ctrl, persistent=False)
        self.head.weight.data.zero_()
        self.head.bias.data.copy_(ctrl.view(-1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        out_h = self.out_h or h
        out_w = self.out_w or w

        feats = self.loc(x)
        c_dst = self.head(feats).view(b, self.k, 2)
        c_src = self.ctrl_src.unsqueeze(0).expand(b, -1, -1)

        T = _tps_solver(c_src, c_dst)
        grid = _tps_grid(T, c_src, out_h, out_w)
        rectified = F.grid_sample(x, grid, align_corners=False)
        return rectified, c_dst


class STNRectifier(nn.Module):
    """Unified STN: either affine or TPS, returning ``(rectified, params)``.

    ``params`` is the affine theta [B, 2, 3] in affine mode, or the predicted
    destination control points [B, K, 2] in TPS mode. Caller passes these to
    ``stn_regularization_loss`` (see ``src/losses/stn_loss.py``) if desired.
    """

    def __init__(self, mode: str = "tps", in_channels: int = 3, **kwargs):
        super().__init__()
        if mode == "affine":
            self.impl = AffineSTN(in_channels=in_channels)
        elif mode == "tps":
            self.impl = TPSSTN(in_channels=in_channels, **kwargs)
        else:
            raise ValueError(f"Unknown STN mode: {mode!r}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.impl(x)
