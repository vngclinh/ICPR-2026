"""Full LP-Diff wrapper: MTA + conditional diffusion U-Net.

Owns:
  * MTA — produces a coarse HR estimate from three LR frames.
  * GaussianDiffusion — learns/samples the residual HR - MTA(LR).

The wrapper provides ``training_loss(lr1, lr2, lr3, hr)`` for training and
``infer(lr1, lr2, lr3)`` for inference with DDIM sampling.

Image domain: tensors are expected in **[-1, 1]** (centered & scaled), which
matches the upstream LP-Diff data normalization. Convert from / to [0,1] at
the IO boundary in the training/inference scripts.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lpdiff.diffusion import GaussianDiffusion
from src.models.lpdiff.mta import MTA
from src.models.lpdiff.unet_diff import DiffusionUNet


class LPDiffNet(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        in_channel: int = 6,
        out_channel: int = 3,
        inner_channel: int = 32,
        norm_groups: int = 32,
        channel_mults: tuple = (1, 2, 4, 4),
        attn_res: tuple = (16,),
        res_blocks: int = 2,
        dropout: float = 0.1,
        beta_schedule: Optional[dict] = None,
        loss_type: str = "l1",
    ):
        super().__init__()
        if beta_schedule is None:
            beta_schedule = {
                "schedule": "linear",
                "n_timestep": 1000,
                "linear_start": 1e-6,
                "linear_end": 1e-2,
            }
        self.beta_schedule = beta_schedule

        self.mta = MTA(embed_dim=64, num_heads=8)
        unet = DiffusionUNet(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            channel_mults=channel_mults,
            attn_res=attn_res,
            res_blocks=res_blocks,
            dropout=dropout,
            image_size=image_size,
        )
        self.diffusion = GaussianDiffusion(
            denoise_fn=unet,
            image_size=image_size,
            channels=out_channel,
            loss_type=loss_type,
        )

    def configure_for_device(self, device: torch.device) -> None:
        """One-shot setup: move buffers to device and instantiate the loss object."""
        self.to(device)
        self.diffusion.set_loss(device)
        self.diffusion.set_new_noise_schedule(self.beta_schedule, device)

    def training_loss(
        self, lr1: torch.Tensor, lr2: torch.Tensor, lr3: torch.Tensor, hr: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.mta(lr1, lr2, lr3)
        return self.diffusion.p_losses(hr=hr, condition=condition)

    @torch.no_grad()
    def infer(
        self,
        lr1: torch.Tensor, lr2: torch.Tensor, lr3: torch.Tensor,
        sampler: str = "ddim", num_steps: int = 50,
    ) -> torch.Tensor:
        condition = self.mta(lr1, lr2, lr3)
        return self.diffusion.super_resolution(
            condition, sampler=sampler, num_steps=num_steps,
        )

    @staticmethod
    def to_diffusion_domain(x_01: torch.Tensor) -> torch.Tensor:
        """Map images from [0, 1] to [-1, 1] (diffusion convention)."""
        return x_01 * 2.0 - 1.0

    @staticmethod
    def from_diffusion_domain(x_pm1: torch.Tensor) -> torch.Tensor:
        """Map images from [-1, 1] back to [0, 1]."""
        return (x_pm1.clamp(-1.0, 1.0) + 1.0) * 0.5
