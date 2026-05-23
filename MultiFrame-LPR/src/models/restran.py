"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN.

Optionally augmented with a lightweight RRDB super-resolution frontend
that runs end-to-end before the OCR pipeline.
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    AttentionFusion,
    PositionalEncoding,
    ResNetFeatureExtractor,
    STNBlock,
)
from src.models.sr_model import RRDBNet


# ImageNet normalization constants used to undo/redo normalization
# when the SR module operates in the [0,1] image domain.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ResTranOCR(nn.Module):
    """OCR pipeline with optional super-resolution frontend.

    Pipeline (use_sr=False, baseline):
        Input [B,F,3,H,W] -> [Optional STN] -> ResNet34 -> AttentionFusion
        -> Transformer -> CTC head

    Pipeline (use_sr=True):
        Input [B,F,3,H,W] -> denormalize -> RRDB(scale=2) -> [B*F,3,2H,2W]
        (HR-like) -> optionally downsample back to (H,W) -> renormalize -> OCR path
    """

    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        pretrained: bool = True,
        use_sr: bool = True,
        sr_num_blocks: int = 8,
        sr_scale: int = 2,
        sr_nf: int = 32,
        sr_gc: int = 16,
        sr_feed_hr: bool = False,
        sr_blend: float = 1.0,
        sr_use_checkpoint: bool = True,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.use_sr = use_sr
        self.sr_scale = sr_scale
        self.sr_feed_hr = sr_feed_hr
        self.sr_blend = float(sr_blend)

        # 0. Optional super-resolution frontend
        if self.use_sr:
            self.sr = RRDBNet(
                in_channels=3,
                out_channels=3,
                nf=sr_nf,
                gc=sr_gc,
                num_blocks=sr_num_blocks,
                scale=sr_scale,
                use_checkpoint=sr_use_checkpoint,
            )
            self.register_buffer(
                "_imagenet_mean",
                torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_imagenet_std",
                torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
                persistent=False,
            )

        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone: ResNet34
        self.backbone = ResNetFeatureExtractor(pretrained=pretrained)

        # 3. Attention Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)

        # 4. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 5. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._imagenet_std + self._imagenet_mean

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._imagenet_mean) / self._imagenet_std

    def _blend_sr(self, base: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """Blend SR output with the original image-domain input for OCR stability."""
        alpha = max(0.0, min(1.0, self.sr_blend))
        if alpha <= 0.0:
            return base
        if alpha >= 1.0:
            return enhanced
        return base + alpha * (enhanced - base)

    def set_sr_requires_grad(self, requires_grad: bool) -> None:
        """Freeze / unfreeze the SR module (used by the trainer for curriculum)."""
        if not self.use_sr:
            return
        for p in self.sr.parameters():
            p.requires_grad = requires_grad

    def set_ocr_requires_grad(self, requires_grad: bool) -> None:
        """Freeze / unfreeze all OCR parameters (everything except SR)."""
        for name, p in self.named_parameters():
            if not name.startswith("sr."):
                p.requires_grad = requires_grad

    def forward(
        self,
        x: torch.Tensor,
        return_sr: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            x: [Batch, Frames, 3, H, W] - ImageNet-normalized.
            return_sr: when True, also return the SR output tensor when SR is enabled.

        Returns:
            logits [B, Seq_Len, Num_Classes] (default), or
            (logits, sr_output [B*F, 3, sr_scale*H, sr_scale*W] or None)
            when ``return_sr``.
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)

        sr_output: torch.Tensor | None = None
        if self.use_sr:
            sr_in = self._denormalize(x_flat).clamp(0.0, 1.0)
            sr_output = self.sr(sr_in)  # [B*F, 3, scale*H, scale*W], image domain

            if self.sr_feed_hr:
                base_hr = F.interpolate(
                    sr_in,
                    scale_factor=self.sr_scale,
                    mode="bilinear",
                    align_corners=False,
                )
                x_flat = self._normalize(self._blend_sr(base_hr, sr_output))
            else:
                # Downscale back to (H, W) for strict baseline compatibility.
                sr_down = F.interpolate(
                    sr_output,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                x_flat = self._normalize(self._blend_sr(sr_in, sr_down))

        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat

        features = self.backbone(x_aligned)  # [B*F, 512, 1, W']
        fused = self.fusion(features)        # [B, 512, 1, W']

        # Pool to the sequence length derived from the original (pre-SR) input width so
        # that sr_feed_hr=True (which doubles the spatial resolution) does not produce a
        # 2× longer CTC sequence and cause character over-prediction.
        expected_w = w // 8  # e.g. 128-wide input → seq_len 16
        if fused.shape[-1] != expected_w:
            fused = F.adaptive_avg_pool2d(fused, (1, expected_w))

        seq_input = fused.squeeze(2).permute(0, 2, 1)  # [B, W', 512]
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input)
        logits = self.head(seq_out).log_softmax(2)

        if return_sr:
            return logits, sr_output
        return logits
