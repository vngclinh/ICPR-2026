"""Five model variants (V1-V4 + V5 SVTR-based) for the ICPR 2026 LPR pipeline.

All four variants share the same STN + SE-ResNet34-C backbone but differ in
**where** multi-frame fusion happens and **how deep** their Transformer
encoder is.

============  ================================  ===============================
Variant       Fusion                            Head
============  ================================  ===============================
V1            early (on feature maps)           Linear -> CTC
V2            late (on encoder tokens)          Decoder -> LN + Linear
V3            late, deeper encoder              Decoder -> LN + Linear
V4            cross-attention over 5 frames     Cross-Attn Decoder -> LN+Linear
============  ================================  ===============================

Each variant ``forward`` returns a dict so the trainer can compute every loss
in a single pass:

    {
        "log_probs":     [B, T, C],
        "features":      [B, T, D]      # input to the main head (for center loss)
        "aux_features":  [B, T, D]      # encoder intermediate tap (for aux CTC)
        "stn_params":    [B, *],
        "stn_mode":      str,
    }
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.lpr_decoder import CrossAttnCTCDecoder, CTCDecoder
from src.models.lpr_encoder import LPREncoder
from src.models.lpr_fusion import (
    FactorizedTemporalAttention,
    QualityFusionMap,
    QualityFusionSeq,
    stack_frames_as_memory,
)
from src.models.lpr_stn import STNRectifier
from src.models.se_resnet34c import SEResNet34C
from src.models.svtr import SVTRBackbone


@dataclass
class VariantConfig:
    """Hyperparameters describing one variant.

    Defaults reflect the suggestions in the design doc (Section 4).
    """

    num_classes: int
    num_frames: int = 5

    # Backbone shared across variants.
    in_channels: int = 3

    # STN.
    use_stn: bool = True
    stn_mode: str = "tps"               # "affine" or "tps"
    stn_tps_x: int = 10
    stn_tps_y: int = 2

    # Encoder.
    d_model: int = 512
    nhead: int = 8
    encoder_layers: int = 4
    encoder_ff: int = 2048
    encoder_dropout: float = 0.1
    aux_tap_layer: int | None = 2       # halfway is a reasonable default

    # Decoder (V2/V3/V4).
    decoder_layers: int = 2
    decoder_ff: int = 2048
    decoder_dropout: float = 0.1
    decoder_num_queries: int = 16

    # Fusion.
    fusion_per_position: bool = True

    # SVTR (V5).
    svtr_img_h: int = 32
    svtr_img_w: int = 128
    svtr_out_channels: int = 192
    svtr_drop_path: float = 0.1
    factemp_layers: int = 3
    factemp_ff: int = 1536
    factemp_heads: int = 8


class _SharedTrunk(nn.Module):
    """STN + SE-ResNet34-C applied per-frame."""

    def __init__(self, cfg: VariantConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_stn:
            if cfg.stn_mode == "tps":
                self.stn = STNRectifier(
                    mode="tps",
                    in_channels=cfg.in_channels,
                    num_x=cfg.stn_tps_x,
                    num_y=cfg.stn_tps_y,
                )
            else:
                self.stn = STNRectifier(mode="affine", in_channels=cfg.in_channels)
        else:
            self.stn = None
        self.backbone = SEResNet34C(in_channels=cfg.in_channels)

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args: x_flat [B*F, 3, H, W]. Returns (feats [B*F, C, 1, W'], stn_params)."""
        stn_params = None
        if self.stn is not None:
            x_flat, stn_params = self.stn(x_flat)
        feats = self.backbone(x_flat)  # [B*F, 512, 1, W']
        return feats, stn_params


def _maps_to_tokens(feat_map: torch.Tensor) -> torch.Tensor:
    """Convert backbone output [B, C, 1, W'] -> token sequence [B, W', C]."""
    return feat_map.squeeze(2).permute(0, 2, 1).contiguous()


class _BaseVariant(nn.Module):
    """Shared scaffolding: trunk, main head, optional aux head."""

    def __init__(self, cfg: VariantConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _SharedTrunk(cfg)
        # Aux head sits on encoder tokens (same dim as encoder d_model).
        self.aux_head = nn.Linear(cfg.d_model, cfg.num_classes)

    def _make_encoder(self) -> LPREncoder:
        return LPREncoder(
            d_model=self.cfg.d_model,
            nhead=self.cfg.nhead,
            num_layers=self.cfg.encoder_layers,
            ff_dim=self.cfg.encoder_ff,
            dropout=self.cfg.encoder_dropout,
            aux_tap_layer=self.cfg.aux_tap_layer,
        )

    def _project_backbone_to_dmodel(self) -> nn.Module:
        if self.cfg.d_model == self.trunk.backbone.out_channels:
            return nn.Identity()
        return nn.Linear(self.trunk.backbone.out_channels, self.cfg.d_model)


class LPRVariantV1(_BaseVariant):
    """V1 — early fusion + CTC head (no decoder)."""

    def __init__(self, cfg: VariantConfig):
        super().__init__(cfg)
        self.fusion = QualityFusionMap(
            channels=self.trunk.backbone.out_channels,
            per_position=cfg.fusion_per_position,
            num_frames=cfg.num_frames,
        )
        self.project = self._project_backbone_to_dmodel()
        self.encoder = self._make_encoder()
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        b, f, c, h, w = x.shape
        x_flat = x.view(b * f, c, h, w)
        feats, stn_params = self.trunk(x_flat)            # [B*F, C, 1, W']
        fused = self.fusion(feats)                         # [B, C, 1, W']
        tokens = _maps_to_tokens(fused)                    # [B, T, C]
        tokens = self.project(tokens)                      # [B, T, D]
        enc_out, aux = self.encoder(tokens)
        log_probs = self.head(enc_out).log_softmax(dim=-1)
        return {
            "log_probs": log_probs,
            "features": enc_out,
            "aux_features": aux,
            "stn_params": stn_params,
            "stn_mode": self.cfg.stn_mode if self.cfg.use_stn else None,
        }


class _LateFusionVariant(_BaseVariant):
    """Shared body for V2 and V3 (late-fusion + refining decoder)."""

    def __init__(self, cfg: VariantConfig):
        super().__init__(cfg)
        self.project = self._project_backbone_to_dmodel()
        self.encoder = self._make_encoder()
        self.fusion = QualityFusionSeq(
            dim=cfg.d_model,
            per_position=cfg.fusion_per_position,
            num_frames=cfg.num_frames,
        )
        self.decoder = CTCDecoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.decoder_layers,
            ff_dim=cfg.decoder_ff,
            dropout=cfg.decoder_dropout,
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        b, f, c, h, w = x.shape
        x_flat = x.view(b * f, c, h, w)
        feats, stn_params = self.trunk(x_flat)            # [B*F, C, 1, W']
        tokens = _maps_to_tokens(feats)                   # [B*F, T, C]
        tokens = self.project(tokens)                     # [B*F, T, D]
        enc_out, aux = self.encoder(tokens)               # [B*F, T, D], aux [B*F, T, D]
        fused = self.fusion(enc_out)                      # [B, T, D]
        dec_out = self.decoder(fused)                     # [B, T, D]
        feat_final = self.final_ln(dec_out)
        log_probs = self.head(feat_final).log_softmax(dim=-1)

        # Aux is on per-frame tokens. Fuse it the same way so the aux head can
        # be applied at batch B rather than B*F.
        aux_fused = self.fusion(aux) if aux is not None else None

        return {
            "log_probs": log_probs,
            "features": feat_final,
            "aux_features": aux_fused,
            "stn_params": stn_params,
            "stn_mode": self.cfg.stn_mode if self.cfg.use_stn else None,
        }


class LPRVariantV2(_LateFusionVariant):
    """V2 — late fusion + decoder, standard encoder depth."""


class LPRVariantV3(_LateFusionVariant):
    """V3 — same architecture as V2; vary depth via VariantConfig."""


class LPRVariantV4(_BaseVariant):
    """V4 — fusion baked into a cross-attention decoder."""

    def __init__(self, cfg: VariantConfig):
        super().__init__(cfg)
        self.project = self._project_backbone_to_dmodel()
        self.encoder = self._make_encoder()
        self.decoder = CrossAttnCTCDecoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.decoder_layers,
            ff_dim=cfg.decoder_ff,
            dropout=cfg.decoder_dropout,
            num_queries=cfg.decoder_num_queries,
        )
        # Aux for V4 also uses an in-encoder tap; we fuse it with mean-pool
        # over frames so the aux head sees [B, T, D].
        self.fusion_aux = QualityFusionSeq(
            dim=cfg.d_model,
            per_position=cfg.fusion_per_position,
            num_frames=cfg.num_frames,
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        b, f, c, h, w = x.shape
        x_flat = x.view(b * f, c, h, w)
        feats, stn_params = self.trunk(x_flat)            # [B*F, C, 1, W']
        tokens = _maps_to_tokens(feats)                   # [B*F, T, C]
        tokens = self.project(tokens)                     # [B*F, T, D]
        enc_out, aux = self.encoder(tokens)               # [B*F, T, D]
        memory = stack_frames_as_memory(enc_out, num_frames=self.cfg.num_frames)  # [B, F*T, D]
        dec_out = self.decoder(memory)                    # [B, num_queries, D]
        feat_final = self.final_ln(dec_out)
        log_probs = self.head(feat_final).log_softmax(dim=-1)

        aux_fused = self.fusion_aux(aux) if aux is not None else None
        return {
            "log_probs": log_probs,
            "features": feat_final,
            "aux_features": aux_fused,
            "stn_params": stn_params,
            "stn_mode": self.cfg.stn_mode if self.cfg.use_stn else None,
        }


class LPRVariantV5(nn.Module):
    """V5 — SVTR backbone + Factorized Temporal Attention.

    Pipeline: TPS → SVTR per-frame → FactorizedTemporalAttention → LPREncoder → CTC head.

    Uses its own SVTR-sized embedding dim (192 by default), so it does NOT
    inherit from ``_BaseVariant`` (which assumes ``cfg.d_model`` everywhere).
    """

    def __init__(self, cfg: VariantConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_stn:
            if cfg.stn_mode == "tps":
                self.stn = STNRectifier(
                    mode="tps",
                    in_channels=cfg.in_channels,
                    num_x=cfg.stn_tps_x,
                    num_y=cfg.stn_tps_y,
                )
            else:
                self.stn = STNRectifier(mode="affine", in_channels=cfg.in_channels)
        else:
            self.stn = None

        self.backbone = SVTRBackbone(
            img_size=(cfg.svtr_img_h, cfg.svtr_img_w),
            in_channels=cfg.in_channels,
            out_channels=cfg.svtr_out_channels,
            drop_path_rate=cfg.svtr_drop_path,
        )

        self.fusion = FactorizedTemporalAttention(
            channels=cfg.svtr_out_channels,
            num_frames=cfg.num_frames,
            num_heads=cfg.factemp_heads,
            num_layers=cfg.factemp_layers,
            ff_dim=cfg.factemp_ff,
            dropout=cfg.encoder_dropout,
        )

        # Refining encoder on the fused token sequence (no per-frame batching).
        self.encoder = LPREncoder(
            d_model=cfg.svtr_out_channels,
            nhead=cfg.factemp_heads,
            num_layers=cfg.encoder_layers,
            ff_dim=cfg.encoder_ff,
            dropout=cfg.encoder_dropout,
            aux_tap_layer=cfg.aux_tap_layer if cfg.aux_tap_layer else None,
        )

        self.aux_head = nn.Linear(cfg.svtr_out_channels, cfg.num_classes)
        self.head = nn.Linear(cfg.svtr_out_channels, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        b, f, c, h, w = x.shape
        x_flat = x.view(b * f, c, h, w)
        stn_params = None
        if self.stn is not None:
            x_flat, stn_params = self.stn(x_flat)
        feats = self.backbone(x_flat)        # [B*F, 192, 1, W']
        fused = self.fusion(feats)           # [B, 192, 1, W']
        tokens = fused.squeeze(2).permute(0, 2, 1).contiguous()  # [B, W', 192]
        enc_out, aux = self.encoder(tokens)
        log_probs = self.head(enc_out).log_softmax(dim=-1)
        return {
            "log_probs": log_probs,
            "features": enc_out,
            "aux_features": aux,
            "stn_params": stn_params,
            "stn_mode": self.cfg.stn_mode if self.cfg.use_stn else None,
        }


VARIANT_REGISTRY: dict[str, type] = {
    "v1": LPRVariantV1,
    "v2": LPRVariantV2,
    "v3": LPRVariantV3,
    "v4": LPRVariantV4,
    "v5": LPRVariantV5,
}


def build_variant(name: str, cfg: VariantConfig) -> nn.Module:
    """Factory: ``build_variant("v3", cfg)``."""
    key = name.lower()
    if key not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant {name!r}; choose from {list(VARIANT_REGISTRY)}")
    return VARIANT_REGISTRY[key](cfg)
