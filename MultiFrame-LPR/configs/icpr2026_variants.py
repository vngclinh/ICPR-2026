"""Per-variant overrides for V1-V4.

The paper does NOT publish the exact (depth, dim, head) configuration for
each variant — it only states that the encoder depths differ across V1-V4
and that 2/4 variants use OHEM-CTC + length penalty.

The values below are reasonable defaults that:
* Create a clear depth gradient (V1<V2<V3<V4) so the ensemble benefits from
  diverse capacities.
* Reserve OHEM + length penalty for V3 and V4 (the two stronger backbones),
  matching the "2/4 model" remark in Section 5.
* Stay within reach of a mid-range GPU (~6-8 GB at batch 32).

Override anything you like via the dataclass when launching training.
"""
from __future__ import annotations

from dataclasses import replace

from configs.icpr2026_base import ICPR2026Config


def variant_v1(base: ICPR2026Config | None = None) -> ICPR2026Config:
    base = base or ICPR2026Config()
    return replace(
        base,
        VARIANT="v1",
        EXPERIMENT_NAME="icpr2026_v1",
        ENCODER_LAYERS=2,
        AUX_TAP_LAYER=1,
        DECODER_LAYERS=0,           # V1 has no decoder (CTC head straight on encoder).
        USE_OHEM=False,
        USE_LENGTH_PENALTY=False,
    )


def variant_v2(base: ICPR2026Config | None = None) -> ICPR2026Config:
    base = base or ICPR2026Config()
    return replace(
        base,
        VARIANT="v2",
        EXPERIMENT_NAME="icpr2026_v2",
        ENCODER_LAYERS=4,
        AUX_TAP_LAYER=2,
        DECODER_LAYERS=2,
        USE_OHEM=False,
        USE_LENGTH_PENALTY=False,
    )


def variant_v3(base: ICPR2026Config | None = None) -> ICPR2026Config:
    base = base or ICPR2026Config()
    return replace(
        base,
        VARIANT="v3",
        EXPERIMENT_NAME="icpr2026_v3",
        ENCODER_LAYERS=6,
        AUX_TAP_LAYER=3,
        DECODER_LAYERS=2,
        USE_OHEM=True,              # OHEM for 2/4 variants -> V3 + V4
        USE_LENGTH_PENALTY=True,
    )


def variant_v4(base: ICPR2026Config | None = None) -> ICPR2026Config:
    base = base or ICPR2026Config()
    return replace(
        base,
        VARIANT="v4",
        EXPERIMENT_NAME="icpr2026_v4",
        ENCODER_LAYERS=6,           # "Encoder*" — different depth from V2/V3 path
        AUX_TAP_LAYER=3,
        DECODER_LAYERS=3,
        DECODER_NUM_QUERIES=16,
        USE_OHEM=True,
        USE_LENGTH_PENALTY=True,
    )


def variant_v5(base: ICPR2026Config | None = None) -> ICPR2026Config:
    """V5 — SVTR backbone + Factorized Temporal Attention."""
    base = base or ICPR2026Config()
    return replace(
        base,
        VARIANT="v5",
        EXPERIMENT_NAME="icpr2026_v5",
        D_MODEL=192,                # SVTR backbone output
        NHEAD=8,
        ENCODER_LAYERS=2,
        ENCODER_FF=768,
        AUX_TAP_LAYER=1,
        DECODER_LAYERS=0,
        USE_OHEM=False,
        USE_LENGTH_PENALTY=False,
    )


VARIANT_BUILDERS = {
    "v1": variant_v1,
    "v2": variant_v2,
    "v3": variant_v3,
    "v4": variant_v4,
    "v5": variant_v5,
}


def build_config(name: str, base: ICPR2026Config | None = None) -> ICPR2026Config:
    """``build_config("v3")`` -> a ready-to-use ICPR2026Config."""
    key = name.lower()
    if key not in VARIANT_BUILDERS:
        raise ValueError(f"Unknown variant {name!r}; choose from {list(VARIANT_BUILDERS)}")
    return VARIANT_BUILDERS[key](base)
