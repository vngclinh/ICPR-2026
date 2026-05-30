"""Base configuration for the ICPR 2026 LPR re-implementation.

This is a separate dataclass from ``configs/config.py`` so it does NOT break
the legacy ``restran`` / SR training pipeline. Variant-specific overrides
live in ``configs/icpr2026_variants.py``.

Hyperparameters not specified by the paper are flagged in comments and
defaulted to sensible values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class ICPR2026Config:
    """Hyperparameters for one variant + shared training settings."""

    # ----- Variant -----
    VARIANT: str = "v1"                 # one of {"v1", "v2", "v3", "v4"}
    EXPERIMENT_NAME: str = "icpr2026_v1"

    # ----- Data paths (mirrors configs/config.py) -----
    DATA_ROOT: str = "data/LRLPR-26-5opEvJTW/train"
    TEST_DATA_ROOT: str = "data/LRLPR-26-5opEvJTW/test"
    # Validation = Scenario-B only, 90/10 split (Section 7).
    VAL_SCENARIO: str = "Scenario-B"
    VAL_SPLIT_FILE: str = "data/LRLPR-26-5opEvJTW/val_tracks_scenarioB.json"
    SUBMISSION_FILE: str = "submission_icpr2026.txt"

    # ----- Input -----
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    NUM_FRAMES: int = 5
    HR_HEIGHT: int = 64
    HR_WIDTH: int = 256
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # ----- Training -----
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30                    # Section 7: train from scratch, 30 epochs
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SEED: int = 42
    NUM_WORKERS: int = 4
    SPLIT_RATIO: float = 0.9            # 90/10 on Scenario-B
    USE_CUDNN_BENCHMARK: bool = True

    # OneCycleLR (Section 7).
    SCHEDULER: str = "onecycle"
    PCT_START: float = 0.1
    DIV_FACTOR: float = 25.0
    FINAL_DIV_FACTOR: float = 1e4

    # ----- STN -----
    USE_STN: bool = True
    STN_MODE: str = "tps"               # {"affine", "tps"}; paper unspecified
    STN_TPS_X: int = 10
    STN_TPS_Y: int = 2

    # ----- Encoder / decoder defaults (per-variant overrides in icpr2026_variants.py) -----
    D_MODEL: int = 512
    NHEAD: int = 8
    ENCODER_LAYERS: int = 4
    ENCODER_FF: int = 2048
    ENCODER_DROPOUT: float = 0.1
    AUX_TAP_LAYER: int = 2
    DECODER_LAYERS: int = 2
    DECODER_FF: int = 2048
    DECODER_DROPOUT: float = 0.1
    DECODER_NUM_QUERIES: int = 16
    FUSION_PER_POSITION: bool = True

    # ----- Loss weights (Section 5; values not given in paper) -----
    LAMBDA_CTC: float = 1.0
    LAMBDA_AUX_CTC: float = 0.2
    LAMBDA_CENTER: float = 0.05
    LAMBDA_STN: float = 0.01
    USE_OHEM: bool = False              # toggled True for 2/4 variants
    OHEM_TOP_K: float = 0.7
    LAMBDA_OHEM: float = 0.5
    USE_LENGTH_PENALTY: bool = False    # toggled True for some variants
    LAMBDA_LENGTH: float = 0.05
    TARGET_PLATE_LENGTH: int = 7

    # ----- Augmentation -----
    DEGRADATION_DOUBLE: bool = True     # Section 7: synth-degrade HR to double train set

    # ----- Misc -----
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    OUTPUT_DIR: str = "results"
    TENSORBOARD_LOG_DIR: str | None = None

    # ----- Derived -----
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.CHAR2IDX = {ch: i + 1 for i, ch in enumerate(self.CHARS)}
        self.IDX2CHAR = {i + 1: ch for i, ch in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank
