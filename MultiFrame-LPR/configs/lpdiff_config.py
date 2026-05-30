"""Standalone training config for LP-Diff super-resolution.

Defaults are tuned for a mid-range GPU (~8-12 GB VRAM):
  * inner_channel halved to 32 (vs upstream 64).
  * 4 stages instead of 5 (channel_mults [1,2,4,4]).
  * batch_size 4 by default; bump up if VRAM allows.
  * DDIM 50-step sampling at inference (vs DDPM 1000 in the paper).
"""
from dataclasses import dataclass, field
from typing import Tuple

import torch


@dataclass
class LPDiffConfig:
    # Data
    DATASET_ROOT: str = "data/LRLPR-26-5opEvJTW"
    DATA_ROOT: str = "data/LRLPR-26-5opEvJTW/train"
    TEST_DATA_ROOT: str = "data/LRLPR-26-5opEvJTW/test"
    VAL_SPLIT_FILE: str = "data/LRLPR-26-5opEvJTW/val_tracks.json"

    # Resolution. LR is bicubic-upscaled to (HR_HEIGHT, HR_WIDTH) before MTA.
    # Must be divisible by 2 ** (num_stages - 1). For [1,2,4,4] -> 8.
    HR_HEIGHT: int = 64
    HR_WIDTH: int = 256

    # Frame selection. MTA expects 3 frames; the dataset gives us 5 so we
    # pick indices below. Defaults to first / middle / last for diversity.
    FRAME_INDICES: Tuple[int, int, int] = (0, 2, 4)

    # Diffusion U-Net architecture
    INNER_CHANNEL: int = 32
    NORM_GROUPS: int = 32
    CHANNEL_MULTS: Tuple[int, ...] = (1, 2, 4, 4)
    ATTN_RES: Tuple[int, ...] = (16,)
    RES_BLOCKS: int = 2
    DROPOUT: float = 0.1

    # Beta schedule (linear in [1e-6, 1e-2] over 1000 steps -- matches upstream)
    BETA_SCHEDULE: str = "linear"
    N_TIMESTEP: int = 1000
    LINEAR_START: float = 1e-6
    LINEAR_END: float = 1e-2

    # Training
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4  # Upstream uses 5e-3 but Adam with our small UNet diverges
    N_ITERATIONS: int = 60_000
    NUM_WORKERS: int = 2
    SEED: int = 42
    PRINT_FREQ: int = 100
    SAVE_FREQ: int = 5_000
    VAL_FREQ: int = 5_000
    GRAD_CLIP: float = 1.0
    EMA_DECAY: float = 0.9999
    USE_EMA: bool = True
    USE_AMP: bool = True  # Mixed precision; turn off if loss explodes.

    # Inference / DDIM
    DDIM_STEPS: int = 50

    # Output
    OUTPUT_DIR: str = "results/lpdiff"
    EXPERIMENT_NAME: str = "lpdiff_v1"

    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    @property
    def beta_schedule_dict(self) -> dict:
        return {
            "schedule": self.BETA_SCHEDULE,
            "n_timestep": self.N_TIMESTEP,
            "linear_start": self.LINEAR_START,
            "linear_end": self.LINEAR_END,
        }


def get_default_config() -> LPDiffConfig:
    return LPDiffConfig()
