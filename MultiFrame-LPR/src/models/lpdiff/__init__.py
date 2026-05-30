"""LP-Diff: diffusion-based license plate super-resolution.

Port of CVPR 2025 paper "LP-Diff: Towards Improved Restoration of Real-World
Degraded License Plate" (Gong et al.), adapted to this codebase. The original
implementation lives at https://github.com/haoyGONG/LP-Diff.

The diffusion model learns the *residual* between the HR ground truth and a
multi-frame fused estimate produced by the MTA module. Inference adds the
residual back to the MTA output.
"""
from src.models.lpdiff.mta import MTA
from src.models.lpdiff.unet_diff import DiffusionUNet
from src.models.lpdiff.diffusion import GaussianDiffusion
from src.models.lpdiff.lpdiff_net import LPDiffNet

__all__ = ["MTA", "DiffusionUNet", "GaussianDiffusion", "LPDiffNet"]
