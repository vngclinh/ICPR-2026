"""Residual Gaussian diffusion wrapper used by LP-Diff.

Differences from the upstream reference:
- The MTA module is *not* owned by this class. The training script feeds the
  precomputed MTA condition directly into ``p_losses`` / ``super_resolution``,
  so the same diffusion can be reused with different conditioning modules
  during ablation.
- A DDIM sampler is provided so inference can run in 20-50 steps instead of
  1000 — critical for offline preprocessing on a mid-range GPU.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def _exists(x) -> bool:
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if callable(d) else d


def make_beta_schedule(
    schedule: str, n_timestep: int,
    linear_start: float = 1e-4, linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    if schedule == "linear":
        return np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    if schedule == "quad":
        return np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    if schedule == "const":
        return linear_end * np.ones(n_timestep, dtype=np.float64)
    if schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        return betas.clamp(max=0.999).numpy()
    raise NotImplementedError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    """Residual DDPM trainer/sampler.

    Conditioning: the residual target is ``x_start = HR - condition`` where the
    condition is supplied by the caller (typically MTA(LR1,LR2,LR3)).
    """

    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: int,
        channels: int = 3,
        loss_type: str = "l1",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.channels = channels
        self.loss_type = loss_type
        self.num_timesteps: int = 0

    def set_loss(self, device: torch.device) -> None:
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="sum").to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="sum").to(device)
        else:
            raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")

    def set_new_noise_schedule(self, schedule_opt: dict, device: torch.device) -> None:
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
            linear_start=schedule_opt["linear_start"],
            linear_end=schedule_opt["linear_end"],
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        # sqrt_alphas_cumprod_prev[t] gives the noise level used during training
        # (kept as a numpy array to support random uniform sampling between
        # consecutive timesteps as in the upstream implementation).
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: int, noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t] * noise
        )

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start
            + self.posterior_mean_coef2[t] * x_t
        )
        return posterior_mean, self.posterior_log_variance_clipped[t]

    def p_mean_variance(
        self, x: torch.Tensor, t: int, clip_denoised: bool, condition_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]],
        ).repeat(batch_size, 1).to(x.device)
        x_recon = self.predict_start_from_noise(
            x, t=t,
            noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level),
        )
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, t: int, condition_x: torch.Tensor, clip_denoised: bool = True,
    ) -> torch.Tensor:
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x,
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, condition: torch.Tensor) -> torch.Tensor:
        """Full DDPM sampling (1000 steps if num_timesteps=1000).

        Returns the predicted HR = condition + denoised_residual.
        """
        img = torch.randn_like(condition)
        for i in tqdm(
            reversed(range(self.num_timesteps)),
            desc="DDPM sampling", total=self.num_timesteps, leave=False,
        ):
            img = self.p_sample(img, i, condition_x=condition)
        return img + condition

    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling (Song et al. 2021). Much faster than the DDPM loop.

        ``eta=0`` is deterministic (recommended for offline preprocessing so the
        cached SR images are reproducible).
        """
        assert self.num_timesteps > 0, "Call set_new_noise_schedule() first."
        device = condition.device
        num_steps = min(num_steps, self.num_timesteps)

        # Uniformly spaced timestep indices into [0, num_timesteps - 1].
        step_indices = np.linspace(0, self.num_timesteps - 1, num_steps, dtype=np.int64)
        step_indices = step_indices[::-1]  # reverse for sampling (t -> 0)

        img = torch.randn_like(condition)
        for idx, t in enumerate(tqdm(step_indices, desc="DDIM sampling", leave=False)):
            batch_size = img.shape[0]
            noise_level = torch.full(
                (batch_size, 1), float(self.sqrt_alphas_cumprod_prev[t + 1]),
                device=device,
            )
            eps = self.denoise_fn(torch.cat([condition, img], dim=1), noise_level)

            alpha_cumprod_t = self.alphas_cumprod[t]
            x0 = (img - torch.sqrt(1.0 - alpha_cumprod_t) * eps) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)

            if idx == len(step_indices) - 1:
                img = x0
            else:
                t_prev = step_indices[idx + 1]
                alpha_cumprod_prev = self.alphas_cumprod[t_prev]
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
                    * (1.0 - alpha_cumprod_t / alpha_cumprod_prev)
                )
                dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev - sigma * sigma) * eps
                noise = torch.randn_like(img) if eta > 0 else 0.0
                img = torch.sqrt(alpha_cumprod_prev) * x0 + dir_xt + sigma * noise

        return img + condition

    @torch.no_grad()
    def super_resolution(
        self, condition: torch.Tensor, sampler: str = "ddim", num_steps: int = 50,
    ) -> torch.Tensor:
        if sampler == "ddpm":
            return self.p_sample_loop(condition)
        if sampler == "ddim":
            return self.ddim_sample(condition, num_steps=num_steps)
        raise ValueError(f"Unknown sampler: {sampler}")

    def q_sample(
        self, x_start: torch.Tensor, continuous_sqrt_alpha_cumprod: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = _default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start
            + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses(
        self,
        hr: torch.Tensor,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Residual diffusion loss: target residual = HR - condition."""
        x_start = hr - condition
        b = x_start.shape[0]
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b,
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = _default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise,
        )
        x_recon = self.denoise_fn(
            torch.cat([condition, x_noisy], dim=1), continuous_sqrt_alpha_cumprod,
        )
        return self.loss_func(noise, x_recon)
