"""Gaussian diffusion (DDPM/DDIM) — schedule, training loss, and sampling.

Self-contained, framework-agnostic wrapper around any noise-predicting network
with signature ``model(x_t, t) -> pred``.  Supports epsilon-, v-, and
x0-parameterisation; DDPM ancestral and deterministic DDIM sampling.  All
schedule quantities are registered as buffers so ``.to(device)`` / checkpoint
round-trips carry them along.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GaussianDiffusion", "EMA", "make_beta_schedule"]


def make_beta_schedule(kind: str, num_timesteps: int) -> torch.Tensor:
    """Return betas for ``num_timesteps`` steps.  kind in {"linear","cosine"}."""
    if kind == "linear":
        scale = 1000.0 / num_timesteps
        return torch.linspace(scale * 1e-4, scale * 2e-2, num_timesteps, dtype=torch.float64)
    if kind == "cosine":  # Nichol & Dhariwal (2021)
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        acp = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"unknown beta schedule {kind!r}")


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """Gather a[t] and broadcast to ``shape`` (B,1,1,1)."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        parameterization: str = "eps",  # "eps" | "v" | "x0"
    ):
        super().__init__()
        assert parameterization in ("eps", "v", "x0")
        self.num_timesteps = int(num_timesteps)
        self.parameterization = parameterization

        betas = make_beta_schedule(beta_schedule, num_timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)

        reg = lambda n, v: self.register_buffer(n, v.to(torch.float32))
        reg("betas", betas)
        reg("alphas_cumprod", acp)
        reg("alphas_cumprod_prev", acp_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(acp))
        reg("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))
        reg("sqrt_recip_acp", torch.sqrt(1.0 / acp))
        reg("sqrt_recipm1_acp", torch.sqrt(1.0 / acp - 1))
        post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        reg("posterior_variance", post_var)
        reg("posterior_log_var_clipped", torch.log(post_var.clamp(min=1e-20)))
        reg("posterior_mean_coef1", betas * torch.sqrt(acp_prev) / (1.0 - acp))
        reg("posterior_mean_coef2", (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp))

    # ---- forward (training) ----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise
        )

    def _target(self, x0, noise, t):
        if self.parameterization == "eps":
            return noise
        if self.parameterization == "x0":
            return x0
        # v-parameterization (Salimans & Ho, 2022)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * noise
            - _extract(self.sqrt_one_minus_acp, t, x0.shape) * x0
        )

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t)
        return F.mse_loss(pred, self._target(x0, noise, t))

    def loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device)
        return self.p_losses(model, x0, t)

    # ---- prediction helpers ----
    def _pred_x0_from(self, x_t, t, pred):
        if self.parameterization == "x0":
            x0 = pred
        elif self.parameterization == "eps":
            x0 = (
                _extract(self.sqrt_recip_acp, t, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_acp, t, x_t.shape) * pred
            )
        else:  # v
            x0 = (
                _extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
                - _extract(self.sqrt_one_minus_acp, t, x_t.shape) * pred
            )
        return x0

    def _pred_eps_from_x0(self, x_t, t, x0):
        return (
            _extract(self.sqrt_recip_acp, t, x_t.shape) * x_t - x0
        ) / _extract(self.sqrt_recipm1_acp, t, x_t.shape)

    # ---- DDPM ancestral sampling ----
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip: bool = True):
        pred = model(x_t, t)
        x0 = self._pred_x0_from(x_t, t, pred)
        if clip:
            x0 = x0.clamp(-1.0, 1.0)
        mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = _extract(self.posterior_log_var_clipped, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float().reshape(-1, *((1,) * (x_t.ndim - 1)))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, model, shape, device, clip: bool = True,
               progress: Optional[Callable[[int], None]] = None) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, clip=clip)
            if progress is not None:
                progress(i)
        return x

    # ---- DDIM (deterministic, fewer steps) ----
    @torch.no_grad()
    def ddim_sample(self, model, shape, device, num_steps: int = 50,
                    eta: float = 0.0, clip: bool = True) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        return self.ddim_sample_from(model, x, self.num_timesteps - 1, num_steps, eta, clip)

    @torch.no_grad()
    def ddim_sample_from(self, model, x: torch.Tensor, t_start: int, num_steps: int = 50,
                         eta: float = 0.0, clip: bool = True) -> torch.Tensor:
        """DDIM from an existing noisy state at ``t_start`` down to 0.

        ``ddim_sample`` always starts from pure noise at T-1.  Reconstruction QC
        starts part-way: noise a real sample to t, then walk it back and see how
        much of it comes home.  That signal appears long before unconditional
        samples stop looking like noise.
        """
        times = torch.linspace(int(t_start), 0, num_steps + 1, dtype=torch.long).tolist()
        for cur, nxt in zip(times[:-1], times[1:]):
            t = torch.full((x.shape[0],), cur, device=x.device, dtype=torch.long)
            pred = model(x, t)
            x0 = self._pred_x0_from(x, t, pred)
            if clip:
                x0 = x0.clamp(-1.0, 1.0)
            eps = self._pred_eps_from_x0(x, t, x0)
            a_next = self.alphas_cumprod[nxt]
            sigma = eta * torch.sqrt((1 - self.alphas_cumprod[cur] / a_next).clamp(min=0) *
                                     (1 - a_next) / (1 - self.alphas_cumprod[cur]).clamp(min=1e-20))
            x = torch.sqrt(a_next) * x0 + torch.sqrt((1 - a_next - sigma ** 2).clamp(min=0)) * eps
            if eta > 0 and nxt > 0:
                x = x + sigma * torch.randn_like(x)
        return x


class EMA:
    """Exponential moving average of model parameters, kept on the model device."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)
