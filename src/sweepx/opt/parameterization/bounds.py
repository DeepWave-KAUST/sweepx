"""Reparameterizations that enforce physical bounds smoothly.

Three FWI-flavoured reparameterizations:

- :class:`SigmoidBounds` - latent in R, physical in (vmin, vmax) via sigmoid.
- :class:`LogVelocity` - latent = log(physical) for strictly positive models.
- :class:`PerturbationFromBackground` - latent = physical - background,
  optimizer iterates the perturbation rather than the absolute value.

Each is a :class:`Parameterization` and round-trips
``inverse(forward(latent)) == latent`` to numerical precision.
"""
from __future__ import annotations

import torch
from torch import Tensor

from sweep_opt.parameterization.base import Parameterization


class SigmoidBounds(Parameterization):
    """Smooth box constraint via a logistic sigmoid.

    physical = vmin + (vmax - vmin) * sigmoid(latent)

    The forward map is C-infinity and bijective onto (vmin, vmax). The
    inverse is the logit. Use this when you want bounds without the
    nondifferentiability of a hard projection (e.g. for L-BFGS without
    bound-handling, or for AD-friendly DIP architectures).
    """

    def __init__(self, vmin: float, vmax: float, eps: float = 1e-6) -> None:
        if not (vmin < vmax):
            raise ValueError(f"require vmin < vmax, got vmin={vmin}, vmax={vmax}")
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.span = self.vmax - self.vmin
        self.eps = eps

    def forward(self, latent: Tensor) -> Tensor:
        return self.vmin + self.span * torch.sigmoid(latent)

    def inverse(self, physical: Tensor) -> Tensor:
        u = (physical - self.vmin) / self.span
        u = u.clamp(self.eps, 1.0 - self.eps)
        return torch.log(u / (1.0 - u))


class LogVelocity(Parameterization):
    """Latent = log(physical). Requires physical > 0."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def forward(self, latent: Tensor) -> Tensor:
        return torch.exp(latent)

    def inverse(self, physical: Tensor) -> Tensor:
        return torch.log(physical.clamp(min=self.eps))


class PerturbationFromBackground(Parameterization):
    """Latent = physical - background; optimizer iterates dv.

    This is what most FWI workflows actually want: invert for the
    perturbation around a smooth starting model rather than the absolute
    velocity. The starting latent is therefore zero, and the optimizer
    sees a problem whose unperturbed forward already matches the data
    (or comes close).
    """

    def __init__(self, background: Tensor) -> None:
        self.background = background.detach().clone()

    def forward(self, latent: Tensor) -> Tensor:
        return self.background.to(latent.device, latent.dtype) + latent

    def inverse(self, physical: Tensor) -> Tensor:
        return physical - self.background.to(physical.device, physical.dtype)
