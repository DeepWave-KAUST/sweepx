"""Denoising diffusion models for seismic velocity priors.

A self-contained DDPM/DDIM implementation (no external diffusion library) that
can be trained on a corpus of velocity models and later reused as a learned
prior for FWI (plug-and-play regularisation / posterior sampling).
"""

from __future__ import annotations

from .unet import UNet2D, UNet3D
from .ddpm import GaussianDiffusion, EMA, make_beta_schedule
from .prior import DiffusionVelocityPrior

__all__ = ["UNet2D", "UNet3D", "GaussianDiffusion", "EMA", "make_beta_schedule",
           "DiffusionVelocityPrior"]
