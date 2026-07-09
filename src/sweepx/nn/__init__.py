"""sweep-nn — PyTorch neural reparameterization and priors for seismic FWI."""

from __future__ import annotations

__version__ = "0.1.0"

from .reparam import Reparameterizer
from .siren import SIREN, SirenMLP, SineLayer
from .dip import DIPReparam
from .priors import LearnedPrior, SeabedFreezeMask, TVPrior
from .hash_encoding import MultiResHashGrid
from .growing_hash_grid import GrowingHashGrid
from .velocity_inr import VelocityINR
from .wavelet import SirenWavelet
from . import diffusion
from .diffusion import UNet2D, GaussianDiffusion, EMA

__all__ = [
    "Reparameterizer",
    "SIREN",
    "SirenMLP",
    "SineLayer",
    "DIPReparam",
    "LearnedPrior",
    "MultiResHashGrid",
    "GrowingHashGrid",
    "SeabedFreezeMask",
    "TVPrior",
    "VelocityINR",
    "SirenWavelet",
    "diffusion",
    "UNet2D",
    "GaussianDiffusion",
    "EMA",
    "__version__",
]
