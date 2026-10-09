"""sweep-nn — PyTorch neural reparameterization and priors for seismic FWI."""

from __future__ import annotations

__version__ = "0.1.1"

from .reparam import Reparameterizer
from .siren import SIREN, SirenMLP, SineLayer
from .dip import DIPReparam
from .priors import LearnedPrior, SeabedFreezeMask, TVPrior
from .hash_encoding import MultiResHashGrid
from .triton_hash_encoding import have_triton
from .growing_hash_grid import GrowingHashGrid
from .velocity_inr import VelocityINR
from .multi_param_inr import MultiParamINR
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
    "have_triton",
    "GrowingHashGrid",
    "SeabedFreezeMask",
    "TVPrior",
    "VelocityINR",
    "MultiParamINR",
    "SirenWavelet",
    "diffusion",
    "UNet2D",
    "GaussianDiffusion",
    "EMA",
    "__version__",
]
