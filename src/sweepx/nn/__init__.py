"""sweep-nn — PyTorch neural reparameterization and priors for seismic FWI."""

from __future__ import annotations

__version__ = "0.1.0"

from .reparam import Reparameterizer
from .siren import SIREN, SirenMLP, SineLayer
from .dip import DIPReparam
from .priors import LearnedPrior
from .hash_encoding import MultiResHashGrid
from .velocity_inr import VelocityINR
from .wavelet import SirenWavelet

__all__ = [
    "Reparameterizer",
    "SIREN",
    "SirenMLP",
    "SineLayer",
    "DIPReparam",
    "LearnedPrior",
    "MultiResHashGrid",
    "VelocityINR",
    "SirenWavelet",
    "__version__",
]
