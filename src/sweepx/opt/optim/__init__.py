"""First-order, quasi-Newton, Newton-type, and modern adaptive optimizers for FWI."""

from sweep_opt.optim.adam_fwi import AdamFWI
from sweep_opt.optim.gauss_newton import TruncatedGaussNewton
from sweep_opt.optim.lbfgs import LBFGS
from sweep_opt.optim.lbfgsb import LBFGSB
from sweep_opt.optim.lion import Lion
from sweep_opt.optim.ncg import NCG
from sweep_opt.optim.newton import TruncatedNewton
from sweep_opt.optim.sam import SAM
from sweep_opt.optim.shampoo import Shampoo
from sweep_opt.optim.soap import SOAP
from sweep_opt.optim.steepest_descent import SteepestDescent
from sweep_opt.optim.trust_region import TrustRegion

__all__ = [
    "AdamFWI",
    "LBFGS",
    "LBFGSB",
    "Lion",
    "NCG",
    "SAM",
    "SOAP",
    "Shampoo",
    "SteepestDescent",
    "TrustRegion",
    "TruncatedGaussNewton",
    "TruncatedNewton",
]
