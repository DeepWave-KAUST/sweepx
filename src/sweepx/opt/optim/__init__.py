"""First-order, quasi-Newton, Newton-type, and modern adaptive optimizers for FWI."""

from sweepx.opt.optim.adam_fwi import AdamFWI
from sweepx.opt.optim.gauss_newton import TruncatedGaussNewton
from sweepx.opt.optim.lbfgs import LBFGS
from sweepx.opt.optim.lbfgsb import LBFGSB
from sweepx.opt.optim.lion import Lion
from sweepx.opt.optim.ncg import NCG
from sweepx.opt.optim.newton import TruncatedNewton
from sweepx.opt.optim.sam import SAM
from sweepx.opt.optim.shampoo import Shampoo
from sweepx.opt.optim.soap import SOAP
from sweepx.opt.optim.steepest_descent import SteepestDescent
from sweepx.opt.optim.trust_region import TrustRegion

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
