"""Gradient preconditioners (pseudo-Hessian, depth weighting, ...)."""

from sweepx.opt.precond.base import IdentityPrecond, Preconditioner
from sweepx.opt.precond.depth_weighting import Compose, DepthWeighting
from sweepx.opt.precond.pseudo_hessian import PseudoHessianPrecond

__all__ = [
    "Compose",
    "DepthWeighting",
    "IdentityPrecond",
    "Preconditioner",
    "PseudoHessianPrecond",
]
