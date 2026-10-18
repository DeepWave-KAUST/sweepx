"""Matrix-free linear-algebra helpers (HVP, JVP/VJP, CG, Steihaug-CG)."""

from sweep_opt.linalg.cg import CGInfo, cg, steihaug_cg
from sweep_opt.linalg.hvp import gauss_newton_vector_product, hessian_vector_product

__all__ = [
    "CGInfo",
    "cg",
    "gauss_newton_vector_product",
    "hessian_vector_product",
    "steihaug_cg",
]
