"""Kronecker-factor helpers for Shampoo.

Shampoo views a parameter as a matrix ``(d1, d2)`` and keeps one ``d x d``
statistic per axis. The helpers here decide which axes get a factor, and
turn a factor into an inverse root robustly (SOAP follows the official
implementation instead, see ``soap.py``):

* the eigendecomposition runs in float64 and retries with a jitter when
  LAPACK/cuSOLVER fails to converge; if it still fails the caller keeps
  its previous factor instead of crashing the run;
* the damping ``eps`` is *relative* to the largest eigenvalue, so the
  preconditioner does not depend on the overall gradient scale (FWI
  gradients routinely have ``|G G^T| ~ 1e-10``, far below any fixed ridge);
* axes longer than ``max_precond_dim`` (e.g. a hash-grid table of shape
  ``(1e5, 1)``) get no factor, so no ``d x d`` matrix is ever allocated.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def as_matrix(t: Tensor) -> Tensor:
    """View ``t`` as 2-D: scalars ``(1, 1)``, vectors ``(n, 1)``, tensors ``(d0, -1)``."""
    if t.dim() == 0:
        return t.view(1, 1)
    if t.dim() == 1:
        return t.view(-1, 1)
    return t.reshape(t.shape[0], -1)


def factored_axes(d1: int, d2: int, max_precond_dim: int) -> Tuple[bool, bool]:
    """Which of the two axes carry a Kronecker factor (length > 1 and not too long)."""
    return (1 < d1 <= max_precond_dim, 1 < d2 <= max_precond_dim)


def _eigh(mat: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
    """float64 symmetric eigendecomposition with one jittered retry; ``None`` on failure."""
    m = 0.5 * (mat + mat.T).double()
    try:
        return torch.linalg.eigh(m)
    except torch.linalg.LinAlgError:
        pass
    n = m.shape[0]
    scale = float(m.diagonal().abs().max())
    if not (scale > 0.0 and scale < float("inf")):
        return None
    try:
        return torch.linalg.eigh(m + 1e-10 * scale * torch.eye(n, dtype=m.dtype, device=m.device))
    except torch.linalg.LinAlgError:
        return None


def inverse_root(mat: Tensor, p: float, eps: float) -> Optional[Tensor]:
    """float64 ``(mat + eps * lambda_max * I)^(-1/p)``, or ``None`` if ``mat`` is zero or eigh fails."""
    out = _eigh(mat)
    if out is None:
        return None
    w, v = out
    w = w.clamp(min=0.0)
    top = float(w.max())
    if not (top > 0.0 and top < float("inf")):
        return None
    inv = (w + eps * top).pow(-1.0 / p)
    # Kept in float64: the damped null-space entries are up to eps^(-1/p) times the
    # range entries, and applying them in float32 leaks rounding error into that
    # null space, where the next step amplifies it again.
    return (v * inv) @ v.T
