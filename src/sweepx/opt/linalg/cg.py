"""Matrix-free conjugate gradient solvers.

Two flavours:

- :func:`cg` - standard preconditioned CG for SPD systems ``A x = b``,
  given a callable ``A v`` (matvec) and an optional preconditioner.
- :func:`steihaug_cg` - truncated CG inside a trust radius (Steihaug 1983),
  used by trust-region Newton-type methods. Returns the trust-boundary
  point if the iteration encounters non-positive curvature or escapes the
  trust radius.

Inputs / outputs are flat 1-D tensors. Optimizers that hold tensor
parameters should flatten / unflatten via the helpers in
:mod:`sweepx.opt._utils`.

References:
    Hestenes, M. R., and Stiefel, E. (1952). "Methods of conjugate gradients
    for solving linear systems." Journal of Research of the National Bureau
    of Standards 49(6), 409-436.
    DOI:10.6028/jres.049.044.

    Steihaug, T. (1983). "The conjugate gradient method and trust regions in
    large scale optimization." SIAM Journal on Numerical Analysis 20(3),
    626-637. DOI:10.1137/0720042.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor


@dataclass
class CGInfo:
    """Convergence info returned by the CG solvers."""

    iters: int
    residual: float
    converged: bool
    boundary: bool = False  # only meaningful for steihaug_cg
    negative_curvature: bool = False  # only meaningful for steihaug_cg


def cg(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    x0: Optional[Tensor] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    preconditioner: Optional[Callable[[Tensor], Tensor]] = None,
) -> tuple[Tensor, CGInfo]:
    """Solve ``A x = b`` via preconditioned conjugate gradient.

    Args:
        matvec: callable returning ``A v`` for a 1-D vector ``v``. ``A`` must
            be symmetric positive (semi-)definite.
        b: right-hand side, 1-D tensor.
        x0: initial guess (default zeros).
        tol: stopping tolerance on relative residual ``||r|| / ||b||``.
        max_iter: maximum iterations (default ``10 * b.numel()``).
        preconditioner: optional callable ``r -> M^{-1} r``. Default identity.

    Returns:
        ``(x, info)``.
    """
    if b.dim() != 1:
        raise ValueError(f"b must be 1-D, got shape {tuple(b.shape)}")
    n = b.numel()
    if max_iter is None:
        max_iter = 10 * n

    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    r = b - matvec(x) if x0 is not None else b.clone()
    b_norm = float(b.norm())
    if b_norm == 0.0:
        return x, CGInfo(iters=0, residual=0.0, converged=True)
    if float(r.norm()) / b_norm < tol:
        return x, CGInfo(iters=0, residual=float(r.norm()) / b_norm, converged=True)
    z = preconditioner(r) if preconditioner is not None else r
    p = z.clone()
    rz_old = float(torch.dot(r, z))

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(torch.dot(p, Ap))
        if pAp <= 0.0:
            # Should not happen for SPD; bail out with current iterate.
            return x, CGInfo(
                iters=k + 1, residual=float(r.norm()) / b_norm,
                converged=False, negative_curvature=True,
            )
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residual = float(r.norm()) / b_norm
        if residual < tol:
            return x, CGInfo(iters=k + 1, residual=residual, converged=True)
        z = preconditioner(r) if preconditioner is not None else r
        rz_new = float(torch.dot(r, z))
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x, CGInfo(
        iters=max_iter, residual=float(r.norm()) / b_norm, converged=False,
    )


def steihaug_cg(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    trust_radius: float,
    *,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    preconditioner: Optional[Callable[[Tensor], Tensor]] = None,
) -> tuple[Tensor, CGInfo]:
    """Truncated CG inside a trust radius (Steihaug 1983).

    Solves the trust-region subproblem ``min 0.5 x^T A x - b^T x``
    subject to ``||x|| <= trust_radius`` approximately. Stops when:

    - the residual is small enough (regular CG convergence), or
    - the iteration generates a direction of non-positive curvature, in
      which case it returns the boundary point along that direction, or
    - the iterate would leave the trust region, in which case it returns
      the boundary point along the current search direction.

    Args:
        matvec: callable returning ``A v``.
        b: right-hand side.
        trust_radius: positive trust radius.
        tol, max_iter, preconditioner: as in :func:`cg`.

    Returns:
        ``(x, info)``.
    """
    if trust_radius <= 0.0:
        raise ValueError(f"trust_radius must be positive, got {trust_radius}")
    if b.dim() != 1:
        raise ValueError(f"b must be 1-D, got shape {tuple(b.shape)}")
    n = b.numel()
    if max_iter is None:
        max_iter = 10 * n

    x = torch.zeros_like(b)
    r = b.clone()
    z = preconditioner(r) if preconditioner is not None else r
    p = z.clone()
    rz_old = float(torch.dot(r, z))
    b_norm = float(b.norm())
    if b_norm == 0.0:
        return x, CGInfo(iters=0, residual=0.0, converged=True)

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(torch.dot(p, Ap))
        if pAp <= 0.0:
            tau = _steihaug_boundary_tau(x, p, trust_radius)
            return x + tau * p, CGInfo(
                iters=k + 1, residual=float(r.norm()) / b_norm,
                converged=False, boundary=True, negative_curvature=True,
            )
        alpha = rz_old / pAp
        x_new = x + alpha * p
        if float(x_new.norm()) >= trust_radius:
            tau = _steihaug_boundary_tau(x, p, trust_radius)
            return x + tau * p, CGInfo(
                iters=k + 1, residual=float(r.norm()) / b_norm,
                converged=False, boundary=True,
            )
        r = r - alpha * Ap
        residual = float(r.norm()) / b_norm
        if residual < tol:
            return x_new, CGInfo(iters=k + 1, residual=residual, converged=True)
        z = preconditioner(r) if preconditioner is not None else r
        rz_new = float(torch.dot(r, z))
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        x = x_new

    return x, CGInfo(
        iters=max_iter, residual=float(r.norm()) / b_norm, converged=False,
    )


def _steihaug_boundary_tau(x: Tensor, p: Tensor, trust_radius: float) -> float:
    """Find tau >= 0 such that ||x + tau p|| = trust_radius."""
    a = float(torch.dot(p, p))
    bb = 2.0 * float(torch.dot(x, p))
    c = float(torch.dot(x, x)) - trust_radius * trust_radius
    disc = max(bb * bb - 4.0 * a * c, 0.0)
    return (-bb + disc**0.5) / (2.0 * a)
