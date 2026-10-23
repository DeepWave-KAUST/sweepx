"""Strong-Wolfe line search.

Reference: Nocedal, J., and Wright, S. J. (2006), Numerical Optimization, 2nd ed.,
Springer, Algorithms 3.5 (search) and 3.6 (zoom). DOI:10.1007/978-0-387-40065-5.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _zoom(
    eval_fn: Callable[[float], Tuple[float, float]],
    alpha_lo: float,
    phi_lo: float,
    alpha_hi: float,
    init_loss: float,
    init_dphi: float,
    c1: float,
    c2: float,
    max_iter: int,
) -> Tuple[float, float]:
    """Zoom subroutine (Nocedal & Wright Alg 3.6) using bisection interpolation."""
    for _ in range(max_iter):
        alpha_j = 0.5 * (alpha_lo + alpha_hi)
        phi_j, dphi_j = eval_fn(alpha_j)
        # A non-finite trial (NaN compares False everywhere) counts as an
        # Armijo failure, so the bracket shrinks toward alpha_lo.
        if (not _finite(phi_j, dphi_j)
                or phi_j > init_loss + c1 * alpha_j * init_dphi or phi_j >= phi_lo):
            alpha_hi = alpha_j
        else:
            if abs(dphi_j) <= -c2 * init_dphi:
                return alpha_j, phi_j
            if dphi_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
            phi_lo = phi_j
        # Relative: alpha carries the units of 1 / |d|, anything from 1e-12 to 1e8.
        if abs(alpha_hi - alpha_lo) <= 1e-12 * max(abs(alpha_hi), abs(alpha_lo)):
            break
    return alpha_lo, phi_lo


def strong_wolfe(
    eval_fn: Callable[[float], Tuple[float, float]],
    init_loss: float,
    init_dphi: float,
    *,
    alpha_init: float = 1.0,
    alpha_max: float = 10.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 25,
) -> Tuple[float, float]:
    """Strong-Wolfe line search.

    Args:
        eval_fn: callable mapping alpha -> (phi(alpha), phi'(alpha)) where
            phi(alpha) = f(x + alpha d) and phi'(alpha) = grad f(x + alpha d) . d.
            A non-finite value or slope (e.g. a solver driven unstable by a
            long trial step) is treated as a failed sufficient-decrease test:
            the search shrinks the step and never accepts such a point.
        init_loss: phi(0). Must be finite.
        init_dphi: phi'(0). Must be finite and strictly negative (descent
            direction).
        alpha_init: first trial step size (typically 1.0 for Newton-like
            methods). ``alpha`` multiplies the search direction, so it carries
            the units of ``1 / |d|``.
        alpha_max: maximum step size.
        c1, c2: Wolfe constants. Must satisfy 0 < c1 < c2 < 1; defaults
            (1e-4, 0.9) follow the Nocedal & Wright recommendation for
            quasi-Newton / Newton methods.
        max_iter: maximum bracket / zoom iterations.

    Returns:
        (alpha_star, phi(alpha_star)). ``alpha_star`` is the best accepted
        step, ``0.0`` if no trial decreased phi enough. It is not necessarily
        the last point ``eval_fn`` was called at, so the caller must move the
        parameters to ``x + alpha_star d`` itself.
    """
    if not _finite(init_loss, init_dphi):
        raise ValueError(
            f"line search needs a finite phi(0) and phi'(0); got phi(0)={init_loss}, "
            f"phi'(0)={init_dphi} (the loss or the gradient at the current "
            "parameters is not finite)"
        )
    if init_dphi >= 0.0:
        raise ValueError(
            f"line search requires a descent direction (phi'(0) < 0); got {init_dphi}"
        )
    if not (0.0 < c1 < c2 < 1.0):
        raise ValueError(f"require 0 < c1 < c2 < 1; got c1={c1}, c2={c2}")

    alpha_prev = 0.0
    phi_prev = init_loss
    alpha_i = alpha_init

    for i in range(max_iter):
        phi_i, dphi_i = eval_fn(alpha_i)
        if (not _finite(phi_i, dphi_i)
                or phi_i > init_loss + c1 * alpha_i * init_dphi
                or (i > 0 and phi_i >= phi_prev)):
            return _zoom(
                eval_fn, alpha_prev, phi_prev, alpha_i,
                init_loss, init_dphi, c1, c2, max_iter,
            )
        if abs(dphi_i) <= -c2 * init_dphi:
            return alpha_i, phi_i
        if dphi_i >= 0.0:
            return _zoom(
                eval_fn, alpha_i, phi_i, alpha_prev,
                init_loss, init_dphi, c1, c2, max_iter,
            )
        alpha_prev = alpha_i
        phi_prev = phi_i
        if alpha_i >= alpha_max * (1.0 - 1e-12):
            return alpha_i, phi_i
        alpha_i = min(2.0 * alpha_i, alpha_max)

    # Out of iterations: the last point that was evaluated and accepted.
    return alpha_prev, phi_prev
