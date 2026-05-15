"""Shared fixtures for sweep-opt unit tests.

These intentionally use small CPU tensors so the suite runs quickly and
hardware-independent. GPU-only / FWI-on-Marmousi checks live under
examples/marmousi/ instead.
"""
from __future__ import annotations

from typing import Callable, Tuple

import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Tiny quadratic: f(x) = 0.5 * x^T A x - b^T x  with SPD A.
# Unique minimizer x* = A^{-1} b.
# ---------------------------------------------------------------------------
@pytest.fixture
def quadratic_2d() -> Tuple[Callable[[Tensor], Tensor], Tensor, Tensor]:
    """Return (f, x_init, x_star) for a 2-D well-conditioned quadratic."""
    A = torch.tensor([[3.0, 0.5], [0.5, 2.0]])
    b = torch.tensor([1.0, -2.0])

    def f(x: Tensor) -> Tensor:
        return 0.5 * x @ A @ x - b @ x

    x_init = torch.zeros(2, requires_grad=True)
    x_star = torch.linalg.solve(A, b)
    return f, x_init, x_star


@pytest.fixture
def quadratic_illconditioned() -> Tuple[Callable[[Tensor], Tensor], Tensor, Tensor]:
    """A 5-D quadratic with condition number ~100 to stress first-order methods."""
    eigs = torch.tensor([1.0, 5.0, 20.0, 50.0, 100.0])
    A = torch.diag(eigs)
    b = torch.arange(1.0, 6.0)

    def f(x: Tensor) -> Tensor:
        return 0.5 * x @ A @ x - b @ x

    x_init = torch.zeros(5, requires_grad=True)
    x_star = b / eigs
    return f, x_init, x_star


# ---------------------------------------------------------------------------
# Rosenbrock: classic non-convex test, slow valley.
# Unique minimizer at (1, 1).
# ---------------------------------------------------------------------------
@pytest.fixture
def rosenbrock_2d() -> Tuple[Callable[[Tensor], Tensor], Tensor, Tensor]:
    """Return (f, x_init, x_star) for the 2-D Rosenbrock function."""
    def f(x: Tensor) -> Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    x_init = torch.tensor([-1.2, 1.0], requires_grad=True)
    x_star = torch.tensor([1.0, 1.0])
    return f, x_init, x_star


# ---------------------------------------------------------------------------
# Bounded quadratic: same as quadratic but with an active bound at the optimum.
# ---------------------------------------------------------------------------
@pytest.fixture
def bounded_quadratic_2d() -> Tuple[
    Callable[[Tensor], Tensor], Tensor, Tensor, Tuple[float, float]
]:
    """A 2-D quadratic whose unconstrained min is outside the box [0, 1]^2.

    Returns (f, x_init, x_star_constrained, (lo, hi)).
    """
    A = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([3.0, -1.0])

    def f(x: Tensor) -> Tensor:
        return 0.5 * x @ A @ x - b @ x

    x_init = torch.tensor([0.5, 0.5], requires_grad=True)
    # Unconstrained min is (1.5, -0.5); projected onto [0, 1]^2 -> (1.0, 0.0)
    x_star = torch.tensor([1.0, 0.0])
    return f, x_init, x_star, (0.0, 1.0)


# ---------------------------------------------------------------------------
# 1-D Born-style integrand fixture used by the pseudo-Hessian preconditioner test.
# Generates synthetic source / receiver illumination tensors.
# ---------------------------------------------------------------------------
@pytest.fixture
def born_illumination_1d() -> Tuple[Tensor, Tensor, Tensor]:
    """Return (raw_grad, source_illum, receiver_illum) on a 1-D depth axis.

    The expected preconditioned gradient is raw_grad / (source_illum * receiver_illum + eps),
    matching the Plessix-Mulder diagonal pseudo-Hessian construction.
    """
    nz = 64
    z = torch.linspace(0.0, 1.0, nz)
    raw_grad = torch.sin(2 * torch.pi * z)
    # Geometrical-spreading-like illumination: stronger near surface.
    source_illum = (1.0 + 5.0 * (1.0 - z)) ** 2
    receiver_illum = (1.0 + 5.0 * (1.0 - z)) ** 2
    return raw_grad, source_illum, receiver_illum


# ---------------------------------------------------------------------------
# Helpers shared by line-search tests.
# ---------------------------------------------------------------------------
@pytest.fixture
def make_closure():
    """Factory: build a torch.optim-compatible closure around f and a tensor param."""

    def _make(f: Callable[[Tensor], Tensor], param: Tensor) -> Callable[[], Tensor]:
        def closure() -> Tensor:
            if param.grad is not None:
                param.grad.zero_()
            loss = f(param)
            loss.backward()
            return loss

        return closure

    return _make
