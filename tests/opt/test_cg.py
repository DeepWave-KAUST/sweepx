"""Tests for sweep_opt.linalg.cg and steihaug_cg."""
import torch

from sweep_opt.linalg import cg, steihaug_cg


def _matvec(A: torch.Tensor):
    return lambda v: A @ v


def test_cg_solves_spd_system():
    """Theoretically <= n iters; allow a small float-precision slack."""
    torch.manual_seed(0)
    n = 8
    M = torch.randn(n, n, dtype=torch.float64)
    A = M @ M.T + 0.5 * torch.eye(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    x, info = cg(_matvec(A), b, tol=1e-10)
    assert info.converged
    assert info.iters <= n
    assert torch.allclose(A @ x, b, atol=1e-8)


def test_cg_warmstart_x0():
    n = 5
    A = torch.eye(n) * 4.0
    b = torch.arange(1.0, n + 1.0)
    x0 = b / 4.0  # exact solution; should converge in 0 iters.
    x, info = cg(_matvec(A), b, x0=x0, tol=1e-10)
    assert info.iters == 0
    assert torch.allclose(x, x0)


def test_cg_with_preconditioner():
    """A perfect preconditioner (M = A^{-1}) collapses CG to one iteration."""
    A = torch.tensor([[3.0, 0.0], [0.0, 100.0]])
    b = torch.tensor([1.0, 1.0])
    A_inv = torch.linalg.inv(A)

    def precond(r):
        return A_inv @ r

    x, info = cg(_matvec(A), b, tol=1e-10, preconditioner=precond)
    assert info.iters <= 1
    assert torch.allclose(x, A_inv @ b, atol=1e-6)


def test_steihaug_returns_inner_solution_when_within_radius():
    """If the unconstrained solution lies inside the trust radius, return it."""
    A = torch.eye(3) * 4.0
    b = torch.tensor([1.0, 1.0, 1.0])
    # Inner solution norm = ||A^{-1} b|| = ||(0.25, 0.25, 0.25)|| ~= 0.433
    x, info = steihaug_cg(_matvec(A), b, trust_radius=2.0, tol=1e-10)
    assert not info.boundary
    assert torch.allclose(x, b / 4.0, atol=1e-6)


def test_steihaug_hits_boundary_on_negative_curvature():
    """A direction of negative curvature should send Steihaug to the boundary."""
    A = torch.diag(torch.tensor([1.0, -1.0]))  # indefinite
    b = torch.tensor([1.0, 1.0])
    x, info = steihaug_cg(_matvec(A), b, trust_radius=1.0)
    assert info.boundary
    assert info.negative_curvature
    assert abs(float(x.norm()) - 1.0) < 1e-6


def test_steihaug_hits_boundary_when_inner_solution_too_large():
    """When the unconstrained min lies outside the trust radius, hit the boundary."""
    A = torch.eye(3) * 1.0
    b = torch.tensor([5.0, 0.0, 0.0])
    x, info = steihaug_cg(_matvec(A), b, trust_radius=1.0, tol=1e-10)
    assert info.boundary
    assert abs(float(x.norm()) - 1.0) < 1e-6
