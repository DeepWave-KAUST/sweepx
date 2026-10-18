"""Tests for sweep_opt.linalg.hessian_vector_product / gauss_newton_vector_product."""
import torch

from sweep_opt.linalg import gauss_newton_vector_product, hessian_vector_product


def test_hvp_matches_explicit_hessian_on_quadratic():
    """f(x) = 0.5 x^T A x - b^T x  =>  H = A constant."""
    A = torch.tensor([[3.0, 0.5], [0.5, 2.0]])
    b = torch.tensor([1.0, -2.0])
    x = torch.tensor([0.7, -1.3], requires_grad=True)

    def loss():
        return 0.5 * x @ A @ x - b @ x

    for v in [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
              torch.tensor([0.5, -0.7])]:
        hv, = hessian_vector_product(loss, [x], [v])
        expected = A @ v
        assert torch.allclose(hv, expected, atol=1e-6)


def test_hvp_matches_finite_difference_on_rosenbrock():
    """HVP via autograd should match a centered finite-difference HVP."""
    x = torch.tensor([-0.7, 1.4], requires_grad=True)

    def loss():
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    v = torch.tensor([0.4, -0.6])
    hv, = hessian_vector_product(loss, [x], [v])

    eps = 1e-3
    # Finite difference: H v ~ (grad(x + eps v) - grad(x - eps v)) / (2 eps).
    def grad_at(point: torch.Tensor) -> torch.Tensor:
        p = point.clone().requires_grad_(True)
        l = (1.0 - p[0]) ** 2 + 100.0 * (p[1] - p[0] ** 2) ** 2
        (g,) = torch.autograd.grad(l, p)
        return g.detach()

    g_plus = grad_at(x.detach() + eps * v)
    g_minus = grad_at(x.detach() - eps * v)
    fd_hv = (g_plus - g_minus) / (2.0 * eps)
    # FD approximation of second derivative is O(eps^2). For Rosenbrock with
    # cubic + quartic terms, the truncation error at eps=1e-3 is ~ ||v||^2 * eps^2 ~ 1e-6
    # multiplied by the leading coefficient ~ 800, so ~1e-3 absolute tolerance.
    assert torch.allclose(hv, fd_hv, atol=2e-2)


def test_hvp_multi_param():
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([3.0, -1.0], requires_grad=True)

    def loss():
        return (a * b).sum() + 0.5 * (a ** 2).sum() + (b ** 2).sum()

    va = torch.tensor([0.3, 0.4])
    vb = torch.tensor([0.5, -0.2])
    ha, hb = hessian_vector_product(loss, [a, b], [va, vb])
    # H = [[I, I], [I, 2 I]] => H [va; vb] = [va + vb, va + 2 vb]
    assert torch.allclose(ha, va + vb, atol=1e-6)
    assert torch.allclose(hb, va + 2.0 * vb, atol=1e-6)


def test_gnvp_matches_jt_j_on_linear_residual():
    """For a linear residual r = M x - y, J = M, so J^T J vec = M^T M vec."""
    M = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([0.5, -0.5, 1.0])
    x = torch.tensor([0.7, -1.3], requires_grad=True)

    def residual():
        return M @ x - y

    v = torch.tensor([0.3, -0.4])
    gnv, = gauss_newton_vector_product(residual, [x], [v])
    expected = M.T @ (M @ v)
    assert torch.allclose(gnv, expected, atol=1e-6)
