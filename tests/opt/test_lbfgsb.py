"""Tests for sweepx.opt.optim.LBFGSB."""
import pytest
import torch

from sweepx.opt.optim import LBFGSB


def _run(opt, closure, n_iters):
    losses = []
    for _ in range(n_iters):
        loss = opt.step(closure)
        losses.append(loss.item())
    return losses


def test_lbfgsb_requires_bounds():
    x = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError):
        LBFGSB([x], bounds=None)


def test_lbfgsb_requires_closure():
    x = torch.zeros(2, requires_grad=True)
    opt = LBFGSB([x], bounds=(0.0, 1.0))
    with pytest.raises(RuntimeError):
        opt.step()


def test_lbfgsb_unconstrained_match(quadratic_2d, make_closure):
    """When bounds are wide enough, L-BFGS-B should hit the same optimum as L-BFGS."""
    f, _, x_star = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = LBFGSB([x], bounds=(-100.0, 100.0))
    closure = make_closure(f, x)
    _run(opt, closure, 30)
    assert torch.allclose(x.detach(), x_star, atol=1e-4)


def test_lbfgsb_constrained_quadratic(bounded_quadratic_2d, make_closure):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = LBFGSB([x], bounds=(lo, hi))
    closure = make_closure(f, x)
    _run(opt, closure, 50)
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()
    assert torch.allclose(x.detach(), x_star, atol=1e-4)


def test_lbfgsb_matches_scipy_on_bounded_quadratic():
    """Reference: scipy.optimize.minimize(method='L-BFGS-B') on a small problem."""
    pytest.importorskip("scipy")
    from scipy.optimize import minimize

    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    b = torch.tensor([3.0, -1.0])
    bounds_tuple = (0.0, 1.0)

    def f_np(x):
        x_t = torch.tensor(x, dtype=torch.float64)
        return float(0.5 * x_t @ A.double() @ x_t - b.double() @ x_t)

    def grad_np(x):
        x_t = torch.tensor(x, dtype=torch.float64)
        g = (A.double() @ x_t - b.double()).numpy()
        return g

    res = minimize(
        f_np, [0.5, 0.5], jac=grad_np, method="L-BFGS-B",
        bounds=[(bounds_tuple[0], bounds_tuple[1])] * 2,
    )
    scipy_x = torch.tensor(res.x, dtype=torch.float64)

    x = torch.tensor([0.5, 0.5], requires_grad=True, dtype=torch.float64)
    opt = LBFGSB([x], bounds=bounds_tuple)

    def closure():
        opt.zero_grad()
        loss = 0.5 * x @ A.double() @ x - b.double() @ x
        loss.backward()
        return loss

    _run(opt, closure, 50)
    assert torch.allclose(x.detach(), scipy_x, atol=1e-4)
