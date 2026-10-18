"""Tests for sweep_opt.optim.TruncatedGaussNewton."""
import pytest
import torch

from sweep_opt.optim import LBFGS, TruncatedGaussNewton


def test_tgn_requires_residual_fn():
    x = torch.zeros(2, requires_grad=True)
    opt = TruncatedGaussNewton([x])
    with pytest.raises(RuntimeError):
        opt.step()


def test_tgn_one_step_on_linear_least_squares():
    """Linear residual r = M x - y => one GN step solves exactly."""
    M = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    y = torch.tensor([1.0, 2.0])
    x = torch.zeros(2, requires_grad=True)
    opt = TruncatedGaussNewton(
        [x], forcing="constant", forcing_constant=1e-12, max_inner_iter=20
    )
    opt.step(lambda: M @ x - y)
    expected = torch.linalg.solve(M, y)
    assert torch.allclose(x.detach(), expected, atol=1e-6)


def test_tgn_converges_on_nonlinear_least_squares():
    """Fit r_i(x) = sin(x) - sin(x_star) for x near x_star."""
    x_star = torch.tensor([0.7, -0.5])
    target = torch.sin(x_star)

    x = torch.tensor([0.4, -0.3], requires_grad=True)
    opt = TruncatedGaussNewton(
        [x], forcing="constant", forcing_constant=1e-8, max_inner_iter=10
    )
    losses = []
    for _ in range(40):
        loss = opt.step(lambda: torch.sin(x) - target)
        losses.append(loss.item())
        if losses[-1] < 1e-12:
            break
    assert losses[-1] < 1e-8
    assert torch.allclose(x.detach(), x_star, atol=1e-3)


def test_tgn_damping_acts_as_levenberg_marquardt():
    """Strong damping should make GN behave like preconditioned SD."""
    x_star = torch.tensor([1.0, -1.0])
    M = torch.tensor([[3.0, 0.0], [0.0, 0.5]])

    x = torch.zeros(2, requires_grad=True)
    opt = TruncatedGaussNewton(
        [x], forcing="constant", forcing_constant=1e-10,
        damping=0.0, max_inner_iter=20
    )
    opt.step(lambda: M @ x - M @ x_star)
    assert torch.allclose(x.detach(), x_star, atol=1e-6)

    # With heavy damping, one step under-shoots the optimum.
    x2 = torch.zeros(2, requires_grad=True)
    opt2 = TruncatedGaussNewton(
        [x2], forcing="constant", forcing_constant=1e-10,
        damping=10.0, max_inner_iter=20
    )
    opt2.step(lambda: M @ x2 - M @ x_star)
    assert (x2.detach() - x_star).norm() > 0.1
    # And further iterations still converge to the optimum.
    for _ in range(150):
        opt2.step(lambda: M @ x2 - M @ x_star)
    assert torch.allclose(x2.detach(), x_star, atol=1e-2)


def test_tgn_preconditioner_used():
    """A counter preconditioner must be invoked inside the inner CG."""

    class Counter:
        def __init__(self):
            self.calls = 0

        def update(self, *_a, **_k):
            return

        def __call__(self, grad, param):
            self.calls += 1
            return grad

    c = Counter()
    M = torch.randn(4, 3)
    y = torch.randn(4)
    x = torch.zeros(3, requires_grad=True)
    opt = TruncatedGaussNewton(
        [x], preconditioner=c, forcing="constant", forcing_constant=1e-10,
        max_inner_iter=10,
    )
    opt.step(lambda: M @ x - y)
    assert c.calls >= 1
