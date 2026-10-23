"""Tests for sweepx.opt.optim.TruncatedNewton."""
import pytest
import torch

from sweepx.opt.optim import LBFGS, TruncatedNewton


def _make_loss_fn(f, x):
    def loss_fn():
        return f(x)
    return loss_fn


def test_tn_requires_loss_fn():
    x = torch.zeros(2, requires_grad=True)
    opt = TruncatedNewton([x])
    with pytest.raises(RuntimeError):
        opt.step()


def test_tn_one_step_on_quadratic(quadratic_2d):
    """One Newton step solves a quadratic exactly."""
    f, _, x_star = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = TruncatedNewton([x], forcing="constant", forcing_constant=1e-12,
                          max_inner_iter=20)
    opt.step(_make_loss_fn(f, x))
    assert torch.allclose(x.detach(), x_star, atol=1e-6)


def test_tn_converges_on_rosenbrock(rosenbrock_2d):
    """TN with tight forcing should reach near-machine precision on Rosenbrock."""
    f, x_init, _ = rosenbrock_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = TruncatedNewton(
        [x], forcing="constant", forcing_constant=1e-8, max_inner_iter=50
    )
    losses = []
    for _ in range(100):
        loss = opt.step(_make_loss_fn(f, x))
        losses.append(loss.item())
        if losses[-1] < 1e-10:
            break
    assert losses[-1] < 1e-6


def test_tn_eisenstat_walker_forcing_decreases(quadratic_illconditioned):
    """eta_k = min(0.5, sqrt(||g||)) must shrink as ||g|| does.

    On a quadratic, TN converges fast and ||g|| -> 0, so sqrt(||g||) drops
    below the 0.5 cap quickly.
    """
    f, _, _ = quadratic_illconditioned
    x = torch.zeros(5, requires_grad=True)
    opt = TruncatedNewton([x], forcing="eisenstat_walker")
    forcings = []
    for _ in range(10):
        opt.step(_make_loss_fn(f, x))
        forcings.append(opt.last_forcing)
    assert forcings[-1] < forcings[0]
    assert forcings[-1] < 0.1


def test_tn_preconditioner_used():
    """A trivial preconditioner that doubles the residual must be invoked."""

    class Counter:
        def __init__(self):
            self.calls = 0

        def update(self, *_a, **_k):
            return

        def __call__(self, grad, param):
            self.calls += 1
            return grad

    c = Counter()
    x = torch.tensor([2.0, -1.5], requires_grad=True)
    opt = TruncatedNewton([x], preconditioner=c, max_inner_iter=5)

    def f(x):
        return 0.5 * (x ** 2).sum()

    opt.step(_make_loss_fn(f, x))
    assert c.calls >= 1


def test_tn_beats_lbfgs_on_strict_quadratic(quadratic_illconditioned):
    """On a quadratic, TN reaches optimum in much fewer iters than L-BFGS."""
    f, _, x_star = quadratic_illconditioned

    x_tn = torch.zeros(5, requires_grad=True)
    opt_tn = TruncatedNewton([x_tn], forcing="constant", forcing_constant=1e-10,
                             max_inner_iter=20)
    opt_tn.step(_make_loss_fn(f, x_tn))
    assert torch.allclose(x_tn.detach(), x_star, atol=1e-5)


class _FirstOrderOnly(torch.autograd.Function):
    """sum(x^2) whose backward has no graph - like sweep's impl='c' gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x * x).sum()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return (2.0 * x * grad).detach()


def test_tn_rejects_gradient_without_graph():
    x = torch.tensor([1.0, -1.0], requires_grad=True)
    opt = TruncatedNewton([x])
    with pytest.raises(RuntimeError, match="twice-differentiable"):
        opt.step(lambda: _FirstOrderOnly.apply(x))
