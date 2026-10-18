"""Tests for sweepx.opt.optim.LBFGS."""
import pytest
import torch

from sweepx.opt.optim import LBFGS, NCG
from sweepx.opt.precond import PseudoHessianPrecond


def _run(opt, closure, n_iters):
    losses = []
    for _ in range(n_iters):
        loss = opt.step(closure)
        losses.append(loss.item())
    return losses


def test_lbfgs_requires_closure():
    x = torch.zeros(2, requires_grad=True)
    opt = LBFGS([x])
    with pytest.raises(RuntimeError):
        opt.step()


def test_lbfgs_converges_on_quadratic(quadratic_2d, make_closure):
    f, _, x_star = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = LBFGS([x], history_size=5)
    closure = make_closure(f, x)
    _run(opt, closure, 20)
    assert torch.allclose(x.detach(), x_star, atol=1e-5)


def test_lbfgs_beats_ncg_on_rosenbrock(rosenbrock_2d, make_closure):
    """L-BFGS with default 10 history pairs should beat NCG on Rosenbrock."""
    f, x_init, _ = rosenbrock_2d
    x_cg = x_init.clone().detach().requires_grad_(True)
    cg = NCG([x_cg])
    cg_losses = _run(cg, make_closure(f, x_cg), 100)
    x_bf = x_init.clone().detach().requires_grad_(True)
    bf = LBFGS([x_bf])
    bf_losses = _run(bf, make_closure(f, x_bf), 100)
    assert bf_losses[-1] <= cg_losses[-1]
    assert bf_losses[-1] < 1e-8


def test_lbfgs_respects_bounds(bounded_quadratic_2d, make_closure):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = LBFGS([x], bounds=(lo, hi))
    closure = make_closure(f, x)
    _run(opt, closure, 50)
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()
    assert torch.allclose(x.detach(), x_star, atol=1e-3)


def test_lbfgs_history_grows_then_caps():
    x = torch.zeros(3, requires_grad=True)
    opt = LBFGS([x], history_size=3)

    def closure():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss

    for _ in range(10):
        opt.step(closure)
    assert opt.history_length <= 3


def test_lbfgs_restart_clears_history():
    x = torch.tensor([2.0, -1.5], requires_grad=True)
    opt = LBFGS([x], history_size=10, restart_every=3)

    def closure():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss

    for k in range(8):
        opt.step(closure)
    # Iter just past restart: history should be small (just the pairs since the last reset).
    assert opt.history_length < 5


def test_lbfgs_preconditioner_used_as_initial_hessian():
    """A preconditioner = H_0 should be queried during the two-loop recursion."""

    class Counter:
        def __init__(self):
            self.calls = 0

        def update(self, *_a, **_k):  # match precond contract
            return

        def __call__(self, grad, param):
            self.calls += 1
            return grad

    c = Counter()
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = LBFGS([x], preconditioner=c)

    def closure():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    # Each step calls _apply_initial_inv_hessian at least once.
    assert c.calls >= 3


def test_lbfgs_pseudo_hessian_accelerates_illconditioned(
    quadratic_illconditioned, make_closure
):
    """PH preconditioner using sqrt(eigs) should hit the optimum in very few iters."""
    f, _, x_star = quadratic_illconditioned
    eigs = torch.tensor([1.0, 5.0, 20.0, 50.0, 100.0])
    sqrt_eigs = eigs.sqrt()

    x = torch.zeros(5, requires_grad=True)
    precond = PseudoHessianPrecond(eps=1e-12, normalize=False)
    precond.update([x], source_illum=sqrt_eigs, receiver_illum=sqrt_eigs)
    opt = LBFGS([x], preconditioner=precond)
    closure = make_closure(f, x)
    _run(opt, closure, 15)
    assert torch.allclose(x.detach(), x_star, atol=1e-5)
