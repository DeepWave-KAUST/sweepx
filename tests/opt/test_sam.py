"""Tests for sweepx.opt.optim.SAM."""
import pytest
import torch

import sweepx.opt
from sweepx.opt.optim import SAM, AdamFWI, SteepestDescent


def _make_closure(f, x, opt):
    def closure():
        opt.zero_grad()
        loss = f(x)
        loss.backward()
        return loss
    return closure


def test_sam_requires_closure():
    x = torch.zeros(2, requires_grad=True)
    base = SteepestDescent([x], lr=1e-2)
    opt = SAM(base, rho=0.05)
    with pytest.raises(RuntimeError):
        opt.step()


def test_sam_with_rho_zero_matches_base():
    """rho=0 must reduce SAM to the underlying base."""
    torch.manual_seed(0)
    x_a = torch.tensor([1.0, -2.0], requires_grad=True)
    x_b = x_a.clone().detach().requires_grad_(True)

    base_a = SteepestDescent([x_a], lr=1e-2)
    base_b = SteepestDescent([x_b], lr=1e-2)
    opt_b = SAM(base_b, rho=0.0)

    for _ in range(50):
        base_a.zero_grad(); (x_a ** 2).sum().backward(); base_a.step()
        opt_b.step(_make_closure(lambda y: (y ** 2).sum(), x_b, opt_b))
    assert torch.allclose(x_a.detach(), x_b.detach(), atol=1e-9)


def test_sam_finds_flatter_minimum():
    """On a 1-D function with a sharp and a flat valley, SAM should bias toward the flat one."""
    # f(x) has a sharp minimum at x = -1 (depth -2) and a flat minimum at x = +1 (depth -2).
    # Defined symmetrically so plain GD lands in either depending on init; SAM should prefer flat.
    def f(p):
        x = p[0]
        sharp = -2.0 * torch.exp(-30.0 * (x + 1.0) ** 2)
        flat = -2.0 * torch.exp(-2.0 * (x - 1.0) ** 2)
        return sharp + flat

    # Start near zero (between the two minima); SAM with positive rho should find the flat one.
    x = torch.tensor([0.0], requires_grad=True)
    base = SteepestDescent([x], lr=0.01)
    opt = SAM(base, rho=0.5)
    for _ in range(2000):
        opt.step(_make_closure(f, x, opt))
    assert x.item() > 0.5  # landed near the flat minimum at +1


def test_sam_with_adam_base_runs_end_to_end():
    """Exercise SAM(AdamFWI(...)) on a simple quadratic."""
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    base = AdamFWI([x], lr=0.1, eps=1e-8)
    opt = SAM(base, rho=0.05)
    for _ in range(500):
        opt.step(_make_closure(lambda y: (y ** 2).sum(), x, opt))
    assert (x ** 2).sum().item() < 1.0


@pytest.mark.parametrize("make_base", [
    lambda x: sweepx.opt.optim.LBFGS([x]),
    lambda x: sweepx.opt.optim.LBFGSB([x], bounds=(-5.0, 5.0)),
    lambda x: sweepx.opt.optim.NCG([x]),
    lambda x: sweepx.opt.optim.SteepestDescent([x], line_search="strong_wolfe"),
    lambda x: sweepx.opt.optim.TruncatedNewton([x]),
    lambda x: sweepx.opt.optim.TruncatedGaussNewton([x]),
    lambda x: sweepx.opt.optim.TrustRegion([x]),
    lambda x: torch.optim.LBFGS([x]),
])
def test_sam_rejects_base_that_needs_a_closure(make_base):
    """SAM calls base.step() without a closure; such a base would fail (or,
    given SAM's closure, re-evaluate g at theta and drop g_temp)."""
    x = torch.zeros(2, requires_grad=True)
    with pytest.raises(TypeError, match="closure"):
        SAM(make_base(x), rho=0.05)
