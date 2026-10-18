"""Tests for sweepx.opt.optim.SteepestDescent."""
import torch

from sweepx.opt.optim import SteepestDescent


def test_sd_converges_on_well_conditioned_quadratic(quadratic_2d):
    f, _, x_star = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = SteepestDescent([x], lr=0.3)
    for _ in range(500):
        opt.zero_grad()
        f(x).backward()
        opt.step()
    assert torch.allclose(x.detach(), x_star, atol=1e-4)


def test_sd_with_wolfe_strict_descent(quadratic_illconditioned, make_closure):
    """Wolfe line search must produce monotone descent and substantial loss reduction.

    Full convergence on a cond=100 quadratic requires hundreds of SD steps; the
    point of this test is to verify the line search itself, not SD's well-known
    slowness. We assert (a) strict descent every step and (b) ~99% loss
    reduction within 80 iterations.
    """
    f, _, _ = quadratic_illconditioned
    x = torch.zeros(5, requires_grad=True)
    opt = SteepestDescent([x], line_search="strong_wolfe", c1=1e-4, c2=0.9)
    closure = make_closure(f, x)
    losses = []
    for _ in range(80):
        loss = opt.step(closure)
        losses.append(loss.item())
    for prev, cur in zip(losses, losses[1:]):
        assert cur <= prev + 1e-9
    assert losses[-1] < 0.01 * losses[0]


def test_sd_with_wolfe_rosenbrock_progress(rosenbrock_2d, make_closure):
    """Wolfe SD should make substantial progress on Rosenbrock at fixed budget."""
    f, x_init, _ = rosenbrock_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = SteepestDescent([x], line_search="strong_wolfe")
    closure = make_closure(f, x)
    initial = float(f(x).detach())
    for _ in range(500):
        opt.step(closure)
    final = float(f(x).detach())
    # Initial Rosenbrock loss at (-1.2, 1.0) is 24.2; SD with Wolfe should
    # get under 1.0 within 500 iters (still slow on Rosenbrock).
    assert final < 1.0
    assert final < initial


def test_sd_respects_bounds(bounded_quadratic_2d):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = SteepestDescent([x], lr=0.3, bounds=(lo, hi))
    for _ in range(500):
        opt.zero_grad()
        f(x).backward()
        opt.step()
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()
    assert torch.allclose(x.detach(), x_star, atol=1e-3)


def test_sd_preconditioner_used():
    """A trivial preconditioner that scales by 2 should change the step size."""

    class ScalePrecond:
        def __init__(self, s):
            self.s = s

        def __call__(self, grad, param):
            return self.s * grad

    x_a = torch.tensor([1.0], requires_grad=True)
    x_b = torch.tensor([1.0], requires_grad=True)
    opt_a = SteepestDescent([x_a], lr=0.1)
    opt_b = SteepestDescent([x_b], lr=0.1, preconditioner=ScalePrecond(2.0))
    for _ in range(5):
        opt_a.zero_grad()
        (x_a ** 2).sum().backward()
        opt_a.step()
        opt_b.zero_grad()
        (x_b ** 2).sum().backward()
        opt_b.step()
    # With 2x preconditioner, x_b should have decayed faster toward 0.
    assert abs(x_b.item()) < abs(x_a.item())
