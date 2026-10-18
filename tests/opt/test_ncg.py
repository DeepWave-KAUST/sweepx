"""Tests for sweep_opt.optim.NCG."""
import pytest
import torch

from sweep_opt.optim import NCG, SteepestDescent


def _run(opt, closure, n_iters):
    losses = []
    for _ in range(n_iters):
        loss = opt.step(closure)
        losses.append(loss.item())
    return losses


def test_ncg_requires_closure():
    x = torch.zeros(2, requires_grad=True)
    opt = NCG([x])
    with pytest.raises(RuntimeError):
        opt.step()


def test_ncg_converges_on_quadratic(quadratic_2d, make_closure):
    """NCG on a 2-D quadratic converges fast; allow a Wolfe-tolerance margin."""
    f, _, x_star = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = NCG([x])
    closure = make_closure(f, x)
    _run(opt, closure, 100)
    assert torch.allclose(x.detach(), x_star, atol=1e-3)


def test_ncg_beats_sd_on_rosenbrock(rosenbrock_2d, make_closure):
    """NCG with Wolfe must reach a lower loss than SD with Wolfe in fewer iterations."""
    f, x_init, _ = rosenbrock_2d

    x_sd = x_init.clone().detach().requires_grad_(True)
    sd = SteepestDescent([x_sd], line_search="strong_wolfe")
    sd_losses = _run(sd, make_closure(f, x_sd), 200)

    x_cg = x_init.clone().detach().requires_grad_(True)
    cg = NCG([x_cg])
    cg_losses = _run(cg, make_closure(f, x_cg), 200)

    assert cg_losses[-1] < sd_losses[-1]
    # And NCG should reach near-optimal Rosenbrock loss.
    assert cg_losses[-1] < 1e-4


def test_ncg_restarts_when_pr_negative(rosenbrock_2d, make_closure):
    """Across a Rosenbrock run, at least one restart (beta = 0 from PR+) must fire."""
    f, x_init, _ = rosenbrock_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = NCG([x])
    closure = make_closure(f, x)
    saw_restart = False
    for _ in range(50):
        opt.step(closure)
        # Skip the very first iteration (always a forced restart).
        if opt._iter > 1 and opt.last_restart:
            saw_restart = True
            break
    assert saw_restart, "expected PR+ to trigger at least one restart on Rosenbrock"


def test_ncg_periodic_restart_every(quadratic_2d, make_closure):
    """restart_every=1 forces a restart on every step (NCG degenerates to SD)."""
    f, _, _ = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = NCG([x], restart_every=1)
    closure = make_closure(f, x)
    for _ in range(5):
        opt.step(closure)
        assert opt.last_restart


def test_ncg_respects_bounds(bounded_quadratic_2d, make_closure):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = NCG([x], bounds=(lo, hi))
    closure = make_closure(f, x)
    _run(opt, closure, 100)
    assert (x.detach() >= lo - 1e-9).all()
    assert (x.detach() <= hi + 1e-9).all()
    assert torch.allclose(x.detach(), x_star, atol=1e-3)
