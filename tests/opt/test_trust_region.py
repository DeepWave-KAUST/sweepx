"""Tests for sweepx.opt.optim.TrustRegion (TTR and ASTR variants)."""
import pytest
import torch

from sweepx.opt.optim import TrustRegion


def _make_loss_fn(f, x):
    def loss_fn():
        return f(x)
    return loss_fn


def test_tr_requires_loss_fn():
    x = torch.zeros(2, requires_grad=True)
    opt = TrustRegion([x])
    with pytest.raises(RuntimeError):
        opt.step()


def test_ttr_converges_on_quadratic(quadratic_2d):
    f, _, x_star = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = TrustRegion([x], variant="ttr", initial_radius=10.0, cg_tol=1e-10)
    for _ in range(10):
        opt.step(_make_loss_fn(f, x))
    assert torch.allclose(x.detach(), x_star, atol=1e-5)


def test_ttr_radius_grows_when_full_step_accepted_at_boundary():
    """When the optimum lies far away and the first step hits the trust
    boundary with rho > 0.75, the radius must grow."""
    A = torch.eye(2)
    b = torch.tensor([10.0, -10.0])  # optimum at (10, -10), far from origin

    x = torch.zeros(2, requires_grad=True)
    opt = TrustRegion([x], variant="ttr", initial_radius=1.0,
                      max_radius=100.0, cg_tol=1e-12)
    initial_radius = opt.trust_radius
    for _ in range(5):
        opt.step(lambda: 0.5 * x @ A @ x - b @ x)
    assert opt.trust_radius > initial_radius


def test_astr_converges_on_quadratic(quadratic_2d):
    f, _, x_star = quadratic_2d
    x = torch.zeros(2, requires_grad=True)
    opt = TrustRegion([x], variant="astr", initial_radius=10.0)
    for _ in range(20):
        opt.step(_make_loss_fn(f, x))
    assert torch.allclose(x.detach(), x_star, atol=1e-3)


def test_astr_uses_two_dim_subspace_after_first_step():
    """After the first step, ASTR must use span(g, prev_step) - tested via accept rate."""
    x = torch.tensor([2.0, -1.5], requires_grad=True)
    A = torch.tensor([[5.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([1.0, -1.0])

    def f(p):
        return 0.5 * p @ A @ p - b @ p

    opt = TrustRegion([x], variant="astr", initial_radius=2.0)
    accepts = 0
    for _ in range(10):
        opt.step(_make_loss_fn(f, x))
        if opt.last_accepted:
            accepts += 1
    assert accepts >= 5  # most steps must succeed on a quadratic


def test_tr_projects_bounds(bounded_quadratic_2d):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = TrustRegion([x], variant="ttr", bounds=(lo, hi), initial_radius=2.0)
    for _ in range(50):
        opt.step(_make_loss_fn(f, x))
    assert (x.detach() >= lo - 1e-9).all()
    assert (x.detach() <= hi + 1e-9).all()
    assert torch.allclose(x.detach(), x_star, atol=1e-2)


def test_tr_rejects_nonfinite_trial_and_shrinks():
    """(x - 2)^2 is NaN past x = 1.5: the full Newton step (to x = 2) must be
    rejected with a smaller radius, not retried with the same one."""
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def loss_fn():
        return ((x - 2.0) ** 2).sum() + (0.0 * torch.sqrt(1.5 - x)).sum()

    opt = TrustRegion([x], variant="ttr", initial_radius=10.0)
    opt.step(loss_fn)
    assert not opt.last_accepted and opt.trust_radius < 10.0 and float(x) == 0.0
    for _ in range(5):
        opt.step(loss_fn)
    assert 0.0 < float(x) <= 1.5


def test_tr_trial_point_is_projected_onto_bounds():
    x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    seen = []

    def loss_fn():
        seen.append(float(x))
        return ((x - 3.0) ** 2).sum()

    opt = TrustRegion([x], variant="ttr", bounds=(0.0, 1.0), initial_radius=10.0)
    opt.step(loss_fn)
    assert all(0.0 <= v <= 1.0 for v in seen) and float(x) == 1.0


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


def test_tr_rejects_gradient_without_graph():
    x = torch.tensor([1.0, -1.0], requires_grad=True)
    opt = TrustRegion([x])
    with pytest.raises(RuntimeError, match="twice-differentiable"):
        opt.step(lambda: _FirstOrderOnly.apply(x))
