"""hessian='difference' / jacobian='difference': curvature products from first derivatives.

The autograd routes of TruncatedNewton, TrustRegion and TruncatedGaussNewton
differentiate a gradient (or a vector-Jacobian product) a second time, so a
loss whose gradient carries no graph -- sweep's impl='c', boundary saving or
the compiled eager step -- was refused.  The difference routes take central
differences of two gradients (two residuals) and run with such a loss.
"""
import pytest
import torch

from sweepx.opt.optim import TruncatedGaussNewton, TruncatedNewton, TrustRegion
from sweepx.opt.optim._difference import (_room, difference_step, gradient_difference,
                                          residual_difference, shifted)

A = torch.tensor([[3.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.5]], dtype=torch.float64)
B = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)


class _FirstOrderQuadratic(torch.autograd.Function):
    """0.5 x^T A x - b^T x with a correct gradient that autograd cannot
    differentiate again (as sweep's impl='c' gradient)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return 0.5 * x @ A @ x - B @ x

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return ((A @ x - B) * grad).detach()


class _FirstOrderLinear(torch.autograd.Function):
    """r = M x - y with a backward off the tape."""

    @staticmethod
    def forward(ctx, x, M, y):
        ctx.M = M
        return M @ x - y

    @staticmethod
    def backward(ctx, grad):
        return (ctx.M.T @ grad).detach(), None, None


def test_tn_difference_runs_where_autograd_refuses_and_solves_a_quadratic():
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError, match="hessian='difference'"):
        TruncatedNewton([x]).step(lambda: _FirstOrderQuadratic.apply(x))
    opt = TruncatedNewton([x], hessian="difference", forcing="constant", forcing_constant=1e-12,
                          max_inner_iter=20)
    opt.step(lambda: _FirstOrderQuadratic.apply(x))
    assert torch.allclose(x.detach(), torch.linalg.solve(A, B), atol=1e-8)


def test_tr_difference_converges_on_a_first_order_only_quadratic():
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = TrustRegion([x], hessian="difference", initial_radius=10.0, cg_tol=1e-10)
    for _ in range(5):
        opt.step(lambda: _FirstOrderQuadratic.apply(x))
    assert torch.allclose(x.detach(), torch.linalg.solve(A, B), atol=1e-6)


def test_tgn_difference_solves_linear_least_squares_off_the_tape():
    M = torch.tensor([[2.0, 1.0], [1.0, 3.0], [0.5, -1.0]], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    opt = TruncatedGaussNewton([x], jacobian="difference", forcing="constant",
                               forcing_constant=1e-12, max_inner_iter=20)
    opt.step(lambda: _FirstOrderLinear.apply(x, M, y))
    expected = torch.linalg.lstsq(M, y.unsqueeze(-1)).solution.squeeze(-1)
    assert torch.allclose(x.detach(), expected, atol=1e-8)


def test_tn_difference_converges_on_rosenbrock(rosenbrock_2d):
    f, x_init, _ = rosenbrock_2d
    x = x_init.detach().double().requires_grad_(True)
    opt = TruncatedNewton([x], hessian="difference", rel_step=1e-4, forcing="constant",
                          forcing_constant=1e-8, max_inner_iter=50)
    for _ in range(200):
        loss = opt.step(lambda: f(x))
        if loss.item() < 1e-12:
            break
    assert torch.allclose(x.detach(), torch.ones(2, dtype=torch.float64), atol=1e-5)


def test_difference_and_autograd_products_agree():
    torch.manual_seed(0)
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    loss_fn = lambda: (torch.sin(x) * x ** 2).sum() + (x[0] * x[1]) ** 2
    v = torch.randn(4, dtype=torch.float64)
    (g,) = torch.autograd.grad(loss_fn(), x, create_graph=True)
    (hv,) = torch.autograd.grad((g * v).sum(), x)
    (hv_dif,) = gradient_difference(loss_fn, [x], [v], rel_step=1e-5)
    assert torch.allclose(hv_dif, hv, rtol=1e-6, atol=1e-8)


def test_shifted_restores_the_parameters_bit_for_bit():
    torch.manual_seed(0)
    params = [torch.randn(5, requires_grad=True), torch.randn(2, 3, requires_grad=True)]
    before = [p.detach().clone() for p in params]
    with shifted(params, [0.1 * torch.randn_like(p) for p in params]):
        assert not torch.equal(params[0].detach(), before[0])
    assert all(torch.equal(p.detach(), b) for p, b in zip(params, before))


def test_the_step_respects_each_parameters_own_scale():
    vp = torch.full((3,), 2000.0)
    eps = torch.full((3,), 0.1)
    e = difference_step([vp, eps], [torch.ones(3), torch.ones(3)], rel_step=1e-2)
    assert e == pytest.approx(1e-3)              # eps may change by 1 % of 0.1, not of 2000
    assert difference_step([vp], [torch.zeros(3)], rel_step=1e-2) is None


def test_room_is_taken_on_the_flattened_parameters():
    params = [torch.tensor([0.0, 1.0], dtype=torch.float64), torch.ones(2, 2, dtype=torch.float64)]
    v = [torch.tensor([1.0, 1.0]), torch.full((2, 2), -1.0)]
    assert _room(params, v, (0.0, 10.0), 1.0) == pytest.approx(1.0)       # the (2, 2) block hits 0
    assert _room(params, v, (0.0, 10.0), -1.0) == 0.0                     # the first entry is on 0
    lo = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=torch.float64)   # one bound per coordinate
    assert _room(params, v, (lo, None), 1.0) == pytest.approx(0.5)
    outside = [torch.tensor([-0.1, 1.0], dtype=torch.float64), params[1]]
    assert _room(outside, v, (0.0, 10.0), 1.0) == 0.0


_T = torch.tensor([-1.0, 0.5, 2.0, 3.0], dtype=torch.float64)


def _hv_exact(x, v):
    return (2.0 + 3.75 * x.clamp(min=0).sqrt()) * v          # Hessian of sum((x - t)^2 + x^2.5)


@pytest.mark.parametrize("x0", [0.0, 2e-3], ids=["on_the_bound", "next_to_the_bound"])
def test_a_probe_outside_where_the_loss_is_nan_falls_back_to_one_side(x0):
    x = torch.tensor([x0, 1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)
    loss_fn = lambda: ((x - _T) ** 2).sum() + (x ** 2.5).sum()
    v = torch.tensor([-1.0, 0.5, -0.3, 0.2], dtype=torch.float64)
    (central,) = gradient_difference(loss_fn, [x], [v], rel_step=1e-2)
    assert not torch.isfinite(central).all()                 # the minus probe is below 0
    (hv,) = gradient_difference(loss_fn, [x], [v], rel_step=1e-2, bounds=(0.0, 10.0))
    # one-sided: O(e), and x^2.5 has no third derivative at 0
    assert torch.allclose(hv, _hv_exact(x.detach(), v), rtol=0.15, atol=0.0)


def test_a_loss_defined_outside_the_box_keeps_the_central_difference():
    """As an FWI misfit is: the bounds change nothing."""
    x = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)
    loss_fn = lambda: ((x - _T) ** 2).sum() + (x ** 4).sum()
    v = torch.tensor([-1.0, 0.5, -0.3, 0.2], dtype=torch.float64)
    (free,) = gradient_difference(loss_fn, [x], [v], rel_step=1e-2)
    (bounded,) = gradient_difference(loss_fn, [x], [v], rel_step=1e-2, bounds=(0.0, 10.0))
    assert torch.equal(free, bounded)
    r_fn = lambda: torch.cat([x - _T, x ** 2])
    assert torch.equal(residual_difference(r_fn, [x], [v], 1e-2),
                       residual_difference(r_fn, [x], [v], 1e-2, bounds=(0.0, 10.0)))


# A loss that is NaN below 0 (x ** 2.5), from an iterate on the lower bound: the
# difference probes used to step outside, so TrustRegion stalled at the start
# and TruncatedNewton fell back to steepest descent.


def _on_the_bound():
    return torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)


@pytest.mark.parametrize("make, steps", [
    (lambda x, route: TrustRegion([x], hessian=route, bounds=(0.0, 10.0), initial_radius=1.0), 15),
    (lambda x, route: TruncatedNewton([x], hessian=route, bounds=(0.0, 10.0)), 1),
], ids=["tr", "tn"])
def test_difference_probes_stay_inside_the_bounds(make, steps):
    loss_of = lambda x: ((x - _T) ** 2).sum() + (x ** 2.5).sum()
    final = {}
    for route in ("autograd", "difference"):
        x = _on_the_bound()
        opt = make(x, route)
        for _ in range(steps):
            opt.step(lambda: loss_of(x))
        final[route] = x.detach()
    assert torch.allclose(final["difference"], final["autograd"], atol=2e-2)
    assert float(loss_of(final["difference"])) < float(loss_of(_on_the_bound().detach())) - 1.0


def test_gauss_newton_difference_probes_stay_inside_the_bounds():
    residual_of = lambda x: torch.cat([x - _T, x ** 1.25])
    final = {}
    for route in ("autograd", "difference"):
        x = _on_the_bound()
        opt = TruncatedGaussNewton([x], jacobian=route, bounds=(0.0, 10.0))
        opt.step(lambda: residual_of(x))
        final[route] = x.detach()
    assert torch.allclose(final["difference"], final["autograd"], atol=2e-2)


@pytest.mark.parametrize("make", [
    lambda x: TruncatedNewton([x], hessian="newton"),
    lambda x: TrustRegion([x], hessian="exact"),
    lambda x: TruncatedGaussNewton([x], jacobian="forward"),
    lambda x: TruncatedNewton([x], hessian="difference", rel_step=0.0),
], ids=["tn_route", "tr_route", "tgn_route", "rel_step"])
def test_bad_options_are_refused(make):
    with pytest.raises(ValueError):
        make(torch.zeros(2, requires_grad=True))
