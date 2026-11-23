"""``reuse_accepted_grad``: the line-search optimizers start each step from the
loss and gradient their previous line search computed at the accepted point,
instead of calling the closure there a second time."""
import pytest
import torch

from sweepx.opt.optim import LBFGS, LBFGSB, NCG, SteepestDescent
from sweepx.opt.optim._wolfe_step import wolfe_step


class _Problem:
    """A small non-convex least-squares problem over several parameters, one
    of which the loss never touches (its ``.grad`` stays ``None``)."""

    def __init__(self):
        gen = torch.Generator().manual_seed(0)
        self.u = torch.randn(4, 6, generator=gen, dtype=torch.float64)
        self.y = torch.randn(3, 6, generator=gen, dtype=torch.float64)
        self.W0 = 0.3 * torch.randn(3, 4, generator=gen, dtype=torch.float64)
        self.v0 = torch.full((3, 1), 0.5, dtype=torch.float64)

    def params(self):
        W = self.W0.clone().requires_grad_(True)
        v = self.v0.clone().requires_grad_(True)
        unused = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        return [W, v, unused]

    def closure(self, opt, params, calls):
        W, v, _ = params

        def closure():
            calls[0] += 1
            opt.zero_grad()
            r = torch.tanh(W @ self.u) * v - self.y
            loss = (r ** 2).sum() + 0.1 * ((v - 1.0) ** 4).sum()
            loss.backward()
            return loss

        return closure


_OPTIMIZERS = {
    "LBFGS": lambda p, **kw: LBFGS(p, **kw),
    "LBFGS-precond-initial_step": lambda p, **kw: LBFGS(
        p, preconditioner=lambda g, _p: 2.0 * g, initial_step=0.2, **kw),
    "LBFGSB": lambda p, **kw: LBFGSB(p, bounds=(-0.4, 0.8), initial_step=0.2, **kw),
    "NCG": lambda p, **kw: NCG(p, initial_step=0.2, **kw),
    "SteepestDescent": lambda p, **kw: SteepestDescent(
        p, line_search="strong_wolfe", initial_step=0.2, **kw),
}


def _trajectory(make_opt, reuse, n_steps, perturb_at=()):
    prob = _Problem()
    params = prob.params()
    opt = make_opt(params, reuse_accepted_grad=reuse)
    calls = [0]
    closure = prob.closure(opt, params, calls)
    losses, states = [], []
    for k in range(n_steps):
        if k in perturb_at:
            with torch.no_grad():
                params[0][0, 0] += 1e-3
        losses.append(opt.step(closure).detach().clone())
        states.append([p.detach().clone() for p in params]
                      + [p.grad.clone() if p.grad is not None else None for p in params])
    return losses, states, calls[0]


def _assert_same(a, b):
    la, sa, _ = a
    lb, sb, _ = b
    for x, y in zip(la, lb):
        assert torch.equal(x, y)
    for xs, ys in zip(sa, sb):
        for x, y in zip(xs, ys):
            assert (x is None and y is None) or torch.equal(x, y)


@pytest.mark.parametrize("name", list(_OPTIMIZERS))
def test_reuse_is_bitwise_identical_and_saves_one_call_per_step(name):
    n = 12
    fresh = _trajectory(_OPTIMIZERS[name], False, n)
    reused = _trajectory(_OPTIMIZERS[name], True, n)
    _assert_same(fresh, reused)
    # Every step after the first starts from the accepted point's cached
    # evaluation, so exactly one closure call is saved per step.
    assert reused[2] == fresh[2] - (n - 1)


@pytest.mark.parametrize("name", list(_OPTIMIZERS))
def test_cache_is_dropped_when_parameters_change_between_steps(name):
    n, perturb = 8, (3, 6)
    fresh = _trajectory(_OPTIMIZERS[name], False, n, perturb)
    reused = _trajectory(_OPTIMIZERS[name], True, n, perturb)
    # Same trajectory as recomputing everywhere: the perturbed steps did not
    # use the gradient cached at the unperturbed point.
    _assert_same(fresh, reused)
    assert reused[2] == fresh[2] - (n - 1 - len(perturb))


@pytest.mark.parametrize("name", list(_OPTIMIZERS))
def test_cached_loss_and_grad_match_a_fresh_evaluation(name):
    prob = _Problem()
    params = prob.params()
    opt = _OPTIMIZERS[name](params)
    closure = prob.closure(opt, params, [0])
    for _ in range(3):
        opt.step(closure)
    loss = opt._grad_cache.load(opt.param_groups, closure)
    cached = [p.grad.clone() if p.grad is not None else None for p in params]
    fresh = closure().detach()
    assert torch.equal(loss, fresh)
    for c, p in zip(cached, params):
        assert (c is None and p.grad is None) or torch.equal(c, p.grad)
    assert cached[2] is None  # the unused parameter's .grad stays None


class _StartWatch:
    """Wraps a closure and records, per call, whether the parameters are still
    those at the start of the current step (i.e. the call is the step's own
    start-point evaluation, not a line-search trial)."""

    def __init__(self, params, closure):
        self.params, self.closure = params, closure
        self.first_call_at_start = []

    def mark(self):
        self._x = [p.detach().clone() for p in self.params]
        self._first = True

    def __call__(self):
        if self._first:
            self.first_call_at_start.append(
                all(torch.equal(p, x) for p, x in zip(self.params, self._x)))
            self._first = False
        return self.closure()


def _step(opt, watch, fn=None):
    watch.mark()
    opt.step(fn if fn is not None else watch)
    return watch.first_call_at_start[-1]


def test_cache_applies_only_to_same_closure_until_invalidated():
    prob = _Problem()
    params = prob.params()
    opt = LBFGS(params, initial_step=0.2)
    calls = [0]
    a = _StartWatch(params, prob.closure(opt, params, calls))
    b = _StartWatch(params, prob.closure(opt, params, calls))
    assert _step(opt, a) is True   # nothing cached yet
    assert _step(opt, a) is False  # same closure at the accepted point: reused
    assert _step(opt, b) is True   # another closure object: recomputed
    assert _step(opt, b) is False
    opt.invalidate_grad_cache()
    assert _step(opt, b) is True   # same closure, but invalidated


def test_bound_method_closure_counts_as_the_same_closure():
    prob = _Problem()
    params = prob.params()
    opt = NCG(params, initial_step=0.2)
    watch = _StartWatch(params, prob.closure(opt, params, [0]))

    class Model:
        def closure(self):
            return watch()

    m = Model()
    assert _step(opt, watch, m.closure) is True
    # ``m.closure`` builds a new bound-method object on every access.
    assert _step(opt, watch, m.closure) is False


def test_disabled_reuse_keeps_nothing():
    prob = _Problem()
    params = prob.params()
    opt = LBFGS(params, initial_step=0.2, reuse_accepted_grad=False)
    opt.step(prob.closure(opt, params, [0]))
    assert opt._grad_cache._x is None


def test_steepest_descent_rejects_reuse_without_line_search():
    x = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError, match="reuse_accepted_grad"):
        SteepestDescent([x], line_search=None, reuse_accepted_grad=True)
    with pytest.raises(ValueError, match="reuse_accepted_grad"):
        SteepestDescent([x], line_search=None, reuse_accepted_grad=False)
    assert SteepestDescent([x]).reuse_accepted_grad is False
    assert SteepestDescent([x], line_search="strong_wolfe").reuse_accepted_grad is True


def _wolfe_on_parabola(keep_accepted, alpha_init, max_iter=2):
    # phi(alpha) = (alpha - 1)^2 from x0 = 0 along d = 1.
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    trials = []

    def evaluate():
        xv = x.detach().clone()
        trials.append(float(xv))
        return (xv - 1.0).pow(2).sum(), 2.0 * (xv - 1.0), (False,)

    zero = torch.zeros(1, dtype=torch.float64)
    alpha, accepted = wolfe_step(
        [{"params": [x]}], evaluate, zero, torch.ones(1, dtype=torch.float64),
        1.0, -2.0, bounds=None, alpha_init=alpha_init, alpha_max=10.0, c1=1e-4, c2=0.1,
        max_iter=max_iter, owner="test", keep_accepted=keep_accepted,
    )
    return alpha, accepted, trials, float(x.detach())


def test_wolfe_step_returns_the_accepted_last_trial():
    alpha, accepted, trials, x = _wolfe_on_parabola(True, alpha_init=0.95)
    assert trials == [0.95] and alpha == 0.95 and x == 0.95
    loss, grad, none = accepted
    assert float(loss) == (0.95 - 1.0) ** 2
    assert float(grad) == 2.0 * (0.95 - 1.0) and none == (False,)
    assert _wolfe_on_parabola(False, alpha_init=0.95)[1] is None


def test_wolfe_step_does_not_return_an_earlier_trial():
    # The first trial (2.5) fails sufficient decrease; zoom tries 1.25 (its
    # alpha_lo; curvature fails with c2 = 0.1), then 0.625 (loss above
    # alpha_lo's) and runs out of iterations: it accepts 1.25, which is not the
    # last trial, so nothing is returned for reuse.
    alpha, accepted, trials, x = _wolfe_on_parabola(True, alpha_init=2.5)
    assert trials == [2.5, 1.25, 0.625]
    assert alpha == 1.25 and x == 1.25
    assert accepted is None
