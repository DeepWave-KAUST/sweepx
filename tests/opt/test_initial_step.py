"""``initial_step``: line searches on parameters in physical units (m/s).

A velocity model in m/s has gradients of 1e-7..1e-5, so a search that starts
at ``alpha = 1`` along ``-g`` moves it by less than float32 resolution. The
problems below mimic that: float32 values around 2000 with a tiny gradient.
"""
import pytest
import torch

import sweepx.opt.optim._wolfe_step as wolfe_step_mod
from sweepx.opt.optim import LBFGS, LBFGSB, NCG, SteepestDescent

V0 = 2000.0


def _ms_problem(scale=1e-9, n=64):
    """f = scale * sum w (x - t)^2 around x0 = 2000 m/s, t = x0 +- 50 m/s."""
    w = torch.linspace(1.0, 4.0, n)
    target = V0 + 50.0 * torch.linspace(-1.0, 1.0, n)

    def f(x):
        return scale * (w * (x - target) ** 2).sum()

    return f, target


def _closure(f, x, opt):
    def closure():
        opt.zero_grad()
        loss = f(x)
        loss.backward()
        return loss
    return closure


def _make(kind, x, **kw):
    if kind == "lbfgs":
        return LBFGS([x], bounds=(1500.0, 5500.0), **kw)
    if kind == "lbfgsb":
        return LBFGSB([x], bounds=(1500.0, 5500.0), **kw)
    if kind == "ncg":
        return NCG([x], bounds=(1500.0, 5500.0), **kw)
    return SteepestDescent([x], line_search="strong_wolfe", bounds=(1500.0, 5500.0), **kw)


KINDS = ["lbfgs", "lbfgsb", "ncg", "sd"]


@pytest.mark.parametrize("kind", KINDS)
def test_default_step_does_not_move_ms_model_and_warns(kind):
    f, _ = _ms_problem()
    x = torch.full((64,), V0, requires_grad=True)
    opt = _make(kind, x)
    with pytest.warns(RuntimeWarning, match="floating-point resolution"):
        for _ in range(3):
            opt.step(_closure(f, x, opt))
    assert torch.equal(x.detach(), torch.full((64,), V0))


@pytest.mark.parametrize("kind", KINDS)
def test_initial_step_moves_ms_model(kind):
    f, target = _ms_problem()
    x = torch.full((64,), V0, requires_grad=True)
    opt = _make(kind, x, initial_step=10.0)
    f0 = float(f(x.detach()))
    first = None
    for k in range(30):
        opt.step(_closure(f, x, opt))
        if k == 0:
            first = float((x.detach() - V0).abs().max())
    # The first search starts at a 10 m/s trial and may expand 10x.
    assert 0.0 < first <= 100.0 + 1e-3
    assert float(f(x.detach())) < 1e-2 * f0


def test_small_moves_capped_by_alpha_max_warn():
    """Gradients large enough to move float32 but far too small for alpha <= 10."""
    f, _ = _ms_problem(scale=1e-6)
    x = torch.full((64,), V0, dtype=torch.float64, requires_grad=True)
    opt = LBFGS([x])
    with pytest.warns(RuntimeWarning, match="alpha_max"):
        opt.step(_closure(f, x, opt))
    assert float((x.detach() - V0).abs().max()) < 0.1


def test_box_blocked_step_does_not_warn(recwarn):
    """At a bound with -g pointing out of the box, staying put is not a stall."""
    x = torch.tensor([1.0], requires_grad=True)
    opt = LBFGS([x], bounds=(0.0, 1.0))
    opt.step(_closure(lambda y: ((y - 3.0) ** 2).sum(), x, opt))
    assert float(x) == 1.0
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def _record_alpha_init(monkeypatch):
    calls = []
    orig = wolfe_step_mod.strong_wolfe

    def spy(*args, **kwargs):
        calls.append(kwargs["alpha_init"])
        return orig(*args, **kwargs)

    monkeypatch.setattr(wolfe_step_mod, "strong_wolfe", spy)
    return calls


def test_lbfgs_initial_step_only_without_history(monkeypatch):
    calls = _record_alpha_init(monkeypatch)
    f, _ = _ms_problem()
    x = torch.full((64,), V0, requires_grad=True)
    opt = LBFGS([x], initial_step=10.0, alpha_init=0.5)
    for _ in range(3):
        opt.step(_closure(f, x, opt))
    g0 = 2e-9 * 4.0 * 50.0  # max |g| at x0 (w = 4, |x0 - t| = 50)
    assert calls[0] == pytest.approx(10.0 / g0, rel=1e-3)
    assert calls[1:] == [0.5, 0.5]


def test_explicit_alpha_init_is_kept_without_initial_step(monkeypatch):
    calls = _record_alpha_init(monkeypatch)
    w = torch.tensor([1.0, 10.0])
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    opt = NCG([x], alpha_init=0.25)
    for _ in range(2):
        opt.step(_closure(lambda y: (w * y ** 2).sum(), x, opt))
    assert calls == [0.25, 0.25]


def test_lbfgs_scaled_preconditioner_matches_unpreconditioned():
    """H_0 = gamma_k P: with P = c I the iterates must not depend on c."""
    A = torch.diag(torch.tensor([1.0, 3.0, 10.0], dtype=torch.float64))
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    def f(y):
        return 0.5 * y @ A @ y - b @ y

    def run(precond):
        x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        opt = LBFGS([x], preconditioner=precond, initial_step=0.5)
        path = []
        for _ in range(6):
            opt.step(_closure(f, x, opt))
            path.append(x.detach().clone())
        return torch.stack(path)

    ref = run(None)
    scaled = run(lambda g, p: 1e-6 * g)
    assert torch.allclose(scaled, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_initial_step_must_be_positive(bad):
    x = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError, match="initial_step"):
        LBFGS([x], initial_step=bad)


def test_sd_initial_step_needs_line_search():
    x = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError, match="initial_step"):
        SteepestDescent([x], lr=0.1, initial_step=1.0)
