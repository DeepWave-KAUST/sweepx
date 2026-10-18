"""Tests for sweepx.opt.optim.SOAP (bit-for-bit parity with the official code is in test_soap_parity.py)."""
import pytest
import torch

from sweepx.opt.optim import SOAP
from sweepx.opt.optim.soap import _project_back


def test_soap_converges_on_quadratic():
    x = torch.tensor([3.0, -2.0, 1.5], requires_grad=True)
    opt = SOAP([x], lr=1e-1, betas=(0.9, 0.999), update_freq=5)
    for _ in range(500):
        opt.zero_grad()
        (x ** 2).sum().backward()
        opt.step()
    assert (x ** 2).sum().item() < 1e-3


def test_soap_2d_parameter_converges():
    torch.manual_seed(0)
    M = torch.randn(4, 5, requires_grad=True)
    M_target = torch.randn(4, 5)
    opt = SOAP([M], lr=1e-1, betas=(0.9, 0.999), update_freq=5)
    for _ in range(500):
        opt.zero_grad()
        ((M - M_target) ** 2).sum().backward()
        opt.step()
    assert ((M - M_target) ** 2).sum().item() < 1e-2


def test_soap_bounds_clamp(bounded_quadratic_2d):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = SOAP([x], lr=1e-2, bounds=(lo, hi), update_freq=10)
    for _ in range(2000):
        opt.zero_grad(); f(x).backward(); opt.step()
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()


def test_soap_defaults_do_not_decay_weights():
    """FWI parameters (e.g. a velocity grid) must not be weight-decayed by default."""
    p = torch.zeros(3, 3, requires_grad=True)
    assert SOAP([p]).defaults["weight_decay"] == 0.0


def test_soap_first_step_only_builds_the_basis_and_per_group_lr():
    torch.manual_seed(0)
    a, b = torch.zeros(4, 5, requires_grad=True), torch.zeros(4, 5, requires_grad=True)
    opt = SOAP([{"params": [a], "lr": 1e-2}, {"params": [b], "lr": 2e-2}], update_freq=1)
    g = torch.randn(4, 5)
    a.grad, b.grad = g.clone(), g.clone()
    opt.step()                                     # official: no update on the first step
    assert not a.detach().any() and not b.detach().any()
    a.grad, b.grad = g.clone(), g.clone()
    opt.step()
    assert a.detach().abs().max() > 0
    assert torch.allclose(b.detach(), 2 * a.detach(), rtol=1e-5, atol=0)


def test_soap_long_axis_gets_no_factor():
    """A (1e5, 1) hash table must not allocate a 1e5 x 1e5 factor."""
    torch.manual_seed(3)
    p = torch.randn(100_000, 1, requires_grad=True)
    p0 = p.detach().clone()
    opt = SOAP([p], lr=1e-3)
    for _ in range(3):
        p.grad = torch.randn(100_000, 1)
        opt.step()
    GG = opt.state[p]["GG"]
    assert GG[0] is None and GG[1].shape == (1, 1)
    assert torch.isfinite(p).all() and not torch.equal(p.detach(), p0)


def test_soap_momentum_is_kept_in_parameter_space_across_refresh():
    """Paper/official: m lives in the original space; a basis refresh must not change it."""
    torch.manual_seed(4)
    p = torch.randn(4, 5, requires_grad=True)
    opt = SOAP([p], lr=1e-3, betas=(0.9, 0.999), update_freq=3)
    for _ in range(3):                                 # first step + 2 updates (state step = 2)
        p.grad = torch.randn(4, 5)
        opt.step()
    st, group = opt.state[p], opt.param_groups[0]
    m_before = _project_back(st["exp_avg"], st, group).clone()
    A, B = torch.randn(4, 4), torch.randn(5, 5)
    st["GG"][0].copy_(A @ A.T); st["GG"][1].copy_(B @ B.T)   # force a very different basis
    Q_old = [q.clone() for q in st["Q"]]
    p.grad = torch.zeros(4, 5)
    opt.step()                                         # state step 3 refreshes the basis
    assert not torch.allclose(st["Q"][0].abs(), Q_old[0].abs(), atol=1e-3)
    m_after = _project_back(st["exp_avg"], st, group)
    assert torch.allclose(m_after, 0.9 * m_before, atol=1e-6)


def test_soap_eigh_falls_back_to_float64(monkeypatch):
    real_eigh = torch.linalg.eigh

    def fp32_fails(m, *a, **k):
        if m.dtype == torch.float32:
            raise torch.linalg.LinAlgError("forced")
        return real_eigh(m, *a, **k)
    monkeypatch.setattr(torch.linalg, "eigh", fp32_fails)
    torch.manual_seed(5)
    p = torch.randn(4, 5, requires_grad=True)
    opt = SOAP([p], lr=1e-3, update_freq=1)
    for _ in range(3):
        p.grad = torch.randn(4, 5)
        opt.step()
    assert all(q.dtype == torch.float32 for q in opt.state[p]["Q"])
    assert torch.isfinite(p).all()


def test_soap_frequent_refresh_is_stable():
    """Rank-2 gradients, basis refreshed every step. An earlier port that refreshed the
    basis before the step (current gradient inside its own projection) blew W up to ~1e3."""
    torch.manual_seed(7)
    W = torch.zeros(32, 16, requires_grad=True)
    T = torch.randn(32, 16) * 1e-2
    opt = SOAP([W], lr=1e-3, betas=(0.9, 0.999), update_freq=1)
    for _ in range(100):
        opt.zero_grad()
        x = torch.randn(16, 2)
        (1e-2 * ((W - T) @ x).pow(2).sum()).backward()
        opt.step()
        assert torch.isfinite(W).all() and W.abs().max() < 0.2


def test_soap_float64_parameters():
    """The official code fails on float64 (factors in the default dtype); sweepx.opt keeps
    the factors in the parameter dtype."""
    torch.manual_seed(8)
    p = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    opt = SOAP([p], lr=1e-3, update_freq=2)
    for _ in range(6):
        p.grad = torch.randn(6, 5, dtype=torch.float64)
        opt.step()
    st = opt.state[p]
    assert st["GG"][0].dtype == torch.float64 and st["Q"][0].dtype == torch.float64
    assert torch.isfinite(p).all()


def test_soap_rejects_low_precision():
    p = torch.zeros(4, 4, dtype=torch.bfloat16, requires_grad=True)
    p.grad = torch.ones(4, 4, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        SOAP([p]).step()
