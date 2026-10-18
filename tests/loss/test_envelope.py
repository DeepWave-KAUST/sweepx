"""Tests for the envelope FWI misfits."""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    from scipy.signal import hilbert as scipy_hilbert
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sweep_loss import EnvelopeLoss, envelope_loss
from sweep_loss._utils import envelope, hilbert


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_envelope_matches_scipy():
    """sweep_loss.envelope == |scipy.signal.hilbert| up to the eps stabiliser
    we add inside the square root for autograd safety."""
    if not HAS_SCIPY:
        import pytest
        pytest.skip("scipy not installed")
    nt = 512
    t = np.arange(nt) * 1e-3
    x = _ricker(t, 0.15)
    ref = np.abs(scipy_hilbert(x))
    ours = envelope(torch.tensor(x, dtype=torch.float64), eps=0.0).numpy()
    assert np.allclose(ours, ref, rtol=1e-10, atol=1e-12)


def test_envelope_zero_when_signals_equal():
    nt = 256
    t = np.arange(nt) * 1e-3
    x = torch.tensor(_ricker(t, 0.1)).view(1, nt, 1, 1).float()
    for kwargs in [{"p": 2}, {"p": 1}, {"log": True}, {"squared": True}]:
        loss = EnvelopeLoss(reduction="sum", **kwargs)(x, x)
        assert float(loss) < 1e-8, kwargs


def test_envelope_p2_matches_manual_formula():
    nt = 128
    t = np.arange(nt) * 1e-3
    s = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).float()
    o = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1).float()
    Es = envelope(s, dim=-3)
    Eo = envelope(o, dim=-3)
    manual = 0.5 * ((Es - Eo) ** 2).sum()
    auto = EnvelopeLoss(p=2, reduction="sum")(s, o)
    assert torch.allclose(manual, auto, rtol=1e-5, atol=1e-7)


def test_envelope_log_matches_manual_formula():
    nt = 128
    t = np.arange(nt) * 1e-3
    s = torch.tensor(_ricker(t, 0.05) + 0.1).view(1, nt, 1, 1).float()
    o = torch.tensor(_ricker(t, 0.07) + 0.1).view(1, nt, 1, 1).float()
    eps = 1e-12
    Es = envelope(s, dim=-3, eps=eps).clamp_min(eps)
    Eo = envelope(o, dim=-3, eps=eps).clamp_min(eps)
    manual = 0.5 * (torch.log(Es) - torch.log(Eo)).pow(2).sum()
    auto = EnvelopeLoss(log=True, reduction="sum")(s, o)
    assert torch.allclose(manual, auto, rtol=1e-5, atol=1e-6)


def test_envelope_squared_matches_manual_formula():
    nt = 128
    t = np.arange(nt) * 1e-3
    s = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).float()
    o = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1).float()
    Es = envelope(s, dim=-3)
    Eo = envelope(o, dim=-3)
    manual = 0.5 * (Es * Es - Eo * Eo).pow(2).sum()
    auto = EnvelopeLoss(squared=True, reduction="sum")(s, o)
    assert torch.allclose(manual, auto, rtol=1e-5, atol=1e-7)


def test_envelope_invariant_to_polarity():
    """Envelope is sign-invariant: E(-d) = E(d).  Polarity flip in syn alone
    should still leave the envelope-difference misfit close to 0 if obs is
    the same up to sign."""
    nt = 256
    t = np.arange(nt) * 1e-3
    s = torch.tensor(_ricker(t, 0.1)).view(1, nt, 1, 1).float()
    loss = EnvelopeLoss(p=2, reduction="sum")(s, -s)
    assert float(loss) < 1e-8


def test_envelope_gradient_flow():
    nt = 256
    t = np.arange(nt) * 1e-3
    s = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1).float()
    EnvelopeLoss(p=2, reduction="sum")(s, o).backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        EnvelopeLoss(p=3)
    with pytest.raises(ValueError):
        EnvelopeLoss(log=True, squared=True)


def test_hilbert_imag_orthogonal_to_real():
    """Property of the analytic signal: <Re(a), Im(a)> = 0 for sufficiently
    long signals up to boundary effects."""
    nt = 1024
    t = np.arange(nt) * 1e-3
    x = torch.tensor(_ricker(t, 0.5)).double()
    a = hilbert(x)
    # The inner product should be O(boundary) which is small for a centred ricker.
    inner = float((a.real * a.imag).sum())
    energy = float((a.real * a.real).sum())
    assert abs(inner) < 1e-3 * energy


def test_functional_alias_matches_module():
    s = torch.randn(1, 64, 3, 1)
    o = torch.randn(1, 64, 3, 1)
    a = EnvelopeLoss(p=2)(s, o)
    b = envelope_loss(s, o, p=2)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
