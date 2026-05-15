"""Tests for the Optimal Transport of the Matching Filter (OTMF) misfit
(Sun & Alkhalifah 2019)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import L2Loss, OTMFLoss, otmf_loss


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _shifted(nt=512, dt=1e-3, t0=0.15, dt_shift=0.0):
    t = np.arange(nt) * dt
    return (
        torch.tensor(_ricker(t, t0 + dt_shift), dtype=torch.float64).view(1, nt, 1, 1),
        torch.tensor(_ricker(t, t0), dtype=torch.float64).view(1, nt, 1, 1),
        dt,
    )


def test_baseline_small_for_identical_signals():
    """For syn == obs the Wiener filter is a *low-pass-filtered* delta (the
    Tikhonov stabiliser broadens the strict delta into a sinc-like kernel),
    so OTMF(o, o) is small but not exactly zero.  We pin it as a baseline
    that subsequent shift tests subtract out.
    """
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()
    baseline = float(OTMFLoss(dt=dt, epsilon=1e-8, reduction="sum")(o, o))
    # Bound below by 0 (positivity) and above by something a few-dt-wide kernel
    # would produce on a Ricker spectrum.
    assert 0.0 <= baseline < 1e-2


def test_amplitude_invariance():
    """OTMF is invariant under positive scaling of either input."""
    _, o, dt = _shifted(nt=512)
    a = OTMFLoss(dt=dt, epsilon=1e-8, reduction="sum")(o, o)
    b = OTMFLoss(dt=dt, epsilon=1e-8, reduction="sum")(0.5 * o, o)
    c = OTMFLoss(dt=dt, epsilon=1e-8, reduction="sum")(o, 5.0 * o)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-9)
    assert torch.allclose(a, c, rtol=1e-4, atol=1e-9)


def test_grows_with_shift():
    losses = []
    for ms in (0.0, 0.003, 0.008, 0.020, 0.030):
        s, o, dt = _shifted(nt=1024, dt_shift=ms)
        losses.append(float(OTMFLoss(dt=dt, epsilon=1e-8, reduction="sum")(s, o)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_otmf_monotone_across_l2_cycle_skipping():
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()
    shifts = [0.0, 0.020, 0.030, 0.040, 0.060, 0.080]
    l2_vals, m_vals = [], []
    for ms in shifts:
        s = torch.tensor(_ricker(t, 0.5 + ms)).view(1, nt, 1, 1).double()
        l2_vals.append(float(L2Loss(reduction="sum")(s, o)))
        m_vals.append(float(OTMFLoss(dt=dt, epsilon=1e-8, reduction="sum")(s, o)))
    pairs = list(zip(l2_vals, l2_vals[1:]))
    assert any(b < a for a, b in pairs), f"L2 unexpectedly monotone: {l2_vals}"
    pairs = list(zip(m_vals, m_vals[1:]))
    assert all(b > a for a, b in pairs), f"OTMF not monotone: {m_vals}"


def test_order_2_quadratic_in_shift():
    """For a pure time shift, the order-2 OTMF should grow as
    baseline + shift^2 (the Wiener filter is approx delta(tau + shift)
    convolved with a small low-pass kernel of fixed shape).
    """
    nt, dt = 2048, 5e-4
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()
    baseline = float(OTMFLoss(dt=dt, epsilon=1e-8, order=2, reduction="sum")(o, o))
    for shift in (0.005, 0.010, 0.020, 0.040):
        s = torch.tensor(_ricker(t, 0.5 + shift)).view(1, nt, 1, 1).double()
        val = float(OTMFLoss(dt=dt, epsilon=1e-8, order=2, reduction="sum")(s, o))
        # val - baseline should equal shift^2 to relative 15%
        expected = shift ** 2
        actual = val - baseline
        assert abs(actual - expected) < 0.15 * expected + 1e-7, (shift, val, baseline)


def test_order1_super_linear_or_linear_in_shift():
    """Order-1 OTMF grows monotonically with shift. With ``positive='square'``
    the Wiener filter is squared first, which produces a doubly-peaked
    density and the growth is between linear and super-linear in shift.
    Verify strict monotonicity.
    """
    nt, dt = 2048, 5e-4
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()
    baseline = float(OTMFLoss(dt=dt, epsilon=1e-8, order=1, reduction="sum")(o, o))
    diffs = []
    for shift in (0.010, 0.020, 0.040, 0.060):
        s = torch.tensor(_ricker(t, 0.5 + shift)).view(1, nt, 1, 1).double()
        val = float(OTMFLoss(dt=dt, epsilon=1e-8, order=1, reduction="sum")(s, o))
        diffs.append(val - baseline)
    assert all(b > a for a, b in zip(diffs, diffs[1:])), diffs


def test_gradient_flow():
    s, o, dt = _shifted(nt=512, dt_shift=0.005)
    s = s.float().clone().requires_grad_(True)
    o = o.float()
    OTMFLoss(dt=dt, epsilon=1e-4, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        OTMFLoss(dt=0)
    with pytest.raises(ValueError):
        OTMFLoss(epsilon=-0.1)
    with pytest.raises(ValueError):
        OTMFLoss(positive="garbage")
    with pytest.raises(ValueError):
        OTMFLoss(order=3)


def test_functional_alias():
    s, o, dt = _shifted()
    a = OTMFLoss(dt=dt, epsilon=1e-4)(s, o)
    b = otmf_loss(s, o, dt=dt, epsilon=1e-4)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
