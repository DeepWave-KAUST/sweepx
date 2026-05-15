"""Tests for the Adaptive Waveform Inversion (AWI) misfit."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import AWILoss, awi_loss


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


def test_amplitude_invariance():
    """Wiener filter for syn = c * obs is c * delta(0); the temporal penalty
    is therefore the same regardless of c."""
    _, o, dt = _shifted(nt=512)
    a = AWILoss(dt=dt, epsilon=1e-8, reduction="sum")(o, o)
    b = AWILoss(dt=dt, epsilon=1e-8, reduction="sum")(0.5 * o, o)
    c = AWILoss(dt=dt, epsilon=1e-8, reduction="sum")(7.3 * o, o)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-9)
    assert torch.allclose(a, c, rtol=1e-5, atol=1e-9)


def test_grows_with_time_shift():
    """For shifts where L2 is monotone, AWI is too."""
    losses = []
    for dt_shift in (0.0, 0.003, 0.008, 0.020, 0.030):
        s, o, dt = _shifted(nt=1024, dt_shift=dt_shift)
        losses.append(float(AWILoss(dt=dt, epsilon=1e-8, reduction="sum")(s, o)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_awi_monotone_across_l2_cycle_skipping():
    """The defining feature of AWI: it stays monotone over a shift range
    where L2 oscillates due to cycle skipping.

    For a 15 Hz Ricker (dominant wavelength ~ 70 ms), L2 grows up to a
    half-wavelength shift (~ 30 ms) then *decreases* (cycle skipping).
    AWI keeps growing.
    """
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5), dtype=torch.float64).view(1, nt, 1, 1)
    from fwiloss import L2Loss
    shifts = [0.0, 0.020, 0.030, 0.040, 0.060, 0.080]
    l2_vals, awi_vals = [], []
    for ms in shifts:
        s = torch.tensor(_ricker(t, 0.5 + ms), dtype=torch.float64).view(1, nt, 1, 1)
        l2_vals.append(float(L2Loss(reduction="sum")(s, o)))
        awi_vals.append(float(AWILoss(dt=dt, epsilon=1e-8, reduction="sum")(s, o)))
    # L2 should be NON-monotone (cycle skipping)
    pairs = list(zip(l2_vals, l2_vals[1:]))
    assert any(b < a for a, b in pairs), f"L2 unexpectedly monotone: {l2_vals}"
    # AWI should be strictly monotone
    pairs = list(zip(awi_vals, awi_vals[1:]))
    assert all(b > a for a, b in pairs), f"AWI not monotone: {awi_vals}"


def test_gradient_flow():
    s, o, dt = _shifted(nt=512, dt_shift=0.005)
    s = s.float().clone().requires_grad_(True)
    o = o.float()
    AWILoss(dt=dt, epsilon=1e-4, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        AWILoss(dt=0.0)
    with pytest.raises(ValueError):
        AWILoss(epsilon=-1.0)


def test_functional_alias():
    s, o, dt = _shifted()
    a = AWILoss(dt=dt, epsilon=1e-4)(s, o)
    b = awi_loss(s, o, dt=dt, epsilon=1e-4)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
