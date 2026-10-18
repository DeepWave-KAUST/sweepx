"""Tests for the 1-Wasserstein FWI misfit (Metivier 2016)."""

from __future__ import annotations

import math

import numpy as np
import torch

from sweep_loss import L2Loss, Wasserstein1Loss, w1_loss


def _gaussian(t, t0, sigma):
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_zero_when_signals_equal():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_gaussian(t, 0.1, 0.02)).view(1, nt, 1, 1).double()
    for positive in ("square", "abs", "exp", "linear"):
        loss = Wasserstein1Loss(positive=positive, dt=dt, reduction="sum")(o, o)
        assert float(loss) < 1e-10, positive


def test_translation_invariance_property():
    """For a translated Gaussian, W1 = |shift| exactly (up to discretisation).

    With ``positive='square'`` a translated Gaussian is still a translated
    (now narrower) Gaussian density, and W1 between two pure translates is
    the absolute translation distance.
    """
    nt, dt = 2048, 1e-3
    t = np.arange(nt) * dt
    obs = torch.tensor(_gaussian(t, 1.0, 0.05)).view(1, nt, 1, 1).double()
    for shift in (0.0, 0.005, 0.020, 0.050):
        syn = torch.tensor(_gaussian(t, 1.0 + shift, 0.05)).view(1, nt, 1, 1).double()
        w1 = float(Wasserstein1Loss(positive="square", dt=dt, reduction="sum")(syn, obs))
        # Expected ~ |shift|; discretisation gives 1-2 dt of error.
        assert abs(w1 - shift) < 3 * dt, (shift, w1)


def test_grows_with_shift():
    losses = []
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    obs = torch.tensor(_gaussian(t, 0.5, 0.03)).view(1, nt, 1, 1).double()
    for shift in (0.0, 0.005, 0.010, 0.020, 0.040):
        syn = torch.tensor(_gaussian(t, 0.5 + shift, 0.03)).view(1, nt, 1, 1).double()
        losses.append(float(Wasserstein1Loss(positive="square", dt=dt, reduction="sum")(syn, obs)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_w1_monotone_across_l2_cycle_skipping():
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()
    shifts = [0.0, 0.020, 0.030, 0.040, 0.060, 0.080]
    l2_vals, w1_vals = [], []
    for ms in shifts:
        s = torch.tensor(_ricker(t, 0.5 + ms)).view(1, nt, 1, 1).double()
        l2_vals.append(float(L2Loss(reduction="sum")(s, o)))
        w1_vals.append(float(Wasserstein1Loss(positive="square", dt=dt, reduction="sum")(s, o)))
    pairs = list(zip(l2_vals, l2_vals[1:]))
    assert any(b < a for a, b in pairs), f"L2 unexpectedly monotone: {l2_vals}"
    pairs = list(zip(w1_vals, w1_vals[1:]))
    assert all(b > a for a, b in pairs), f"W1 not monotone: {w1_vals}"


def test_gradient_flow():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.1)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_ricker(t, 0.12)).view(1, nt, 1, 1).float()
    Wasserstein1Loss(positive="square", dt=dt, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        Wasserstein1Loss(positive="bogus")
    with pytest.raises(ValueError):
        Wasserstein1Loss(dt=-1)


def test_functional_alias():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1)
    o = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1)
    a = Wasserstein1Loss(positive="abs", dt=dt)(s, o)
    b = w1_loss(s, o, positive="abs", dt=dt)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
