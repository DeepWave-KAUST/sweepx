"""Tests for the 2-Wasserstein FWI misfit (Engquist-Froese 2014)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import L2Loss, Wasserstein2Loss, w2_loss


def _gaussian(t, t0, sigma):
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_zero_when_signals_equal():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_gaussian(t, 0.1, 0.02)).view(1, nt, 1, 1).double()
    for positive in ("square", "abs", "exp"):
        loss = Wasserstein2Loss(positive=positive, dt=dt, reduction="sum")(o, o)
        assert float(loss) < 1e-8, positive


def test_w2_squared_equals_shift_squared_for_translated_gaussian():
    """For two translated Gaussians (densities), W2^2 = (translation)^2.

    Our loss returns 0.5 * W2^2, hence expected = 0.5 * shift^2.
    """
    nt, dt = 4096, 5e-4
    t = np.arange(nt) * dt
    obs = torch.tensor(_gaussian(t, 1.0, 0.05)).view(1, nt, 1, 1).double()
    for shift in (0.0, 0.005, 0.020, 0.050):
        syn = torch.tensor(_gaussian(t, 1.0 + shift, 0.05)).view(1, nt, 1, 1).double()
        w2 = float(Wasserstein2Loss(
            positive="square", dt=dt, n_quantiles=1024, reduction="sum"
        )(syn, obs))
        expected = 0.5 * shift ** 2
        # Discretisation tolerance: quantile MC + Gaussian truncation
        assert abs(w2 - expected) < max(5e-5, 0.05 * expected + 1e-6), (shift, w2, expected)


def test_grows_with_shift():
    losses = []
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    obs = torch.tensor(_gaussian(t, 0.5, 0.03)).view(1, nt, 1, 1).double()
    for shift in (0.0, 0.005, 0.010, 0.020, 0.040):
        syn = torch.tensor(_gaussian(t, 0.5 + shift, 0.03)).view(1, nt, 1, 1).double()
        losses.append(float(Wasserstein2Loss(
            positive="square", dt=dt, n_quantiles=512, reduction="sum"
        )(syn, obs)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_w2_monotone_across_l2_cycle_skipping():
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()
    shifts = [0.0, 0.020, 0.030, 0.040, 0.060, 0.080]
    l2_vals, w2_vals = [], []
    for ms in shifts:
        s = torch.tensor(_ricker(t, 0.5 + ms)).view(1, nt, 1, 1).double()
        l2_vals.append(float(L2Loss(reduction="sum")(s, o)))
        w2_vals.append(float(Wasserstein2Loss(
            positive="square", dt=dt, n_quantiles=512, reduction="sum"
        )(s, o)))
    pairs = list(zip(l2_vals, l2_vals[1:]))
    assert any(b < a for a, b in pairs), f"L2 unexpectedly monotone: {l2_vals}"
    pairs = list(zip(w2_vals, w2_vals[1:]))
    assert all(b > a for a, b in pairs), f"W2 not monotone: {w2_vals}"


def test_gradient_flow():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_gaussian(t, 0.1, 0.02)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_gaussian(t, 0.12, 0.02)).view(1, nt, 1, 1).float()
    Wasserstein2Loss(positive="square", dt=dt, n_quantiles=128, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        Wasserstein2Loss(positive="bogus")
    with pytest.raises(ValueError):
        Wasserstein2Loss(dt=-1)
    with pytest.raises(ValueError):
        Wasserstein2Loss(n_quantiles=3)


def test_functional_alias():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1)
    o = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1)
    a = Wasserstein2Loss(positive="abs", dt=dt, n_quantiles=128)(s, o)
    b = w2_loss(s, o, positive="abs", dt=dt, n_quantiles=128)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
