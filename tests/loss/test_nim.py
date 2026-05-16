"""Tests for the Normalised Integration Method (NIM) FWI misfit."""

from __future__ import annotations

import math

import numpy as np
import torch

from sweep_loss import NIMLoss, nim_loss


def _gaussian(t, t0, sigma):
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_zero_when_signals_equal():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    g = torch.tensor(_gaussian(t, 0.1, 0.02)).view(1, nt, 1, 1).float()
    for positive in ("square", "abs", "exp", "linear"):
        loss = NIMLoss(positive=positive, dt=dt, reduction="sum")(g, g)
        assert float(loss) < 1e-10, positive


def test_cdf_squared_increases_with_translation():
    """For a translated Gaussian, the NIM misfit equals the integrated
    squared CDF distance ~ (dt_shift)^2 * 1 (plus higher-order)."""
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    losses = []
    for shift in (0.0, 0.005, 0.010, 0.020):
        obs = torch.tensor(_gaussian(t, 0.5, 0.02)).view(1, nt, 1, 1).float()
        syn = torch.tensor(_gaussian(t, 0.5 + shift, 0.02)).view(1, nt, 1, 1).float()
        losses.append(float(NIMLoss(positive="square", dt=dt, reduction="sum")(syn, obs)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_nim_matches_manual_cdf_l2():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    syn = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).double()
    obs = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1).double()

    # Recompute the loss manually with the same positive transform.
    fs = syn * syn
    fo = obs * obs
    ps = fs / fs.sum(dim=-3, keepdim=True).clamp_min(1e-12)
    po = fo / fo.sum(dim=-3, keepdim=True).clamp_min(1e-12)
    Fs = torch.cumsum(ps, dim=-3)
    Fo = torch.cumsum(po, dim=-3)
    d2 = (Fs - Fo) ** 2
    manual = 0.5 * dt * (d2.sum(dim=-3) - 0.5 * (d2[..., 0:1, :, :] + d2[..., -1:, :, :])).sum()

    auto = NIMLoss(positive="square", dt=dt, reduction="sum")(syn, obs)
    assert torch.allclose(manual, auto, rtol=1e-6, atol=1e-9)


def test_nim_avoids_cycle_skipping():
    """When the shift is greater than half a wavelength, L2 saturates but
    NIM keeps growing — that's the whole point of NIM."""
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    obs = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()

    # Half-wavelength shift to put L2 in the cycle-skipping regime
    syn_close = torch.tensor(_ricker(t, 0.5 + 0.005)).view(1, nt, 1, 1).double()
    syn_far = torch.tensor(_ricker(t, 0.5 + 0.06)).view(1, nt, 1, 1).double()
    from sweep_loss import L2Loss
    L2_close = float(L2Loss(reduction="sum", half=True)(syn_close, obs))
    L2_far = float(L2Loss(reduction="sum", half=True)(syn_far, obs))
    nim_close = float(NIMLoss(positive="square", dt=dt, reduction="sum")(syn_close, obs))
    nim_far = float(NIMLoss(positive="square", dt=dt, reduction="sum")(syn_far, obs))
    # L2 typically saturates / drops; NIM should keep increasing.
    assert nim_far > nim_close
    # And the *ratio* of nim_far / nim_close should be much larger than L2_far / L2_close
    assert (nim_far / nim_close) > (L2_far / L2_close)


def test_gradient_flow():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1).float()
    NIMLoss(positive="square", dt=dt, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_positive_or_dt():
    import pytest
    with pytest.raises(ValueError):
        NIMLoss(positive="bogus")
    with pytest.raises(ValueError):
        NIMLoss(dt=0)


def test_functional_alias():
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.02)).view(1, nt, 1, 1)
    o = torch.tensor(_ricker(t, 0.03)).view(1, nt, 1, 1)
    a = NIMLoss(positive="square", dt=dt)(s, o)
    b = nim_loss(s, o, positive="square", dt=dt)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
