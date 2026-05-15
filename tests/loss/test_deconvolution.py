"""Tests for the deconvolution-based misfit (Luo-Sava 2011)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import DeconvolutionLoss, L2Loss, deconvolution_loss


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


def test_amplitude_invariance_with_normalize():
    """Normalised Luo-Sava is invariant under positive scaling of inputs."""
    _, o, dt = _shifted(nt=512)
    a = DeconvolutionLoss(dt=dt, epsilon=1e-8, normalize=True, reduction="sum")(o, o)
    b = DeconvolutionLoss(dt=dt, epsilon=1e-8, normalize=True, reduction="sum")(0.4 * o, o)
    c = DeconvolutionLoss(dt=dt, epsilon=1e-8, normalize=True, reduction="sum")(3.7 * o, o)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-9)
    assert torch.allclose(a, c, rtol=1e-5, atol=1e-9)


def test_unnormalized_scales_with_amplitude():
    """Without normalisation, scaling the syn amplitude changes the loss."""
    _, o, dt = _shifted(nt=512)
    base = DeconvolutionLoss(dt=dt, epsilon=1e-8, normalize=False, reduction="sum")(o, o)
    scaled = DeconvolutionLoss(dt=dt, epsilon=1e-8, normalize=False, reduction="sum")(0.5 * o, o)
    assert not torch.allclose(base, scaled, rtol=1e-3, atol=1e-12)


def test_grows_with_time_shift_normalized():
    losses = []
    for ms in (0.0, 0.005, 0.010, 0.020, 0.030):
        s, o, dt = _shifted(nt=1024, dt_shift=ms)
        losses.append(float(
            DeconvolutionLoss(dt=dt, epsilon=1e-8, normalize=True, reduction="sum")(s, o)
        ))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_decon_monotone_across_l2_cycle_skipping():
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5), dtype=torch.float64).view(1, nt, 1, 1)
    shifts = [0.0, 0.020, 0.030, 0.040, 0.060, 0.080]
    l2_vals, d_vals = [], []
    for ms in shifts:
        s = torch.tensor(_ricker(t, 0.5 + ms), dtype=torch.float64).view(1, nt, 1, 1)
        l2_vals.append(float(L2Loss(reduction="sum")(s, o)))
        d_vals.append(float(
            DeconvolutionLoss(dt=dt, epsilon=1e-8, normalize=True, reduction="sum")(s, o)
        ))
    pairs = list(zip(l2_vals, l2_vals[1:]))
    assert any(b < a for a, b in pairs), f"L2 unexpectedly monotone: {l2_vals}"
    pairs = list(zip(d_vals, d_vals[1:]))
    assert all(b > a for a, b in pairs), f"Decon not monotone: {d_vals}"


def test_gradient_flow():
    s, o, dt = _shifted(nt=512, dt_shift=0.005)
    s = s.float().clone().requires_grad_(True)
    o = o.float()
    DeconvolutionLoss(dt=dt, epsilon=1e-4, normalize=False, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        DeconvolutionLoss(dt=-1.0)
    with pytest.raises(ValueError):
        DeconvolutionLoss(epsilon=-1.0)


def test_functional_alias():
    s, o, dt = _shifted()
    a = DeconvolutionLoss(dt=dt, epsilon=1e-4, normalize=True)(s, o)
    b = deconvolution_loss(s, o, dt=dt, epsilon=1e-4, normalize=True)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
