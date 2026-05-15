"""Tests for the Time-Frequency phase/envelope misfit
(Fichtner 2008; Kristekova et al. 2006/2009)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import TimeFrequencyPhaseLoss, time_frequency_phase_loss


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _shifted_pair(nt=512, dt=1e-3, t0=0.25, dt_shift=0.0, fc=15.0):
    t = np.arange(nt) * dt
    return (
        torch.tensor(_ricker(t, t0 + dt_shift, fc), dtype=torch.float64).view(1, nt, 1, 1),
        torch.tensor(_ricker(t, t0, fc), dtype=torch.float64).view(1, nt, 1, 1),
    )


def test_zero_for_identical_signals():
    s, o = _shifted_pair()
    for alpha in (0.0, 0.5, 1.0):
        loss = float(
            TimeFrequencyPhaseLoss(alpha=alpha, n_fft=64, reduction="sum")(o, o)
        )
        assert loss < 1e-10, alpha


def test_alpha_endpoints_decompose_loss():
    s, o = _shifted_pair(dt_shift=0.003)
    env = float(TimeFrequencyPhaseLoss(alpha=0.0, n_fft=128, reduction="sum")(s, o))
    phi = float(TimeFrequencyPhaseLoss(alpha=1.0, n_fft=128, reduction="sum")(s, o))
    mix = float(TimeFrequencyPhaseLoss(alpha=0.5, n_fft=128, reduction="sum")(s, o))
    # alpha=0.5: (1-0.5)*env + 0.5*phi
    expected = 0.5 * env + 0.5 * phi
    assert abs(mix - expected) < 1e-6 * max(abs(mix), abs(expected))


def test_envelope_part_zero_for_pure_polarity_flip():
    """|STFT(d)| = |STFT(-d)| ⇒ TF-envelope misfit is invariant to a global
    sign flip of either signal.  TF-phase, of course, is not."""
    s, _ = _shifted_pair()
    env = float(TimeFrequencyPhaseLoss(alpha=0.0, n_fft=64, reduction="sum")(s, -s))
    phi = float(TimeFrequencyPhaseLoss(alpha=1.0, n_fft=64, reduction="sum")(s, -s))
    assert env < 1e-10
    assert phi > 0.05


def test_grows_with_shift():
    losses = []
    for shift in (0.0, 0.001, 0.003, 0.007, 0.015):
        s, o = _shifted_pair(dt_shift=shift, nt=1024)
        losses.append(float(
            TimeFrequencyPhaseLoss(alpha=0.5, n_fft=128, reduction="sum")(s, o)
        ))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_phase_residual_bounded():
    """Unit-circle L2 residual |e^{i phi_s} - e^{i phi_o}|^2 in [0, 4],
    scaled by w in [0, 1], scaled by 1/2 ⇒ per-sample phase loss in [0, 2].
    """
    s = torch.randn(1, 1024, 2, 1, dtype=torch.float64)
    o = torch.randn(1, 1024, 2, 1, dtype=torch.float64)
    pw = TimeFrequencyPhaseLoss(alpha=1.0, n_fft=64, reduction="none")(s, o)
    assert float(pw.max()) <= 2.0 + 1e-6


def test_gradient_flow():
    s, o = _shifted_pair(dt_shift=0.003)
    s = s.float().clone().requires_grad_(True)
    o = o.float()
    TimeFrequencyPhaseLoss(alpha=0.5, n_fft=64, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        TimeFrequencyPhaseLoss(alpha=-0.1)
    with pytest.raises(ValueError):
        TimeFrequencyPhaseLoss(alpha=1.5)
    with pytest.raises(ValueError):
        TimeFrequencyPhaseLoss(n_fft=4)


def test_n_fft_larger_than_trace_raises():
    import pytest
    s = torch.randn(1, 32, 1, 1)
    o = torch.randn_like(s)
    with pytest.raises(ValueError):
        TimeFrequencyPhaseLoss(n_fft=128)(s, o)


def test_functional_alias():
    s, o = _shifted_pair()
    a = TimeFrequencyPhaseLoss(alpha=0.7, n_fft=64)(s, o)
    b = time_frequency_phase_loss(s, o, alpha=0.7, n_fft=64)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)
