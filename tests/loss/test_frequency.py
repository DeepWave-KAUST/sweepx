"""Tests for the frequency- and Laplace-domain FWI misfits."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import (
    FrequencyAmplitudeLoss,
    FrequencyDomainL2Loss,
    FrequencyPhaseLoss,
    LaplaceL2Loss,
    LogarithmicShinMinLoss,
    L2Loss,
)


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _shifted_pair(nt=512, dt=1e-3, t0=0.15, dt_shift=0.0, fc=15.0):
    t = np.arange(nt) * dt
    return (
        torch.tensor(_ricker(t, t0 + dt_shift, fc), dtype=torch.float64).view(1, nt, 1, 1),
        torch.tensor(_ricker(t, t0, fc), dtype=torch.float64).view(1, nt, 1, 1),
        dt,
    )


# -------------------------------------------------------------------------
# Frequency-domain L2  (Pratt-Shin-Hicks)
# -------------------------------------------------------------------------
def test_freqL2_matches_explicit_rfft_formula():
    """Direct numerical check against 0.5 * sum |rfft(s) - rfft(o)|^2."""
    s, o, _ = _shifted_pair(dt_shift=0.005)
    Ds = torch.fft.rfft(s, dim=-3)
    Do = torch.fft.rfft(o, dim=-3)
    manual = 0.5 * ((Ds - Do).real ** 2 + (Ds - Do).imag ** 2).sum()
    auto = FrequencyDomainL2Loss(reduction="sum")(s, o)
    assert torch.allclose(manual, auto, rtol=1e-5, atol=1e-7)


def test_freqL2_zero_for_identical():
    s = torch.randn(1, 256, 3, 1)
    assert float(FrequencyDomainL2Loss(reduction="sum")(s, s)) < 1e-10


def test_freqL2_band_restriction():
    s, o, _ = _shifted_pair(dt_shift=0.005)
    full = FrequencyDomainL2Loss(reduction="sum")(s, o)
    band = FrequencyDomainL2Loss(freq_band=(5, 30), reduction="sum")(s, o)
    assert 0 < float(band) < float(full)


def test_freqL2_invalid_band():
    import pytest
    with pytest.raises(ValueError):
        FrequencyDomainL2Loss(freq_band=(5, 4))(torch.randn(1, 8, 1, 1),
                                                 torch.randn(1, 8, 1, 1))


# -------------------------------------------------------------------------
# Phase-only / amplitude-only / Shin-Min log
# -------------------------------------------------------------------------
def test_phase_only_zero_for_identical():
    s, o, _ = _shifted_pair()
    assert float(FrequencyPhaseLoss(reduction="sum")(s, s)) < 1e-12


def test_amplitude_only_zero_for_identical():
    s, o, _ = _shifted_pair()
    assert float(FrequencyAmplitudeLoss(reduction="sum")(s, s)) < 1e-12


def test_shin_min_log_zero_for_identical():
    s = torch.randn(1, 256, 2, 1, dtype=torch.float64) + 0.5
    assert float(LogarithmicShinMinLoss(reduction="sum")(s, s)) < 1e-10


def test_shin_min_log_decomposes_amp_plus_phase():
    """|log Ds - log Do|^2 = (log|Ds|-log|Do|)^2 + wrap(arg Ds - arg Do)^2."""
    s, o, _ = _shifted_pair(dt_shift=0.005)
    Ds = torch.fft.rfft(s, dim=-3)
    Do = torch.fft.rfft(o, dim=-3)
    log_amp = torch.log(torch.sqrt(Ds.real ** 2 + Ds.imag ** 2 + 1e-8)) - \
              torch.log(torch.sqrt(Do.real ** 2 + Do.imag ** 2 + 1e-8))
    Phi_s = torch.atan2(Ds.imag, Ds.real)
    Phi_o = torch.atan2(Do.imag, Do.real)
    dphi = torch.atan2(torch.sin(Phi_s - Phi_o), torch.cos(Phi_s - Phi_o))
    manual = 0.5 * (log_amp ** 2 + dphi ** 2).sum()
    auto = LogarithmicShinMinLoss(reduction="sum")(s, o)
    assert torch.allclose(manual, auto, rtol=1e-5, atol=1e-7)


def test_amplitude_only_invariant_to_time_shift():
    """|D(omega)| is shift-invariant => amplitude misfit on a pure shift = 0."""
    s, o, _ = _shifted_pair(dt_shift=0.005, nt=1024)
    loss = FrequencyAmplitudeLoss(reduction="sum")(s, o)
    # very small but not exactly zero because the FFT of a finite shifted
    # window picks up boundary energy
    assert float(loss) < 1e-3


def test_phase_only_grows_with_shift():
    losses = []
    for dt_shift in [0.0, 0.001, 0.003, 0.007]:
        s, o, _ = _shifted_pair(dt_shift=dt_shift, nt=1024)
        losses.append(float(FrequencyPhaseLoss(amplitude_weight=True, reduction="sum")(s, o)))
    assert all(b >= a - 1e-9 for a, b in zip(losses, losses[1:])), losses


# -------------------------------------------------------------------------
# Laplace L2
# -------------------------------------------------------------------------
def test_laplace_l2_zero_for_identical():
    s = torch.randn(1, 256, 2, 1, dtype=torch.float64)
    loss = LaplaceL2Loss(s=2.0, dt=1e-3, reduction="sum")(s, s)
    assert float(loss) < 1e-12


def test_laplace_l2_matches_manual_damped_l2():
    s, o, dt = _shifted_pair(dt_shift=0.003, nt=512)
    damp_s = 5.0
    t = torch.arange(s.shape[-3], dtype=s.dtype) * dt
    damp = torch.exp(-damp_s * t).view(1, -1, 1, 1)
    manual = 0.5 * ((damp * (s - o)) ** 2).sum()
    auto = LaplaceL2Loss(s=damp_s, dt=dt, reduction="sum")(s, o)
    assert torch.allclose(manual, auto, rtol=1e-5, atol=1e-7)


def test_laplace_l2_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        LaplaceL2Loss(s=0.0)
    with pytest.raises(ValueError):
        LaplaceL2Loss(dt=-1.0)


# -------------------------------------------------------------------------
# Autograd
# -------------------------------------------------------------------------
def test_all_have_gradients():
    s = torch.randn(1, 64, 2, 1, requires_grad=True)
    o = torch.randn(1, 64, 2, 1)
    for cls in (FrequencyDomainL2Loss, FrequencyAmplitudeLoss, FrequencyPhaseLoss,
                LogarithmicShinMinLoss):
        s.grad = None
        cls(reduction="sum")(s, o).backward()
        assert s.grad is not None and torch.isfinite(s.grad).all(), cls.__name__
    s.grad = None
    LaplaceL2Loss(s=1.0, dt=1e-3, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
