"""Tests for the instantaneous-phase and envelope+phase combined misfits."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import (
    EnvelopeLoss,
    EnvelopePhaseLoss,
    InstantaneousPhaseLoss,
    envelope_phase_loss,
    instantaneous_phase_loss,
)
from fwiloss._utils import instantaneous_phase


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _shifted_pair(nt=512, dt=1e-3, t0=0.15, dt_shift=0.0, fc=15.0):
    t = np.arange(nt) * dt
    obs = _ricker(t, t0, fc)
    syn = _ricker(t, t0 + dt_shift, fc)
    return (
        torch.tensor(syn, dtype=torch.float64).view(1, nt, 1, 1),
        torch.tensor(obs, dtype=torch.float64).view(1, nt, 1, 1),
    )


# -------------------------------------------------------------------------
# Instantaneous phase
# -------------------------------------------------------------------------
def test_inst_phase_zero_when_signals_equal():
    s, o = _shifted_pair()
    loss = InstantaneousPhaseLoss(reduction="sum")(s, s)
    assert float(loss) < 1e-12


def test_inst_phase_invariant_to_amplitude_scaling():
    """phi(c d) = phi(d) for c > 0 (constant amplitude scaling leaves the
    instantaneous phase unchanged).

    The envelope weight ``w = E_o / max E_o`` depends only on the obs
    signal, so multiplying syn by a positive scalar should leave the
    envelope-weighted phase misfit invariant up to round-off.  Without
    the envelope weight, samples with tiny |a(t)| produce essentially
    random atan2 values - that branch is not amplitude-invariant in
    finite precision and is not asserted here.
    """
    s, o = _shifted_pair(dt_shift=0.003)
    a = InstantaneousPhaseLoss(envelope_weight=True, reduction="sum")(s, o)
    b = InstantaneousPhaseLoss(envelope_weight=True, reduction="sum")(s * 3.7, o)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_inst_phase_monotone_in_small_shift():
    losses = []
    for dt_shift in [0.0, 0.001, 0.003, 0.007, 0.012]:
        s, o = _shifted_pair(dt_shift=dt_shift, nt=1024)
        losses.append(float(
            InstantaneousPhaseLoss(envelope_weight=True, reduction="sum")(s, o)
        ))
    assert all(b >= a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_inst_phase_polarity_sensitive():
    """Unlike the envelope misfit, instantaneous phase IS polarity sensitive."""
    s, _ = _shifted_pair()
    # flipping polarity adds pi to the phase => not zero
    loss = InstantaneousPhaseLoss(reduction="sum", envelope_weight=False)(s, -s)
    assert float(loss) > 0.5


def test_wrap_in_minus_pi_pi():
    """Phase residuals must stay in (-pi, pi]; check on a long random signal."""
    s = torch.randn(1, 1024, 4, 1, dtype=torch.float64)
    o = torch.randn(1, 1024, 4, 1, dtype=torch.float64)
    out = InstantaneousPhaseLoss(envelope_weight=False, reduction="none")(s, o)
    # out = 0.5 * dphi^2;  dphi in (-pi, pi] => out <= pi^2/2
    assert float(out.max()) <= 0.5 * math.pi ** 2 + 1e-9


# -------------------------------------------------------------------------
# Envelope + phase combined
# -------------------------------------------------------------------------
def test_envelope_phase_alpha_endpoints():
    s, o = _shifted_pair(dt_shift=0.003)
    e_only = EnvelopePhaseLoss(alpha=0.0, reduction="sum")(s, o)
    p_only = EnvelopePhaseLoss(alpha=1.0, reduction="sum")(s, o)
    env = EnvelopeLoss(p=2, reduction="sum")(s, o)
    phi = InstantaneousPhaseLoss(reduction="sum")(s, o)
    assert torch.allclose(e_only, env, rtol=1e-5, atol=1e-7)
    assert torch.allclose(p_only, phi, rtol=1e-5, atol=1e-7)


def test_envelope_phase_linear_combination():
    s, o = _shifted_pair(dt_shift=0.003)
    for alpha in [0.25, 0.5, 0.75]:
        mixed = EnvelopePhaseLoss(alpha=alpha, reduction="sum")(s, o)
        env = EnvelopeLoss(p=2, reduction="sum")(s, o)
        phi = InstantaneousPhaseLoss(reduction="sum")(s, o)
        expected = (1 - alpha) * env + alpha * phi
        assert torch.allclose(mixed, expected, rtol=1e-5, atol=1e-7), alpha


def test_envelope_phase_gradient_flow():
    s, o = _shifted_pair(dt_shift=0.003)
    s = s.float().clone().requires_grad_(True)
    o = o.float()
    EnvelopePhaseLoss(alpha=0.5, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_invalid_alpha():
    import pytest
    with pytest.raises(ValueError):
        EnvelopePhaseLoss(alpha=-0.1)
    with pytest.raises(ValueError):
        EnvelopePhaseLoss(alpha=1.5)


def test_functional_alias():
    s, o = _shifted_pair()
    a = InstantaneousPhaseLoss()(s, o)
    b = instantaneous_phase_loss(s, o)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
    c = EnvelopePhaseLoss(alpha=0.3)(s, o)
    d = envelope_phase_loss(s, o, alpha=0.3)
    assert torch.allclose(c, d, rtol=1e-7, atol=1e-9)
