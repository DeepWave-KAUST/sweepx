"""Tests for the cross-correlation travel-time misfit (Luo & Schuster 1991)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import CrossCorrelationTraveltimeLoss, cross_correlation_traveltime_loss


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _shifted_pair(nt=512, dt=1e-3, t0=0.15, shift_samples=10, fc=15.0):
    t = np.arange(nt) * dt
    obs = _ricker(t, t0, fc)
    syn = _ricker(t, t0 + shift_samples * dt, fc)
    return (
        torch.tensor(syn, dtype=torch.float64).view(1, nt, 1, 1),
        torch.tensor(obs, dtype=torch.float64).view(1, nt, 1, 1),
        dt,
    )


def test_zero_for_identical_traces():
    syn, obs, dt = _shifted_pair(shift_samples=0)
    loss = CrossCorrelationTraveltimeLoss(dt=dt, power=4.0, reduction="sum")(syn, obs)
    assert float(loss) < 1e-10


def test_recovers_known_shift_for_unimodal_ricker():
    """For a single Ricker shifted by k samples, the smooth-centroid estimate
    should recover k * dt to within a fraction of dt."""
    shift = 12
    syn, obs, dt = _shifted_pair(shift_samples=shift, nt=1024)
    # power large + small gate so centroid is sharp
    loss = CrossCorrelationTraveltimeLoss(
        dt=dt, power=8.0, sigma=200.0, reduction="none"
    )(syn, obs)
    tau_star = math.sqrt(2.0 * float(loss[0]))  # tau* * dt
    # syn is observed at lag +shift relative to obs => tau* should equal -shift*dt
    expected = abs(shift) * dt
    assert math.isclose(tau_star, expected, rel_tol=0.05, abs_tol=0.5 * dt)


def test_monotone_in_shift_magnitude():
    """Larger absolute shifts give larger loss (other things equal)."""
    losses = []
    for k in [1, 4, 9, 16, 25]:
        syn, obs, dt = _shifted_pair(shift_samples=k, nt=1024)
        losses.append(
            float(
                CrossCorrelationTraveltimeLoss(
                    dt=dt, power=4.0, sigma=300.0, reduction="sum"
                )(syn, obs)
            )
        )
    # Strictly increasing
    assert all(b > a for a, b in zip(losses, losses[1:])), losses


def test_gradient_finite_and_nontrivial():
    syn, obs, dt = _shifted_pair(shift_samples=5, nt=512)
    syn = syn.float().clone().requires_grad_(True)
    obs = obs.float()
    CrossCorrelationTraveltimeLoss(
        dt=dt, power=4.0, sigma=100.0, reduction="sum"
    )(syn, obs).backward()
    assert syn.grad is not None
    assert torch.isfinite(syn.grad).all()
    assert syn.grad.abs().sum() > 0


def test_functional_alias_matches_module():
    syn, obs, dt = _shifted_pair(shift_samples=3)
    a = CrossCorrelationTraveltimeLoss(dt=dt, power=2.0, reduction="sum")(syn, obs)
    b = cross_correlation_traveltime_loss(syn, obs, dt=dt, power=2.0, reduction="sum")
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        CrossCorrelationTraveltimeLoss(dt=0.0)
    with pytest.raises(ValueError):
        CrossCorrelationTraveltimeLoss(power=-1)
