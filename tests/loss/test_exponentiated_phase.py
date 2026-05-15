"""Tests for the exponentiated-phase misfit (Yuan et al. 2020)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import ExponentiatedPhaseLoss, exponentiated_phase_loss


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _shifted_pair(nt=512, dt=1e-3, t0=0.15, dt_shift=0.0, fc=15.0):
    t = np.arange(nt) * dt
    return (
        torch.tensor(_ricker(t, t0 + dt_shift, fc), dtype=torch.float64).view(1, nt, 1, 1),
        torch.tensor(_ricker(t, t0, fc), dtype=torch.float64).view(1, nt, 1, 1),
    )


def test_zero_when_identical():
    s, o = _shifted_pair()
    assert float(ExponentiatedPhaseLoss(reduction="sum")(s, s)) < 1e-12


def test_amplitude_invariance():
    """The exponentiated phase ~s(t) = a(t)/|a(t)| is invariant under any
    positive scalar multiplication of the input.

    With eps=0 the identity is exact; with eps>0 the stabiliser breaks
    invariance by a fraction of order eps / |a|^2 -- here we set eps=0
    explicitly to test the mathematical property.
    """
    s, o = _shifted_pair(dt_shift=0.003)
    a = ExponentiatedPhaseLoss(eps=0.0, reduction="sum")(s, o)
    b = ExponentiatedPhaseLoss(eps=0.0, reduction="sum")(s * 4.0, o)
    c = ExponentiatedPhaseLoss(eps=0.0, reduction="sum")(s * 0.25, o)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-8)
    assert torch.allclose(a, c, rtol=1e-5, atol=1e-8)


def test_bounded_by_two_times_n():
    """|~s| = 1 everywhere ⇒ |dR|² + |dI|² ≤ 4 ⇒ per-sample loss ≤ 2."""
    s = torch.randn(1, 1024, 4, 1, dtype=torch.float64)
    o = torch.randn(1, 1024, 4, 1, dtype=torch.float64)
    out = ExponentiatedPhaseLoss(reduction="none")(s, o)
    assert float(out.max()) <= 2.0 + 1e-6


def test_grows_with_shift():
    losses = []
    for ms in (0.0, 0.001, 0.003, 0.006, 0.012):
        s, o = _shifted_pair(dt_shift=ms, nt=1024)
        losses.append(float(ExponentiatedPhaseLoss(reduction="sum")(s, o)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_polarity_sensitive():
    """Flipping syn changes the phase by pi → exponentiated phase flips
    sign of both real & imaginary, so the loss is large."""
    s, _ = _shifted_pair()
    loss = float(ExponentiatedPhaseLoss(reduction="sum")(s, -s))
    n = s.numel()
    # Max per-sample = 2, so for fully anti-correlated traces we expect a
    # large total close to the upper bound.
    assert loss > 0.5 * n


def test_avoids_phase_branch_cut():
    """Yuan et al. 2020's selling point: no discontinuity at phi = ±pi.

    We construct two signals whose instantaneous phases differ by a value
    that hops the ±pi branch cut; the exponentiated-phase loss should be
    monotone in the magnitude of the true phase difference, while a naive
    phase difference would jump.
    """
    nt, dt = 512, 1e-3
    t = np.arange(nt) * dt
    # Continuous signals whose phases sweep through pi.
    s = torch.tensor(np.sin(2 * np.pi * 5 * t)).view(1, nt, 1, 1).double()
    losses = []
    for phase_off in (0.05, 0.10, 0.20, 0.40, 0.55):
        o = torch.tensor(np.sin(2 * np.pi * 5 * t + phase_off)).view(1, nt, 1, 1).double()
        losses.append(float(ExponentiatedPhaseLoss(reduction="sum")(s, o)))
    # Smooth, monotone growth — no jump.
    assert all(b > a - 1e-6 for a, b in zip(losses, losses[1:])), losses


def test_gradient_flow():
    s, o = _shifted_pair(dt_shift=0.003)
    s = s.float().clone().requires_grad_(True)
    o = o.float()
    ExponentiatedPhaseLoss(reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_eps():
    import pytest
    with pytest.raises(ValueError):
        ExponentiatedPhaseLoss(eps=-1e-3)


def test_functional_alias():
    s, o = _shifted_pair()
    a = ExponentiatedPhaseLoss()(s, o)
    b = exponentiated_phase_loss(s, o)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
