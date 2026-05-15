"""Tests for the Sinkhorn / entropic-OT FWI misfit."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import L2Loss, SinkhornLoss, sinkhorn_loss


def _gaussian(t, t0, sigma):
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_debiased_zero_for_identical():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_gaussian(t, 0.05, 0.01)).view(1, nt, 1, 1).double()
    eps = (10 * dt) ** 2  # spatial-scale-of-cost^2
    loss = SinkhornLoss(epsilon=eps, n_iter=128, positive="square",
                        dt=dt, debiased=True, reduction="sum")(o, o)
    assert float(loss) < 1e-5


def test_undebiased_differs_from_debiased():
    """Sanity check that the ``debiased`` flag actually changes the value."""
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_gaussian(t, 0.05, 0.01)).view(1, nt, 1, 1).double()
    o = torch.tensor(_gaussian(t, 0.07, 0.01)).view(1, nt, 1, 1).double()
    eps = (10 * dt) ** 2
    biased = float(SinkhornLoss(epsilon=eps, n_iter=128, positive="square",
                                dt=dt, debiased=False, reduction="sum")(s, o))
    debiased = float(SinkhornLoss(epsilon=eps, n_iter=128, positive="square",
                                  dt=dt, debiased=True, reduction="sum")(s, o))
    assert abs(biased - debiased) > 1e-6


def test_debiased_nonnegative_for_distinct_signals():
    """Debiased Sinkhorn is a divergence: >= 0, with equality iff p == q."""
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_gaussian(t, 0.05, 0.01)).view(1, nt, 1, 1).double()
    o = torch.tensor(_gaussian(t, 0.07, 0.01)).view(1, nt, 1, 1).double()
    eps = (10 * dt) ** 2
    loss = float(SinkhornLoss(epsilon=eps, n_iter=128, positive="square",
                              dt=dt, debiased=True, reduction="sum")(s, o))
    assert loss > 0


def test_grows_with_shift():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    obs = torch.tensor(_gaussian(t, 0.1, 0.02)).view(1, nt, 1, 1).double()
    eps = (10 * dt) ** 2
    losses = []
    for ms in (0.0, 0.005, 0.010, 0.020, 0.030):
        syn = torch.tensor(_gaussian(t, 0.1 + ms, 0.02)).view(1, nt, 1, 1).double()
        losses.append(float(SinkhornLoss(
            epsilon=eps, n_iter=128, positive="square", dt=dt,
            debiased=True, reduction="sum"
        )(syn, obs)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_sinkhorn_approaches_w2_squared_for_small_epsilon():
    """As epsilon -> 0, the debiased Sinkhorn divergence with cost
    C(t,t') = (t - t')^2 approaches the 2-Wasserstein W2^2."""
    nt, dt = 512, 5e-4
    t = np.arange(nt) * dt
    shift = 0.030
    obs = torch.tensor(_gaussian(t, 0.13, 0.005)).view(1, nt, 1, 1).double()
    syn = torch.tensor(_gaussian(t, 0.13 + shift, 0.005)).view(1, nt, 1, 1).double()
    # for a translated Gaussian, W2^2 = shift^2 exactly
    expected = shift ** 2
    eps_small = (2 * dt) ** 2
    sink = float(SinkhornLoss(
        epsilon=eps_small, n_iter=400, positive="square", dt=dt,
        debiased=True, reduction="sum"
    )(syn, obs))
    # Tolerance: numerical sinkhorn at eps=2dt^2 should hit within 25% of W2^2
    assert abs(sink - expected) < 0.25 * expected, (sink, expected)


def test_sinkhorn_monotone_across_l2_cycle_skipping():
    nt, dt = 1024, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.5)).view(1, nt, 1, 1).double()
    eps = (8 * dt) ** 2
    shifts = [0.0, 0.020, 0.030, 0.040, 0.060, 0.080]
    l2_vals, s_vals = [], []
    for ms in shifts:
        s = torch.tensor(_ricker(t, 0.5 + ms)).view(1, nt, 1, 1).double()
        l2_vals.append(float(L2Loss(reduction="sum")(s, o)))
        s_vals.append(float(SinkhornLoss(
            epsilon=eps, n_iter=80, positive="square", dt=dt,
            debiased=True, reduction="sum"
        )(s, o)))
    pairs = list(zip(l2_vals, l2_vals[1:]))
    assert any(b < a for a, b in pairs), f"L2 unexpectedly monotone: {l2_vals}"
    pairs = list(zip(s_vals, s_vals[1:]))
    assert all(b > a for a, b in pairs), f"Sinkhorn not monotone: {s_vals}"


def test_gradient_flow():
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_gaussian(t, 0.03, 0.01)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_gaussian(t, 0.035, 0.01)).view(1, nt, 1, 1).float()
    SinkhornLoss(epsilon=(3*dt)**2, n_iter=32, positive="square", dt=dt,
                 debiased=True, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        SinkhornLoss(epsilon=0)
    with pytest.raises(ValueError):
        SinkhornLoss(n_iter=0)
    with pytest.raises(ValueError):
        SinkhornLoss(positive="bogus")
    with pytest.raises(ValueError):
        SinkhornLoss(dt=-1)


def test_functional_alias():
    nt, dt = 32, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_gaussian(t, 0.01, 0.005)).view(1, nt, 1, 1)
    o = torch.tensor(_gaussian(t, 0.012, 0.005)).view(1, nt, 1, 1)
    a = SinkhornLoss(epsilon=(2*dt)**2, n_iter=16, dt=dt, positive="square")(s, o)
    b = sinkhorn_loss(s, o, epsilon=(2*dt)**2, n_iter=16, dt=dt, positive="square")
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)
