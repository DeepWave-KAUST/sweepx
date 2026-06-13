"""Tests for the Soft-DTW FWI misfit (Cuturi & Blondel 2017)."""

from __future__ import annotations

import math

import numpy as np
import torch

from sweep_loss import L2Loss, SoftDTWLoss, soft_dtw_loss


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_zero_for_identical_small_gamma():
    """As gamma -> 0, Soft-DTW with identical signals approaches the classical
    DTW cost, which is 0 along the diagonal."""
    nt, dt = 32, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.01)).view(1, nt, 1, 1).double()
    loss = SoftDTWLoss(gamma=0.05, normalize_by_length=False, reduction="sum")(o, o)
    # gamma * log(K) tail term, K = #valid paths.  Bound loosely.
    assert float(loss) < 1e-3, float(loss)


def test_grows_with_shift():
    """Soft-DTW should grow with small shifts (where DTW alignment cost ~ shift^2)."""
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.03)).view(1, nt, 1, 1).double()
    losses = []
    for ms in (0.0, 0.002, 0.005, 0.010, 0.020):
        s = torch.tensor(_ricker(t, 0.03 + ms)).view(1, nt, 1, 1).double()
        losses.append(float(SoftDTWLoss(gamma=0.1, reduction="sum")(s, o)))
    # Soft-DTW is symmetric -- with gamma>0 a shift gives non-trivial gap
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_soft_dtw_smaller_than_l2_for_warped_signals():
    """Soft-DTW absorbs the time warp; L2 does not."""
    nt, dt = 48, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.02)).view(1, nt, 1, 1).double()
    # warp = constant time shift
    s = torch.tensor(_ricker(t, 0.020 + 0.005)).view(1, nt, 1, 1).double()
    l2 = float(L2Loss(reduction="sum")(s, o))
    dtw = float(SoftDTWLoss(gamma=0.05, normalize_by_length=False, reduction="sum")(s, o))
    assert dtw < l2


def test_gradient_flow():
    nt, dt = 32, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.01)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_ricker(t, 0.012)).view(1, nt, 1, 1).float()
    SoftDTWLoss(gamma=0.1, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_gamma_larger_smoother():
    """Larger gamma should yield smaller *raw* soft-DTW because the soft-min
    blends more paths (Cuturi & Blondel 2017, prop. 1).

    This monotonicity is a property of the raw soft-DTW, so we pin
    ``divergence=False`` here.  The default ``divergence=True`` subtracts the
    gamma-dependent self-terms sDTW(x,x)/sDTW(y,y) and is *not* monotone in
    gamma (see Blondel-Mensch-Vert 2020), so the assertion below only holds
    for the raw variant.
    """
    nt, dt = 32, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.012)).view(1, nt, 1, 1).double()
    o = torch.tensor(_ricker(t, 0.014)).view(1, nt, 1, 1).double()
    cold = float(SoftDTWLoss(gamma=0.01, divergence=False, normalize_by_length=False, reduction="sum")(s, o))
    hot = float(SoftDTWLoss(gamma=10.0, divergence=False, normalize_by_length=False, reduction="sum")(s, o))
    assert hot < cold


def test_invalid_params():
    import pytest
    with pytest.raises(ValueError):
        SoftDTWLoss(gamma=0)
    with pytest.raises(ValueError):
        SoftDTWLoss(gamma=-1)


def test_functional_alias():
    nt, dt = 16, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.004)).view(1, nt, 1, 1)
    o = torch.tensor(_ricker(t, 0.006)).view(1, nt, 1, 1)
    a = SoftDTWLoss(gamma=0.3)(s, o)
    b = soft_dtw_loss(s, o, gamma=0.3)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)
