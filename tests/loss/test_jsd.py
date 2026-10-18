"""Tests for the Jensen-Shannon divergence FWI misfit (Yan et al. 2024)."""

from __future__ import annotations

import math

import numpy as np
import torch

from sweep_loss import JensenShannonLoss, L2Loss, jensen_shannon_loss


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _gaussian(t, t0, sigma):
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def test_zero_when_identical():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_gaussian(t, 0.05, 0.01)).view(1, nt, 1, 1).double()
    for positive in ("square", "abs", "exp", "linear"):
        loss = JensenShannonLoss(positive=positive, reduction="sum")(o, o)
        assert float(loss) < 1e-10, positive


def test_symmetric():
    """JSD(p, q) == JSD(q, p)."""
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_gaussian(t, 0.04, 0.012)).view(1, nt, 1, 1).double()
    o = torch.tensor(_gaussian(t, 0.05, 0.010)).view(1, nt, 1, 1).double()
    a = JensenShannonLoss(positive="square", reduction="sum")(s, o)
    b = JensenShannonLoss(positive="square", reduction="sum")(o, s)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)


def test_bounded_by_log2():
    """0 <= JSD(p, q) <= log 2 per trace."""
    nt, dt = 128, 1e-3
    s = torch.randn(2, nt, 3, 1, dtype=torch.float64)
    o = torch.randn(2, nt, 3, 1, dtype=torch.float64)
    out = JensenShannonLoss(positive="square", reduction="none")(s, o)
    assert (out >= -1e-10).all()
    assert (out <= math.log(2) + 1e-6).all()


def test_normalize_by_log2_scales_to_unit_interval():
    nt = 128
    s = torch.randn(1, nt, 2, 1, dtype=torch.float64)
    o = torch.randn(1, nt, 2, 1, dtype=torch.float64)
    out = JensenShannonLoss(
        positive="square", normalize_by_log2=True, reduction="none"
    )(s, o)
    assert (out >= -1e-10).all()
    assert (out <= 1.0 + 1e-6).all()


def test_grows_with_shift():
    losses = []
    nt, dt = 512, 1e-3
    t = np.arange(nt) * dt
    obs = torch.tensor(_gaussian(t, 0.2, 0.02)).view(1, nt, 1, 1).double()
    for shift in (0.0, 0.002, 0.005, 0.015, 0.030):
        syn = torch.tensor(_gaussian(t, 0.2 + shift, 0.02)).view(1, nt, 1, 1).double()
        losses.append(float(
            JensenShannonLoss(positive="square", reduction="sum")(syn, obs)
        ))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_jsd_saturates_at_log2_for_disjoint_signals():
    """For two signals with disjoint supports, p and q are disjoint
    distributions and JSD attains its upper bound log 2."""
    nt, dt = 512, 1e-3
    t = np.arange(nt) * dt
    # Two Gaussians far apart so their squared envelopes don't overlap.
    p = torch.tensor(_gaussian(t, 0.05, 0.005)).view(1, nt, 1, 1).double()
    q = torch.tensor(_gaussian(t, 0.40, 0.005)).view(1, nt, 1, 1).double()
    jsd = float(JensenShannonLoss(positive="square", reduction="sum")(p, q))
    assert abs(jsd - math.log(2)) < 1e-2


def test_gradient_flow():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.04)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).float()
    JensenShannonLoss(positive="square", reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_invalid_positive():
    import pytest
    with pytest.raises(ValueError):
        JensenShannonLoss(positive="garbage")


def test_functional_alias():
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.02)).view(1, nt, 1, 1)
    o = torch.tensor(_ricker(t, 0.025)).view(1, nt, 1, 1)
    a = JensenShannonLoss(positive="abs")(s, o)
    b = jensen_shannon_loss(s, o, positive="abs")
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)
