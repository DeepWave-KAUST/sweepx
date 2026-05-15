"""Tests for the local-similarity FWI misfit (Fomel 2007)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fwiloss import (
    GlobalCorrelationLoss,
    LocalSimilarityLoss,
    local_similarity_loss,
)


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_zero_for_identical():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.1)).view(1, nt, 1, 1).double()
    loss = LocalSimilarityLoss(sigma_samples=8.0, reduction="sum")(o, o)
    assert float(loss) < 1e-6


def test_invariant_to_per_trace_amplitude():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).double()
    o = torch.tensor(_ricker(t, 0.07)).view(1, nt, 1, 1).double()
    a = LocalSimilarityLoss(sigma_samples=8.0, reduction="sum")(s, o)
    b = LocalSimilarityLoss(sigma_samples=8.0, reduction="sum")(2.5 * s, 0.1 * o)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_grows_with_shift():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.1)).view(1, nt, 1, 1).double()
    losses = []
    for ms in (0.0, 0.001, 0.003, 0.007, 0.015):
        s = torch.tensor(_ricker(t, 0.1 + ms)).view(1, nt, 1, 1).double()
        losses.append(float(LocalSimilarityLoss(sigma_samples=8.0, reduction="sum")(s, o)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_local_more_local_than_global():
    """Local similarity highlights time-localised mismatches that global
    NCC washes out by averaging."""
    nt, dt = 512, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.25)).view(1, nt, 1, 1).double()
    # syn = obs except in a small time window where it is flipped
    s = o.clone()
    mask = (t > 0.2) & (t < 0.3)
    s[0, mask, 0, 0] = -s[0, mask, 0, 0]
    g_loss = float(GlobalCorrelationLoss(reduction="sum")(s, o))
    l_loss = float(LocalSimilarityLoss(sigma_samples=5.0, reduction="sum")(s, o))
    # local picks up the local sign flip strongly; global averages it down
    assert l_loss > 0.5
    assert g_loss < l_loss


def test_gradient_flow():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.04)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).float()
    LocalSimilarityLoss(sigma_samples=4.0, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_sigma():
    import pytest
    with pytest.raises(ValueError):
        LocalSimilarityLoss(sigma_samples=0)


def test_functional_alias():
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.03)).view(1, nt, 1, 1)
    o = torch.tensor(_ricker(t, 0.035)).view(1, nt, 1, 1)
    a = LocalSimilarityLoss(sigma_samples=4.0)(s, o)
    b = local_similarity_loss(s, o, sigma_samples=4.0)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)
