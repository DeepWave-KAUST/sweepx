"""Tests for :class:`fwiloss.HybridL1L2Loss`."""

from __future__ import annotations

import torch

from fwiloss import HybridL1L2Loss, PseudoHuberLoss, hybrid_l1l2_loss


def test_matches_pseudo_huber_numerically():
    syn = torch.randn(2, 32, 4, 1)
    obs = torch.randn_like(syn)
    for delta in (0.1, 1.0, 5.0):
        a = HybridL1L2Loss(delta=delta, reduction="sum")(syn, obs)
        b = PseudoHuberLoss(delta=delta, reduction="sum")(syn, obs)
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-7), delta


def test_quadratic_for_small_residuals():
    delta = 2.0
    r = torch.linspace(-0.01, 0.01, 7)
    syn = r.view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    loss = HybridL1L2Loss(delta=delta, reduction="none")(syn, obs).view(-1)
    assert torch.allclose(loss, 0.5 * r * r, atol=1e-6)


def test_linear_for_large_residuals():
    delta = 0.3
    r = torch.tensor([1.0e3, -1.0e3])
    syn = r.view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    loss = HybridL1L2Loss(delta=delta, reduction="none")(syn, obs).view(-1)
    expected = delta * r.abs() - delta * delta
    assert torch.allclose(loss, expected, rtol=1e-4)


def test_gradient_bounded_by_delta():
    delta = 0.5
    syn = torch.linspace(-20.0, 20.0, 100).view(1, -1, 1, 1).clone()
    syn.requires_grad_(True)
    obs = torch.zeros_like(syn)
    HybridL1L2Loss(delta=delta, reduction="sum")(syn, obs).backward()
    assert syn.grad.abs().max() <= delta + 1e-6


def test_functional_alias():
    syn = torch.randn(1, 16, 2, 1)
    obs = torch.randn_like(syn)
    assert torch.allclose(
        hybrid_l1l2_loss(syn, obs, delta=0.4),
        HybridL1L2Loss(delta=0.4)(syn, obs),
    )
