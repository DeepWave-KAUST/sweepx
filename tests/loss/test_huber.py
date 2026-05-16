"""Tests for :class:`sweep_loss.HuberLoss` and :class:`sweep_loss.PseudoHuberLoss`."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sweep_loss import HuberLoss, PseudoHuberLoss, huber_loss, pseudo_huber_loss


# ----------------------------------------------------------------------------
# Huber
# ----------------------------------------------------------------------------
def test_huber_matches_formula_pointwise():
    delta = 0.5
    r = torch.tensor([-2.0, -0.7, -0.1, 0.0, 0.3, 1.5])
    syn = r.view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    ours = HuberLoss(delta=delta, reduction="none")(syn, obs).view(-1)

    expected = torch.where(
        r.abs() <= delta,
        0.5 * r * r,
        delta * (r.abs() - 0.5 * delta),
    )
    assert torch.allclose(ours, expected, rtol=1e-6, atol=1e-7)


def test_huber_matches_torch_reference():
    delta = 0.9
    syn = torch.randn(2, 32, 4, 1)
    obs = torch.randn_like(syn)
    ours = HuberLoss(delta=delta, reduction="mean")(syn, obs)
    ref = F.huber_loss(syn, obs, reduction="mean", delta=delta)
    assert torch.allclose(ours, ref, rtol=1e-6, atol=1e-8)


def test_huber_reduces_to_l2_for_large_delta():
    syn = torch.randn(1, 64, 3, 1)
    obs = torch.randn_like(syn)
    huber = HuberLoss(delta=1e6, reduction="sum")(syn, obs)
    from sweep_loss import L2Loss

    l2 = L2Loss(half=True, reduction="sum")(syn, obs)
    assert torch.allclose(huber, l2, rtol=1e-5, atol=1e-6)


def test_huber_reduces_to_l1_for_small_delta_large_residual():
    """For |r| >> delta, h_delta(r) = delta*(|r| - delta/2) ~ delta*|r|."""
    delta = 1e-3
    syn = torch.full((1, 8, 1, 1), 10.0)
    obs = torch.zeros_like(syn)
    huber = HuberLoss(delta=delta, reduction="sum")(syn, obs)
    expected = delta * (syn.abs().sum() - 0.5 * delta * syn.numel())
    assert torch.allclose(huber, expected, rtol=1e-5, atol=1e-6)


def test_huber_grad_bounded_by_delta():
    """|dJ/dr| <= delta everywhere (the defining property of the Huber norm)."""
    delta = 0.4
    syn = torch.tensor([-5.0, -0.1, 0.0, 0.2, 3.0]).view(1, 5, 1, 1).clone()
    syn.requires_grad_(True)
    obs = torch.zeros_like(syn)
    HuberLoss(delta=delta, reduction="sum")(syn, obs).backward()
    assert syn.grad.abs().max() <= delta + 1e-6


# ----------------------------------------------------------------------------
# Pseudo-Huber
# ----------------------------------------------------------------------------
def test_pseudo_huber_quadratic_for_small_r():
    """tilde h(r) ~ 1/2 r^2 for |r| << delta."""
    delta = 1.0
    r = torch.linspace(-0.01, 0.01, 11)
    ours = PseudoHuberLoss(delta=delta, reduction="none")(
        r.view(1, -1, 1, 1), torch.zeros(1, 11, 1, 1)
    ).view(-1)
    # Leading-order: pseudo_huber = 1/2 r^2 - 1/8 r^4/delta^2 + ...
    assert torch.allclose(ours, 0.5 * r * r, atol=1e-6)


def test_pseudo_huber_linear_for_large_r():
    delta = 0.5
    r = torch.tensor([1e3])
    ours = float(PseudoHuberLoss(delta=delta, reduction="sum")(
        r.view(1, -1, 1, 1), torch.zeros(1, 1, 1, 1)
    ))
    expected = delta * float(r.abs().sum()) - delta * delta
    assert math.isclose(ours, expected, rel_tol=1e-3)


def test_pseudo_huber_zero_at_zero():
    syn = torch.zeros(1, 16, 2, 1)
    obs = torch.zeros_like(syn)
    assert torch.allclose(PseudoHuberLoss(0.7)(syn, obs), torch.tensor(0.0))


def test_functional_aliases():
    syn = torch.randn(1, 32, 3, 1)
    obs = torch.randn_like(syn)
    assert torch.allclose(huber_loss(syn, obs, delta=0.5), HuberLoss(delta=0.5)(syn, obs))
    assert torch.allclose(pseudo_huber_loss(syn, obs, delta=0.5),
                          PseudoHuberLoss(delta=0.5)(syn, obs))
