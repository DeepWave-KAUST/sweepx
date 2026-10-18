"""Tests for :class:`sweep_loss.L1Loss`."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sweep_loss import L1Loss, l1_loss


def test_zero_when_equal(small_gather):
    syn, obs, _ = small_gather
    assert torch.allclose(L1Loss()(obs.clone(), obs), torch.tensor(0.0))


def test_matches_torch_l1_loss():
    syn = torch.randn(2, 64, 4, 2)
    obs = torch.randn_like(syn)
    ours = L1Loss(reduction="mean")(syn, obs)
    ref = F.l1_loss(syn, obs, reduction="mean")
    assert torch.allclose(ours, ref, rtol=1e-6, atol=1e-8)

    ours_sum = L1Loss(reduction="sum")(syn, obs)
    ref_sum = F.l1_loss(syn, obs, reduction="sum")
    assert torch.allclose(ours_sum, ref_sum, rtol=1e-6, atol=1e-8)


def test_gradient_is_sign():
    """For J = sum |r|,  dJ/dsyn = sign(syn - obs)  (a.e.)."""
    syn = torch.tensor([[[[0.0, 2.0, -3.0]]]]).permute(0, 3, 1, 2).requires_grad_(True)
    # shape (1, 3, 1, 1) -- canonical
    obs = torch.tensor([[[[1.0, 1.0, 1.0]]]]).permute(0, 3, 1, 2)
    L1Loss(reduction="sum")(syn, obs).backward()
    # residual = syn - obs = (-1, 1, -4) so sign = (-1, 1, -1)
    expected = torch.tensor([[[[-1.0, 1.0, -1.0]]]]).permute(0, 3, 1, 2)
    assert torch.allclose(syn.grad, expected)


def test_robust_to_outliers_vs_l2():
    """L1 < L2 contribution from a single huge outlier."""
    syn = torch.zeros(1, 8, 1, 1)
    obs = torch.zeros(1, 8, 1, 1)
    obs[0, 0, 0, 0] = 1e3
    from sweep_loss import L2Loss

    l1 = L1Loss(reduction="sum")(syn, obs)
    l2 = L2Loss(reduction="sum", half=False)(syn, obs)
    assert float(l1) == 1e3
    assert float(l2) == 1e6


def test_functional_alias():
    syn = torch.randn(1, 32, 3, 1)
    obs = torch.randn_like(syn)
    a = L1Loss()(syn, obs)
    b = l1_loss(syn, obs)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
