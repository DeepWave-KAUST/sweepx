"""Sanity tests for :class:`fwiloss.L2Loss`.

These tests pin down the *formula* (1/2 sum (syn - obs)^2) and the analytic
properties everyone relies on:

* loss(syn=obs) == 0
* gradient w.r.t. syn equals the residual (syn - obs)  (under half=True, reduction='sum')
* numerical equality with ``torch.nn.functional.mse_loss`` up to the 1/2 factor
* shape-broadcasting through the canonical (nshots, nt, nrec, nchan) layout
* differentiability through ``loss.backward()``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fwiloss import L2Loss, l2_loss


def test_zero_when_equal(small_gather):
    syn, obs, _ = small_gather
    syn = obs.clone()
    assert torch.allclose(L2Loss()(syn, obs), torch.tensor(0.0))


def test_matches_mse_up_to_half_factor(small_gather):
    syn, obs, _ = small_gather
    ours = L2Loss(half=False, reduction="mean")(syn, obs)
    ref = F.mse_loss(syn, obs, reduction="mean")
    assert torch.allclose(ours, ref, rtol=1e-6, atol=1e-8)


def test_half_factor():
    syn = torch.randn(1, 64, 4, 1)
    obs = torch.randn(1, 64, 4, 1)
    full = L2Loss(half=False, reduction="sum")(syn, obs)
    half = L2Loss(half=True, reduction="sum")(syn, obs)
    assert torch.allclose(half * 2, full, rtol=1e-6, atol=1e-8)


def test_gradient_equals_residual():
    """For J = 1/2 sum r^2 with r = syn - obs, dJ/dsyn = r exactly."""
    torch.manual_seed(123)
    syn = torch.randn(2, 32, 3, 1, requires_grad=True)
    obs = torch.randn(2, 32, 3, 1)
    L2Loss(half=True, reduction="sum")(syn, obs).backward()
    assert torch.allclose(syn.grad, syn.detach() - obs, rtol=1e-6, atol=1e-8)


def test_accepts_lower_rank_inputs():
    nt = 128
    syn = torch.randn(nt, requires_grad=True)
    obs = torch.randn(nt)
    loss = L2Loss(reduction="mean")(syn, obs)
    loss.backward()
    assert loss.ndim == 0


def test_reduction_none_keeps_shape():
    syn = torch.randn(2, 64, 4, 1)
    obs = torch.randn(2, 64, 4, 1)
    out = L2Loss(reduction="none", half=True)(syn, obs)
    assert out.shape == syn.shape


def test_mask_excludes_elements():
    syn = torch.zeros(1, 8, 2, 1)
    obs = torch.zeros(1, 8, 2, 1)
    obs[0, 0, 0, 0] = 10.0     # this sample contributes 50.0 to the misfit
    mask = torch.ones_like(syn)
    mask[0, 0, 0, 0] = 0.0
    masked = L2Loss(mask=mask, reduction="sum", half=True)(syn, obs)
    assert torch.allclose(masked, torch.tensor(0.0))


def test_functional_alias_matches_module(small_gather):
    syn, obs, _ = small_gather
    a = L2Loss()(syn, obs)
    b = l2_loss(syn, obs)
    assert torch.allclose(a, b, rtol=1e-7, atol=1e-9)
