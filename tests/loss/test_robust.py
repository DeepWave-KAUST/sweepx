"""Tests for the Cauchy / Tukey / Geman-McClure robust misfits."""

from __future__ import annotations

import math

import torch

from fwiloss import (
    CauchyLoss,
    GemanMcClureLoss,
    L2Loss,
    TukeyLoss,
    cauchy_loss,
    geman_mcclure_loss,
    tukey_loss,
)


# ---------- Cauchy ---------------------------------------------------------
def test_cauchy_formula_pointwise():
    c = 0.7
    r = torch.tensor([-3.0, 0.0, 0.5, 2.0])
    syn = r.view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    ours = CauchyLoss(c=c, reduction="none")(syn, obs).view(-1)
    expected = 0.5 * c * c * torch.log1p((r * r) / (c * c))
    assert torch.allclose(ours, expected, rtol=1e-6, atol=1e-7)


def test_cauchy_zero_at_zero():
    assert float(CauchyLoss(c=1.0, reduction="sum")(torch.zeros(1, 8, 1, 1),
                                                   torch.zeros(1, 8, 1, 1))) == 0.0


def test_cauchy_grows_log_for_large_r():
    """As |r|/c -> inf, rho_C ~ c^2 log(|r|/c)."""
    c = 0.5
    r = torch.tensor([1e3])
    val = float(CauchyLoss(c=c, reduction="sum")(r.view(1, 1, 1, 1),
                                                  torch.zeros(1, 1, 1, 1)))
    expected = c * c * math.log(r.item() / c)
    assert math.isclose(val, expected, rel_tol=1e-2)


# ---------- Tukey ----------------------------------------------------------
def test_tukey_formula_inside_threshold():
    c = 1.0
    r = torch.tensor([-0.5, 0.0, 0.3])
    syn = r.view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    ours = TukeyLoss(c=c, reduction="none")(syn, obs).view(-1)
    inside = 1.0 - (r / c) ** 2
    expected = (c * c / 6.0) * (1.0 - inside ** 3)
    assert torch.allclose(ours, expected, rtol=1e-6, atol=1e-7)


def test_tukey_clamped_outside_threshold():
    c = 0.4
    syn = torch.tensor([10.0, -10.0]).view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    ours = TukeyLoss(c=c, reduction="none")(syn, obs).view(-1)
    expected = torch.full_like(ours, c * c / 6.0)
    assert torch.allclose(ours, expected, atol=1e-7)


def test_tukey_grad_vanishes_for_outliers():
    """Gradient is exactly zero for |r| > c (full outlier rejection)."""
    c = 0.3
    syn = torch.tensor([5.0, -7.0, 0.1]).view(1, -1, 1, 1).clone()
    syn.requires_grad_(True)
    obs = torch.zeros_like(syn)
    TukeyLoss(c=c, reduction="sum")(syn, obs).backward()
    g = syn.grad.view(-1)
    assert torch.allclose(g[:2], torch.zeros(2))
    assert g[2].abs() > 0


# ---------- Geman-McClure -------------------------------------------------
def test_geman_mcclure_bounded_by_csq():
    c = 0.5
    syn = torch.linspace(-100.0, 100.0, 200).view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    out = GemanMcClureLoss(c=c, reduction="none")(syn, obs)
    assert (out <= c * c + 1e-6).all()


def test_geman_mcclure_zero_at_zero():
    assert float(GemanMcClureLoss(c=1.0, reduction="sum")(torch.zeros(1, 4, 1, 1),
                                                          torch.zeros(1, 4, 1, 1))) == 0.0


def test_geman_mcclure_quadratic_for_small_r():
    """rho_GM(r) = r^2 + O(r^4/c^2) for |r|<<c."""
    c = 5.0
    r = torch.linspace(-0.001, 0.001, 5)
    syn = r.view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    loss = GemanMcClureLoss(c=c, reduction="none")(syn, obs).view(-1)
    assert torch.allclose(loss, r * r, atol=1e-9)


# ---------- All: comparison against L2 in small-r regime ------------------
def test_robust_losses_close_to_l2_for_tiny_residuals():
    """All three losses agree with L2 to leading order for tiny residuals.

    Use float64 so the O(u^2) truncation actually shows up rather than being
    swallowed by float32 round-off.
    """
    eps = 1e-3
    syn = torch.full((1, 64, 4, 1), eps, dtype=torch.float64)
    obs = torch.zeros_like(syn)
    l2 = L2Loss(half=True, reduction="sum")(syn, obs)
    # Each rho is c^2/2 log(1+u), c^2/6(1-(1-u)^3) ~ c^2/2 * u, c^2 u/(1+u)
    # where u = (r/c)^2 << 1.  All match L2 = 1/2 sum r^2 up to O(u^2).
    assert torch.allclose(CauchyLoss(c=1.0, reduction="sum")(syn, obs), l2, rtol=1e-3)
    assert torch.allclose(TukeyLoss(c=1.0, reduction="sum")(syn, obs), l2, rtol=1e-3)
    assert torch.allclose(GemanMcClureLoss(c=1.0, reduction="sum")(syn, obs),
                          2 * l2, rtol=1e-3)  # GM definition is 2x


# ---------- Functional API -------------------------------------------------
def test_functional_aliases():
    syn = torch.randn(1, 16, 2, 1)
    obs = torch.randn_like(syn)
    assert torch.allclose(cauchy_loss(syn, obs, c=0.5),
                          CauchyLoss(c=0.5)(syn, obs))
    assert torch.allclose(tukey_loss(syn, obs, c=0.5),
                          TukeyLoss(c=0.5)(syn, obs))
    assert torch.allclose(geman_mcclure_loss(syn, obs, c=0.5),
                          GemanMcClureLoss(c=0.5)(syn, obs))
