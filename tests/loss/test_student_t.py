"""Tests for :class:`fwiloss.StudentTLoss`."""

from __future__ import annotations

import math

import torch

from fwiloss import CauchyLoss, L2Loss, StudentTLoss, student_t_loss


def test_formula_pointwise():
    nu, sigma = 3.0, 0.5
    r = torch.tensor([-2.0, 0.0, 0.4, 1.7])
    syn = r.view(1, -1, 1, 1)
    obs = torch.zeros_like(syn)
    ours = StudentTLoss(nu=nu, sigma=sigma, reduction="none")(syn, obs).view(-1)
    u = (r / sigma) ** 2
    expected = 0.5 * (nu + 1.0) * torch.log1p(u / nu)
    assert torch.allclose(ours, expected, rtol=1e-6, atol=1e-7)


def test_zero_at_zero():
    syn = torch.zeros(1, 8, 2, 1)
    obs = torch.zeros_like(syn)
    assert float(StudentTLoss(nu=2.0, sigma=1.0, reduction="sum")(syn, obs)) == 0.0


def test_nu1_matches_cauchy_up_to_factor():
    """Student-t with nu=1 reproduces the Cauchy form up to a constant factor.

    Cauchy:   rho_C(r) = (sigma^2/2) log(1 + (r/sigma)^2)
    Student-1:rho_t(r) = log(1 + (r/sigma)^2)
              =>  rho_t = (2/sigma^2) * rho_C
    """
    sigma = 0.7
    syn = torch.randn(1, 32, 4, 1)
    obs = torch.randn_like(syn)
    st = StudentTLoss(nu=1.0, sigma=sigma, reduction="sum")(syn, obs)
    ca = CauchyLoss(c=sigma, reduction="sum")(syn, obs)
    assert torch.allclose(st, (2.0 / sigma ** 2) * ca, rtol=1e-5, atol=1e-7)


def test_nu_large_approaches_l2_up_to_factor():
    """For nu -> inf, rho_t(r) -> (1/2) (r/sigma)^2 = (1/sigma^2) * J_L2."""
    nu = 1e5
    sigma = 1.0
    syn = torch.randn(1, 64, 4, 1, dtype=torch.float64)
    obs = torch.randn_like(syn)
    st = StudentTLoss(nu=nu, sigma=sigma, reduction="sum")(syn, obs)
    l2 = L2Loss(half=True, reduction="sum")(syn, obs)
    assert torch.allclose(st, l2 / (sigma ** 2), rtol=1e-3)


def test_full_nll_adds_constant():
    nu, sigma = 4.0, 1.2
    syn = torch.randn(1, 16, 2, 1)
    obs = torch.randn_like(syn)
    nll = StudentTLoss(nu=nu, sigma=sigma, full_nll=True, reduction="sum")(syn, obs)
    base = StudentTLoss(nu=nu, sigma=sigma, full_nll=False, reduction="sum")(syn, obs)
    const_per_sample = (
        math.lgamma(nu / 2.0)
        - math.lgamma((nu + 1.0) / 2.0)
        + 0.5 * math.log(nu * math.pi)
        + math.log(sigma)
    )
    n = syn.numel()
    assert torch.allclose(nll - base, torch.tensor(n * const_per_sample), rtol=1e-5)


def test_gradient_bounded_for_large_residual():
    """Influence function saturates for large |r|."""
    syn = torch.tensor([1e3]).view(1, 1, 1, 1).clone()
    syn.requires_grad_(True)
    obs = torch.zeros_like(syn)
    StudentTLoss(nu=2.0, sigma=1.0, reduction="sum")(syn, obs).backward()
    # rho_t'(r) = (nu+1) r / (nu sigma^2 + r^2) -> (nu+1)/r for large r.
    # For r=1e3, the gradient should be ~3/1000.
    assert syn.grad.abs().item() < 1e-1


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        StudentTLoss(nu=0.0)
    with pytest.raises(ValueError):
        StudentTLoss(sigma=-1.0)


def test_functional_alias():
    syn = torch.randn(1, 16, 2, 1)
    obs = torch.randn_like(syn)
    assert torch.allclose(
        student_t_loss(syn, obs, nu=2.0, sigma=0.5),
        StudentTLoss(nu=2.0, sigma=0.5)(syn, obs),
    )
