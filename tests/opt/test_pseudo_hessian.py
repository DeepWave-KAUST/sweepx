"""Tests for sweepx.opt.precond.PseudoHessianPrecond."""
import pytest
import torch

from sweepx.opt.precond import IdentityPrecond, PseudoHessianPrecond
from sweepx.opt.optim import SteepestDescent


def test_identity_when_no_illumination_passed():
    precond = PseudoHessianPrecond(eps=1e-3)
    p = torch.zeros(8, requires_grad=True)
    g = torch.randn(8)
    out = precond(g, p)
    # No update() call -> identity behaviour.
    assert torch.equal(out, g)


def test_matches_closed_form_on_born_fixture(born_illumination_1d):
    raw_grad, source_illum, receiver_illum = born_illumination_1d
    p = torch.zeros_like(raw_grad, requires_grad=True)

    eps = 1e-3
    precond = PseudoHessianPrecond(eps=eps, normalize=False)
    precond.update([p], source_illum=source_illum, receiver_illum=receiver_illum)

    expected = raw_grad / (source_illum * receiver_illum + eps)
    assert torch.allclose(precond(raw_grad, p), expected, atol=1e-6)


def test_normalize_keeps_max_at_one(born_illumination_1d):
    _, source_illum, receiver_illum = born_illumination_1d
    p = torch.zeros_like(source_illum, requires_grad=True)
    precond = PseudoHessianPrecond(eps=1e-3, normalize=True)
    precond.update([p], source_illum=source_illum, receiver_illum=receiver_illum)
    diag = precond.diagonal_for(p)
    assert diag is not None
    assert torch.isclose(diag.max(), torch.tensor(1.0))


def test_idempotent_update():
    p = torch.zeros(4, requires_grad=True)
    precond = PseudoHessianPrecond()
    s = torch.tensor([1.0, 2.0, 4.0, 8.0])
    r = torch.tensor([1.0, 1.0, 1.0, 1.0])
    precond.update([p], source_illum=s, receiver_illum=r)
    diag1 = precond.diagonal_for(p).clone()
    precond.update([p], source_illum=s, receiver_illum=r)
    diag2 = precond.diagonal_for(p).clone()
    assert torch.equal(diag1, diag2)


def test_dict_keying_for_multi_param():
    p1 = torch.zeros(3, requires_grad=True)
    p2 = torch.zeros(2, requires_grad=True)
    precond = PseudoHessianPrecond(normalize=False)
    s = {id(p1): torch.tensor([1.0, 1.0, 1.0]),
         id(p2): torch.tensor([2.0, 4.0])}
    r = {id(p1): torch.tensor([1.0, 2.0, 3.0]),
         id(p2): torch.tensor([1.0, 1.0])}
    precond.update([p1, p2], source_illum=s, receiver_illum=r)
    g1 = torch.ones(3)
    g2 = torch.ones(2)
    eps = precond.eps
    expected1 = g1 / (s[id(p1)] * r[id(p1)] + eps)
    expected2 = g2 / (s[id(p2)] * r[id(p2)] + eps)
    assert torch.allclose(precond(g1, p1), expected1)
    assert torch.allclose(precond(g2, p2), expected2)


def test_single_tensor_form_rejects_multi_param():
    p1 = torch.zeros(3, requires_grad=True)
    p2 = torch.zeros(2, requires_grad=True)
    precond = PseudoHessianPrecond()
    with pytest.raises(ValueError):
        precond.update([p1, p2], source_illum=torch.ones(3))


def test_composes_with_steepest_descent_on_quadratic(quadratic_illconditioned, make_closure):
    """Pseudo-Hessian-style preconditioning should make a poorly-scaled SD converge faster.

    Use the diagonal of the Hessian itself as the 'illumination' product so the
    preconditioner becomes the exact Newton diagonal scaling. SD with that
    preconditioner converges in O(1) iterations.
    """
    f, _, x_star = quadratic_illconditioned
    eigs = torch.tensor([1.0, 5.0, 20.0, 50.0, 100.0])

    x = torch.zeros(5, requires_grad=True)
    precond = PseudoHessianPrecond(eps=1e-12, normalize=False)
    # Use sqrt(eigs) for both S and R so S * R + eps approx eigs (the Hessian diagonal).
    sqrt_eigs = eigs.sqrt()
    precond.update([x], source_illum=sqrt_eigs, receiver_illum=sqrt_eigs)

    opt = SteepestDescent([x], lr=1.0, preconditioner=precond)
    for _ in range(20):
        opt.zero_grad()
        f(x).backward()
        opt.step()
    assert torch.allclose(x.detach(), x_star, atol=1e-4)


def test_identity_precond_acts_as_identity():
    p = torch.zeros(3, requires_grad=True)
    g = torch.tensor([1.0, -2.0, 3.0])
    precond = IdentityPrecond()
    precond.update([p])  # no-op
    assert torch.equal(precond(g, p), g)
