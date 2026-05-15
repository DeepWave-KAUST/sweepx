"""Tests for sweep_opt.optim.SOAP."""
import torch

from sweep_opt.optim import SOAP


def test_soap_converges_on_quadratic():
    x = torch.tensor([3.0, -2.0, 1.5], requires_grad=True)
    opt = SOAP([x], lr=1e-1, update_freq=5)
    for _ in range(500):
        opt.zero_grad()
        (x ** 2).sum().backward()
        opt.step()
    assert (x ** 2).sum().item() < 1e-3


def test_soap_2d_parameter_converges():
    torch.manual_seed(0)
    M = torch.randn(4, 5, requires_grad=True)
    M_target = torch.randn(4, 5)
    opt = SOAP([M], lr=1e-1, update_freq=5)
    for _ in range(500):
        opt.zero_grad()
        ((M - M_target) ** 2).sum().backward()
        opt.step()
    assert ((M - M_target) ** 2).sum().item() < 1e-2


def test_soap_bounds_clamp(bounded_quadratic_2d):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = SOAP([x], lr=1e-2, bounds=(lo, hi), update_freq=10)
    for _ in range(2000):
        opt.zero_grad(); f(x).backward(); opt.step()
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()
