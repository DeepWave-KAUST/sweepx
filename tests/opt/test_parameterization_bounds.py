"""Tests for parameterization bounds."""
import torch

from sweepx.opt.parameterization import (
    IdentityParameterization,
    LogVelocity,
    PerturbationFromBackground,
    SigmoidBounds,
)


def test_identity_round_trip():
    z = torch.randn(8)
    p = IdentityParameterization()
    assert torch.equal(p.inverse(p.forward(z)), z)


def test_sigmoid_bounds_within_range():
    p = SigmoidBounds(vmin=1500.0, vmax=5500.0)
    z = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    out = p.forward(z)
    assert (out > 1500.0).all() and (out < 5500.0).all()


def test_sigmoid_bounds_round_trip():
    p = SigmoidBounds(vmin=0.0, vmax=10.0)
    z0 = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert torch.allclose(p.inverse(p.forward(z0)), z0, atol=1e-6)


def test_sigmoid_bounds_gradient_flows():
    p = SigmoidBounds(vmin=0.0, vmax=1.0)
    z = torch.tensor([0.0, 1.0], requires_grad=True)
    physical = p.forward(z)
    loss = (physical ** 2).sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().max().item() > 0.0


def test_log_velocity_round_trip():
    p = LogVelocity()
    physical = torch.tensor([1500.0, 3000.0, 5500.0])
    z = p.inverse(physical)
    assert torch.allclose(p.forward(z), physical, atol=1e-3)


def test_perturbation_round_trip():
    bg = torch.linspace(1500.0, 5500.0, 10)
    p = PerturbationFromBackground(bg)
    physical = bg + torch.randn(10) * 100.0
    z = p.inverse(physical)
    assert torch.allclose(p.forward(z), physical, atol=1e-6)


def test_perturbation_zero_latent_returns_background():
    bg = torch.linspace(1500.0, 5500.0, 10)
    p = PerturbationFromBackground(bg)
    z = torch.zeros(10)
    assert torch.equal(p.forward(z), bg)
