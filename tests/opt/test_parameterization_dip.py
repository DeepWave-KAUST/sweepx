"""Tests for sweepx.opt.parameterization.DIPParameterization and SmallUNet2D."""
import pytest
import torch
import torch.nn as nn

from sweepx.opt.parameterization import DIPParameterization, SmallUNet2D


def test_identity_generator_round_trip_via_random_latent():
    """A linear generator with init weights produces deterministic forward output."""
    gen = nn.Linear(4, 6)
    p = DIPParameterization(gen, latent_shape=(1, 4))
    z = torch.zeros(1, 4)
    out = p.forward(z)
    assert out.shape == (1, 6)


def test_inverse_raises():
    gen = nn.Linear(4, 6)
    p = DIPParameterization(gen, latent_shape=(1, 4))
    with pytest.raises(NotImplementedError):
        p.inverse(torch.zeros(1, 6))


def test_bounded_output_via_sigmoid():
    gen = nn.Linear(4, 6)
    p = DIPParameterization(gen, latent_shape=(1, 4), out_min=1500.0, out_max=5500.0)
    z = torch.randn(1, 4) * 5
    out = p.forward(z)
    assert (out > 1500.0).all() and (out < 5500.0).all()


def test_gradient_flows_through_generator():
    """The optimizer iterates the latent; gradient must flow through the generator."""
    gen = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 6))
    p = DIPParameterization(gen, latent_shape=(1, 4))
    z = torch.randn(1, 4, requires_grad=True)
    out = p.forward(z)
    loss = out.pow(2).sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().max().item() > 0.0


def test_random_latent_helper_shape():
    gen = nn.Linear(8, 6)
    p = DIPParameterization(gen, latent_shape=(1, 8))
    z = p.random_latent(scale=0.5)
    assert z.shape == (1, 8)
    assert z.requires_grad


def test_small_unet_preserves_spatial_shape():
    net = SmallUNet2D(in_channels=4, out_channels=1, base_channels=8)
    x = torch.randn(1, 4, 16, 16)
    out = net(x)
    # SmallUNet2D returns (B, H, W) (squeeze channel dim) when out_channels=1
    assert out.shape == (1, 16, 16)


def test_small_unet_dip_optim_loop_runs():
    """End-to-end smoke test: optimize the latent to match a target via SGD."""
    torch.manual_seed(0)
    net = SmallUNet2D(in_channels=4, out_channels=1, base_channels=8)
    p = DIPParameterization(net, latent_shape=(1, 4, 8, 8))
    z = p.random_latent(scale=0.1)
    target = torch.randn(1, 8, 8) * 0.1
    opt = torch.optim.Adam([z, *net.parameters()], lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        out = p.forward(z)
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
    final = (p.forward(z) - target).pow(2).mean().item()
    assert final < 0.1  # any decrease is fine; DIP fits the low-frequency content fast
