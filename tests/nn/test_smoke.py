"""Smoke tests for sweep-nn."""

import pytest
import torch

import sweep_nn
from sweep_nn.dip import DIPReparam
from sweep_nn.priors import LearnedPrior, perceptual_distance
from sweep_nn.reparam import Reparameterizer
from sweep_nn.siren import SIREN


def test_version():
    assert isinstance(sweep_nn.__version__, str)


def test_siren_output_shape_and_range():
    net = SIREN(out_shape=(16, 32), hidden_features=32, hidden_layers=2, vp_min=1500.0, vp_max=4500.0)
    vp = net()
    assert vp.shape == (16, 32)
    assert torch.all(vp >= 1500.0 - 1e-3)
    assert torch.all(vp <= 4500.0 + 1e-3)


def test_siren_3d():
    net = SIREN(out_shape=(8, 8, 8), hidden_features=16, hidden_layers=2)
    vp = net()
    assert vp.shape == (8, 8, 8)


def test_siren_backprop():
    net = SIREN(out_shape=(8, 16), hidden_features=16, hidden_layers=2)
    vp = net()
    loss = vp.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_dip_output_shape():
    net = DIPReparam(out_shape=(32, 32), latent_channels=4, base_ch=8, depth=2)
    vp = net()
    assert vp.shape == (32, 32)


def test_dip_latent_is_buffer_not_param():
    net = DIPReparam(out_shape=(16, 16), latent_channels=4, base_ch=8, depth=2)
    param_ids = {id(p) for p in net.parameters()}
    assert id(net.z) not in param_ids


def test_reparam_invalid_bounds():
    with pytest.raises(ValueError):
        SIREN(out_shape=(8, 8), vp_min=4000, vp_max=3000)


def test_reparam_invalid_squash():
    with pytest.raises(ValueError):
        SIREN(out_shape=(8, 8), squash="not-a-squash")


def test_learned_prior_freezes():
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 4, 3, padding=1),
    )
    prior = LearnedPrior(backbone)
    for p in prior.backbone.parameters():
        assert not p.requires_grad


def test_perceptual_distance_zero_when_identical():
    backbone = torch.nn.Conv2d(1, 4, 3, padding=1)
    prior = LearnedPrior(backbone)
    x = torch.randn(1, 1, 16, 16)
    d = perceptual_distance(prior, x, x)
    assert float(d) < 1e-6
