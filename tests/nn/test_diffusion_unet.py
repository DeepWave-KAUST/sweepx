"""Tests for the DDPM U-Net: 2D and 3D share one dims-parameterised implementation."""

import pytest
import torch

from sweep_nn.diffusion import GaussianDiffusion, UNet2D, UNet3D


def _tiny(cls):
    """Same architecture either side of the dims switch, small enough for CI."""
    return cls(base_channels=8, channel_mults=(1, 2), attn_resolutions=(8,),
               image_size=16, num_heads=2)


def test_unet2d_shape():
    net = _tiny(UNet2D)
    x = torch.randn(2, 1, 16, 16)
    assert net(x, torch.tensor([0, 5])).shape == x.shape


def test_unet3d_shape():
    net = _tiny(UNet3D)
    x = torch.randn(2, 1, 16, 16, 16)
    assert net(x, torch.tensor([0, 5])).shape == x.shape


def test_unet3d_is_all_conv3d():
    net = _tiny(UNet3D)
    assert any(isinstance(m, torch.nn.Conv3d) for m in net.modules())
    assert not any(isinstance(m, torch.nn.Conv2d) for m in net.modules())


def test_2d_and_3d_share_parameter_names():
    """The dims switch must not rename anything, or 2D checkpoints stop loading."""
    assert set(_tiny(UNet2D).state_dict()) == set(_tiny(UNet3D).state_dict())


def test_3d_kernels_carry_the_extra_axis():
    sd2, sd3 = _tiny(UNet2D).state_dict(), _tiny(UNet3D).state_dict()
    k = "init_conv.weight"
    assert sd2[k].ndim == 4 and sd3[k].ndim == 5
    assert sd3[k].shape[-3:] == (3, 3, 3)


@pytest.mark.parametrize("cls,shape", [(UNet2D, (1, 1, 16, 16)), (UNet3D, (1, 1, 16, 16, 16))])
def test_out_conv_zero_init(cls, shape):
    net = _tiny(cls)
    assert net(torch.randn(*shape), torch.tensor([3])).abs().max() == 0


@pytest.mark.parametrize("param", ["eps", "v", "x0"])
def test_diffusion_is_dimension_agnostic(param):
    net = _tiny(UNet3D)
    gd = GaussianDiffusion(num_timesteps=5, parameterization=param)
    loss = gd.loss(net, torch.randn(2, 1, 16, 16, 16))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_sampling_3d():
    net = _tiny(UNet3D)
    gd = GaussianDiffusion(num_timesteps=5)
    shape = (1, 1, 16, 16, 16)
    with torch.no_grad():
        assert gd.ddim_sample(net, shape, device="cpu", num_steps=2).shape == shape
        assert gd.sample(net, shape, device="cpu").shape == shape
