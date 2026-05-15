"""DIP-FWI: deep-image-prior reparameterization.

Represents the physical model as the output of a randomly-initialized neural
network ``generator(latent)``. The optimizer iterates ``latent`` (and,
optionally, the generator weights), while the FWI solver consumes
``generator(latent)``. The implicit regularization comes from the
*spectral bias* of the network: low-frequency structure is fitted faster
than high-frequency noise.

This module provides:

- :class:`DIPParameterization` - the user-facing wrapper.
- :class:`SmallUNet2D` - a minimal reference generator that the user can
  swap in for FWI experiments.

Refs:
    Lempitsky, V., Vedaldi, A., and Ulyanov, D. (2018). "Deep Image Prior."
    CVPR. DOI:10.1109/CVPR.2018.00984.

    He, T., et al. (2024). "Deep image prior full-waveform inversion."
    JGR: Machine Learning and Computation 1, e2024JH000227.
    DOI:10.1029/2024JH000227.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sweep_opt.parameterization.base import Parameterization


class DIPParameterization(Parameterization):
    """Wrap a neural-network generator as a reparameterization.

    Args:
        generator: an ``nn.Module`` mapping the latent (shape
            ``latent_shape``, batch axis included) to a physical-model
            tensor of the desired output shape.
        latent_shape: shape of the latent input the generator expects
            (e.g. ``(1, 16, 8, 8)`` for a small 2-D generator).
        out_min, out_max: optional output rescaling. If both supplied,
            ``forward`` adds an affine map ``physical = out_min + (out_max - out_min) * sigmoid(generator_output)``
            so the physical model is bounded.

    Calling convention:
        ``forward(latent)`` returns the physical tensor; ``inverse`` is
        not naturally invertible for arbitrary networks, so calling it
        raises NotImplementedError. Initialise the latent randomly and
        either fit ``generator(latent_init)`` to the initial physical
        model offline, or just start from the random initialization.
    """

    def __init__(
        self,
        generator: nn.Module,
        latent_shape: Tuple[int, ...],
        *,
        out_min: float = None,
        out_max: float = None,
    ) -> None:
        if (out_min is None) != (out_max is None):
            raise ValueError("specify both out_min and out_max, or neither")
        self.generator = generator
        self.latent_shape = tuple(latent_shape)
        self.out_min = out_min
        self.out_max = out_max

    def forward(self, latent: Tensor) -> Tensor:
        out = self.generator(latent)
        if self.out_min is not None:
            span = self.out_max - self.out_min
            out = self.out_min + span * torch.sigmoid(out)
        return out

    def inverse(self, physical: Tensor) -> Tensor:
        raise NotImplementedError(
            "DIPParameterization.inverse is not generally invertible. "
            "Initialise the latent randomly (e.g. torch.randn(latent_shape) * 0.1) "
            "and accept that generator(latent_init) does not match the initial model."
        )

    def random_latent(self, scale: float = 0.1, device=None,
                      dtype: torch.dtype = torch.float32) -> Tensor:
        """Return a randomly-initialised latent of the right shape."""
        return scale * torch.randn(*self.latent_shape, dtype=dtype, device=device,
                                   requires_grad=True)


class SmallUNet2D(nn.Module):
    """A minimal 2-D U-Net generator suitable for small-scale DIP-FWI tests.

    Args:
        in_channels: number of input channels of the latent.
        out_channels: number of channels of the physical model (1 for vp).
        base_channels: width of the feature maps; default 16.

    The output spatial shape equals the input latent's spatial shape, so
    the user is responsible for choosing a latent whose spatial size
    matches the desired model grid (or interpolating after).
    """

    def __init__(self, in_channels: int = 16, out_channels: int = 1,
                 base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.enc1 = block(in_channels, c)
        self.enc2 = block(c, 2 * c)
        self.bottleneck = block(2 * c, 4 * c)
        self.dec2 = block(4 * c + 2 * c, 2 * c)
        self.dec1 = block(2 * c + c, c)
        self.out = nn.Conv2d(c, out_channels, kernel_size=1)

        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1).squeeze(1)
