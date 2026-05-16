"""Deep Image Prior — a small U-Net fed by a fixed random latent.

Ulyanov, Vedaldi, Lempitsky (2018), "Deep Image Prior".
DOI: https://doi.org/10.1109/CVPR.2018.00984

For FWI, the latent is fixed and only the network's weights are optimized.
Empirically this gives a strong implicit smoothness prior on `vp`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reparam import Reparameterizer


def _conv_block(in_ch: int, out_ch: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class _UNet2D(nn.Module):
    """Minimal 2-D U-Net used by :class:`DIPReparam`."""

    def __init__(self, in_ch: int = 8, base_ch: int = 32, depth: int = 4) -> None:
        super().__init__()
        chs = [base_ch * (2 ** i) for i in range(depth)]
        # encoder
        self.encs = nn.ModuleList()
        c_prev = in_ch
        for c in chs:
            self.encs.append(_conv_block(c_prev, c))
            c_prev = c
        # bottleneck
        self.bot = _conv_block(chs[-1], chs[-1] * 2)
        # decoder
        self.decs = nn.ModuleList()
        c_prev = chs[-1] * 2
        for c in reversed(chs):
            self.decs.append(_conv_block(c_prev + c, c))
            c_prev = c
        self.out_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bot(x)
        for dec, skip in zip(self.decs, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.out_conv(x)


class DIPReparam(Reparameterizer):
    """Deep Image Prior reparameterizer (2-D).

    Parameters
    ----------
    out_shape
        ``(nz, nx)``.
    latent_channels
        Number of channels in the fixed input noise.
    base_ch, depth
        U-Net width and depth.
    """

    def __init__(
        self,
        out_shape: Tuple[int, int],
        *,
        latent_channels: int = 8,
        base_ch: int = 32,
        depth: int = 4,
        vp_min: float = 1500.0,
        vp_max: float = 4500.0,
        squash: str = "tanh",
    ) -> None:
        if len(out_shape) != 2:
            raise ValueError(f"DIPReparam is 2-D; got out_shape={out_shape}")
        super().__init__(out_shape=out_shape, vp_min=vp_min, vp_max=vp_max, squash=squash)
        self.unet = _UNet2D(in_ch=latent_channels, base_ch=base_ch, depth=depth)
        # latent is registered as a buffer (frozen, not learned)
        z = torch.randn(1, latent_channels, *out_shape)
        self.register_buffer("z", z, persistent=True)

    def _raw_forward(self) -> torch.Tensor:
        y = self.unet(self.z).squeeze(0).squeeze(0)  # -> (nz, nx)
        return y


__all__ = ["DIPReparam"]
