"""A compact, self-contained U-Net for DDPM-style diffusion models, 2D and 3D.

Standard ingredients: sinusoidal timestep embedding, GroupNorm+SiLU residual
blocks with time conditioning, strided-conv down / nearest-up sampling, and
optional multi-head self-attention at chosen resolutions.  No external
dependencies beyond PyTorch — kept deliberately generic so the same network can
later serve as a learned prior in FWI.

The blocks are parameterised by ``dims`` (2 or 3) rather than duplicated, so
:class:`UNet2D` and :class:`UNet3D` share one implementation and one parameter
naming scheme.  Everything except the convolutions is dimension-agnostic:
GroupNorm, ``nn.Linear`` time projection and ``F.interpolate`` all take the
extra axis for free.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet2D", "UNet3D"]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a batch of (integer) timesteps -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # zero-pad the odd tail
        emb = F.pad(emb, (0, 1))
    return emb


def _conv(dims: int, *args, **kwargs) -> nn.Module:
    """``nn.Conv2d`` or ``nn.Conv3d`` depending on ``dims``."""
    return {2: nn.Conv2d, 3: nn.Conv3d}[dims](*args, **kwargs)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv residual block with additive time embedding."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        groups: int = 32,
        dropout: float = 0.0,
        dims: int = 2,
    ):
        super().__init__()
        self.dims = dims
        g_in = math.gcd(groups, in_ch)
        g_out = math.gcd(groups, out_ch)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = _conv(dims, in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _conv(dims, out_ch, out_ch, 3, padding=1)
        self.skip = _conv(dims, in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        emb = self.emb_proj(F.silu(t_emb))
        h = h + emb.reshape(*emb.shape, *((1,) * self.dims))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Multi-head self-attention over the spatial grid.

    Cost is quadratic in the number of cells, which bites much harder in 3D: a
    64-cube is 262144 tokens where a 64-square is 4096.  Keep 3D attention to
    the deepest (smallest) stages only.
    """

    def __init__(self, ch: int, num_heads: int = 4, groups: int = 32, dims: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(math.gcd(groups, ch), ch)
        self.qkv = _conv(dims, ch, ch * 3, 1)
        self.proj = _conv(dims, ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        n = math.prod(spatial)
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        nh = self.num_heads
        # (B, nh, N, c/nh)
        q, k, v = (t.reshape(b, nh, c // nh, n).transpose(-1, -2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, *spatial)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int, dims: int = 2):
        super().__init__()
        self.op = _conv(dims, ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int, dims: int = 2):
        super().__init__()
        self.conv = _conv(dims, ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class _UNetND(nn.Module):
    """U-Net that predicts a per-cell target (noise / v / x0) from (x_t, t).

    Args:
        in_channels: input/output channels (1 for single-parameter velocity).
        base_channels: width of the first stage.
        channel_mults: width multiplier per resolution stage.
        num_res_blocks: residual blocks per stage.
        attn_resolutions: grid sizes (cells per side) at which to insert attention.
        dropout: dropout inside residual blocks.
        image_size: nominal training resolution per side (used only to resolve attention).
        num_heads: attention heads.
        dims: 2 for images/sections, 3 for volumes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16, 8),
        dropout: float = 0.0,
        image_size: int = 64,
        num_heads: int = 4,
        groups: int = 32,
        dims: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.dims = dims
        t_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.time_embed_dim = base_channels
        attn_resolutions = set(attn_resolutions)

        self.init_conv = _conv(dims, in_channels, base_channels, 3, padding=1)

        # ---- encoder ----
        self.downs = nn.ModuleList()
        skip_chs = [base_channels]
        ch = base_channels
        res = image_size
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks = nn.ModuleList([ResBlock(ch, out_ch, t_dim, groups, dropout, dims)])
                ch = out_ch
                if res in attn_resolutions:
                    blocks.append(AttnBlock(ch, num_heads, groups, dims))
                self.downs.append(blocks)
                skip_chs.append(ch)
            if i != len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch, dims)]))
                skip_chs.append(ch)
                res //= 2

        # ---- bottleneck ----
        self.mid = nn.ModuleList(
            [
                ResBlock(ch, ch, t_dim, groups, dropout, dims),
                AttnBlock(ch, num_heads, groups, dims),
                ResBlock(ch, ch, t_dim, groups, dropout, dims),
            ]
        )

        # ---- decoder ----
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                blocks = nn.ModuleList(
                    [ResBlock(ch + skip_chs.pop(), out_ch, t_dim, groups, dropout, dims)]
                )
                ch = out_ch
                if res in attn_resolutions:
                    blocks.append(AttnBlock(ch, num_heads, groups, dims))
                self.ups.append(blocks)
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(ch, dims)]))
                res *= 2

        self.out_norm = nn.GroupNorm(math.gcd(groups, ch), ch)
        self.out_conv = _conv(dims, ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))
        h = self.init_conv(x)
        skips = [h]
        for stage in self.downs:
            if isinstance(stage[0], Downsample):
                h = stage[0](h)
            else:
                h = stage[0](h, t_emb)
                if len(stage) > 1:
                    h = stage[1](h)
            skips.append(h)

        h = self.mid[0](h, t_emb)
        h = self.mid[1](h)
        h = self.mid[2](h, t_emb)

        for stage in self.ups:
            if isinstance(stage[0], Upsample):
                h = stage[0](h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = stage[0](h, t_emb)
                if len(stage) > 1:
                    h = stage[1](h)

        return self.out_conv(F.silu(self.out_norm(h)))


class UNet2D(_UNetND):
    """Two-dimensional U-Net (velocity sections)."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("dims", None)
        super().__init__(*args, dims=2, **kwargs)


class UNet3D(_UNetND):
    """Three-dimensional U-Net (velocity volumes).

    Same architecture as :class:`UNet2D` with Conv3d throughout.  The defaults
    are deliberately narrower: a 3x3x3 kernel holds 3x the weights of a 3x3 one
    and every feature map carries an extra axis, so matching the 2D width would
    cost ~3x the parameters and ~S x 3 the FLOPs at side length S.  Attention
    also defaults to the deepest stage only, since its cost is quadratic in the
    cell count.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (8,),
        dropout: float = 0.0,
        image_size: int = 64,
        num_heads: int = 4,
        groups: int = 32,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            image_size=image_size,
            num_heads=num_heads,
            groups=groups,
            dims=3,
        )
