"""SIREN — Sinusoidal Representation Network.

Sitzmann et al. (2020), "Implicit Neural Representations with Periodic
Activation Functions", NeurIPS.
DOI: https://doi.org/10.48550/arXiv.2006.09661

A small MLP that maps spatial coordinates to a scalar field, with
sin activations. We cache the coordinate grid as a buffer so a single
``net()`` call is a forward pass with no Python-level loops.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .reparam import Reparameterizer


class _SineLayer(nn.Module):
    """Linear + sin(w0 * x), with paper-faithful initialization."""

    def __init__(
        self, in_features: int, out_features: int, *, w0: float = 30.0, is_first: bool = False
    ) -> None:
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(Reparameterizer):
    """SIREN reparameterizer.

    Parameters
    ----------
    out_shape
        ``(nz, nx)`` 2-D or ``(nz, ny, nx)`` 3-D output grid.
    hidden_features, hidden_layers
        Width and depth of the MLP.
    w0
        Initial-layer frequency. The paper recommends 30 for image-scale tasks;
        higher values fit higher spatial frequencies at the cost of slower
        convergence.
    """

    def __init__(
        self,
        out_shape: Tuple[int, ...],
        *,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        w0: float = 30.0,
        vp_min: float = 1500.0,
        vp_max: float = 4500.0,
        squash: str = "tanh",
    ) -> None:
        super().__init__(out_shape=out_shape, vp_min=vp_min, vp_max=vp_max, squash=squash)
        ndim = len(out_shape)
        if ndim not in (2, 3):
            raise ValueError(f"SIREN supports 2-D or 3-D out_shape; got {out_shape}")

        layers: list[nn.Module] = [
            _SineLayer(ndim, hidden_features, w0=w0, is_first=True)
        ]
        for _ in range(hidden_layers - 1):
            layers.append(_SineLayer(hidden_features, hidden_features, w0=w0))
        # Output layer: linear, with the same paper-style init.
        out_lin = nn.Linear(hidden_features, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / w0
            out_lin.weight.uniform_(-bound, bound)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

        # Cache normalized coordinate grid in [-1, 1].
        grids = [torch.linspace(-1.0, 1.0, s) for s in out_shape]
        coord = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1)  # (..., ndim)
        self.register_buffer("coord", coord.reshape(-1, ndim), persistent=False)

    def _raw_forward(self) -> torch.Tensor:
        y = self.net(self.coord).squeeze(-1)
        return y.reshape(self.out_shape)


__all__ = ["SIREN"]
