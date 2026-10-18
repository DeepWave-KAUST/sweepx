"""SIREN-based source-wavelet representation.

A 1-D SIREN that maps normalized time coordinates ``t ∈ [-1, 1]`` to a
scalar wavelet sample. Used to learn / smooth source signatures for FWI
when the field source wavelet is unknown.

Notes
-----
``bias=True`` is mandatory (the default here is ``True``, opposite to
:class:`SirenMLP`). Without bias, every layer is odd through the origin,
so the network can only represent odd-symmetric functions ``f(-t) = -f(t)``
— which makes it impossible to fit a localized causal wavelet whose peak
sits near one end of the time window.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .siren import SirenMLP


class SirenWavelet(nn.Module):
    """1-D SIREN parameterization of a time-domain source wavelet.

    Parameters
    ----------
    nt
        Number of time samples in the output wavelet.
    hidden_features
        SIREN width. See :class:`SirenMLP`.
    hidden_layers
        Number of hidden sine layers after the first one.
    first_omega0
        Sine frequency of the first layer.
    hidden_omega0
        Sine frequency of the hidden layers.
    bias
        Per the docstring: keep this True. ``False`` is exposed only for
        completeness (e.g. fitting an explicitly odd wavelet).
    """

    def __init__(
        self,
        nt: int,
        *,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        first_omega0: float = 30.0,
        hidden_omega0: float = 30.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.nt = int(nt)
        coords = torch.linspace(-1.0, 1.0, self.nt, dtype=torch.float32).reshape(-1, 1)
        self.register_buffer("coords", coords, persistent=False)
        self.mlp = SirenMLP(
            in_features=1,
            out_features=1,
            hidden_features=int(hidden_features),
            hidden_layers=int(hidden_layers),
            first_omega0=float(first_omega0),
            hidden_omega0=float(hidden_omega0),
            bias=bool(bias),
        )

    def forward(self) -> torch.Tensor:
        return self.mlp(self.coords).reshape(self.nt)


__all__ = ["SirenWavelet"]
