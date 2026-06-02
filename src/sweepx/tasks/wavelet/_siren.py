"""SIREN (sinusoidal-activation implicit representation) building blocks.

Extracted verbatim from ``fwi_workflow.models.inr`` so that
``sweep_tasks.wavelet`` is self-contained (no dependency on
``fwi_workflow``). Only the wavelet-relevant slice is included —
``VelocityINR`` / ``VelocityINR3D`` were the velocity-side equivalents
and are not needed here.

The classes are deliberately kept thin: each lazily imports ``torch`` so
that simply ``import sweep_tasks.wavelet`` doesn't drag torch into
CPU-only contexts (e.g. ``sweep-tasks analyze-wavelet`` or
``sweep-tasks convert-farfield`` work without torch installed).
"""

from __future__ import annotations

import math


class SirenLayer:
    """Factory for SIREN linear layers with the paper initialization."""

    @staticmethod
    def init_linear(linear, in_features: int, omega0: float, is_first: bool) -> None:
        """Initialize a linear layer following the SIREN paper."""

        import torch

        with torch.no_grad():
            if is_first:
                bound = 1.0 / float(in_features)
            else:
                bound = math.sqrt(6.0 / float(in_features)) / float(omega0)
            linear.weight.uniform_(-bound, bound)
            if linear.bias is not None:
                linear.bias.uniform_(-bound, bound)


class SirenMLP:
    """SIREN MLP that maps coordinates or encoded coordinates to scalar values."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        first_omega0: float = 30.0,
        hidden_omega0: float = 30.0,
        use_bias: bool = False,
        device=None,
    ):
        """Create a SIREN MLP."""

        import torch

        self.first_omega0 = float(first_omega0)
        self.hidden_omega0 = float(hidden_omega0)
        self.layers = torch.nn.ModuleList()
        first = torch.nn.Linear(int(in_features), int(hidden_features), bias=bool(use_bias))
        SirenLayer.init_linear(first, int(in_features), self.first_omega0, is_first=True)
        self.layers.append(first)
        for _ in range(int(hidden_layers)):
            layer = torch.nn.Linear(int(hidden_features), int(hidden_features), bias=bool(use_bias))
            SirenLayer.init_linear(layer, int(hidden_features), self.hidden_omega0, is_first=False)
            self.layers.append(layer)
        self.final = torch.nn.Linear(int(hidden_features), int(out_features), bias=bool(use_bias))
        SirenLayer.init_linear(self.final, int(hidden_features), self.hidden_omega0, is_first=False)
        self.module = torch.nn.Module()
        self.module.layers = self.layers
        self.module.final = self.final
        self.module.to(device)

    def parameters(self):
        """Return trainable parameters."""

        return self.module.parameters()

    def __call__(self, coords):
        """Evaluate the SIREN MLP."""

        import torch

        x = torch.sin(self.first_omega0 * self.layers[0](coords))
        for layer in self.layers[1:]:
            x = torch.sin(self.hidden_omega0 * layer(x))
        return self.final(x)


class SirenWavelet:
    """Small SIREN that maps normalized time coordinates to wavelet samples.

    The MLP is built with ``use_bias=True`` so that the wavelet does not have
    to be odd-symmetric in the time coordinate. Without bias, every layer is
    odd through the origin and the network can only represent
    ``f(-coord) = -f(coord)`` shapes, which makes it impossible to fit a
    localized causal wavelet whose peak is near one end of the coordinate
    range.
    """

    def __init__(
        self,
        nt: int,
        hidden_features: int,
        hidden_layers: int,
        first_omega0: float,
        hidden_omega0: float,
        device,
        use_bias: bool = True,
    ):
        """Create a SIREN wavelet representation."""

        import torch

        self.nt = int(nt)
        self.coords = torch.linspace(-1.0, 1.0, int(nt), device=device, dtype=torch.float32).reshape(-1, 1)
        self.mlp = SirenMLP(
            in_features=1,
            out_features=1,
            hidden_features=int(hidden_features),
            hidden_layers=int(hidden_layers),
            first_omega0=float(first_omega0),
            hidden_omega0=float(hidden_omega0),
            use_bias=bool(use_bias),
            device=device,
        )

    def parameters(self):
        """Return trainable INR parameters."""

        return self.mlp.parameters()

    def __call__(self):
        """Return wavelet samples."""

        return self.mlp(self.coords).reshape(self.nt)
