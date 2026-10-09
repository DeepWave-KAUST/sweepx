"""Base class for any "produces a velocity tensor" module.

The pattern: subclasses implement ``_raw_forward()`` that returns a
tensor in some normalized range. The base class scales that tensor to
``[vp_min, vp_max]`` and shapes it correctly.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class Reparameterizer(nn.Module):
    """Abstract base: ``net() -> vp_tensor`` in physical units.

    Subclasses implement :meth:`_raw_forward`, which must return a tensor
    with shape ``out_shape`` whose values lie roughly in ``[-1, 1]`` (or
    any range — see :meth:`_to_physical`).

    Parameters
    ----------
    out_shape
        Spatial shape of the velocity model, e.g. ``(nz, nx)`` for 2-D.
    vp_min
        Lower end of the physical-unit range.
    vp_max
        Upper end. The raw output is rescaled and (by default, when
        ``squash="tanh"``) tanh-squashed to live in ``[vp_min, vp_max]``.
    squash
        How the raw output is mapped to ``[vp_min, vp_max]``:
        - ``"tanh"``  — `vp_min + (vp_max-vp_min) * 0.5*(tanh(raw)+1)`
        - ``"sigmoid"`` — `vp_min + (vp_max-vp_min) * sigmoid(raw)`
        - ``"linear"`` — no squashing; raw values used directly (loss
          must enforce bounds).
    """

    def __init__(
        self,
        out_shape: Tuple[int, ...],
        *,
        vp_min: float = 1500.0,
        vp_max: float = 4500.0,
        squash: str = "tanh",
    ) -> None:
        super().__init__()
        if vp_max <= vp_min:
            raise ValueError(f"need vp_min < vp_max; got {vp_min} >= {vp_max}")
        if squash not in ("tanh", "sigmoid", "linear"):
            raise ValueError(f"unknown squash {squash!r}")
        self.out_shape = tuple(out_shape)
        self.vp_min = float(vp_min)
        self.vp_max = float(vp_max)
        self.squash = squash

    # subclasses implement this
    def _raw_forward(self) -> torch.Tensor:
        raise NotImplementedError

    def _to_physical(self, raw: torch.Tensor) -> torch.Tensor:
        if self.squash == "tanh":
            u = 0.5 * (torch.tanh(raw) + 1.0)
        elif self.squash == "sigmoid":
            u = torch.sigmoid(raw)
        else:  # linear
            u = raw
        return self.vp_min + (self.vp_max - self.vp_min) * u

    def forward(self) -> torch.Tensor:
        raw = self._raw_forward()
        if raw.shape[-len(self.out_shape):] != self.out_shape:
            raise RuntimeError(
                f"{type(self).__name__}._raw_forward must yield shape ending in "
                f"{self.out_shape}; got {tuple(raw.shape)}"
            )
        return self._to_physical(raw)


__all__ = ["Reparameterizer"]
