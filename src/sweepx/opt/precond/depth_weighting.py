"""Depth-weighting preconditioner.

A ``(z / z_max) ** p`` mask applied along the depth axis, used to compensate
for the surface-illumination bias of seismic experiments and reach deeper
into the model.

This is composable with any other preconditioner via simple multiplication
of the cached diagonals, e.g. with the pseudo-Hessian preconditioner:

    precond_total = PseudoHessianPrecond(...) * DepthWeighting(...)

is realized via :class:`Compose`, defined here for convenience.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor

from sweepx.opt.precond.base import Preconditioner


class DepthWeighting(Preconditioner):
    """Multiply the gradient by ``(z / z_max) ** p`` along the depth axis.

    Args:
        depth_axis: index of the depth axis in the parameter tensor.
            For sweep's 2-D ``(nx, nz)`` models this is ``-1``; for 3-D
            ``(nx, ny, nz)`` it is also ``-1``. Default: ``-1``.
        p: exponent. ``p > 0`` boosts deep components, ``p < 0`` boosts
            shallow. Default: ``1.0``.
        z_max_strategy: ``"size"`` uses the depth-axis size as ``z_max``
            (per Mulder & Plessix); ``"unit"`` normalizes to 1 directly.
    """

    def __init__(self, depth_axis: int = -1, p: float = 1.0,
                 z_max_strategy: str = "size") -> None:
        if z_max_strategy not in ("size", "unit"):
            raise ValueError(f"unknown z_max_strategy: {z_max_strategy!r}")
        self.depth_axis = depth_axis
        self.p = float(p)
        self.z_max_strategy = z_max_strategy
        self._weights: dict = {}

    def update(self, params: Iterable[Tensor], **_kwargs) -> None:
        for p in params:
            self._weights[id(p)] = self._build_weights(p)

    def _build_weights(self, param: Tensor) -> Tensor:
        nz = param.shape[self.depth_axis]
        z = torch.arange(nz, dtype=param.dtype, device=param.device)
        z_max = float(nz - 1) if self.z_max_strategy == "size" else 1.0
        w_1d = (z / max(z_max, 1.0)) ** self.p
        # Broadcast to the full param shape.
        view_shape = [1] * param.dim()
        view_shape[self.depth_axis] = nz
        return w_1d.view(view_shape).expand_as(param).contiguous()

    def __call__(self, grad: Tensor, param: Tensor) -> Tensor:
        w = self._weights.get(id(param))
        if w is None:
            # Lazy build.
            w = self._build_weights(param)
            self._weights[id(param)] = w
        return grad * w


class Compose(Preconditioner):
    """Compose multiple preconditioners by multiplying their gradients.

    Calls every wrapped preconditioner's ``update`` in order, then chains
    their ``__call__`` so that ``precond(grad, param)`` returns the gradient
    after each one has been applied in sequence.
    """

    def __init__(self, *preconds: Preconditioner) -> None:
        if not preconds:
            raise ValueError("Compose needs at least one preconditioner")
        self.preconds = list(preconds)

    def update(self, params, **kwargs) -> None:
        for pc in self.preconds:
            pc.update(params, **kwargs)

    def __call__(self, grad: Tensor, param: Tensor) -> Tensor:
        out = grad
        for pc in self.preconds:
            out = pc(out, param)
        return out
