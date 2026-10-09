"""Learned-prior helpers.

The intended use is for *perceptual* / feature-based losses: wrap a frozen
feature extractor (e.g. a CNN pretrained on velocity-model patches) and
compute distances in feature space inside your loss function.

The wrapper deliberately freezes the underlying module's parameters so
gradients flow to whatever produced the input, never to the prior itself.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LearnedPrior(nn.Module):
    """Wrap a feature extractor; freeze its parameters; expose ``features(x)``.

    Parameters
    ----------
    backbone
        An ``nn.Module`` whose ``forward(x)`` produces either a single
        feature tensor or a tuple/list of multi-scale tensors.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "LearnedPrior":  # type: ignore[override]
        # keep backbone in eval regardless
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        out = self.backbone(x)
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(out)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.features(x)


def perceptual_distance(
    prior: LearnedPrior,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    weights: Iterable[float] | None = None,
    p: int = 2,
) -> torch.Tensor:
    """L_p distance averaged across the prior's feature maps.

    Parameters
    ----------
    prior
        A :class:`LearnedPrior`.
    a, b
        Inputs in the same shape the backbone expects.
    weights
        Per-scale weights (same length as the backbone's feature list).
    p
        Distance order. ``2`` for L2, ``1`` for L1.
    """
    fa = prior.features(a)
    fb = prior.features(b)
    if len(fa) != len(fb):
        raise RuntimeError(
            f"prior produced {len(fa)} feature maps for `a` and {len(fb)} for `b`"
        )
    if weights is None:
        ws = [1.0] * len(fa)
    else:
        ws = list(weights)
        if len(ws) != len(fa):
            raise ValueError(f"expected {len(fa)} weights; got {len(ws)}")
    total = a.new_zeros(())
    for w, x, y in zip(ws, fa, fb):
        total = total + w * (x - y).abs().pow(p).mean()
    return total


class TVPrior(nn.Module):
    """Total-variation-style smoothness penalty on a velocity volume.

    Computes a derivative-L2 (Sobolev) penalty along the lateral (x and,
    in 3-D, y) and depth (z) axes. Supports first-order, second-order, or
    both. Used as a soft regularizer in the FWI loss::

        L_total = L_data + weight * TVPrior(...)(velocity)

    The velocity tensor is internally normalized by ``velocity_scale_m_s``
    so the weight stays interpretable across surveys with different vp
    magnitudes (m/s).

    In 2-D the model is ``(nz, nx)``; in 3-D it is ``(nz, ny, nx)``. The
    depth axis is always axis 0.

    Parameters
    ----------
    order
        ``"first"`` (default) — sum of squared first differences.
        ``"second"`` — sum of squared second differences (curvature).
        ``"both"`` / ``"mixed"`` — both first and second.
    x_weight
        Multiplier for the lateral (x) differences.
    z_weight
        Multiplier for the depth (z) differences.
    y_weight
        Multiplier for the y differences; ignored on 2-D inputs.
    velocity_scale_m_s
        Normalization for vp before differentiation (m/s). Default
        ``1000.0``.
    """

    def __init__(
        self,
        *,
        order: str = "first",
        x_weight: float = 1.0,
        z_weight: float = 1.0,
        y_weight: float = 1.0,
        velocity_scale_m_s: float = 1000.0,
    ) -> None:
        super().__init__()
        if str(order).lower() not in {"first", "second", "both", "mixed", "first_second"}:
            raise ValueError(
                f"TVPrior.order must be one of "
                f"'first'|'second'|'both'|'mixed'|'first_second'; got {order!r}"
            )
        if velocity_scale_m_s <= 0:
            raise ValueError("velocity_scale_m_s must be > 0")
        self.order = str(order).lower()
        self.x_weight = float(x_weight)
        self.z_weight = float(z_weight)
        self.y_weight = float(y_weight)
        self.velocity_scale_m_s = float(velocity_scale_m_s)

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        if velocity.ndim not in (2, 3):
            raise ValueError(
                f"TVPrior expects 2-D (nz, nx) or 3-D (nz, ny, nx) vp; "
                f"got shape {tuple(velocity.shape)}"
            )
        v = velocity / self.velocity_scale_m_s
        loss = v.new_zeros(())
        use_first = self.order in {"first", "both", "mixed", "first_second"}
        use_second = self.order in {"second", "both", "mixed", "first_second"}
        # 2-D: axes (0=z, 1=x). 3-D: axes (0=z, 1=y, 2=x).
        if v.ndim == 2:
            axes = {"z": 0, "x": 1}
        else:
            axes = {"z": 0, "y": 1, "x": 2}
        weights = {"z": self.z_weight, "x": self.x_weight, "y": self.y_weight}
        for name, ax in axes.items():
            w = weights[name]
            if w == 0.0:
                continue
            if use_first and v.shape[ax] > 1:
                d1 = torch.diff(v, n=1, dim=ax)
                loss = loss + w * torch.mean(d1 * d1)
            if use_second and v.shape[ax] > 2:
                d2 = torch.diff(v, n=2, dim=ax)
                loss = loss + w * torch.mean(d2 * d2)
        return loss


class SeabedFreezeMask:
    """Mask that zeros gradients above the seabed (water column).

    For marine surveys the water-column vp is essentially constant at
    ~1500 m/s and well known a priori — letting FWI update those cells
    introduces noise that doesn't help the inversion. Multiplying the
    gradient by ``mask = (depth_idx >= floor(seabed_idx))`` before the
    optimizer step keeps the water column frozen at its initial value
    while letting everything below the seabed update normally.

    Parameters
    ----------
    seabed_depth_m
        ``(ny, nx)`` (3-D) or ``(nx,)`` (2-D) per-trace seabed depths in
        meters from the model's top edge (z = 0).
    dz_m
        Grid spacing along the depth axis (m).
    buffer_cells
        Optional integer pad — keep the first ``buffer_cells`` rows below
        the seabed also frozen. Useful when the wavelet has a non-zero
        rise time and the seabed reflection straddles a few cells.

    Use
    ---
    Build the mask once, store on the optimizer's device, and call
    ``apply_to(grad)`` (in-place) after backward and before optim.step.
    The mask broadcasts: a 2-D ``(nx,)`` seabed against a 2-D ``(nz, nx)``
    vp; a 2-D ``(ny, nx)`` seabed against a 3-D ``(nz, ny, nx)`` vp.
    """

    def __init__(
        self,
        seabed_depth_m: torch.Tensor | "np.ndarray",
        *,
        dz_m: float,
        buffer_cells: int = 0,
    ) -> None:
        import numpy as np

        if dz_m <= 0:
            raise ValueError("dz_m must be > 0")
        self.dz_m = float(dz_m)
        self.buffer_cells = max(0, int(buffer_cells))
        sb = np.asarray(seabed_depth_m, dtype=np.float64)
        if sb.ndim not in (1, 2):
            raise ValueError(
                f"seabed_depth_m must be (nx,) for 2-D or (ny, nx) for "
                f"3-D; got shape {sb.shape}"
            )
        # Quantize to grid cells (floor — be conservative, freeze one
        # extra row if rounding goes the wrong way).
        sb_idx = np.floor(sb / self.dz_m).astype(np.int64) + self.buffer_cells
        self._sb_idx = sb_idx  # numpy

    def build_mask(self, vp_shape: tuple[int, ...], *, device, dtype) -> torch.Tensor:
        """Return a ``vp_shape`` boolean mask broadcast against the seabed map."""
        import numpy as np

        if len(vp_shape) == 2:
            nz, nx = vp_shape
            if self._sb_idx.ndim != 1 or self._sb_idx.shape[0] != nx:
                raise ValueError(
                    f"SeabedFreezeMask expected 1-D seabed (nx={nx}); "
                    f"got shape {self._sb_idx.shape}"
                )
            z_idx = np.arange(nz, dtype=np.int64)[:, None]
            mask_np = z_idx >= self._sb_idx[None, :]
        elif len(vp_shape) == 3:
            nz, ny, nx = vp_shape
            if self._sb_idx.ndim != 2 or self._sb_idx.shape != (ny, nx):
                raise ValueError(
                    f"SeabedFreezeMask expected 2-D seabed (ny={ny}, nx={nx}); "
                    f"got shape {self._sb_idx.shape}"
                )
            z_idx = np.arange(nz, dtype=np.int64)[:, None, None]
            mask_np = z_idx >= self._sb_idx[None, :, :]
        else:
            raise ValueError(f"vp_shape must be 2-D or 3-D; got {vp_shape}")
        return torch.as_tensor(mask_np, device=device, dtype=dtype)

    def apply_to(self, grad: torch.Tensor) -> None:
        """In-place multiply ``grad`` by the seabed mask."""
        if grad is None:
            return
        mask = self.build_mask(tuple(grad.shape), device=grad.device, dtype=grad.dtype)
        grad.mul_(mask)


__all__ = [
    "LearnedPrior",
    "perceptual_distance",
    "TVPrior",
    "SeabedFreezeMask",
]
