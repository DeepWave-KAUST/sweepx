"""Base classes and shape utilities for FWI loss functions.

All misfits in :mod:`sweep_loss` consume tensors with the shape

    (nshots, nt, nreceivers, nchannel)

We call this the **canonical** layout.  The time axis is always axis ``-3``;
the trailing two axes are receiver index and recorded component.

Many of the FWI literature misfits are intrinsically *per-trace*: they take a
1-D time series and return a scalar.  To make those losses easy to write we
provide a helper :func:`flatten_traces` that reshapes ``(nshots, nt, nrec,
nchan)`` into ``(N, nt)`` with ``N = nshots * nrec * nchan`` and back.

The :class:`BaseFWILoss` class implements the reduction / masking / shape
checking boilerplate that every FWI loss needs and lets subclasses focus on
the actual sample-wise (or trace-wise) misfit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Shape helpers
# ----------------------------------------------------------------------------
def to_canonical(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Promote ``x`` to canonical ``(nshots, nt, nrec, nchan)`` shape.

    Accepted inputs:

    * ``(nt,)``                      - single trace, single shot, single channel
    * ``(nt, nrec)``                 - single shot, single channel
    * ``(nshots, nt, nrec)``         - single channel
    * ``(nshots, nt, nrec, nchan)``  - canonical

    Returns a (possibly unsqueezed) view together with the original shape so the
    caller can squeeze the result back if desired.
    """
    original_shape = tuple(x.shape)
    if x.ndim == 1:
        x = x.view(1, x.shape[0], 1, 1)
    elif x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(-1)  # (1, nt, nrec, 1)
    elif x.ndim == 3:
        x = x.unsqueeze(-1)               # (ns, nt, nrec, 1)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(
            f"FWI loss expects 1D-4D tensor with time on axis -3, got shape {original_shape}"
        )
    return x, original_shape


def flatten_traces(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Reshape canonical-layout tensor into ``(N, nt)`` for per-trace ops.

    Returns the flattened tensor together with the canonical ``(ns, nt, nr, nc)``
    shape so :func:`unflatten_traces` can put it back.
    """
    assert x.ndim == 4, f"Expected canonical 4D, got {x.shape}"
    ns, nt, nr, nc = x.shape
    # Bring time to last axis: (ns, nr, nc, nt) -> (N, nt)
    out = x.permute(0, 2, 3, 1).contiguous().view(ns * nr * nc, nt)
    return out, (ns, nt, nr, nc)


def unflatten_traces(
    x: torch.Tensor, canonical_shape: Tuple[int, int, int, int]
) -> torch.Tensor:
    """Inverse of :func:`flatten_traces`."""
    ns, nt, nr, nc = canonical_shape
    return x.view(ns, nr, nc, nt).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------------
# Base loss
# ----------------------------------------------------------------------------
class BaseFWILoss(nn.Module):
    """Common base class for FWI misfit functions.

    Subclasses implement :meth:`_pointwise` (returning a tensor of the same
    canonical shape as the inputs) or override :meth:`forward` directly for
    losses that cannot be expressed sample-wise.

    Parameters
    ----------
    reduction
        How to reduce the elementwise misfit to a scalar.  One of ``"mean"``,
        ``"sum"`` or ``"none"`` (return the un-reduced tensor).
    mask
        Optional float / bool tensor broadcastable to the inputs.  Elements
        where the mask is ``0`` / ``False`` are excluded from the reduction.
    """

    def __init__(
        self,
        reduction: str = "mean",
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean'|'sum'|'none', got {reduction!r}"
            )
        self.reduction = reduction
        # Register mask as a buffer so it follows .to(device).
        if mask is not None:
            self.register_buffer("mask", mask.to(torch.float32))
        else:
            self.mask = None  # type: ignore[assignment]

    # -- subclass hook -------------------------------------------------------
    def _pointwise(
        self, syn: torch.Tensor, obs: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - abstract
        """Return an element-wise misfit tensor with shape == ``syn.shape``.

        Subclasses that are not naturally element-wise (e.g. cross-correlation,
        envelope, OT) should override :meth:`forward` instead.
        """
        raise NotImplementedError

    # -- public API ----------------------------------------------------------
    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )

        misfit = self._pointwise(syn_c, obs_c)
        return self._reduce(misfit)

    # -- helpers -------------------------------------------------------------
    def _reduce(self, misfit: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            mask = self.mask.to(misfit.dtype).to(misfit.device)
            misfit = misfit * mask
            if self.reduction == "mean":
                denom = mask.sum().clamp(min=1.0)
                return misfit.sum() / denom
            if self.reduction == "sum":
                return misfit.sum()
            return misfit
        if self.reduction == "mean":
            return misfit.mean()
        if self.reduction == "sum":
            return misfit.sum()
        return misfit

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"reduction={self.reduction!r}"
