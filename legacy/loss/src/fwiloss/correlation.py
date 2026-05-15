"""Amplitude-normalised correlation misfits.

Two closely related FWI objectives are bundled here:

1. **Normalised cross-correlation (NCC) / global-correlation misfit**
   (Choi & Alkhalifah 2012; Routh et al. 2011).  For each trace, denote
   the unit-norm normalisation
   :math:`\\hat{d}(t) = d(t) / \\|d\\|_2`.  Then

   .. math::
       \\mathcal{J}_{\\text{NCC}}(m) = -\\!\\sum_{\\text{trace}}
            \\sum_{t} \\hat d_{\\mathrm{syn}}(t)\\,\\hat d_{\\mathrm{obs}}(t).

   Equivalently it minimises the negative inner product of unit-norm
   traces — completely insensitive to per-trace amplitude scaling.  The
   constant ``+1`` offset (``offset_one=True``, default) maps the perfect
   match to zero, i.e. :math:`\\mathcal{J} = \\sum (1 - \\langle \\hat d_{\\mathrm s},
   \\hat d_{\\mathrm o}\\rangle)`.

2. **Trace-normalised L2** (Choi & Alkhalifah 2012, eq. 9):

   .. math::
       \\mathcal{J}_{\\text{nL2}}(m) = \\tfrac{1}{2}\\!\\sum_{\\text{trace}}
            \\sum_t \\bigl(\\hat d_{\\mathrm{syn}}(t) - \\hat d_{\\mathrm{obs}}(t)\\bigr)^2,

   which is mathematically equivalent to the NCC misfit up to the constant
   :math:`\\sum_t \\hat d_{\\mathrm s}^2 + \\hat d_{\\mathrm o}^2 = 2` per trace:
   :math:`\\mathcal{J}_{\\text{nL2}} = N_{\\text{trace}}\\!-\\!\\sum\\langle\\cdot,\\cdot\\rangle
   = \\mathcal{J}_{\\text{NCC}} + N_{\\text{trace}}`.

References
----------
* Choi, Y. & Alkhalifah, T. (2012). *Application of multi-source waveform
  inversion to marine streamer data using the global correlation norm.*
  **Geophys. Prospect.** 60 (4), 748-758.
  doi:10.1111/j.1365-2478.2012.01079.x
* Routh, P., Krebs, J., Lazaratos, S., et al. (2011). *Encoded
  simultaneous source full-wavefield inversion for spectrally-shaped
  marine streamer data.* SEG Tech. Progr. Expanded Abstracts, 2433-2438.
  doi:10.1190/1.3627696
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical
from ._utils import l2_normalize


class GlobalCorrelationLoss(BaseFWILoss):
    """Normalised cross-correlation misfit (Choi & Alkhalifah 2012).

    Parameters
    ----------
    offset_one
        If True (default) compute :math:`1 - \\langle\\hat d_s, \\hat d_o\\rangle`
        per trace so a perfect match gives 0.  Otherwise return the negative
        correlation directly.
    eps
        Small constant to stabilise the L2 normalisation when a trace has
        near-zero amplitude (e.g. muted samples).
    """

    def __init__(
        self,
        offset_one: bool = True,
        eps: float = 1e-12,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.offset_one = bool(offset_one)
        self.eps = float(eps)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )

        # Flatten to (N, nt) and L2-normalise per trace.
        syn_f, canon = flatten_traces(syn_c)
        obs_f, _ = flatten_traces(obs_c)
        syn_n = l2_normalize(syn_f, dim=-1, eps=self.eps)
        obs_n = l2_normalize(obs_f, dim=-1, eps=self.eps)

        per_trace_corr = (syn_n * obs_n).sum(dim=-1)   # (N,)
        if self.offset_one:
            per_trace = 1.0 - per_trace_corr
        else:
            per_trace = -per_trace_corr

        # Trace-level reduction -> a scalar.  We don't have a per-sample mask
        # here, so honour reduction directly on the per-trace values.
        if self.mask is not None:
            # If the user supplied a sample-level mask, collapse it to a
            # per-trace weight by max(|mask|) along time so dead traces are 0.
            mask = self.mask.to(syn_c.dtype).to(syn_c.device)
            mask, _ = flatten_traces(mask.expand_as(syn_c))
            w = mask.abs().amax(dim=-1)                 # (N,)
            num = (per_trace * w).sum()
            if self.reduction == "mean":
                denom = w.sum().clamp_min(1.0)
                return num / denom
            if self.reduction == "sum":
                return num
            return per_trace * w
        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


class TraceNormalizedL2Loss(BaseFWILoss):
    """Trace-by-trace amplitude-normalised L2 (Choi & Alkhalifah 2012, eq. 9)."""

    def __init__(
        self,
        eps: float = 1e-12,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.eps = float(eps)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # L2-normalise per trace along time.  ``syn`` and ``obs`` are
        # canonical (ns, nt, nr, nc).  Time axis = -3.
        syn_n = syn / torch.linalg.vector_norm(syn, dim=-3, keepdim=True).clamp_min(
            self.eps
        )
        obs_n = obs / torch.linalg.vector_norm(obs, dim=-3, keepdim=True).clamp_min(
            self.eps
        )
        d = syn_n - obs_n
        return 0.5 * d * d


def global_correlation_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    offset_one: bool = True,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    return GlobalCorrelationLoss(
        offset_one=offset_one, eps=eps, reduction=reduction
    )(syn, obs)


def trace_normalized_l2_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    return TraceNormalizedL2Loss(eps=eps, reduction=reduction)(syn, obs)


__all__ = [
    "GlobalCorrelationLoss",
    "TraceNormalizedL2Loss",
    "global_correlation_loss",
    "trace_normalized_l2_loss",
]
