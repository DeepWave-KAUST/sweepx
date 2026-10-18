"""Jensen-Shannon divergence FWI misfit (Yan et al. 2024).

Each trace is mapped to a probability density and then compared by the
symmetric Jensen-Shannon divergence

.. math::

    \\mathrm{JSD}(p, q) \\;=\\; \\tfrac{1}{2}\\,\\mathrm{KL}(p\\,\\|\\,m)
        + \\tfrac{1}{2}\\,\\mathrm{KL}(q\\,\\|\\,m),
    \\qquad m = \\tfrac{1}{2}(p + q),

with the standard Kullback-Leibler divergence
:math:`\\mathrm{KL}(p\\,\\|\\,m) = \\sum_t p(t) \\log\\!\\bigl(p(t)/m(t)\\bigr)`.

The misfit summed over traces is

.. math::

    \\mathcal J_{\\mathrm{JSD}}(m_{\\mathrm{model}}) =
        \\sum_{\\text{trace}} \\mathrm{JSD}(p_{\\mathrm s}, p_{\\mathrm o}).

Useful properties

* **Symmetric** (unlike plain KL).
* **Bounded**: :math:`\\mathrm{JSD}(p, q) \\in [0, \\log 2]`.
* **Square root** of JSD is a metric (Endres & Schindelin 2003); we
  return JSD itself (not its square root) because the gradient is
  cleaner.

The 1-D positive transform follows the same convention as
:class:`sweep_loss.NIMLoss` / :class:`sweep_loss.Wasserstein1Loss`:
``"square"`` / ``"abs"`` / ``"linear"`` / ``"exp"``.

References
----------
* Yan, Y., Chen, X., Li, J., et al. (2024). *Multiparameter
  shallow-seismic waveform inversion based on the Jensen-Shannon
  divergence.*  Geophys. J. Int. 238 (1), 132-155.
  doi:10.1093/gji/ggae143
* Endres, D. M. & Schindelin, J. E. (2003). *A new metric for
  probability distributions.*  IEEE Trans. Inf. Theory 49 (7),
  1858-1860.  doi:10.1109/TIT.2003.813506
* Lin, J. (1991). *Divergence measures based on the Shannon entropy.*
  IEEE Trans. Inf. Theory 37 (1), 145-151.  doi:10.1109/18.61115
"""

from __future__ import annotations

import math

import torch

from ._utils import normalize_density, positive_transform
from .base import BaseFWILoss, flatten_traces, to_canonical


class JensenShannonLoss(BaseFWILoss):
    """Jensen-Shannon divergence FWI misfit (Yan et al. 2024)."""

    def __init__(
        self,
        positive: str = "square",
        c: "float | None" = None,
        normalize_by_log2: bool = False,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if positive not in ("square", "abs", "linear", "exp"):
            raise ValueError(f"unknown positive='{positive}'")
        self.positive = positive
        self.c = float(c) if c is not None else None
        self.normalize_by_log2 = bool(normalize_by_log2)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        s, _ = flatten_traces(syn_c)
        o, _ = flatten_traces(obs_c)

        c_param = self.c
        if self.positive == "linear" and c_param is None:
            c_param = float(torch.max(s.abs().max(), o.abs().max()).item()) + 1e-6
        fs = positive_transform(s, method=self.positive, c=c_param)
        fo = positive_transform(o, method=self.positive, c=c_param)
        p = normalize_density(fs, dim=-1)
        q = normalize_density(fo, dim=-1)
        m = 0.5 * (p + q)
        eps = 1e-30
        kl_pm = (p * (torch.log(p.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)
        kl_qm = (q * (torch.log(q.clamp_min(eps)) - torch.log(m.clamp_min(eps)))).sum(dim=-1)
        per_trace = 0.5 * (kl_pm + kl_qm)

        if self.normalize_by_log2:
            per_trace = per_trace / math.log(2.0)

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def jensen_shannon_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    positive: str = "square",
    c: "float | None" = None,
    normalize_by_log2: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    return JensenShannonLoss(
        positive=positive,
        c=c,
        normalize_by_log2=normalize_by_log2,
        reduction=reduction,
    )(syn, obs)


__all__ = ["JensenShannonLoss", "jensen_shannon_loss"]
