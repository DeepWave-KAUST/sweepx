"""1-Wasserstein / Kantorovich-Rubinstein FWI misfit.

For two probability measures :math:`\\mu, \\nu` on :math:`\\mathbb R` with
densities :math:`f, g` and CDFs :math:`F, G`, the 1-Wasserstein distance
admits the explicit one-dimensional formula

.. math::

    W_1(\\mu, \\nu) \\;=\\; \\int_{\\mathbb R} \\bigl|F(x) - G(x)\\bigr|\\,\\mathrm dx.

For two signed seismic traces, Métivier et al. (2016) propose to convert
each trace to a probability density via a positive transform (square /
abs / linear / exp) and then take :math:`W_1` of the two densities.  The
practical 1-D formula above is what we implement here, broadcast over
all traces.

This is sometimes called the **KR-norm** (Kantorovich-Rubinstein dual)
in the optimal-transport literature.  Although Métivier et al. (2016)
develop the full :math:`d`-dimensional case via dual ascent and a
fast-marching / max-flow solver, the **per-trace** version we implement
is the most common usage in FWI codes and shares the anti-cycle-skipping
property.

References
----------
* Métivier, L., Brossier, R., Mérigot, Q., Oudet, E. & Virieux, J.
  (2016). *Measuring the misfit between seismograms using an optimal
  transport distance: application to full waveform inversion.*
  Geophys. J. Int. 205 (1), 345-377.
  doi:10.1093/gji/ggw014
* Engquist, B. & Froese, B. D. (2014). *Application of the Wasserstein
  metric to seismic signals.*  Commun. Math. Sci. 12 (5), 979-988.
  doi:10.4310/CMS.2014.v12.n5.a7
"""

from __future__ import annotations

import torch

from ._utils import normalize_density, positive_transform
from .base import BaseFWILoss, flatten_traces, to_canonical


class Wasserstein1Loss(BaseFWILoss):
    """1-Wasserstein FWI misfit (KR-norm) via the 1-D CDF formula."""

    def __init__(
        self,
        positive: str = "linear",
        c: "float | None" = None,
        dt: float = 1.0,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if positive not in ("square", "abs", "linear", "exp"):
            raise ValueError(f"unknown positive='{positive}'")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.positive = positive
        self.c = float(c) if c is not None else None
        self.dt = float(dt)

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
        ps = normalize_density(fs, dim=-1)
        po = normalize_density(fo, dim=-1)

        Fs = torch.cumsum(ps, dim=-1)
        Fo = torch.cumsum(po, dim=-1)
        # Trapezoidal integration of |Fs - Fo| along time.
        diff = (Fs - Fo).abs()
        per_trace = self.dt * (diff.sum(dim=-1) - 0.5 * (diff[..., 0] + diff[..., -1]))

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def w1_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    positive: str = "linear",
    c: "float | None" = None,
    dt: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    return Wasserstein1Loss(
        positive=positive, c=c, dt=dt, reduction=reduction
    )(syn, obs)


__all__ = ["Wasserstein1Loss", "w1_loss"]
