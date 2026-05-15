"""Normalised Integration Method (NIM) misfit.

Given a real, signed signal :math:`d(t)`, NIM (Liu, Hu & Wang 2012;
Donno, Chauris & Calandra 2013) first maps it to a non-negative density,
normalises to total mass one and then compares the resulting
**cumulative distribution functions** in L2:

.. math::

    f(t) = \\sigma\\bigl(d(t)\\bigr), \\quad
    F(t) = \\frac{\\int_0^t f(\\tau)\\,\\mathrm d\\tau}{\\int_0^T f(\\tau)\\,\\mathrm d\\tau},

.. math::

    \\mathcal J_{\\mathrm{NIM}}(m) = \\tfrac{1}{2}\\!\\sum_{\\text{trace}} \\int_0^T
        \\bigl(F_{\\mathrm{syn}}(t) - F_{\\mathrm{obs}}(t)\\bigr)^2 \\,\\mathrm d t.

The positive map :math:`\\sigma(\\cdot)` is chosen by the user via the
``positive`` argument: ``"square"`` (default; the original Liu 2012
choice), ``"abs"``, ``"linear"`` (signal + offset c) or ``"exp"``.

The integral squared difference of the CDFs is identical to the
**1-Wasserstein** distance between the densities (Bonneel et al. 2011),
which is why NIM is sometimes referred to as the "Wasserstein-1 / KR
misfit on a positive-transformed trace" in modern OT literature.  Here
we implement the time-domain CDF form directly because it is by far the
cheapest formulation per trace.

References
----------
* Liu, F., Hu, X. & Wang, J. (2012). *An optimised waveform inversion
  method based on a non-quadratic misfit function and the normalised
  integration method.*  Geophys. Prospect. 60 (3), 386-394.
  doi:10.1111/j.1365-2478.2011.00993.x
* Donno, D., Chauris, H. & Calandra, H. (2013). *Estimating the
  background velocity model with the normalised integration method.*
  75th EAGE Conference & Exhibition.  doi:10.3997/2214-4609.20130411
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical
from ._utils import normalize_density, positive_transform


class NIMLoss(BaseFWILoss):
    """Normalised Integration Method FWI misfit (Liu 2012; Donno 2013)."""

    def __init__(
        self,
        positive: str = "square",
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

        # Work per trace.
        syn_f, canon = flatten_traces(syn_c)  # (N, nt)
        obs_f, _ = flatten_traces(obs_c)

        # Map to non-negative and normalise to a probability density on time.
        c_param = self.c
        if self.positive == "linear" and c_param is None:
            # Use a common offset for the two signals so they remain comparable.
            c_param = float(torch.max(syn_f.abs().max(), obs_f.abs().max()).item()) + 1e-6
        fs = positive_transform(syn_f, method=self.positive, c=c_param)
        fo = positive_transform(obs_f, method=self.positive, c=c_param)
        ps = normalize_density(fs, dim=-1)
        po = normalize_density(fo, dim=-1)

        Fs = torch.cumsum(ps, dim=-1)
        Fo = torch.cumsum(po, dim=-1)
        # Trapezoidal numerical integration of (Fs - Fo)^2 over time.
        diff2 = (Fs - Fo) ** 2
        per_trace = 0.5 * self.dt * (
            diff2.sum(dim=-1) - 0.5 * (diff2[..., 0] + diff2[..., -1])
        )                                                     # (N,)
        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def nim_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    positive: str = "square",
    c: "float | None" = None,
    dt: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    return NIMLoss(positive=positive, c=c, dt=dt, reduction=reduction)(syn, obs)


__all__ = ["NIMLoss", "nim_loss"]
