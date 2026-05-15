"""Envelope-based FWI misfits.

Given a 1-D real signal :math:`d(t)`, its analytic signal
:math:`a(t)=d(t)+ i\\,\\mathcal H[d](t)` and instantaneous envelope
:math:`E(t)=|a(t)|`, three closely-related envelope misfits have been
proposed for FWI:

* **Envelope difference, p-norm**
  (Wu, Luo & Wu 2014; Chi, Dong & Liu 2014):

  .. math::

      \\mathcal J_E^{(p)}(m)
        = \\tfrac{1}{p}\\sum_{s,t,r,c}
            \\bigl( E_{\\mathrm{syn}}(t) - E_{\\mathrm{obs}}(t)\\bigr)^{p}.

  We support ``p in {1, 2}``.  The ``p=2`` form is Wu's "envelope
  inversion" of Geophysics 2014; the ``p=1`` form is sometimes used as a
  more outlier-resistant variant.

* **Logarithmic envelope ratio** (Bozdağ, Trampert & Tromp 2011, eq. 14):

  .. math::

      \\mathcal J_E^{\\log}(m) = \\tfrac{1}{2}\\sum_{s,t,r,c}
          \\bigl( \\log E_{\\mathrm{syn}}(t) - \\log E_{\\mathrm{obs}}(t)\\bigr)^2.

* **Squared-envelope difference** (Chi, Dong & Liu 2014):

  .. math::

      \\mathcal J_{E^2}(m) = \\tfrac{1}{2}\\sum_{s,t,r,c}
          \\bigl( E_{\\mathrm{syn}}^2(t) - E_{\\mathrm{obs}}^2(t)\\bigr)^2.

  Used because the squared envelope is what enters the Bozdağ-Tromp
  adjoint sensitivity kernel directly.

References
----------
* Bozdağ, E., Trampert, J. & Tromp, J. (2011). *Misfit functions for full
  waveform inversion based on instantaneous phase and envelope
  measurements.*  Geophys. J. Int. 185 (2), 845-870.
  doi:10.1111/j.1365-246X.2011.04970.x
* Wu, R.-S., Luo, J. & Wu, B. (2014). *Seismic envelope inversion and
  modulation signal model.* **Geophysics** 79 (3), WA13-WA24.
  doi:10.1190/geo2013-0294.1
* Chi, B., Dong, L. & Liu, Y. (2014). *Full-waveform inversion based on
  envelope objective function.*  76th EAGE Conference, Expanded Abstracts.
  doi:10.3997/2214-4609.20141008
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss
from ._utils import envelope


class EnvelopeLoss(BaseFWILoss):
    """Envelope-difference FWI misfit (Wu, Luo & Wu 2014; Bozdağ 2011)."""

    def __init__(
        self,
        p: int = 2,
        log: bool = False,
        squared: bool = False,
        eps: float = 1e-12,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if log and squared:
            raise ValueError("`log` and `squared` cannot both be True")
        if p not in (1, 2):
            raise ValueError(f"p must be 1 or 2, got {p}")
        self.p = int(p)
        self.log = bool(log)
        self.squared = bool(squared)
        self.eps = float(eps)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # Envelope is computed along the time axis (-3 in canonical layout).
        Es = envelope(syn, dim=-3, eps=self.eps)
        Eo = envelope(obs, dim=-3, eps=self.eps)
        if self.log:
            r = torch.log(Es.clamp_min(self.eps)) - torch.log(Eo.clamp_min(self.eps))
            return 0.5 * r * r
        if self.squared:
            r = Es * Es - Eo * Eo
            return 0.5 * r * r
        r = Es - Eo
        if self.p == 2:
            return 0.5 * r * r
        return r.abs()


def envelope_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    p: int = 2,
    log: bool = False,
    squared: bool = False,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    return EnvelopeLoss(
        p=p, log=log, squared=squared, eps=eps, reduction=reduction
    )(syn, obs)


__all__ = ["EnvelopeLoss", "envelope_loss"]
