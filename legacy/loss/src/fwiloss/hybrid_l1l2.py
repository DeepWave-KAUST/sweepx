"""Hybrid L1/L2 (a.k.a. "L1-L2" / Bube-Langan) misfit.

For each residual sample :math:`r = d_{\\mathrm{syn}} - d_{\\mathrm{obs}}`,

.. math::

    \\rho_{\\delta}(r) = \\delta^2 \\Bigl(\\sqrt{1 + (r/\\delta)^2} - 1\\Bigr).

Despite being numerically identical to the *pseudo-Huber* loss in
:mod:`fwiloss.huber`, this functional was independently introduced for
**seismic tomography / FWI** by Bube & Langan (1997) - the geophysics
community usually calls it the "hybrid L1-L2 norm" because

* for :math:`|r|\\ll\\delta` it is :math:`\\tfrac{1}{2} r^2` (L2 behaviour),
* for :math:`|r|\\gg\\delta` it is :math:`\\delta|r|-\\delta^2` (L1 behaviour).

We keep it as a separate class so the user-facing FWI vocabulary matches the
literature; under the hood it forwards to :class:`PseudoHuberLoss`.

References
----------
* Bube, K. P. & Langan, R. T. (1997). *Hybrid L1/L2 minimisation with
  applications to tomography.* **Geophysics** 62 (4), 1183-1195.
* Ha, T., Chung, W. & Shin, C. (2009). *Waveform inversion using a
  back-propagation algorithm and a Huber function norm.* **Geophysics**
  74 (3), R15-R24.
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss
from .huber import PseudoHuberLoss


class HybridL1L2Loss(BaseFWILoss):
    """Bube-Langan (1997) hybrid L1/L2 misfit."""

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self.delta = float(delta)
        self._impl = PseudoHuberLoss(delta=delta, reduction="none")

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # Re-implement here rather than reusing _impl.forward to keep the
        # reduction path consistent.
        d = self.delta
        r = syn - obs
        return d * d * (torch.sqrt(1.0 + (r / d) ** 2) - 1.0)


def hybrid_l1l2_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    return HybridL1L2Loss(delta=delta, reduction=reduction)(syn, obs)


__all__ = ["HybridL1L2Loss", "hybrid_l1l2_loss"]
