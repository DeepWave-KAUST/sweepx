"""Huber and pseudo-Huber misfits.

For a residual :math:`r = d_{\\mathrm{syn}} - d_{\\mathrm{obs}}` and a
threshold :math:`\\delta > 0`,

.. math::

    h_{\\delta}(r) =
        \\begin{cases}
            \\tfrac{1}{2} r^2 & |r| \\le \\delta \\\\
            \\delta\\bigl(|r| - \\tfrac{1}{2}\\delta\\bigr) & |r| > \\delta
        \\end{cases}

(Huber, 1964).  A smooth variant that is C^∞ everywhere is the
*pseudo-Huber*

.. math::

    \\tilde h_{\\delta}(r) = \\delta^2
        \\Bigl(\\sqrt{1 + (r/\\delta)^2} - 1\\Bigr),

which behaves like :math:`\\tfrac{1}{2}r^2` for :math:`|r|\\ll \\delta` and like
:math:`\\delta |r| - \\delta^2` for :math:`|r|\\gg \\delta` (Charbonnier et al.
1997; Hartley & Zisserman 2003).

For FWI the Huber norm was introduced as a stable compromise between L2
(efficient for small residuals) and L1 (robust to outliers) by Guitton &
Symes (2003); see also Bube & Langan (1997).

References
----------
* Huber, P. J. (1964). *Robust estimation of a location parameter.*  Ann.
  Math. Stat. 35 (1), 73-101.
* Guitton, A. & Symes, W. W. (2003). *Robust inversion of seismic data using
  the Huber norm.*  **Geophysics** 68 (4), 1310-1319.
* Bube, K. P. & Langan, R. T. (1997). *Hybrid L1/L2 minimisation with
  applications to tomography.* **Geophysics** 62 (4), 1183-1195.
* Charbonnier, P. et al. (1997). *Deterministic edge-preserving regularisation
  in computed imaging.* IEEE T. Image Process. 6, 298-311.
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss


class HuberLoss(BaseFWILoss):
    """Classical Huber misfit (piecewise quadratic / linear)."""

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

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        r = syn - obs
        a = r.abs()
        d = self.delta
        quad = 0.5 * r * r
        lin = d * (a - 0.5 * d)
        return torch.where(a <= d, quad, lin)


class PseudoHuberLoss(BaseFWILoss):
    """Smooth pseudo-Huber misfit (Charbonnier 1997)."""

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

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        r = syn - obs
        d = self.delta
        return d * d * (torch.sqrt(1.0 + (r / d) ** 2) - 1.0)


def huber_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    return HuberLoss(delta=delta, reduction=reduction)(syn, obs)


def pseudo_huber_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    return PseudoHuberLoss(delta=delta, reduction=reduction)(syn, obs)


__all__ = [
    "HuberLoss",
    "PseudoHuberLoss",
    "huber_loss",
    "pseudo_huber_loss",
]
