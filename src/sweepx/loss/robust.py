"""Robust M-estimators commonly used as FWI misfits.

We provide three classical robust losses for the per-sample residual
:math:`r = d_{\\mathrm{syn}} - d_{\\mathrm{obs}}`:

* **Cauchy / Lorentzian** (Black & Anandan 1996; Crase et al. 1990 use it for
  FWI):

  .. math::
      \\rho^{\\text{Cauchy}}_{c}(r) = \\tfrac{c^2}{2} \\,\\log\\bigl(1 + (r/c)^2\\bigr).

* **Tukey biweight** (Beaton & Tukey 1974; advocated for FWI by Bube &
  Nemeth 2007 and Liu et al. 2018):

  .. math::
      \\rho^{\\text{Tukey}}_{c}(r) =
          \\begin{cases}
              \\tfrac{c^2}{6}\\Bigl(1 - \\bigl[1 - (r/c)^2\\bigr]^3\\Bigr) & |r| \\le c \\\\
              \\tfrac{c^2}{6} & |r| > c
          \\end{cases}

  i.e. completely *rejects* residuals beyond the threshold ``c``.

* **Geman-McClure** (Geman & McClure 1985):

  .. math::
      \\rho^{\\text{GM}}_{c}(r) = \\frac{(r/c)^2}{1 + (r/c)^2}\\cdot c^2.

All three reduce to L2 for :math:`|r|\\to 0` (multiplied by suitable constants).

References
----------
* Beaton, A. E. & Tukey, J. W. (1974). *The fitting of power series, meaning
  polynomials, illustrated on band-spectroscopic data.* Technometrics 16,
  147-185.
* Black, M. J. & Anandan, P. (1996). *The robust estimation of multiple
  motions: Parametric and piecewise-smooth flow fields.* CVIU 63 (1),
  75-104.
* Bube, K. P. & Nemeth, T. (2007). *Fast line searches for the robust
  solution of linear systems in the hybrid l1/l2 and Huber norms.*
  **Geophysics** 72 (2), A13-A17.
* Geman, S. & McClure, D. E. (1985). *Bayesian image analysis: An
  application to single photon emission tomography.*  Proc. Stat. Comp.
  Sect., Amer. Stat. Assoc., 12-18.
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss


# ---------------------------------------------------------------------------
# Cauchy / Lorentzian
# ---------------------------------------------------------------------------
class CauchyLoss(BaseFWILoss):
    """Cauchy / Lorentzian robust misfit."""

    def __init__(
        self,
        c: float = 1.0,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = float(c)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        r = syn - obs
        c2 = self.c * self.c
        return 0.5 * c2 * torch.log1p((r * r) / c2)


# ---------------------------------------------------------------------------
# Tukey biweight
# ---------------------------------------------------------------------------
class TukeyLoss(BaseFWILoss):
    """Tukey biweight robust misfit.

    Outliers with :math:`|r| > c` contribute a constant :math:`c^2/6` so they
    cannot influence the gradient at all.
    """

    def __init__(
        self,
        c: float = 1.0,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = float(c)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        r = syn - obs
        c = self.c
        c2_over_6 = (c * c) / 6.0
        inside = 1.0 - (r / c) ** 2
        # Cube of clamped (so that |r| > c gives 0)
        cube = inside.clamp(min=0.0) ** 3
        return c2_over_6 * (1.0 - cube)


# ---------------------------------------------------------------------------
# Geman-McClure
# ---------------------------------------------------------------------------
class GemanMcClureLoss(BaseFWILoss):
    """Geman-McClure robust misfit."""

    def __init__(
        self,
        c: float = 1.0,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = float(c)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        r = syn - obs
        u = (r / self.c) ** 2
        return self.c * self.c * u / (1.0 + u)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
def cauchy_loss(syn, obs, c: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    return CauchyLoss(c=c, reduction=reduction)(syn, obs)


def tukey_loss(syn, obs, c: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    return TukeyLoss(c=c, reduction=reduction)(syn, obs)


def geman_mcclure_loss(
    syn, obs, c: float = 1.0, reduction: str = "mean"
) -> torch.Tensor:
    return GemanMcClureLoss(c=c, reduction=reduction)(syn, obs)


__all__ = [
    "CauchyLoss",
    "GemanMcClureLoss",
    "TukeyLoss",
    "cauchy_loss",
    "geman_mcclure_loss",
    "tukey_loss",
]
