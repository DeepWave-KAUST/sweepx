"""Student's-t negative-log-likelihood misfit (Aravkin et al. 2011).

For residual :math:`r = d_{\\mathrm{syn}} - d_{\\mathrm{obs}}` modelled as
Student's-t with :math:`\\nu` degrees of freedom and scale :math:`\\sigma`,
the negative log-likelihood up to a data-independent constant is

.. math::

    \\rho^{\\nu,\\sigma}(r) \\;=\\; \\tfrac{\\nu+1}{2}\\,
        \\log\\!\\Bigl(1 + \\tfrac{1}{\\nu}(r/\\sigma)^2\\Bigr).

Limits:

* :math:`\\nu = 1`  → Cauchy/Lorentzian (up to a known overall factor).
* :math:`\\nu \\to \\infty`  → Gaussian / L2 (up to a known overall factor).

References
----------
* Aravkin, A. Y., van Leeuwen, T. & Herrmann, F. J. (2011). *Robust FWI
  using Student-t distribution.*  SEG Technical Program Expanded Abstracts,
  pp. 2669-2673.  doi:10.1190/1.3627747
* Aravkin, A., Burke, J. V. & Friedlander, M. P. (2013). *Variational
  properties of value functions.* SIAM J. Optim. 23 (3), 1689-1717.
  doi:10.1137/120899157
"""

from __future__ import annotations

import math

import torch

from .base import BaseFWILoss


class StudentTLoss(BaseFWILoss):
    """Student's-t negative-log-likelihood FWI misfit.

    Parameters
    ----------
    nu
        Degrees of freedom (``nu > 0``).  ``nu = 1`` reproduces Cauchy.
        ``nu -> infinity`` reproduces Gaussian / L2.
    sigma
        Scale parameter (``sigma > 0``).
    full_nll
        If ``True`` include the data-independent normalisation constants of
        the Student-t log-density.  Default is ``False`` (the
        residual-dependent part only — this is what enters the FWI gradient).
    """

    def __init__(
        self,
        nu: float = 1.0,
        sigma: float = 1.0,
        full_nll: bool = False,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.nu = float(nu)
        self.sigma = float(sigma)
        self.full_nll = bool(full_nll)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        r = syn - obs
        nu, sigma = self.nu, self.sigma
        u = (r / sigma) ** 2
        rho = 0.5 * (nu + 1.0) * torch.log1p(u / nu)
        if self.full_nll:
            # -log p(r) = rho(r) + const, where the constant is
            #   log Gamma(nu/2) - log Gamma((nu+1)/2) + 1/2 log(nu pi) + log sigma.
            const = (
                math.lgamma(nu / 2.0)
                - math.lgamma((nu + 1.0) / 2.0)
                + 0.5 * math.log(nu * math.pi)
                + math.log(sigma)
            )
            rho = rho + const
        return rho


def student_t_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    nu: float = 1.0,
    sigma: float = 1.0,
    full_nll: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    return StudentTLoss(
        nu=nu, sigma=sigma, full_nll=full_nll, reduction=reduction
    )(syn, obs)


__all__ = ["StudentTLoss", "student_t_loss"]
