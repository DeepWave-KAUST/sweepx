"""Exponentiated-phase FWI misfit (Yuan, Bozdağ, Ciardelli, Gao & Simons 2020).

Unlike the bare instantaneous-phase misfit (Bozdağ et al. 2011), which has to
deal with phase unwrapping near zero-amplitude samples, the exponentiated
phase normalises the analytic signal by its own envelope so the comparison
is done on the **complex unit circle**:

.. math::

    \\tilde s(t) = \\frac{s(t) + i\\,\\mathcal H[s](t)}{E_s(t)} = e^{i\\phi_s(t)},
    \\qquad E_s(t) = \\sqrt{s^2(t) + \\mathcal H^2[s](t)},

with the misfit (Yuan et al. 2020, eq. 7)

.. math::

    \\chi_{\\mathrm{EP}}(m) = \\tfrac{1}{2}\\sum_{\\text{trace}} \\int_0^T
        \\bigl|\\Re\\{\\tilde s\\} - \\Re\\{\\tilde d\\}\\bigr|^2
        + \\bigl|\\Im\\{\\tilde s\\} - \\Im\\{\\tilde d\\}\\bigr|^2 \\,\\mathrm dt.

Equivalently, this is the squared L2 distance on the **unit-modulus**
analytic signals -- it carries the same phase information as
:class:`InstantaneousPhaseLoss` but is everywhere :math:`C^\\infty` and has
no branch cut at :math:`\\phi = \\pm\\pi`.

References
----------
* Yuan, Y. O., Bozdağ, E., Ciardelli, C., Gao, F. & Simons, F. J. (2020).
  *The exponentiated phase measurement, and objective-function
  hybridisation for adjoint waveform tomography.*  Geophys. J. Int.
  221 (2), 1145-1164.  doi:10.1093/gji/ggaa063
* Gao, F., Yuan, Y. O., Ciardelli, C., Simons, F. J., Bozdağ, E. &
  Tromp, J. (2023). *Review of misfit functions for adjoint full
  waveform inversion in seismology.*  Geophys. J. Int. 235 (3),
  2794-2820.  doi:10.1093/gji/ggad372
"""

from __future__ import annotations

import torch

from ._utils import hilbert
from .base import BaseFWILoss


class ExponentiatedPhaseLoss(BaseFWILoss):
    """Exponentiated-phase FWI misfit (Yuan, Bozdağ et al. 2020).

    Parameters
    ----------
    eps
        Stabiliser added inside the envelope :math:`\\sqrt{s^2 + H[s]^2 + \\epsilon}`
        to keep gradients finite where the envelope is near zero.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        self.eps = float(eps)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        as_ = hilbert(syn, dim=-3)
        ao_ = hilbert(obs, dim=-3)
        Es = torch.sqrt(as_.real * as_.real + as_.imag * as_.imag + self.eps)
        Eo = torch.sqrt(ao_.real * ao_.real + ao_.imag * ao_.imag + self.eps)
        dR = as_.real / Es - ao_.real / Eo
        dI = as_.imag / Es - ao_.imag / Eo
        return 0.5 * (dR * dR + dI * dI)


def exponentiated_phase_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    return ExponentiatedPhaseLoss(eps=eps, reduction=reduction)(syn, obs)


__all__ = ["ExponentiatedPhaseLoss", "exponentiated_phase_loss"]
