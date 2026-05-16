"""Instantaneous-phase and envelope-weighted-phase FWI misfits.

For a real signal :math:`d(t)` with analytic signal
:math:`a(t)=d(t)+i\\,\\mathcal H[d](t) = E(t)\\,e^{i\\phi(t)}`,

* the **instantaneous phase** is :math:`\\phi(t)=\\arctan(\\mathcal H[d]/d)`,
* the **instantaneous envelope** is :math:`E(t)=|a(t)|`.

We provide two principled phase-based misfits:

1. **Instantaneous-phase difference** (Bozdağ, Trampert & Tromp 2011,
   eq. 22; Fichtner et al. 2008):

   .. math::

      \\mathcal J_\\phi(m) = \\tfrac{1}{2}\\sum
          \\bigl[w_\\phi(t)\\,\\Delta\\phi(t)\\bigr]^2,

   where :math:`\\Delta\\phi = \\operatorname{wrap}(\\phi_s-\\phi_o)` is the
   wrapped phase difference and :math:`w_\\phi(t)` an optional envelope
   weight that down-weights samples where the envelope is too small for
   the phase to be reliable (default: :math:`w_\\phi = E_o(t)/\\max E_o`).

2. **Envelope + phase combined** (Fichtner 2008; Yuan, Simons & Tromp
   2016):

   .. math::

      \\mathcal J_{E+\\phi}(m) = (1-\\alpha)\\,\\mathcal J_E + \\alpha\\,\\mathcal J_\\phi.

   The default :math:`\\alpha = 0.5` gives equal weighting; many studies
   pick :math:`\\alpha` per frequency band.

Implementation notes
--------------------
* The wrapped phase difference is computed *without* unwrapping the
  signals — :math:`\\operatorname{wrap}(x)=\\arctan2(\\sin x,\\cos x)` is
  differentiable and never returns values outside :math:`(-\\pi,\\pi]`.
* When ``envelope_weight=True`` (default) we multiply the residual by
  :math:`E_o(t)` before squaring; this is the standard recipe to avoid
  amplifying noise where the envelope is essentially zero.

References
----------
* Bozdağ, E., Trampert, J. & Tromp, J. (2011). *Misfit functions for full
  waveform inversion based on instantaneous phase and envelope
  measurements.*  Geophys. J. Int. 185 (2), 845-870.
  doi:10.1111/j.1365-246X.2011.04970.x
* Fichtner, A., Kennett, B. L. N., Igel, H. & Bunge, H.-P. (2008).
  *Theoretical background for continental- and global-scale full-waveform
  inversion in the time-frequency domain.*  Geophys. J. Int. 175 (2),
  665-685.  doi:10.1111/j.1365-246X.2008.03923.x
* Yuan, Y. O., Simons, F. J. & Tromp, J. (2016). *Double-difference
  adjoint seismic tomography.*  Geophys. J. Int. 206 (3), 1599-1618.
  doi:10.1093/gji/ggw233
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import BaseFWILoss
from ._utils import envelope, hilbert


def _wrap_phase_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapped phase difference d phi = arctan2(sin(a-b), cos(a-b)).

    a, b can be the instantaneous phases of two signals.
    """
    s = torch.sin(a - b)
    c = torch.cos(a - b)
    return torch.atan2(s, c)


class InstantaneousPhaseLoss(BaseFWILoss):
    """Instantaneous-phase difference misfit (Bozdağ 2011, eq. 22)."""

    def __init__(
        self,
        envelope_weight: bool = True,
        eps: float = 1e-6,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.envelope_weight = bool(envelope_weight)
        self.eps = float(eps)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        as_ = hilbert(syn, dim=-3)
        ao_ = hilbert(obs, dim=-3)
        phi_s = torch.atan2(as_.imag, as_.real)
        phi_o = torch.atan2(ao_.imag, ao_.real)
        dphi = _wrap_phase_diff(phi_s, phi_o)
        if self.envelope_weight:
            Eo = torch.sqrt(ao_.real * ao_.real + ao_.imag * ao_.imag + self.eps)
            # Normalise by max along time so the weight is in [0, 1].
            wmax = Eo.amax(dim=-3, keepdim=True).clamp_min(self.eps)
            w = Eo / wmax
            dphi = w * dphi
        return 0.5 * dphi * dphi


class EnvelopePhaseLoss(BaseFWILoss):
    """Weighted combination of envelope + instantaneous-phase misfits."""

    def __init__(
        self,
        alpha: float = 0.5,
        envelope_p: int = 2,
        envelope_log: bool = False,
        envelope_weight_phase: bool = True,
        eps: float = 1e-12,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if envelope_p not in (1, 2):
            raise ValueError(f"envelope_p must be 1 or 2, got {envelope_p}")
        self.alpha = float(alpha)
        self.envelope_p = int(envelope_p)
        self.envelope_log = bool(envelope_log)
        self.envelope_weight_phase = bool(envelope_weight_phase)
        self.eps = float(eps)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        as_ = hilbert(syn, dim=-3)
        ao_ = hilbert(obs, dim=-3)
        Es = torch.sqrt(as_.real * as_.real + as_.imag * as_.imag + self.eps)
        Eo = torch.sqrt(ao_.real * ao_.real + ao_.imag * ao_.imag + self.eps)

        # Envelope term
        if self.envelope_log:
            re = torch.log(Es.clamp_min(self.eps)) - torch.log(Eo.clamp_min(self.eps))
            env_pw = 0.5 * re * re
        elif self.envelope_p == 2:
            re = Es - Eo
            env_pw = 0.5 * re * re
        else:
            env_pw = (Es - Eo).abs()

        # Phase term
        phi_s = torch.atan2(as_.imag, as_.real)
        phi_o = torch.atan2(ao_.imag, ao_.real)
        dphi = _wrap_phase_diff(phi_s, phi_o)
        if self.envelope_weight_phase:
            wmax = Eo.amax(dim=-3, keepdim=True).clamp_min(self.eps)
            w = Eo / wmax
            dphi = w * dphi
        phi_pw = 0.5 * dphi * dphi

        return (1.0 - self.alpha) * env_pw + self.alpha * phi_pw


def instantaneous_phase_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    envelope_weight: bool = True,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    return InstantaneousPhaseLoss(
        envelope_weight=envelope_weight, eps=eps, reduction=reduction
    )(syn, obs)


def envelope_phase_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    alpha: float = 0.5,
    envelope_p: int = 2,
    envelope_log: bool = False,
    envelope_weight_phase: bool = True,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    return EnvelopePhaseLoss(
        alpha=alpha,
        envelope_p=envelope_p,
        envelope_log=envelope_log,
        envelope_weight_phase=envelope_weight_phase,
        eps=eps,
        reduction=reduction,
    )(syn, obs)


__all__ = [
    "EnvelopePhaseLoss",
    "InstantaneousPhaseLoss",
    "envelope_phase_loss",
    "instantaneous_phase_loss",
]
