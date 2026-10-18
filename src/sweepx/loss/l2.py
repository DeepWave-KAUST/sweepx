"""Classical L2 (least-squares) misfit for FWI.

Given a synthetic data tensor :math:`d_{\\text{syn}}` and an observed data
tensor :math:`d_{\\text{obs}}` (both with canonical layout
``(nshots, nt, nreceivers, nchannel)``) the *least-squares* misfit is

.. math::

    \\mathcal{J}_{L_2}(m) = \\tfrac{1}{2}
        \\sum_{s,t,r,c} \\bigl(d_{\\text{syn}}[s,t,r,c] - d_{\\text{obs}}[s,t,r,c]\\bigr)^2

This is the canonical FWI objective introduced by Lailly (1983) and
Tarantola (1984).

References
----------
* Tarantola, A. (1984). *Inversion of seismic reflection data in the
  acoustic approximation*. Geophysics 49 (8), 1259-1266.
* Lailly, P. (1983). *The seismic inverse problem as a sequence of before
  stack migrations*.  Conference on Inverse Scattering, SIAM.
* Virieux, J. & Operto, S. (2009). *An overview of full-waveform inversion
  in exploration geophysics*.  Geophysics 74 (6), WCC1-WCC26.
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss


class L2Loss(BaseFWILoss):
    """Classical least-squares FWI misfit.

    Parameters
    ----------
    reduction
        ``"mean"`` (default), ``"sum"`` or ``"none"``.  Note that the literature
        convention is :math:`\\tfrac{1}{2}\\sum (\\cdot)^2`; when ``half=True``
        (default) the elementwise misfit is multiplied by 1/2 so that the
        gradient w.r.t. :math:`d_{\\text{syn}}` is exactly the residual.
    half
        If ``True`` (default) the leading 1/2 factor is applied.
    mask
        Optional mask broadcastable to the inputs.
    """

    def __init__(
        self,
        reduction: str = "mean",
        half: bool = True,
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.half = bool(half)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        r = syn - obs
        m = r * r
        if self.half:
            m = 0.5 * m
        return m


def l2_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    reduction: str = "mean",
    half: bool = True,
) -> torch.Tensor:
    """Functional form of :class:`L2Loss`."""
    return L2Loss(reduction=reduction, half=half)(syn, obs)


__all__ = ["L2Loss", "l2_loss"]
