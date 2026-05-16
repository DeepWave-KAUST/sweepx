"""L1 (absolute-difference) misfit for FWI.

.. math::

    \\mathcal{J}_{L_1}(m) = \\sum_{s,t,r,c} \\bigl| d_{\\mathrm{syn}}[s,t,r,c]
        - d_{\\mathrm{obs}}[s,t,r,c]\\bigr|.

Compared with the L2 loss this misfit is much less sensitive to outliers in
the data: the influence function (gradient w.r.t. the residual) is the
sign function rather than the residual itself, so a single very large sample
contributes only :math:`\\pm 1` to the gradient.

References
----------
* Brossier, R., Operto, S. & Virieux, J. (2010). *Which data residual norm
  for robust elastic frequency-domain full waveform inversion?* **Geophysics**
  75 (3), R37-R46.  (Detailed comparison of L1 vs. L2 / Cauchy / Huber.)
* Crase, E., Pica, A., Noble, M., McDonald, J. & Tarantola, A. (1990).
  *Robust elastic nonlinear waveform inversion: application to real data.*
  **Geophysics** 55 (5), 527-538.  (Early use of L1 for FWI.)
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss


class L1Loss(BaseFWILoss):
    """L1 / absolute-difference FWI misfit."""

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return (syn - obs).abs()


def l1_loss(
    syn: torch.Tensor, obs: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Functional form of :class:`L1Loss`."""
    return L1Loss(reduction=reduction)(syn, obs)


__all__ = ["L1Loss", "l1_loss"]
