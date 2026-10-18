"""Soft-DTW (dynamic time warping) FWI misfit (Cuturi & Blondel 2017).

Soft-DTW is a smoothed version of classical Dynamic Time Warping that
makes the warping cost *differentiable*:

.. math::

    \\mathrm{sDTW}_\\gamma(d_s, d_o) \\;=\\; -\\gamma\\,\\log\\!\\sum_{A \\in \\mathcal A}
        \\exp\\!\\bigl(-\\tfrac{1}{\\gamma}\\,\\langle A, \\Delta\\rangle\\bigr),

where :math:`\\Delta_{ij} = (d_s(t_i) - d_o(t_j))^2` is the pointwise cost
and :math:`\\mathcal A` is the set of monotone alignment paths.  As
:math:`\\gamma \\to 0^+` we recover the classical DTW; as
:math:`\\gamma \\to \\infty` we recover :math:`-\\gamma\\log|\\mathcal A|` + a
soft-min of all path costs.

This package implements the **O(nt^2)** time, **O(nt^2)** memory forward
recursion of Cuturi & Blondel (2017, alg. 1).  Gradients are obtained
automatically via PyTorch autograd through the soft-min recursion.

Soft-DTW has been used as a differentiable FWI misfit by Chen, Peter &
Ravasi (2022) and is one of the most powerful misfits for matching
wavetrains under unknown, **non-stationary** time warps.

References
----------
* Cuturi, M. & Blondel, M. (2017). *Soft-DTW: a differentiable loss
  function for time-series.*  ICML 70, 894-903.  arXiv:1703.01541
* Chen, F., Peter, D. & Ravasi, M. (2022). *Cycle-skipping mitigation
  using misfit measurements based on differentiable dynamic time
  warping.* **Geophysics** 87 (4), R325-R335.
  doi:10.1190/geo2021-0598.1
* Ma, Y. & Hale, D. (2013). *Wave-equation reflection traveltime
  inversion with dynamic warping and full-waveform inversion.*
  **Geophysics** 78 (6), R223-R233.
  doi:10.1190/geo2013-0004.1
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical


def _soft_min(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Soft-min of three tensors with temperature gamma.

    soft-min_gamma(a, b, c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))
    """
    # Stack to (N, 3) then logsumexp for numerical stability.
    stack = torch.stack([-a / gamma, -b / gamma, -c / gamma], dim=-1)
    return -gamma * torch.logsumexp(stack, dim=-1)


def _soft_dtw_forward(D: torch.Tensor, gamma: float) -> torch.Tensor:
    """Forward Soft-DTW recursion.

    Parameters
    ----------
    D
        Pointwise cost ``(N, nt_s, nt_o)``.
    gamma
        Soft-min temperature; ``gamma -> 0`` recovers classical DTW.

    Returns
    -------
    ``(N,)`` Soft-DTW value per batch entry.
    """
    N, n, m = D.shape
    # Padded DP table with +inf on the boundary so the recursion is clean.
    R = torch.full((N, n + 2, m + 2), float("inf"), device=D.device, dtype=D.dtype)
    R[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            R[:, i, j] = D[:, i - 1, j - 1] + _soft_min(
                R[:, i - 1, j], R[:, i, j - 1], R[:, i - 1, j - 1], gamma
            )
    return R[:, n, m]


class SoftDTWLoss(BaseFWILoss):
    """Soft-DTW FWI misfit (Cuturi & Blondel 2017).

    Parameters
    ----------
    gamma
        Soft-min temperature.  ``gamma -> 0`` is classical DTW (not
        differentiable); ``gamma = 1`` (default) is the smooth Cuturi-
        Blondel setting.
    normalize_by_length
        If True (default), divide the per-trace cost by ``nt`` so the
        loss does not scale with trace length.  Cuturi & Blondel define
        Soft-DTW without this normalisation.
    divergence
        If True (default), return the **soft-DTW divergence** of
        Blondel-Mensch-Vert (2020),

        .. math::

            D_\\gamma(x, y) = \\mathrm{sDTW}_\\gamma(x, y)
                - \\tfrac{1}{2}\\bigl[\\mathrm{sDTW}_\\gamma(x, x)
                + \\mathrm{sDTW}_\\gamma(y, y)\\bigr].

        This is non-negative and zero iff :math:`x=y`, which is the
        property an FWI misfit must have.  Setting ``divergence=False``
        returns the raw soft-DTW (Cuturi-Blondel 2017), which can be
        negative for ``gamma > 0`` (because the soft-min mixes
        exponentially many alignment paths, so the log-partition can
        exceed zero).  Default ``True``.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        normalize_by_length: bool = True,
        divergence: bool = True,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.gamma = float(gamma)
        self.normalize_by_length = bool(normalize_by_length)
        self.divergence = bool(divergence)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        s, _ = flatten_traces(syn_c)
        o, _ = flatten_traces(obs_c)
        N, nt = s.shape

        # Pointwise cost matrix Delta[i, j] = (s[i] - o[j])^2 per trace.
        D_so = (s.unsqueeze(-1) - o.unsqueeze(-2)) ** 2

        per_trace = _soft_dtw_forward(D_so, self.gamma)
        if self.divergence:
            D_ss = (s.unsqueeze(-1) - s.unsqueeze(-2)) ** 2
            D_oo = (o.unsqueeze(-1) - o.unsqueeze(-2)) ** 2
            per_trace = per_trace - 0.5 * (
                _soft_dtw_forward(D_ss, self.gamma)
                + _soft_dtw_forward(D_oo, self.gamma)
            )
        if self.normalize_by_length:
            per_trace = per_trace / nt

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def soft_dtw_loss(
    syn,
    obs,
    gamma: float = 1.0,
    normalize_by_length: bool = True,
    divergence: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    return SoftDTWLoss(
        gamma=gamma,
        normalize_by_length=normalize_by_length,
        divergence=divergence,
        reduction=reduction,
    )(syn, obs)


__all__ = ["SoftDTWLoss", "soft_dtw_loss"]
