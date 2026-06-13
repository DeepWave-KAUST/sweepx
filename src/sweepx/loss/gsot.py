"""Graph-space Optimal Transport (GSOT) FWI misfit.

The graph-space approach of Métivier et al. (2018) embeds each sampled
trace in the *time-amplitude plane*

.. math::

    \\mathcal G[d] = \\bigl\\{(t_i,\\, d(t_i))\\bigr\\}_{i=0}^{n_t-1},

assigns one observed sample to one synthetic sample via a permutation
:math:`\\sigma\\in S_{n_t}`, and pays a 2-D ground cost

.. math::

    c\\bigl((t_i, d_s(t_i)),\\,(t_j, d_o(t_j))\\bigr) \\;=\\;
        \\eta\\,(t_i - t_j)^2 + (d_s(t_i) - d_o(t_j))^2.

The misfit is the minimum-cost permutation:

.. math::

    \\mathcal J_{\\mathrm{GSOT}}(m) \\;=\\;
        \\min_{\\sigma\\in S_{n_t}}\\sum_{i=0}^{n_t-1}
            c\\bigl((t_i, d_s(t_i)),\\,(t_{\\sigma(i)}, d_o(t_{\\sigma(i)}))\\bigr).

The weighting parameter :math:`\\eta = (\\Delta d_{\\max}/\\Delta t_{\\max})^2`
balances the time and amplitude axes; Métivier et al. (2018) recommend
:math:`\\eta = (\\sigma_d/T_{\\mathrm{shift,max}})^2` where
:math:`T_{\\mathrm{shift,max}}` is the largest plausible traveltime
error.

Solver
------
The exact min-cost assignment is solved by the **Hungarian / Jonker-
Volgenant** algorithm.  We delegate that to ``scipy.optimize.linear_sum_assignment``
which is :math:`O(n_t^3)` -- adequate for traces with a few hundred
samples (decimate the data otherwise).

Differentiability
-----------------
Once the permutation :math:`\\sigma^\\star` has been computed (via SciPy),
we still need a gradient.  We use the *envelope theorem*: for the optimal
:math:`\\sigma^\\star`,

.. math::

    \\nabla_{d_s(t_i)}\\,\\mathcal J_{\\mathrm{GSOT}} =
        2\\bigl(d_s(t_i) - d_o(t_{\\sigma^\\star(i)})\\bigr),

so we re-implement the loss in PyTorch *given* the SciPy-computed
permutation -- gradients flow naturally.  This is the standard approach
in differentiable OT (Cuturi 2013; Genevay et al. 2018).

References
----------
* Métivier, L., Allain, A., Brossier, R., Mérigot, Q., Oudet, E. &
  Virieux, J. (2018). *Optimal transport for mitigating cycle skipping
  in FWI: a graph-space transform approach.*  Geophysics 83 (5),
  R515-R540.  doi:10.1190/geo2017-0807.1
* Métivier, L., Brossier, R., Mérigot, Q. & Oudet, E. (2019). *A graph
  space optimal transport distance as a generalisation of Lp distances:
  application to a seismic imaging inverse problem.*  Inverse Problems
  35 (8), 085001.  doi:10.1088/1361-6420/ab206f
* Jonker, R. & Volgenant, A. (1987). *A shortest augmenting path
  algorithm for dense and sparse linear assignment problems.*
  Computing 38 (4), 325-340.  doi:10.1007/BF02278710
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore
    HAS_SCIPY = True
except ImportError:                                                   # pragma: no cover
    HAS_SCIPY = False


def _solve_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Hungarian solver wrapping ``scipy.optimize.linear_sum_assignment``.

    Parameters
    ----------
    cost
        ``(nt, nt)`` cost matrix (non-negative).

    Returns
    -------
    ``(nt,)`` LongTensor giving the column index assigned to each row.
    """
    if not HAS_SCIPY:                                                 # pragma: no cover
        raise RuntimeError(
            "GSOTLoss requires scipy.optimize.linear_sum_assignment. "
            "Install scipy: pip install scipy"
        )
    c = cost.detach().cpu().numpy()
    row_ind, col_ind = _scipy_lsa(c)
    return torch.as_tensor(col_ind, dtype=torch.long, device=cost.device)


class GSOTLoss(BaseFWILoss):
    """Graph-space Optimal Transport FWI misfit (Métivier et al. 2018).

    Parameters
    ----------
    eta
        Weight on the squared *time* cost.  Set to
        ``(max|d| / max_lag)**2`` for a typical seismic gather.  If
        ``None``, we estimate it from each trace automatically.
    dt
        Sampling interval in seconds.
    max_shift_samples
        If given, restrict the assignment to within ``+/- max_shift_samples``
        samples around the identity to make the :math:`O(n_t^3)` Hungarian
        solver cheaper (Métivier et al. 2018, sec. 3.3 -- this is the
        ``banded`` GSOT).  Pass ``None`` (default) for unrestricted GSOT.
    """

    def __init__(
        self,
        eta: Optional[float] = None,
        dt: float = 1.0,
        max_shift_samples: Optional[int] = None,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if eta is not None and eta <= 0:
            raise ValueError(f"eta must be > 0 if given, got {eta}")
        if max_shift_samples is not None and max_shift_samples <= 0:
            raise ValueError(
                f"max_shift_samples must be > 0 if given, got {max_shift_samples}"
            )
        self.eta = float(eta) if eta is not None else None
        self.dt = float(dt)
        self.max_shift_samples = (
            int(max_shift_samples) if max_shift_samples is not None else None
        )

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

        t = torch.arange(nt, device=s.device, dtype=s.dtype) * self.dt
        # Time cost (same for every trace)
        time_cost = (t.unsqueeze(0) - t.unsqueeze(1)) ** 2          # (nt, nt)

        per_trace = torch.zeros(N, device=s.device, dtype=s.dtype)
        BIG = 1.0e30
        for n in range(N):
            # Amplitude cost
            amp_cost = (s[n].unsqueeze(0) - o[n].unsqueeze(1)) ** 2  # (nt, nt) [j=obs_t, i=syn_t]
            # Effective eta
            eta = self.eta
            if eta is None:
                # Métivier (2018): eta = (max|d|/Tshiftmax)^2 with Tshiftmax = nt*dt/4
                d_max = float(torch.maximum(s[n].abs().max(), o[n].abs().max()).item())
                Tshift = nt * self.dt / 4.0
                eta = (d_max / Tshift) ** 2 + 1e-30
            # Total cost C[i, j] = eta * (t_i - t_j)^2 + (s_i - o_j)^2
            # We orient so rows=i (syn time), cols=j (obs time).
            cost = eta * time_cost + amp_cost.t()
            if self.max_shift_samples is not None:
                # Forbid assignments with |i - j| > max_shift_samples.
                idx_i = torch.arange(nt, device=s.device).unsqueeze(1)
                idx_j = torch.arange(nt, device=s.device).unsqueeze(0)
                forbidden = (idx_i - idx_j).abs() > self.max_shift_samples
                cost = cost.masked_fill(forbidden, BIG)

            col = _solve_assignment(cost)
            # Differentiable loss with the optimal permutation held fixed
            o_perm = o[n][col]
            per_trace[n] = (eta * (t - t[col]) ** 2 + (s[n] - o_perm) ** 2).sum()

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def gsot_loss(
    syn,
    obs,
    eta: Optional[float] = None,
    dt: float = 1.0,
    max_shift_samples: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    return GSOTLoss(
        eta=eta,
        dt=dt,
        max_shift_samples=max_shift_samples,
        reduction=reduction,
    )(syn, obs)


__all__ = ["GSOTLoss", "gsot_loss"]
