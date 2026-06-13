"""Sinkhorn / entropic-OT FWI misfit (Cuturi 2013; Sun & Alkhalifah 2020).

For two discrete probability measures :math:`p, q` on the time grid
:math:`\\{t_i\\}_{i=0}^{n_t-1}` and a cost matrix :math:`C_{ij} = (t_i -
t_j)^2`, the entropy-regularised optimal transport ("Sinkhorn") problem
is

.. math::

    \\mathrm{OT}_\\varepsilon(p, q) \\;=\\;
        \\min_{\\Pi \\in \\Pi(p, q)} \\sum_{ij} C_{ij}\\,\\Pi_{ij}
        + \\varepsilon\\sum_{ij} \\Pi_{ij}\\bigl(\\log \\Pi_{ij} - 1\\bigr),

solved by Sinkhorn iterations (matrix-scaling on
:math:`K_{ij} = e^{-C_{ij}/\\varepsilon}`).  We use the **log-domain**
Sinkhorn updates of Schmitzer (2019) for numerical stability.

The *Sinkhorn divergence* (Feydy et al. 2019)

.. math::

    S_\\varepsilon(p, q) = \\mathrm{OT}_\\varepsilon(p, q) - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(p, p) - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(q, q)

is a *true* divergence (non-negative, zero iff :math:`p=q`) and is what
we return by default (``debiased=True``).  Setting ``debiased=False``
returns the raw :math:`\\mathrm{OT}_\\varepsilon`.

References
----------
* Cuturi, M. (2013). *Sinkhorn distances: lightspeed computation of
  optimal transport.*  NeurIPS 26.  arXiv:1306.0895.
* Feydy, J., Séjourné, T., Vialard, F.-X., Amari, S., Trouvé, A. &
  Peyré, G. (2019). *Interpolating between optimal transport and MMD
  using Sinkhorn divergences.*  AISTATS 22, 2681-2690.
  arXiv:1810.08278.
* Schmitzer, B. (2019). *Stabilised sparse scaling algorithms for entropy
  regularised transport problems.*  SIAM J. Sci. Comput. 41 (3),
  A1443-A1481.  doi:10.1137/16M1106018
* Sun, B. & Alkhalifah, T. (2020). *ML-misfit: a neural network-based
  misfit function for full-waveform inversion.*  arXiv:2002.03163.
"""

from __future__ import annotations

import torch

from ._utils import normalize_density, positive_transform
from .base import BaseFWILoss, flatten_traces, to_canonical


def _log_sum_exp(a: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = a.max(dim=dim, keepdim=True)
    return (m.squeeze(dim) + torch.log((a - m).exp().sum(dim=dim) + 1e-30))


def _sinkhorn_log(p: torch.Tensor, q: torch.Tensor, C: torch.Tensor,
                  epsilon: float, n_iter: int) -> torch.Tensor:
    """Log-domain Sinkhorn for batched 1-D problems.

    Parameters
    ----------
    p, q
        ``(N, nt)`` non-negative tensors summing to 1 along the last axis.
    C
        ``(nt, nt)`` cost matrix.
    epsilon
        Entropic regularisation.
    n_iter
        Number of Sinkhorn iterations.

    Returns
    -------
    OT_eps : ``(N,)`` tensor with the entropic OT value for each row.
    """
    N, nt = p.shape
    log_p = torch.log(p.clamp_min(1e-30))
    log_q = torch.log(q.clamp_min(1e-30))
    K = -C / epsilon                              # (nt, nt)
    f = torch.zeros_like(log_p)                   # (N, nt)
    g = torch.zeros_like(log_q)                   # (N, nt)
    for _ in range(n_iter):
        # update g: log mu(j) = log q(j) - lse_i (-C(i,j)/eps + f(i)/eps)
        a = K.unsqueeze(0) + (f / epsilon).unsqueeze(-1)        # (N, nt, nt)
        g = epsilon * (log_q - _log_sum_exp(a, dim=-2))
        b = K.unsqueeze(0) + (g / epsilon).unsqueeze(-2)        # (N, nt, nt)
        f = epsilon * (log_p - _log_sum_exp(b, dim=-1))

    # OT_eps = sum_ij Pi_ij * (C_ij + epsilon * (log Pi_ij - 1))
    # Pi_ij = exp((f_i + g_j - C_ij)/eps + log p_i + log q_j - log p_i - log q_j)
    # simpler: Pi_ij = exp((f_i + g_j - C_ij) / eps) with this convention.
    # Then OT_eps = sum_ij Pi_ij C_ij  +  epsilon * sum_ij Pi_ij (log Pi_ij - 1)
    # In the dual: OT_eps = <f, p> + <g, q>.
    OT = (f * p).sum(dim=-1) + (g * q).sum(dim=-1)
    return OT


class SinkhornLoss(BaseFWILoss):
    """Sinkhorn / entropic-OT FWI misfit.

    Parameters
    ----------
    epsilon
        Entropic regularisation, in **squared time units** (same units as
        :math:`(t_i-t_j)^2`).  Default ``1.0`` is sensible when ``dt=1``;
        scale by ``dt`` accordingly for real seismic data.
    n_iter
        Number of Sinkhorn iterations.  64 is usually enough.
    positive
        Positive transform applied to each signed trace.
    dt
        Sampling interval (used to build the cost matrix).
    debiased
        If ``True`` (default) return the Sinkhorn divergence
        :math:`S_\\varepsilon`.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        n_iter: int = 64,
        positive: str = "linear",
        c: "float | None" = None,
        dt: float = 1.0,
        debiased: bool = True,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if n_iter <= 0:
            raise ValueError(f"n_iter must be > 0, got {n_iter}")
        if positive not in ("square", "abs", "linear", "exp"):
            raise ValueError(f"unknown positive='{positive}'")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.epsilon = float(epsilon)
        self.n_iter = int(n_iter)
        self.positive = positive
        self.c = float(c) if c is not None else None
        self.dt = float(dt)
        self.debiased = bool(debiased)

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

        c_param = self.c
        if self.positive == "linear" and c_param is None:
            c_param = float(torch.max(s.abs().max(), o.abs().max()).item()) + 1e-6
        fs = positive_transform(s, method=self.positive, c=c_param)
        fo = positive_transform(o, method=self.positive, c=c_param)
        p = normalize_density(fs, dim=-1)
        q = normalize_density(fo, dim=-1)

        # Cost matrix (built once per call).
        t = torch.arange(nt, device=s.device, dtype=s.dtype) * self.dt
        C = (t.unsqueeze(0) - t.unsqueeze(1)) ** 2

        OTpq = _sinkhorn_log(p, q, C, self.epsilon, self.n_iter)
        if self.debiased:
            OTpp = _sinkhorn_log(p, p, C, self.epsilon, self.n_iter)
            OTqq = _sinkhorn_log(q, q, C, self.epsilon, self.n_iter)
            per_trace = OTpq - 0.5 * (OTpp + OTqq)
        else:
            per_trace = OTpq

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def sinkhorn_loss(
    syn,
    obs,
    epsilon: float = 1.0,
    n_iter: int = 64,
    positive: str = "linear",
    c: "float | None" = None,
    dt: float = 1.0,
    debiased: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    return SinkhornLoss(
        epsilon=epsilon,
        n_iter=n_iter,
        positive=positive,
        c=c,
        dt=dt,
        debiased=debiased,
        reduction=reduction,
    )(syn, obs)


__all__ = ["SinkhornLoss", "sinkhorn_loss"]
