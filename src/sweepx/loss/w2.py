"""2-Wasserstein FWI misfit via the 1-D CDF transport map.

For two 1-D probability densities :math:`f, g` on :math:`[0, T]` with
strictly-increasing CDFs :math:`F, G`, the 2-Wasserstein distance is

.. math::

    W_2^2(f, g) \\;=\\; \\int_0^1 \\bigl(F^{-1}(z) - G^{-1}(z)\\bigr)^2 \\,\\mathrm dz.

We evaluate this integral by sampling :math:`z` uniformly on
:math:`(0, 1)`, computing the inverse CDFs of the two signals by linear
interpolation of the cumulative sums, and summing the squared
differences.

Each trace is first mapped to a non-negative density via a positive
transform (square / abs / linear / exp, see :func:`positive_transform`).

References
----------
* Engquist, B. & Froese, B. D. (2014). *Application of the Wasserstein
  metric to seismic signals.*  Commun. Math. Sci. 12 (5), 979-988.
  doi:10.4310/CMS.2014.v12.n5.a7
* Engquist, B., Froese, B. D. & Yang, Y. (2016). *Optimal transport for
  seismic full waveform inversion.*  Comm. Math. Sci. 14 (8), 2309-2330.
  doi:10.4310/CMS.2016.v14.n8.a9
* Yang, Y., Engquist, B., Sun, J. & Hamfeldt, B. F. (2018). *Application
  of optimal transport and the quadratic Wasserstein metric to
  full-waveform inversion.* **Geophysics** 83 (1), R43-R62.
  doi:10.1190/geo2016-0663.1
"""

from __future__ import annotations

import torch

from ._utils import normalize_density, positive_transform
from .base import BaseFWILoss, flatten_traces, to_canonical


def _inverse_cdf(F: torch.Tensor, dt: float, z: torch.Tensor) -> torch.Tensor:
    """Inverse CDF (quantile function) by linear interpolation.

    F has shape ``(N, nt)`` and is monotonically non-decreasing along the
    last axis with :math:`F[:, 0]=p_0 \\ge 0` and :math:`F[:, -1] \\le 1`
    (in practice =1 after normalisation).  ``z`` has shape ``(M,)``.

    Returns ``(N, M)`` with the times :math:`t(z)` at which
    :math:`F(t) = z`.
    """
    N, nt = F.shape
    M = z.shape[0]
    # times associated with each F-sample
    t = torch.arange(nt, device=F.device, dtype=F.dtype) * dt  # (nt,)

    # For each (n, m), find the index k such that F[n, k] <= z[m] < F[n, k+1]
    # then linearly interpolate t between t[k] and t[k+1].
    # We use torch.searchsorted on each row.  searchsorted returns the
    # first index where z could be inserted while keeping F sorted, which
    # is what we want (right-most index k+1 such that F[k] <= z).
    idx = torch.searchsorted(F.contiguous(), z.expand(N, M).contiguous(), right=True)
    idx = idx.clamp(min=1, max=nt - 1)
    idx_low = idx - 1
    idx_high = idx

    # Gather F and t along the time axis.
    F_low = torch.gather(F, 1, idx_low)
    F_high = torch.gather(F, 1, idx_high)
    t_low = t[idx_low]
    t_high = t[idx_high]
    # Avoid divide-by-zero where the CDF is flat.
    denom = (F_high - F_low).clamp_min(1e-30)
    frac = (z.expand(N, M) - F_low) / denom
    return t_low + frac * (t_high - t_low)


class Wasserstein2Loss(BaseFWILoss):
    """2-Wasserstein FWI misfit via the inverse-CDF (quantile-difference) form."""

    def __init__(
        self,
        positive: str = "linear",
        c: "float | None" = None,
        dt: float = 1.0,
        n_quantiles: int = 256,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if positive not in ("square", "abs", "linear", "exp"):
            raise ValueError(f"unknown positive='{positive}'")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if n_quantiles < 4:
            raise ValueError(f"n_quantiles must be >= 4, got {n_quantiles}")
        self.positive = positive
        self.c = float(c) if c is not None else None
        self.dt = float(dt)
        self.n_quantiles = int(n_quantiles)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        s, _ = flatten_traces(syn_c)
        o, _ = flatten_traces(obs_c)

        c_param = self.c
        if self.positive == "linear" and c_param is None:
            c_param = float(torch.max(s.abs().max(), o.abs().max()).item()) + 1e-6
        fs = positive_transform(s, method=self.positive, c=c_param)
        fo = positive_transform(o, method=self.positive, c=c_param)
        ps = normalize_density(fs, dim=-1)
        po = normalize_density(fo, dim=-1)

        Fs = torch.cumsum(ps, dim=-1)
        Fo = torch.cumsum(po, dim=-1)
        # Quantile sampling avoids the endpoints
        z = (torch.arange(self.n_quantiles, device=s.device, dtype=s.dtype) + 0.5) / self.n_quantiles
        ts = _inverse_cdf(Fs, self.dt, z)
        to_ = _inverse_cdf(Fo, self.dt, z)
        d2 = (ts - to_) ** 2                         # (N, M)
        per_trace = d2.mean(dim=-1)                  # Monte-Carlo of integral over z
        per_trace = 0.5 * per_trace                  # convention 1/2 W2^2

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def w2_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    positive: str = "linear",
    c: "float | None" = None,
    dt: float = 1.0,
    n_quantiles: int = 256,
    reduction: str = "mean",
) -> torch.Tensor:
    return Wasserstein2Loss(
        positive=positive, c=c, dt=dt, n_quantiles=n_quantiles, reduction=reduction
    )(syn, obs)


__all__ = ["Wasserstein2Loss", "w2_loss"]
