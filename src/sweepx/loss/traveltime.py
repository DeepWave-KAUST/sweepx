"""Cross-correlation travel-time misfit (Luo & Schuster 1991).

For each trace, compute the cross-correlation

.. math::

    c(\\tau) \\;=\\; \\int d_{\\mathrm{syn}}(t)\\,d_{\\mathrm{obs}}(t+\\tau)\\,\\mathrm{d}t,

find the lag :math:`\\tau^*` maximising :math:`c(\\tau)` and define the
per-trace misfit as :math:`\\tfrac{1}{2}(\\tau^*)^2`.  The differentiable
implementation here follows Luo & Schuster (1991) and van Leeuwen & Mulder
(2010): we approximate :math:`\\tau^*` by a centroid-of-the-correlation
estimate

.. math::

    \\tau^* \\;\\approx\\;
        \\frac{\\sum_{\\tau} \\tau\\,c(\\tau)^p\\,w(\\tau)}{\\sum_{\\tau} c(\\tau)^p\\,w(\\tau)},

where :math:`p` is a sharpening exponent (``power``) and :math:`w(\\tau)` an
optional Gaussian gate around :math:`\\tau=0` (``sigma``).  Setting
``power -> infinity`` recovers the original argmax estimator (replace with
``softargmax`` ``temperature``).

For ``power=2`` and ``sigma=None`` this is the *quadratic-weighted centroid*
of the correlation; it is a *smooth* surrogate of ``argmax_tau c(tau)`` that
plays well with PyTorch autograd, while being exact whenever the
correlation is symmetric and unimodal — which is the case for noise-free
matched traces.

For exact-recovery checks see ``tests/test_traveltime.py``.

A closely related smooth surrogate (85th EAGE 2024, *Differentiable
Traveltime Misfit for Wave-Equation Tomography*) replaces the
non-negative ``power``-of-:math:`c(\\tau)` weighting by a **softmax**
of :math:`c(\\tau)`:

.. math::

    \\mathrm{prob}(\\tau)
        \\;=\\; \\frac{\\exp\\bigl(c(\\tau)\\bigr)}{\\sum_{\\tau'} \\exp\\bigl(c(\\tau')\\bigr)},
    \\qquad
    \\tau^\\star
        \\;\\approx\\; \\sum_\\tau \\tau\\,\\mathrm{prob}(\\tau).

Both estimators belong to the same family of smooth argmax surrogates;
in the high-sharpness limit (``power -> infinity`` or
softmax-temperature ``-> 0``) they both collapse to the hard argmax.

References
----------
* Luo, Y. & Schuster, G. T. (1991). *Wave-equation travel-time inversion.*
  **Geophysics** 56 (5), 645-653.  doi:10.1190/1.1443081
* Marquering, H., Dahlen, F. A. & Nolet, G. (1999). *Three-dimensional
  sensitivity kernels for finite-frequency traveltimes.*
  Geophys. J. Int. 137 (3), 805-815.
  doi:10.1046/j.1365-246x.1999.00837.x
* van Leeuwen, T. & Mulder, W. A. (2010). *A correlation-based misfit
  criterion for wave-equation traveltime tomography.*
  Geophys. J. Int. 182 (3), 1383-1394.
  doi:10.1111/j.1365-246X.2010.04681.x
* Wang, S., Song, P., Tan, J., Xia, D., Zhao, B. & Mao, S. (2024).
  *Differentiable Traveltime Misfit for Wave-Equation Tomography.*
  85th EAGE Annual Conference & Exhibition, Oslo, Norway, Expanded
  Abstracts, 1-5.  doi:10.3997/2214-4609.202410170
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical


def _cross_correlation(syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Full cross-correlation :math:`c(\\tau)=\\sum_t syn(t) obs(t+\\tau)`.

    Inputs are 2D ``(N, nt)``; output is ``(N, 2 nt - 1)`` arranged as
    :math:`\\tau\\in[-(nt-1), \\,nt-1]`.

    Implemented via FFT for O(nt log nt).
    """
    N, nt = syn.shape
    n_pad = 2 * nt - 1
    n_fft = 1 << (n_pad - 1).bit_length()
    S = torch.fft.rfft(syn, n=n_fft)
    O = torch.fft.rfft(obs, n=n_fft)
    # Cross-correlation = ifft(conj(S) * O) when we want
    # c(tau) = sum_t syn(t) obs(t+tau).
    C = torch.fft.irfft(torch.conj(S) * O, n=n_fft)
    # rearrange so tau = -(nt-1) .. nt-1
    return torch.cat([C[:, -(nt - 1):], C[:, :nt]], dim=-1)


class CrossCorrelationTraveltimeLoss(BaseFWILoss):
    """Smooth cross-correlation travel-time misfit (Luo-Schuster 1991).

    Parameters
    ----------
    dt
        Sampling interval.  The returned misfit is in time-units squared
        (``[s^2]`` if ``dt`` is in seconds).
    power
        Exponent used to sharpen the centroid estimate.  ``power=2`` is the
        classical correlation-weighted centroid.  Larger values approach the
        argmax estimator; smaller values give a smoother loss landscape.
    sigma
        Optional Gaussian gate (in samples) around :math:`\\tau=0` to confine
        the centroid to a physically plausible lag range.  ``None`` disables
        gating.
    """

    def __init__(
        self,
        dt: float = 1.0,
        power: float = 2.0,
        sigma: Optional[float] = None,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        self.dt = float(dt)
        self.power = float(power)
        self.sigma = float(sigma) if sigma is not None else None

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        syn_f, _canon = flatten_traces(syn_c)            # (N, nt)
        obs_f, _ = flatten_traces(obs_c)
        N, nt = syn_f.shape

        c = _cross_correlation(syn_f, obs_f)             # (N, 2nt-1)
        tau = torch.arange(
            -(nt - 1), nt, device=syn_f.device, dtype=syn_f.dtype
        )                                                # (2nt-1,)

        # Strictly positive weights for the centroid:
        weights = c.clamp(min=0.0) ** self.power
        if self.sigma is not None:
            gate = torch.exp(-(tau / self.sigma) ** 2 / 2.0)
            weights = weights * gate
        Z = weights.sum(dim=-1).clamp_min(1e-12)
        tau_star = (weights * tau).sum(dim=-1) / Z       # (N,)  in samples

        per_trace = 0.5 * (tau_star * self.dt) ** 2      # (N,)

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def cross_correlation_traveltime_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    dt: float = 1.0,
    power: float = 2.0,
    sigma: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    return CrossCorrelationTraveltimeLoss(
        dt=dt, power=power, sigma=sigma, reduction=reduction
    )(syn, obs)


__all__ = [
    "CrossCorrelationTraveltimeLoss",
    "cross_correlation_traveltime_loss",
]
