"""Adaptive Waveform Inversion (AWI) misfit, Warner & Guasch (2014, 2016).

For each trace, the modelled data :math:`d_s(t)` are convolved with a
**Wiener filter** :math:`w(\\tau)` that best maps them onto the observed
data :math:`d_o(t)`:

.. math::

    w \\;=\\; \\arg\\min_{w}\\; \\bigl\\|\\, d_s \\ast w - d_o\\,\\bigr\\|_2^2
            + \\epsilon\\,\\|w\\|_2^2.

If the model is perfect we expect :math:`w` to collapse to a Kronecker
delta at zero lag.  AWI's misfit penalises the *temporal spread* of
:math:`w`:

.. math::

    \\mathcal J_{\\mathrm{AWI}}(m) \\;=\\; \\frac{1}{2}\\sum_{\\text{trace}}
        \\frac{\\sum_{\\tau} (T(\\tau)\\,w(\\tau))^2}{\\sum_{\\tau} w(\\tau)^2},

where :math:`T(\\tau)` is a *lag penalty* (in samples or seconds), the
canonical choice being :math:`T(\\tau)=\\tau` so that a delta at zero lag
gives 0 misfit.

Implementation
--------------
* We solve the Wiener-filter least-squares problem **in the frequency
  domain** with Tikhonov regularisation, exactly as Warner & Guasch
  (2016, eq. 12):

  .. math::

      W(\\omega) \\;=\\; \\frac{\\overline{D_s(\\omega)}\\,D_o(\\omega)}
                              {|D_s(\\omega)|^2 + \\epsilon \\overline{P}(\\omega)}.

  Here :math:`\\epsilon \\overline P` is the Tikhonov stabiliser; we pick
  :math:`\\overline P` to be the average :math:`|D_s|^2` over time samples,
  so :math:`\\epsilon` is dimensionless.
* The Wiener filter is then ``irfft``'ed back to the time domain, the
  temporal penalty is applied with ``T(tau) = (tau * dt)`` in *seconds*
  (centred around zero so positive and negative lags are equally
  penalised), and the per-trace misfit is reduced as above.

Both the Wiener-filter step and the temporal penalty are fully
differentiable, so the misfit can be back-propagated through to the
model parameters.

References
----------
* Warner, M. & Guasch, L. (2014). *Adaptive waveform inversion: theory.*
  SEG Tech. Progr. Expanded Abstracts, pp. 1089-1093.
  doi:10.1190/segam2014-0371.1
* Warner, M. & Guasch, L. (2016). *Adaptive waveform inversion: theory.*
  **Geophysics** 81 (6), R429-R445.  doi:10.1190/geo2015-0387.1
* Guasch, L., Warner, M. & Ravaut, C. (2019). *Adaptive waveform
  inversion: practice.*  **Geophysics** 84 (3), R447-R461.
  doi:10.1190/geo2018-0377.1
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical


class AWILoss(BaseFWILoss):
    """Adaptive Waveform Inversion misfit (Warner & Guasch 2016).

    Parameters
    ----------
    dt
        Sampling interval in seconds.  Used to convert the lag index into
        the temporal penalty ``T(tau) = tau * dt``.
    epsilon
        Dimensionless Tikhonov regularisation for the Wiener filter,
        relative to the trace energy.  Default 1e-4.
    """

    def __init__(
        self,
        dt: float = 1.0,
        epsilon: float = 1e-4,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        self.dt = float(dt)
        self.epsilon = float(epsilon)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        s, _canon = flatten_traces(syn_c)        # (N, nt)
        o, _ = flatten_traces(obs_c)
        N, nt = s.shape

        # Pad to 2 nt to avoid wrap-around in the Wiener-filter convolution.
        n_pad = 2 * nt
        S = torch.fft.rfft(s, n=n_pad)
        O = torch.fft.rfft(o, n=n_pad)
        Sa = S.abs() ** 2
        # Tikhonov stabiliser scaled by the *peak* spectral amplitude (a la
        # Warner-Guasch 2016): out-of-band bins get an absolute floor that
        # prevents the 0/0 noise that would otherwise leak into the
        # time-domain Wiener filter.
        eps_floor = self.epsilon * Sa.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        W = torch.conj(S) * O / (Sa + eps_floor)
        w = torch.fft.irfft(W, n=n_pad)          # (N, n_pad), zero-lag at index 0

        # Re-centre so zero lag is in the middle, and use *all* n_pad lags.
        w_full = torch.fft.fftshift(w, dim=-1)
        lag_idx = torch.arange(n_pad, device=s.device, dtype=s.dtype) - n_pad // 2
        T = lag_idx * self.dt                    # (n_pad,)
        T = T.view(1, n_pad)

        num = ((T * w_full) ** 2).sum(dim=-1)
        den = (w_full ** 2).sum(dim=-1).clamp_min(1e-30)
        per_trace = 0.5 * num / den              # (N,)

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def awi_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    dt: float = 1.0,
    epsilon: float = 1e-4,
    reduction: str = "mean",
) -> torch.Tensor:
    return AWILoss(dt=dt, epsilon=epsilon, reduction=reduction)(syn, obs)


__all__ = ["AWILoss", "awi_loss"]
