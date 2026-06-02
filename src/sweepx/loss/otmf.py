"""Optimal Transport of the Matching Filter (OTMF) misfit.

Sun & Alkhalifah (2019) combine the matching-filter / adaptive-
waveform-inversion idea with optimal transport.  For each trace:

1. Compute the **Wiener matching filter** :math:`w(\\tau)` so that
   :math:`d_{\\mathrm s}\\!\\ast\\!w \\approx d_{\\mathrm o}` (regularised Wiener
   filter, same as in :class:`sweep_loss.AWILoss`).
2. Preprocess :math:`w` to a non-negative density :math:`\\hat w(\\tau)` (by
   squaring, taking absolute value, or shifting; see ``positive``).
3. Measure the Wasserstein distance between :math:`\\hat w` and the *Dirac
   delta at zero lag* :math:`\\delta_0(\\tau)`.  For a 1-D probability
   density on :math:`[-T, T]`, the squared Wasserstein-2 distance to
   :math:`\\delta_0` is the second moment of :math:`\\hat w`:

   .. math::

       W_2^2(\\hat w, \\delta_0) = \\int \\tau^2 \\,\\hat w(\\tau)\\,\\mathrm d\\tau,

   and the 1-Wasserstein distance is the **first absolute moment**:

   .. math::

       W_1(\\hat w, \\delta_0) = \\int |\\tau|\\,\\hat w(\\tau)\\,\\mathrm d\\tau.

   The OTMF misfit is one of these moments, summed over traces.

The resulting objective is convex in the time shift (Sun & Alkhalifah
2019, fig. 4) — unlike a direct L2 misfit, the matching-filter
formulation absorbs amplitude / waveform discrepancies, and the
Wasserstein distance to the delta gives a quadratic convex penalty in
the *kinematic* shift.

References
----------
* Sun, B. & Alkhalifah, T. (2019). *Adaptive traveltime inversion.*
  **Geophysics** 84 (4), U13-U29.  doi:10.1190/geo2018-0595.1
* Sun, B. & Alkhalifah, T. (2019). *The application of an optimal
  transport to a preconditioned data matching function for robust
  waveform inversion.* **Geophysics** 84 (6), R923-R945.
  doi:10.1190/geo2018-0413.1
* Sun, B. & Alkhalifah, T. (2019). *Stereo optimal transport of the
  matching filter.*  SEG Tech. Progr. Expanded Abstracts, pp. 1505-1509.
  doi:10.1190/segam2019-3199662.1
"""

from __future__ import annotations

import torch

from ._utils import normalize_density, positive_transform
from .base import BaseFWILoss, flatten_traces, to_canonical


class OTMFLoss(BaseFWILoss):
    """Optimal Transport of the Matching Filter misfit (Sun & Alkhalifah 2019).

    Parameters
    ----------
    dt
        Sampling interval in seconds.
    epsilon
        Tikhonov stabiliser for the Wiener filter, relative to
        :math:`\\max_\\omega |D_{\\mathrm s}|^2`.
    positive
        Positive transform mapping the Wiener filter :math:`w` to a
        non-negative density.  Default ``"square"``.
    order
        ``1`` for :math:`W_1(\\hat w, \\delta_0) = \\int |\\tau|\\hat w`,
        ``2`` for :math:`W_2^2(\\hat w, \\delta_0) = \\int \\tau^2\\hat w`.
    """

    def __init__(
        self,
        dt: float = 1.0,
        epsilon: float = 1e-4,
        positive: str = "square",
        c: "float | None" = None,
        order: int = 2,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        if positive not in ("square", "abs", "linear", "exp"):
            raise ValueError(f"unknown positive='{positive}'")
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        self.dt = float(dt)
        self.epsilon = float(epsilon)
        self.positive = positive
        self.c = float(c) if c is not None else None
        self.order = int(order)

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

        # ---- Wiener matching filter (same recipe as AWILoss) -------------
        n_pad = 2 * nt
        S = torch.fft.rfft(s, n=n_pad)
        O = torch.fft.rfft(o, n=n_pad)
        Sa = S.abs() ** 2
        eps_floor = self.epsilon * Sa.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        W = torch.conj(S) * O / (Sa + eps_floor)
        w = torch.fft.irfft(W, n=n_pad)
        w = torch.fft.fftshift(w, dim=-1)                     # zero lag at centre

        # ---- preprocess to a non-negative density on lag axis ------------
        c_param = self.c
        if self.positive == "linear" and c_param is None:
            c_param = float(w.abs().max().item()) + 1e-6
        wp = positive_transform(w, method=self.positive, c=c_param)
        wp = normalize_density(wp, dim=-1)

        # ---- moment-of-density wrt delta_0 -------------------------------
        lag = (torch.arange(n_pad, device=s.device, dtype=s.dtype) - n_pad // 2) * self.dt
        if self.order == 2:
            per_trace = (lag * lag * wp).sum(dim=-1)
        else:
            per_trace = (lag.abs() * wp).sum(dim=-1)

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def otmf_loss(
    syn: torch.Tensor,
    obs: torch.Tensor,
    dt: float = 1.0,
    epsilon: float = 1e-4,
    positive: str = "square",
    c: "float | None" = None,
    order: int = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    return OTMFLoss(
        dt=dt, epsilon=epsilon, positive=positive, c=c, order=order, reduction=reduction
    )(syn, obs)


__all__ = ["OTMFLoss", "otmf_loss"]
