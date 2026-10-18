"""Deconvolution-based FWI misfit (Luo & Sava 2011).

This functional is closely related to AWI (Warner & Guasch 2016) but
penalises a different quantity:  the deconvolution :math:`d_o / d_s`
should ideally be a unit impulse, so we measure how far it deviates from
that with a *penalty function* :math:`P(\\tau)`.

For each trace,

.. math::

    \\Psi(\\tau) \\;=\\; \\mathcal F^{-1}\\!\\Bigl(\\frac{D_o(\\omega)}{D_s(\\omega) + \\epsilon}\\Bigr),

    \\mathcal J_{\\mathrm{Decon}}(m) \\;=\\;
        \\tfrac{1}{2}\\sum_{\\text{trace}}\\sum_{\\tau} P(\\tau)\\,\\Psi(\\tau)^2,

with :math:`P(\\tau)=\\tau^2` (the original Luo-Sava choice).  Because
``Psi`` is *not* normalised, the misfit depends on absolute amplitude;
this is intentional in Luo-Sava (it carries amplitude information that
AWI's normalised version discards).

For numerical stability we regularise the deconvolution exactly as in
AWI (see :mod:`sweep_loss.awi`).  An ``"awi-style"`` ``normalize=True`` mode
divides the per-trace numerator by :math:`\\sum_\\tau \\Psi(\\tau)^2`
giving the amplitude-invariant version proposed by Choi & Alkhalifah
(2018).

References
----------
* Luo, S. & Sava, P. (2011). *A deconvolution-based objective function for
  wave-equation inversion.* SEG Tech. Progr. Expanded Abstracts,
  pp. 2788-2792.  doi:10.1190/1.3627773
* Choi, Y. & Alkhalifah, T. (2018). *Time-domain full-waveform inversion
  of exponentially damped wavefield using the deconvolution-based
  objective function.* **Geophysics** 83 (2), R77-R88.
  doi:10.1190/geo2017-0057.1
* Zhu, H. & Fomel, S. (2016). *Building good starting models for FWI
  using adaptive matching filtering misfit.* **Geophysics** 81 (5),
  U61-U72.  doi:10.1190/geo2015-0596.1
"""

from __future__ import annotations

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical


class DeconvolutionLoss(BaseFWILoss):
    """Luo-Sava (2011) deconvolution-based misfit.

    Parameters
    ----------
    dt
        Sampling interval in seconds.
    epsilon
        Tikhonov stabiliser for the spectral division, relative to the
        *peak* :math:`|D_s|^2`.
    normalize
        If True (default False), divide the penalty by
        :math:`\\sum_\\tau \\Psi^2` so the loss is amplitude-invariant
        (Choi-Alkhalifah 2018 variant).
    """

    def __init__(
        self,
        dt: float = 1.0,
        epsilon: float = 1e-4,
        normalize: bool = False,
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
        self.normalize = bool(normalize)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        s, _canon = flatten_traces(syn_c)
        o, _ = flatten_traces(obs_c)
        N, nt = s.shape

        n_pad = 2 * nt
        S = torch.fft.rfft(s, n=n_pad)
        O = torch.fft.rfft(o, n=n_pad)
        Sa = S.abs() ** 2
        eps_floor = self.epsilon * Sa.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        # Spectral division with Tikhonov stabiliser.
        Psi_omega = torch.conj(S) * O / (Sa + eps_floor)
        psi = torch.fft.irfft(Psi_omega, n=n_pad)
        psi = torch.fft.fftshift(psi, dim=-1)              # zero-lag in centre

        lag = (torch.arange(n_pad, device=s.device, dtype=s.dtype) - n_pad // 2) * self.dt
        P = lag * lag                                       # P(tau) = tau^2
        num = (P * psi * psi).sum(dim=-1)
        if self.normalize:
            den = (psi * psi).sum(dim=-1).clamp_min(1e-30)
            per_trace = 0.5 * num / den
        else:
            per_trace = 0.5 * num

        if self.reduction == "mean":
            return per_trace.mean()
        if self.reduction == "sum":
            return per_trace.sum()
        return per_trace


def deconvolution_loss(
    syn, obs, dt=1.0, epsilon=1e-4, normalize=False, reduction="mean"
) -> torch.Tensor:
    return DeconvolutionLoss(
        dt=dt, epsilon=epsilon, normalize=normalize, reduction=reduction
    )(syn, obs)


__all__ = ["DeconvolutionLoss", "deconvolution_loss"]
