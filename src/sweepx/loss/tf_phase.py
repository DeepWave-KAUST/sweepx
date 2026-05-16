"""Time-Frequency phase / envelope misfit (Fichtner 2008; Kristeková 2009).

Following Kristeková et al. (2009) and Fichtner et al. (2008) we transform
each trace into the time-frequency plane with a short-time Fourier
transform (STFT) using a Gaussian (Gabor) window, and then compare the
time-frequency representations sample-by-sample.

Let :math:`G_s(t, \\omega), G_o(t, \\omega)` be the STFT of the synthetic
and observed traces.  Decomposing
:math:`G(t,\\omega) = |G(t,\\omega)|\\,e^{i\\phi(t,\\omega)}` we form two
misfits:

.. math::

    \\mathcal J_{\\mathrm{TF-env}}(m) &=
        \\tfrac12 \\sum_{t,\\omega}\\bigl(|G_s| - |G_o|\\bigr)^2,

    \\mathcal J_{\\mathrm{TF-phi}}(m) &=
        \\tfrac12 \\sum_{t,\\omega}\\bigl[w(t,\\omega)\\,
            \\Delta\\phi(t,\\omega)\\bigr]^2,

with the **wrapped** phase difference
:math:`\\Delta\\phi = \\arctan2(\\sin(\\phi_s-\\phi_o), \\cos(\\phi_s-\\phi_o))`
and an **amplitude weight**
:math:`w(t,\\omega) = |G_o(t,\\omega)|/\\max|G_o|` so that bins with
negligible energy do not contribute spurious phase residuals.

The class :class:`TimeFrequencyPhaseLoss` exposes both terms via the
``alpha`` mixing coefficient, mirroring our :class:`EnvelopePhaseLoss`:

.. math::

    \\mathcal J_{\\mathrm{TF}}(m) = (1-\\alpha)\\mathcal J_{\\mathrm{TF-env}}
        + \\alpha\\,\\mathcal J_{\\mathrm{TF-phi}}.

References
----------
* Fichtner, A., Kennett, B. L. N., Igel, H. & Bunge, H.-P. (2008).
  *Theoretical background for continental- and global-scale full-waveform
  inversion in the time-frequency domain.*  Geophys. J. Int. 175 (2),
  665-685.  doi:10.1111/j.1365-246X.2008.03923.x
* Kristeková, M., Kristek, J. & Moczo, P. (2009). *Time-frequency misfit
  and goodness-of-fit criteria for quantitative comparison of time
  signals.*  Geophys. J. Int. 178 (2), 813-825.
  doi:10.1111/j.1365-246X.2009.04177.x
* Kristeková, M., Kristek, J., Moczo, P. & Day, S. M. (2006). *Misfit
  criteria for quantitative comparison of seismograms.*  Bull. Seismol.
  Soc. Am. 96 (5), 1836-1850.  doi:10.1785/0120060012
"""

from __future__ import annotations

import math

import torch

from .base import BaseFWILoss, flatten_traces, to_canonical


def _gabor_stft(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    sigma_samples: float,
) -> torch.Tensor:
    """Short-time Fourier transform with a Gaussian (Gabor) window.

    Parameters
    ----------
    x
        ``(N, nt)`` real input.
    n_fft
        FFT length per window.
    hop_length
        Window step in samples.
    sigma_samples
        Standard deviation of the Gaussian window in samples.

    Returns
    -------
    ``(N, n_fft//2 + 1, n_frames)`` complex STFT.
    """
    # Build Gaussian window
    win = torch.exp(
        -0.5
        * (torch.arange(n_fft, device=x.device, dtype=x.dtype) - n_fft // 2) ** 2
        / sigma_samples ** 2
    )
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        center=True,
        return_complex=True,
        normalized=False,
        pad_mode="reflect",
    )


class TimeFrequencyPhaseLoss(BaseFWILoss):
    """Time-Frequency phase / envelope misfit (Fichtner-Kristeková family).

    Parameters
    ----------
    alpha
        Weighting between TF envelope (``alpha=0``) and TF phase
        (``alpha=1``).  Default ``0.5``.
    n_fft
        FFT length per Gabor window.  Default ``128``.
    hop_length
        Window step in samples.  ``None`` (default) ⇒ ``n_fft // 4``.
    sigma_samples
        Gaussian-window standard deviation in samples.  Default
        ``n_fft / 6`` so the window is well-contained.
    eps
        Stabiliser for division by the envelope.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        n_fft: int = 128,
        hop_length: "int | None" = None,
        sigma_samples: "float | None" = None,
        eps: float = 1e-12,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if n_fft < 8:
            raise ValueError(f"n_fft must be >= 8, got {n_fft}")
        self.alpha = float(alpha)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else n_fft // 4
        self.sigma_samples = (
            float(sigma_samples) if sigma_samples is not None else n_fft / 6.0
        )
        self.eps = float(eps)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        if syn_c.shape != obs_c.shape:
            raise ValueError(
                f"shape mismatch: syn={tuple(syn.shape)} obs={tuple(obs.shape)}"
            )
        s, _ = flatten_traces(syn_c)
        o, _ = flatten_traces(obs_c)
        nt = s.shape[-1]
        if nt < self.n_fft:
            raise ValueError(
                f"trace length {nt} < n_fft {self.n_fft}; reduce n_fft or "
                f"resample input"
            )

        Gs = _gabor_stft(s, self.n_fft, self.hop_length, self.sigma_samples)
        Go = _gabor_stft(o, self.n_fft, self.hop_length, self.sigma_samples)

        As = torch.sqrt(Gs.real ** 2 + Gs.imag ** 2 + self.eps)
        Ao = torch.sqrt(Go.real ** 2 + Go.imag ** 2 + self.eps)

        # Envelope term
        d_env = As - Ao
        env = 0.5 * d_env * d_env

        # Phase term: compare unit-modulus complex signals (exponentiated-phase
        # form).  This is the same Kristekova / Fichtner residual as
        # (delta phi)^2 to leading order but with a gradient that is well-
        # defined at the zero-amplitude bins.  We multiply by |G_o|/max|G_o|
        # so noise-floor bins don't dominate.
        cs_real = Gs.real / As
        cs_imag = Gs.imag / As
        co_real = Go.real / Ao
        co_imag = Go.imag / Ao
        dR = cs_real - co_real
        dI = cs_imag - co_imag
        d_unit_sq = dR * dR + dI * dI
        wmax = Ao.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        w = Ao / wmax
        phi = 0.5 * (w * w) * d_unit_sq

        per_sample = (1.0 - self.alpha) * env + self.alpha * phi

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample


def time_frequency_phase_loss(
    syn,
    obs,
    alpha: float = 0.5,
    n_fft: int = 128,
    hop_length: "int | None" = None,
    sigma_samples: "float | None" = None,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    return TimeFrequencyPhaseLoss(
        alpha=alpha,
        n_fft=n_fft,
        hop_length=hop_length,
        sigma_samples=sigma_samples,
        eps=eps,
        reduction=reduction,
    )(syn, obs)


__all__ = ["TimeFrequencyPhaseLoss", "time_frequency_phase_loss"]
