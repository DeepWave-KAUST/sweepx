"""Frequency- and Laplace-domain FWI misfits.

We FFT the data along the time axis and operate on the complex spectrum
:math:`D(\\omega)=\\sum_t d(t) e^{-i\\omega t}`.  Four classical
frequency-domain misfits are implemented in this module:

1. **Frequency-domain L2** (Pratt, Shin & Hicks 1998):

   .. math::

       \\mathcal J_{\\Omega}(m) \\;=\\; \\tfrac{1}{2}\\!\\sum_{\\omega\\in\\Omega}
            \\bigl|D_{\\mathrm s}(\\omega) - D_{\\mathrm o}(\\omega)\\bigr|^2.

   :math:`\\Omega` is an optional band of frequency *indices* used to do
   multi-scale / band-limited FWI.

2. **Phase-only frequency-domain** (Bednar, Shin & Pyun 2007):

   .. math::

       \\mathcal J_{\\phi}(m) \\;=\\; \\tfrac{1}{2}\\!\\sum_{\\omega\\in\\Omega}
            \\bigl|\\Delta\\Phi(\\omega)\\bigr|^2,

   :math:`\\Delta\\Phi = \\operatorname{wrap}(\\Phi_{\\mathrm s}-\\Phi_{\\mathrm o})`,
   :math:`\\Phi(\\omega) = \\arg D(\\omega)`.

3. **Amplitude-only frequency-domain** (Shin & Min 2006):

   .. math::

       \\mathcal J_A(m) \\;=\\; \\tfrac{1}{2}\\!\\sum_{\\omega\\in\\Omega}
            \\bigl(|D_{\\mathrm s}(\\omega)| - |D_{\\mathrm o}(\\omega)|\\bigr)^2.

4. **Logarithmic Shin-Min misfit** (Shin & Min 2006, eq. 13):

   .. math::

       \\mathcal J_{\\log}(m) \\;=\\; \\tfrac{1}{2}\\!\\sum_{\\omega\\in\\Omega}
            \\bigl|\\,\\log D_{\\mathrm s}(\\omega) - \\log D_{\\mathrm o}(\\omega)\\,\\bigr|^2.

   With :math:`\\log D = \\log|D| + i\\,\\Phi`, the real part is the log-amplitude
   misfit and the imaginary part is the (unwrapped) phase misfit — Shin &
   Min showed that this single complex objective conveniently decouples
   the two contributions.

We also expose a **Laplace-domain L2** (Shin & Cha 2008): the data are first
multiplied by :math:`e^{-st}` (real damping with rate :math:`s>0`) and then
fed to the time-domain L2.  The Laplace-Fourier hybrid (Shin & Cha 2009)
is the same with complex :math:`s = \\sigma + i\\omega`.

References
----------
* Pratt, R. G., Shin, C. & Hicks, G. J. (1998). *Gauss-Newton and full
  Newton methods in frequency-space seismic waveform inversion.*
  Geophys. J. Int. 133 (2), 341-362.
  doi:10.1046/j.1365-246X.1998.00498.x
* Shin, C. & Min, D.-J. (2006). *Waveform inversion using a logarithmic
  wavefield.* **Geophysics** 71 (3), R31-R42.  doi:10.1190/1.2194523
* Bednar, J. B., Shin, C. & Pyun, S. (2007). *Comparison of waveform
  inversion, part 2: phase approach.*  Geophys. Prospect. 55 (4),
  465-475.  doi:10.1111/j.1365-2478.2007.00618.x
* Shin, C. & Cha, Y. H. (2008). *Waveform inversion in the Laplace
  domain.*  Geophys. J. Int. 173 (3), 922-931.
  doi:10.1111/j.1365-246X.2008.03768.x
* Shin, C. & Cha, Y. H. (2009). *Waveform inversion in the
  Laplace-Fourier domain.*  Geophys. J. Int. 177 (3), 1067-1079.
  doi:10.1111/j.1365-246X.2009.04102.x
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .base import BaseFWILoss, to_canonical


def _band_mask(nf: int, freq_band: Optional[Sequence[int]] = None,
               device=None) -> torch.Tensor:
    """Return a (nf,) bool mask selecting the given inclusive index range.

    ``None`` means "all frequencies".  ``(k0, k1)`` selects indices in
    ``[k0, k1]`` inclusive.
    """
    if freq_band is None:
        return torch.ones(nf, dtype=torch.bool, device=device)
    k0, k1 = freq_band
    if k0 < 0 or k1 < 0 or k0 > k1 or k1 >= nf:
        raise ValueError(f"invalid freq_band {freq_band} for nf={nf}")
    m = torch.zeros(nf, dtype=torch.bool, device=device)
    m[k0 : k1 + 1] = True
    return m


def _rfft_time(x: torch.Tensor) -> torch.Tensor:
    """rFFT along the time axis (-3) of a canonical tensor."""
    return torch.fft.rfft(x, dim=-3)


class FrequencyDomainL2Loss(BaseFWILoss):
    """Frequency-domain L2 misfit (Pratt-Shin-Hicks 1998)."""

    def __init__(
        self,
        freq_band: Optional[Sequence[int]] = None,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.freq_band = tuple(freq_band) if freq_band is not None else None

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        Ds = _rfft_time(syn_c)
        Do = _rfft_time(obs_c)
        nf = Ds.shape[-3]
        m = _band_mask(nf, self.freq_band, device=Ds.device)

        # Broadcast mask to (1, nf, 1, 1)
        shape = [1] * Ds.ndim
        shape[-3] = nf
        m = m.view(shape)

        r = (Ds - Do) * m
        pw = 0.5 * (r.real * r.real + r.imag * r.imag)
        if self.reduction == "mean":
            return pw.mean()
        if self.reduction == "sum":
            return pw.sum()
        return pw


class FrequencyPhaseLoss(BaseFWILoss):
    """Phase-only frequency-domain misfit (Bednar-Shin-Pyun 2007)."""

    def __init__(
        self,
        freq_band: Optional[Sequence[int]] = None,
        amplitude_weight: bool = True,
        eps: float = 1e-6,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.freq_band = tuple(freq_band) if freq_band is not None else None
        self.amplitude_weight = bool(amplitude_weight)
        self.eps = float(eps)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        Ds = _rfft_time(syn_c)
        Do = _rfft_time(obs_c)
        nf = Ds.shape[-3]
        m = _band_mask(nf, self.freq_band, device=Ds.device)
        shape = [1] * Ds.ndim
        shape[-3] = nf
        m = m.view(shape)

        Phi_s = torch.atan2(Ds.imag, Ds.real)
        Phi_o = torch.atan2(Do.imag, Do.real)
        dphi = torch.atan2(torch.sin(Phi_s - Phi_o), torch.cos(Phi_s - Phi_o))
        if self.amplitude_weight:
            Ao = torch.sqrt(Do.real ** 2 + Do.imag ** 2 + self.eps)
            wmax = Ao.amax(dim=-3, keepdim=True).clamp_min(self.eps)
            dphi = dphi * (Ao / wmax)
        pw = 0.5 * dphi * dphi * m.float()
        if self.reduction == "mean":
            return pw.mean()
        if self.reduction == "sum":
            return pw.sum()
        return pw


class FrequencyAmplitudeLoss(BaseFWILoss):
    """Amplitude-only frequency-domain misfit (Shin-Min 2006)."""

    def __init__(
        self,
        freq_band: Optional[Sequence[int]] = None,
        eps: float = 1e-12,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.freq_band = tuple(freq_band) if freq_band is not None else None
        self.eps = float(eps)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        Ds = _rfft_time(syn_c)
        Do = _rfft_time(obs_c)
        nf = Ds.shape[-3]
        m = _band_mask(nf, self.freq_band, device=Ds.device)
        shape = [1] * Ds.ndim
        shape[-3] = nf
        m = m.view(shape)

        As = torch.sqrt(Ds.real ** 2 + Ds.imag ** 2 + self.eps)
        Ao = torch.sqrt(Do.real ** 2 + Do.imag ** 2 + self.eps)
        d = (As - Ao) * m.float()
        pw = 0.5 * d * d
        if self.reduction == "mean":
            return pw.mean()
        if self.reduction == "sum":
            return pw.sum()
        return pw


class LogarithmicShinMinLoss(BaseFWILoss):
    """Shin-Min (2006) logarithmic wavefield misfit.

    :math:`\\tfrac{1}{2}\\sum_\\omega |\\log D_s(\\omega) - \\log D_o(\\omega)|^2`.

    Implemented via the *principal* complex logarithm
    :math:`\\log z = \\log|z| + i\\,\\arg z`, with the phase-difference part
    wrapped to :math:`(-\\pi, \\pi]` to avoid the well-known phase-unwrapping
    pathologies that the unmodified log can introduce.  This wrapped variant
    is sometimes called the "log-spectrum" misfit (Shin-Min 2006, eq. 13;
    Choi & Alkhalifah 2013).
    """

    def __init__(
        self,
        freq_band: Optional[Sequence[int]] = None,
        eps: float = 1e-8,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        self.freq_band = tuple(freq_band) if freq_band is not None else None
        self.eps = float(eps)

    def forward(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        syn_c, _ = to_canonical(syn)
        obs_c, _ = to_canonical(obs)
        Ds = _rfft_time(syn_c)
        Do = _rfft_time(obs_c)
        nf = Ds.shape[-3]
        m = _band_mask(nf, self.freq_band, device=Ds.device)
        shape = [1] * Ds.ndim
        shape[-3] = nf
        m = m.view(shape).float()

        As = torch.sqrt(Ds.real ** 2 + Ds.imag ** 2 + self.eps)
        Ao = torch.sqrt(Do.real ** 2 + Do.imag ** 2 + self.eps)
        log_amp = torch.log(As) - torch.log(Ao)
        Phi_s = torch.atan2(Ds.imag, Ds.real)
        Phi_o = torch.atan2(Do.imag, Do.real)
        dphi = torch.atan2(torch.sin(Phi_s - Phi_o), torch.cos(Phi_s - Phi_o))
        # |log Ds - log Do|^2 = log_amp^2 + dphi^2
        pw = 0.5 * (log_amp * log_amp + dphi * dphi) * m
        if self.reduction == "mean":
            return pw.mean()
        if self.reduction == "sum":
            return pw.sum()
        return pw


class LaplaceL2Loss(BaseFWILoss):
    """Time-domain damped L2 misfit (Laplace-domain flavour, Shin-Cha 2008).

    The data are damped by :math:`e^{-s\\,t}` (with ``s > 0`` the damping
    rate) and then compared in L2.  Equivalent to the time-domain L2 of
    :math:`\\hat d(t) = e^{-s t} d(t)`.

    .. note::

       This is the **time-domain damped-L2** practical flavour
       :math:`\\tfrac12\\sum_t (e^{-st}(d_s-d_o))^2`, *not* the exact
       frequency-domain logarithmic Laplace misfit of Shin & Cha (2008,
       eqs. 4-11), which uses :math:`\\log D(s)` of the complex-frequency
       Helmholtz solution (i.e. :math:`|\\sum_t e^{-st} d|^2`, summing before
       the square).  The two are not equivalent; this damped-L2 variant is
       the one commonly used in time-domain FWI codes.

    Parameters
    ----------
    s
        Real damping rate (units: 1/time).  Larger ``s`` emphasises
        early arrivals.
    dt
        Sampling interval.
    """

    def __init__(
        self,
        s: float = 1.0,
        dt: float = 1.0,
        reduction: str = "mean",
        mask: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(reduction=reduction, mask=mask)
        if s <= 0:
            raise ValueError(f"s must be > 0, got {s}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.s = float(s)
        self.dt = float(dt)

    def _pointwise(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        nt = syn.shape[-3]
        t = torch.arange(nt, device=syn.device, dtype=syn.dtype) * self.dt
        damp_shape = [1] * syn.ndim
        damp_shape[-3] = nt
        damp = torch.exp(-self.s * t).view(damp_shape)
        r = damp * (syn - obs)
        return 0.5 * r * r


def frequency_domain_l2_loss(syn, obs, **kw):
    return FrequencyDomainL2Loss(**kw)(syn, obs)


def frequency_phase_loss(syn, obs, **kw):
    return FrequencyPhaseLoss(**kw)(syn, obs)


def frequency_amplitude_loss(syn, obs, **kw):
    return FrequencyAmplitudeLoss(**kw)(syn, obs)


def shin_min_log_loss(syn, obs, **kw):
    return LogarithmicShinMinLoss(**kw)(syn, obs)


def laplace_l2_loss(syn, obs, **kw):
    return LaplaceL2Loss(**kw)(syn, obs)


__all__ = [
    "FrequencyAmplitudeLoss",
    "FrequencyDomainL2Loss",
    "FrequencyPhaseLoss",
    "LaplaceL2Loss",
    "LogarithmicShinMinLoss",
    "frequency_amplitude_loss",
    "frequency_domain_l2_loss",
    "frequency_phase_loss",
    "laplace_l2_loss",
    "shin_min_log_loss",
]
