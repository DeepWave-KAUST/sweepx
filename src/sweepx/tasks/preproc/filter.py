"""Zero-phase bandpass / lowpass / highpass filters.

Built on ``scipy.signal.butter`` (SOS form) + ``sosfiltfilt`` (zero-phase
forward+reverse). Accepts numpy arrays or torch tensors. For torch input
the operation is **not** differentiable (we drop to numpy for
``sosfiltfilt``); use that side of the pipeline before the forward
solver, not inside the gradient path.

Filter order semantics
----------------------
``order=N`` is the **prototype Butterworth order** passed to
``butter()``. Because ``sosfiltfilt`` applies the filter forward then
reverse, the effective MAGNITUDE response is ``|H_N(f)|²`` (i.e. the
stop-band roll-off is ``2N × 6 dB/oct`` instead of ``N × 6 dB/oct``),
but the prototype is still order N. This matches the convention in
``fwi_workflow-dev`` (which uses ``butter(N, ...)`` + ``filtfilt``;
``filter_order`` in that codebase is also the prototype order).

Padding
-------
The default ``padtype="odd"`` uses scipy's reflective padding (about
``3 * (2*order + 1)`` samples), which suppresses edge transients to
machine precision. Set ``padtype=None`` to match the ``torchaudio.functional.filtfilt``
behavior used inside ``fwi_workflow-dev``'s GPU path (no padding,
visible edge transients that decay across the trace). For numerical
equivalence with that path, pass ``padtype=None``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import _xp


def _design_sos(
    btype: str, dt: float, lo: float | None, hi: float | None, order: int
) -> np.ndarray:
    nyq = 0.5 / dt
    if btype == "low":
        if hi is None:
            raise ValueError("lowpass requires `hi` (cutoff frequency).")
        wn: Any = hi / nyq
    elif btype == "high":
        if lo is None:
            raise ValueError("highpass requires `lo` (cutoff frequency).")
        wn = lo / nyq
    elif btype == "band":
        if lo is None or hi is None:
            raise ValueError("bandpass requires both `lo` and `hi`.")
        if not 0 < lo < hi < nyq:
            raise ValueError(
                f"bandpass needs 0 < lo < hi < Nyquist={nyq}; got lo={lo}, hi={hi}"
            )
        wn = [lo / nyq, hi / nyq]
    else:
        raise ValueError(f"unknown btype {btype!r}")
    return butter(order, wn, btype=btype, output="sos")


def bandpass(
    x, lo: float, hi: float, dt: float, *,
    order: int = 4, axis: int = -1, padtype: str | None = "odd",
):
    """Zero-phase Butterworth bandpass (prototype order ``N`` → ``|H_N(f)|²``).

    Parameters
    ----------
    x
        Input array — numpy or torch tensor.
    lo, hi
        Passband edges in Hz.
    dt
        Time sample interval (s).
    order
        Prototype Butterworth order. Filtfilt application gives the
        equivalent of ``|H_N(f)|²`` (zero-phase, ``2N × 6 dB/oct`` stop-
        band roll-off). Matches ``filter_order`` in ``fwi_workflow-dev``.
    axis
        Axis along which to filter. Default ``-1``.
    padtype
        ``"odd"`` (default) uses scipy's reflective padding — edge
        transients suppressed to machine precision. ``None`` disables
        padding to match ``torchaudio.functional.filtfilt`` (the path
        used in ``fwi_workflow-dev``'s GPU FWI; expect visible edge
        transients).
    """
    sos = _design_sos("band", dt, lo, hi, order)
    y = sosfiltfilt(sos, _xp.to_numpy(x), axis=axis, padtype=padtype)
    return _xp.like(x, np.ascontiguousarray(y))


def lowpass(
    x, fc: float, dt: float, *,
    order: int = 4, axis: int = -1, padtype: str | None = "odd",
):
    """Zero-phase Butterworth lowpass at ``fc`` Hz. See :func:`bandpass`."""
    sos = _design_sos("low", dt, None, fc, order)
    y = sosfiltfilt(sos, _xp.to_numpy(x), axis=axis, padtype=padtype)
    return _xp.like(x, np.ascontiguousarray(y))


def highpass(
    x, fc: float, dt: float, *,
    order: int = 4, axis: int = -1, padtype: str | None = "odd",
):
    """Zero-phase Butterworth highpass at ``fc`` Hz. See :func:`bandpass`."""
    sos = _design_sos("high", dt, fc, None, order)
    y = sosfiltfilt(sos, _xp.to_numpy(x), axis=axis, padtype=padtype)
    return _xp.like(x, np.ascontiguousarray(y))


def _butter_gain_sq(nfft: int, dt_s: float, lo: float, hi: float, order: int) -> np.ndarray:
    """Sampled ``|H(f)|²`` of a Butterworth bandpass.

    Equivalent to ``scipy.signal.sosfreqz`` magnitude squared sampled at
    ``np.fft.rfftfreq(nfft, dt_s)`` — i.e. the frequency-domain analogue
    of a zero-phase forward+reverse filter. Returned as a ``float32``
    1-D array of length ``nfft // 2 + 1``.

    Mirrors the legacy ``_butter_filtfilt_gain_np`` in
    ``fwi_workflow-dev``'s ``crg_worker.py`` so a torch port matches the
    legacy GPU bandpass numerics to floating-point precision.
    """
    from scipy.signal import sosfreqz

    sos = _design_sos("band", dt_s, lo, hi, order)
    freqs = np.fft.rfftfreq(int(nfft), d=float(dt_s))
    # ``sosfreqz`` accepts ``worN`` as a frequency vector in rad/sample.
    worN = 2.0 * np.pi * freqs * float(dt_s)
    _, h = sosfreqz(sos, worN=worN, fs=2.0 * np.pi)
    gain_sq = np.abs(h).astype(np.float32) ** 2
    return gain_sq


def bandpass_torch(
    x, lo: float, hi: float, dt: float, *,
    order: int = 4, axis: int = -1,
):
    """Differentiable zero-phase Butterworth bandpass via FFT (GPU-friendly).

    The Butterworth ``|H(f)|²`` response is precomputed on the CPU (SciPy)
    then applied as a real-valued frequency-domain multiplication of the
    input's ``rfft``, followed by ``irfft``. Equivalent to the forward+
    reverse filter (``filtfilt``) magnitude response without paying the
    cost of two sequential time-domain passes — and crucially without
    leaving the input device, so the result remains autograd-compatible.

    Parameters
    ----------
    x
        Real torch tensor. Filter is applied along ``axis``.
    lo, hi
        Passband edges in Hz.
    dt
        Time sample interval (s).
    order
        Prototype Butterworth order — same convention as :func:`bandpass`,
        i.e. the effective stop-band roll-off is ``2N × 6 dB/oct`` because
        ``|H(f)|²`` is applied.
    axis
        Axis along which to filter. Default ``-1`` (last).

    Notes
    -----
    * Edge transients: no extra padding (matches ``torchaudio.functional.filtfilt``
      semantics in legacy ``fwi_workflow-dev``). If you need scipy's
      reflective-pad behavior, pre-pad the input by ``~3 * (2 * order + 1)``
      samples along the time axis before calling.
    * The gain table is built every call; for hot-path use cache it
      outside (``lo / hi / dt / nfft / order`` is a tiny key).
    """
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"bandpass_torch expects a torch.Tensor; got {type(x).__name__}"
        )
    if x.is_complex():
        raise TypeError("bandpass_torch expects a real-valued input")
    nfft = int(x.shape[axis])
    if nfft < 4:
        raise ValueError(f"bandpass_torch needs nfft >= 4; got {nfft}")
    gain_sq_np = _butter_gain_sq(nfft, float(dt), float(lo), float(hi), int(order))
    gain = torch.as_tensor(gain_sq_np, device=x.device, dtype=x.dtype)
    # Broadcast shape: 1s on every axis except ``axis`` (which gets the
    # rfft bin count nfft//2 + 1).
    bcast = [1] * x.ndim
    bcast[axis] = gain.numel()
    gain = gain.reshape(bcast)
    Xf = torch.fft.rfft(x, n=nfft, dim=axis)
    Yf = Xf * gain
    y = torch.fft.irfft(Yf, n=nfft, dim=axis)
    return y.to(dtype=x.dtype)


__all__ = ["bandpass", "lowpass", "highpass", "bandpass_torch"]
