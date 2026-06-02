"""Basic wavelet utilities used by dataset workflows.

Ported verbatim from ``fwi_workflow.wavelet.estimation`` — pure numpy,
no internal deps.
"""

from __future__ import annotations

import numpy as np


def normalize_wavelet(wavelet: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a wavelet by maximum absolute amplitude."""

    values = np.asarray(wavelet, dtype=float)
    scale = np.max(np.abs(values))
    if scale < eps:
        raise ValueError("Cannot normalize a near-zero wavelet")
    return values / scale


def ricker(frequency_hz: float, dt_s: float, nt: int, peak_time_s: float | None = None) -> np.ndarray:
    """Create a Ricker wavelet for tests and initial experiments."""

    if peak_time_s is None:
        peak_time_s = 1.0 / frequency_hz
    t = np.arange(nt, dtype=float) * dt_s
    arg = np.pi * frequency_hz * (t - peak_time_s)
    return (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))


def butterworth_bandpass_response(
    frequencies_hz: np.ndarray,
    lowcut_hz: float | None,
    highcut_hz: float | None,
    order: int = 4,
) -> np.ndarray:
    """Return a zero-phase Butterworth bandpass amplitude response.

    The response is intended for FFT-domain filtering. It avoids scipy so
    the workflow can run in minimal HPC Python environments.
    """

    freqs = np.asarray(frequencies_hz, dtype=float)
    response = np.ones(freqs.shape, dtype=float)
    filt_order = int(order)
    if filt_order <= 0:
        raise ValueError("Butterworth order must be positive")

    if lowcut_hz is not None and float(lowcut_hz) > 0.0:
        low = float(lowcut_hz)
        highpass = np.zeros_like(response)
        nonzero = freqs > 0.0
        highpass[nonzero] = 1.0 / np.sqrt(1.0 + (low / freqs[nonzero]) ** (2 * filt_order))
        response *= highpass

    if highcut_hz is not None and float(highcut_hz) > 0.0:
        high = float(highcut_hz)
        response *= 1.0 / np.sqrt(1.0 + (freqs / high) ** (2 * filt_order))

    return response


def apply_fft_filter(
    data: np.ndarray,
    dt_s: float,
    lowcut_hz: float | None = None,
    highcut_hz: float | None = None,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter along one array axis.

    The FFT is internally padded to the next power of two so we never hit
    numpy's slow non-power-of-two path (prime nt would otherwise drop FFT
    cost from O(N log N) to O(N²)).
    """

    values = np.asarray(data, dtype=np.float32)
    nt = int(values.shape[axis])
    nfft = 1 << max(int(nt - 1).bit_length(), 1) if nt > 1 else 1
    freqs = np.fft.rfftfreq(nfft, d=float(dt_s))
    response = butterworth_bandpass_response(freqs, lowcut_hz, highcut_hz, order=order)
    shape = [1] * values.ndim
    shape[axis] = response.size
    spectrum = np.fft.rfft(values, n=nfft, axis=axis)
    filtered_full = np.fft.irfft(spectrum * response.reshape(shape), n=nfft, axis=axis)
    sl = [slice(None)] * values.ndim
    sl[axis] = slice(0, nt)
    return filtered_full[tuple(sl)].astype(np.float32, copy=False)
