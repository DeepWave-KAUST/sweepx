"""Tests for :func:`sweep_tasks.preproc.filter.bandpass_torch`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep_tasks.preproc.filter import bandpass, bandpass_torch


def _ricker(nt: int, dt: float, fm: float, delay: float) -> np.ndarray:
    t = np.arange(nt, dtype=np.float32) * dt
    arg = np.pi * fm * (t - delay)
    return ((1.0 - 2.0 * arg ** 2) * np.exp(-(arg ** 2))).astype(np.float32)


def test_bandpass_torch_matches_scipy_filtfilt_in_passband():
    """For a wavelet whose spectrum is well within the passband, the
    differentiable bandpass should agree with scipy's filtfilt within a
    small tolerance (the two use the same |H|² magnitude response)."""
    nt = 1024
    dt = 0.001
    x = _ricker(nt, dt, fm=15.0, delay=0.1)
    lo, hi = 5.0, 30.0
    y_torch = bandpass_torch(
        torch.as_tensor(x), lo=lo, hi=hi, dt=dt, order=4, axis=-1,
    ).numpy()
    y_scipy = bandpass(x, lo=lo, hi=hi, dt=dt, order=4, axis=-1)
    # FFT vs filtfilt differ near edges; compare RMS over the interior.
    interior = slice(50, nt - 50)
    rms = float(np.sqrt(np.mean((y_torch[interior] - y_scipy[interior]) ** 2)))
    peak = float(np.max(np.abs(y_scipy[interior])))
    assert rms < 0.02 * peak, f"rms={rms} peak={peak}"


def test_bandpass_torch_attenuates_out_of_band_tone():
    """A tone well below the passband should be heavily attenuated."""
    nt = 2048
    dt = 0.001
    t = np.arange(nt, dtype=np.float32) * dt
    tone = np.sin(2 * np.pi * 1.0 * t).astype(np.float32)  # 1 Hz
    y = bandpass_torch(
        torch.as_tensor(tone), lo=10.0, hi=40.0, dt=dt, order=4, axis=-1,
    ).numpy()
    # 1 Hz is ~10 octaves below 10 Hz; attenuation should be large.
    rms_in = float(np.sqrt(np.mean(tone ** 2)))
    rms_out = float(np.sqrt(np.mean(y ** 2)))
    assert rms_out < 0.05 * rms_in, f"in={rms_in} out={rms_out}"


def test_bandpass_torch_passes_in_band_tone():
    """A tone inside the passband should survive with near-unity gain."""
    nt = 2048
    dt = 0.001
    t = np.arange(nt, dtype=np.float32) * dt
    tone = np.sin(2 * np.pi * 20.0 * t).astype(np.float32)
    y = bandpass_torch(
        torch.as_tensor(tone), lo=10.0, hi=40.0, dt=dt, order=4, axis=-1,
    ).numpy()
    rms_in = float(np.sqrt(np.mean(tone ** 2)))
    rms_out = float(np.sqrt(np.mean(y[100:-100] ** 2)))
    assert 0.85 * rms_in < rms_out < 1.05 * rms_in


def test_bandpass_torch_axis_kwarg_filters_last_axis_by_default():
    """Multi-dim input: filter operates along the requested axis only."""
    nt = 512
    dt = 0.001
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5, nt)).astype(np.float32)
    y = bandpass_torch(
        torch.as_tensor(x), lo=5.0, hi=30.0, dt=dt, order=4, axis=-1,
    ).numpy()
    assert y.shape == x.shape


def test_bandpass_torch_is_differentiable():
    """gradient flows through bandpass_torch (autograd-compatible)."""
    nt = 256
    dt = 0.001
    x = torch.as_tensor(_ricker(nt, dt, fm=15.0, delay=0.05),
                        dtype=torch.float32).requires_grad_(True)
    y = bandpass_torch(x, lo=5.0, hi=30.0, dt=dt, order=4, axis=-1)
    loss = (y ** 2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_bandpass_torch_rejects_numpy():
    with pytest.raises(TypeError, match="torch.Tensor"):
        bandpass_torch(np.zeros(64, dtype=np.float32),
                       lo=5.0, hi=30.0, dt=0.001)
