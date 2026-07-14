"""Smoke tests for sweep_tasks.preproc (filter + resample, absorbed from the
retired sweep-preproc)."""

import numpy as np
import pytest

import sweep_tasks.preproc
from sweep_tasks.preproc.filter import bandpass, lowpass
from sweep_tasks.preproc.resample import resample_time


def _sine(freq_hz, dt, nt):
    t = np.arange(nt) * dt
    return np.sin(2 * np.pi * freq_hz * t).astype("float32")


def test_preproc_package_exposes_submodules():
    assert set(sweep_tasks.preproc.__all__) == {"filter", "resample"}


def test_bandpass_attenuates_out_of_band():
    dt, nt = 0.001, 2000
    sig_in = _sine(2.0, dt, nt)          # 2 Hz — should be stopped
    sig_pass = _sine(20.0, dt, nt)       # 20 Hz — should pass
    mixed = sig_in + sig_pass
    out = bandpass(mixed, lo=10.0, hi=30.0, dt=dt)
    # 20 Hz content should be largely preserved, 2 Hz strongly attenuated
    err_pass = np.linalg.norm(out - sig_pass) / np.linalg.norm(sig_pass)
    err_in = np.linalg.norm(out - sig_in) / np.linalg.norm(sig_in)
    assert err_pass < 0.1, f"passband not preserved: err={err_pass}"
    assert err_in > 0.8, f"stopband not attenuated: err={err_in}"


def test_lowpass_preserves_dc():
    dt, nt = 0.001, 1000
    sig = np.ones(nt, dtype="float32") + _sine(50.0, dt, nt)
    out = lowpass(sig, fc=5.0, dt=dt)
    assert np.abs(out.mean() - 1.0) < 1e-2


def test_bandpass_invalid_band():
    with pytest.raises(ValueError):
        bandpass(np.zeros(100), lo=20.0, hi=10.0, dt=0.001)


def test_resample_halves_rate():
    dt_in, dt_out = 0.001, 0.002
    sig = _sine(5.0, dt_in, 4000)
    out = resample_time(sig, dt_in, dt_out, axis=0)
    assert out.shape[0] == 2000
