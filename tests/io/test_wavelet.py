"""Unit tests for :func:`sweep_io.wavelet.load_wavelet_npz`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_io.wavelet import WaveletNPZ, load_wavelet_npz


def _save(tmp_path: Path, name: str, **arrays) -> Path:
    p = tmp_path / name
    np.savez(p, **arrays)
    return p


def test_direct_batch_format_wavelet_and_time(tmp_path):
    samples = np.linspace(-1, 1, 64, dtype=np.float32)
    time_s = np.arange(64, dtype=np.float64) * 0.002
    path = _save(tmp_path, "direct.npz", wavelet=samples, time_s=time_s)
    out = load_wavelet_npz(path)
    assert isinstance(out, WaveletNPZ)
    assert out.key == "wavelet"
    np.testing.assert_allclose(out.samples, samples)
    assert out.dt_s == pytest.approx(0.002)
    assert out.source_delay_s is None


def test_siren_pipeline_format_optimized_key(tmp_path):
    samples = np.linspace(0, 1, 100, dtype=np.float32)
    path = _save(
        tmp_path, "siren.npz",
        optimized_siren_wavelet=samples,
        dt_s=np.float64(0.001),
        source_delay_s=np.float64(0.05),
    )
    out = load_wavelet_npz(path)
    assert out.key == "optimized_siren_wavelet"
    assert out.dt_s == pytest.approx(0.001)
    assert out.source_delay_s == pytest.approx(0.05)


def test_explicit_key_overrides_prefer_list(tmp_path):
    samples_init = np.full(50, 7.0, dtype=np.float32)
    samples_opt = np.full(50, 9.0, dtype=np.float32)
    path = _save(
        tmp_path, "both.npz",
        initial_causal_wavelet=samples_init,
        optimized_siren_wavelet=samples_opt,
        dt_s=np.float64(0.001),
    )
    # Default would pick optimized_siren_wavelet first; force initial.
    out = load_wavelet_npz(path, explicit_key="initial_causal_wavelet")
    assert out.key == "initial_causal_wavelet"
    np.testing.assert_allclose(out.samples, samples_init)


def test_no_matching_key_raises(tmp_path):
    path = _save(tmp_path, "bad.npz",
                 not_a_wavelet=np.zeros(10), dt_s=np.float64(0.001))
    with pytest.raises(KeyError, match="none of"):
        load_wavelet_npz(path)


def test_missing_dt_and_time_raises(tmp_path):
    path = _save(tmp_path, "no_dt.npz", wavelet=np.zeros(10, dtype=np.float32))
    with pytest.raises(KeyError, match="neither"):
        load_wavelet_npz(path)


def test_explicit_key_not_present_raises(tmp_path):
    path = _save(tmp_path, "p.npz", wavelet=np.zeros(10, dtype=np.float32),
                 dt_s=np.float64(0.001))
    with pytest.raises(KeyError, match="explicit_key"):
        load_wavelet_npz(path, explicit_key="does_not_exist")
