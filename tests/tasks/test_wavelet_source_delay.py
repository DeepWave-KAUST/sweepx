"""Tests for ``_get_wavelet_source_delay_s`` — the helper used by the
multisource runner to align obs to the SIREN-pipeline wavelet's
zero-prepad frame.

Without this alignment, syn (which fires the wavelet array verbatim —
zero-prepad and all, so the main bang sits at sample
``source_delay_s / dt``) and obs (which has the real event near t=0)
are off by ``source_delay_s`` — a systematic cycle-skip that breaks
gradient convergence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_tasks.runner import _get_wavelet_source_delay_s
from sweep_tasks.schemas import (
    FromNpyWavelet,
    FromSirenPipelineNpzWavelet,
    RickerWavelet,
)


def _write_siren_npz(
    tmp_path: Path,
    *,
    nt: int = 200,
    dt_s: float = 0.001,
    source_delay_s: float | None = 0.05,
    key: str = "optimized_siren_wavelet",
) -> Path:
    out = tmp_path / "siren_wavelet.npz"
    rng = np.random.default_rng(0)
    wave = rng.standard_normal(nt).astype(np.float32)
    payload = {key: wave, "dt_s": np.float64(dt_s)}
    if source_delay_s is not None:
        payload["source_delay_s"] = np.float64(source_delay_s)
    np.savez(out, **payload)
    return out


def test_get_source_delay_s_siren_with_delay(tmp_path):
    p = _write_siren_npz(tmp_path, source_delay_s=0.1)
    spec = FromSirenPipelineNpzWavelet(path=str(p))
    assert _get_wavelet_source_delay_s(spec) == pytest.approx(0.1, abs=1e-6)


def test_get_source_delay_s_siren_without_delay_returns_zero(tmp_path):
    p = _write_siren_npz(tmp_path, source_delay_s=None)
    spec = FromSirenPipelineNpzWavelet(path=str(p))
    assert _get_wavelet_source_delay_s(spec) == 0.0


def test_get_source_delay_s_siren_explicit_zero(tmp_path):
    p = _write_siren_npz(tmp_path, source_delay_s=0.0)
    spec = FromSirenPipelineNpzWavelet(path=str(p))
    assert _get_wavelet_source_delay_s(spec) == 0.0


def test_get_source_delay_s_ricker_kind_returns_zero(tmp_path):
    spec = RickerWavelet(fm=10.0, delay=0.05, scale=1.0)
    assert _get_wavelet_source_delay_s(spec) == 0.0


def test_get_source_delay_s_from_npy_kind_returns_zero(tmp_path):
    """Non-SIREN file format never carries source_delay_s; helper
    must short-circuit without touching the path."""
    spec = FromNpyWavelet(path="/nonexistent/wavelet.npy", scale=1.0)
    assert _get_wavelet_source_delay_s(spec) == 0.0


def test_get_source_delay_s_siren_explicit_key(tmp_path):
    """When ``explicit_key`` is set, the helper still reads source_delay_s
    from the npz scalar (it's a sibling field, not tied to the wavelet
    array key)."""
    p = _write_siren_npz(
        tmp_path, source_delay_s=0.075, key="initial_causal_wavelet",
    )
    spec = FromSirenPipelineNpzWavelet(
        path=str(p), explicit_key="initial_causal_wavelet",
    )
    assert _get_wavelet_source_delay_s(spec) == pytest.approx(0.075, abs=1e-6)
