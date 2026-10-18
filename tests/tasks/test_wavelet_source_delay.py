"""Tests for ``_get_wavelet_source_delay_s`` — the helper used by the
plan-streaming runner to align obs to the SIREN-pipeline wavelet's
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

from sweep_tasks._helpers.wavelet_build import _get_wavelet_source_delay_s
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


# ---------------------------------------------------------------------------
# estimate-wavelet output → FWI runner compatibility (added 2026-05-22)
# ---------------------------------------------------------------------------
def _fake_prepared_data(nt: int = 200, dt_s: float = 0.002,
                        observed_delay_s: float = 0.0):
    """Build a minimal PreparedWaveletData stand-in for save_wavelet_outputs."""
    from sweep_tasks.wavelet.sweep_torch import PreparedWaveletData

    return PreparedWaveletData(
        observed=np.zeros((1, 1, nt), dtype=np.float32),
        sources=np.zeros((1, 1, 2), dtype=np.int64),
        receivers=np.zeros((1, 1, 2), dtype=np.int64),
        model=np.full((10, 10), 1500.0, dtype=np.float32),
        dt_s=dt_s,
        nt=nt,
        shot_ids=np.array([1], dtype=np.int64),
        x_origin_m=0.0,
        geometry_summary={"nshots": 1},
        selected_offsets_m=np.array([100.0], dtype=np.float32),
        observed_delay_s=float(observed_delay_s),
    )


def _fake_config(nt: int = 200, dt_s: float = 0.002, *, initial_path=None):
    """Minimal WaveletInversionConfig instance for the saver."""
    from sweep_tasks.wavelet.sweep_torch import WaveletInversionConfig

    return WaveletInversionConfig(
        plan_path=Path("/nonexistent/plan.npz"),
        output_dir=Path("/tmp/unused"),
        tmax_s=float(nt * dt_s),
        mode="discrete",
        initial_wavelet_path=initial_path,
    )


def test_estimate_wavelet_npz_has_fwi_metadata(tmp_path):
    """``save_wavelet_outputs`` must write all the keys the FWI runner
    consumes via ``kind: siren_pipeline_npz``:
      * ``dt_s`` (scalar) — load_wavelet_npz prefers this over time_s
      * ``source_delay_s`` (scalar) — drives obs prepad-shift
      * ``optimized_siren_wavelet`` (alias of wavelet) — legacy key

    Without this metadata, ``_get_wavelet_source_delay_s`` silently
    returns 0 and syn/obs end up cycle-skipped at the wavelet prepad.
    """
    from sweep_tasks.wavelet.sweep_torch import save_wavelet_outputs

    nt, dt = 256, 0.002
    wavelet = np.zeros(nt, dtype=np.float32)
    # Put a fake peak at sample 50 so the measured-peak fallback gives
    # source_delay_s = 50 × dt = 0.1 s.
    wavelet[50] = 1.0
    out_dir = tmp_path / "siren_pipeline"
    save_wavelet_outputs(
        output_dir=out_dir,
        wavelet=wavelet,
        initial_wavelet=None,
        losses=[1.0, 0.5, 0.1],
        data=_fake_prepared_data(nt=nt, dt_s=dt),
        config=_fake_config(nt=nt, dt_s=dt),
    )

    npz_path = out_dir / "estimated_wavelet.npz"
    assert npz_path.exists()
    with np.load(npz_path) as nz:
        files = set(nz.files)
        assert "wavelet" in files
        assert "optimized_siren_wavelet" in files            # legacy alias
        assert "dt_s" in files                                # FWI loader needs this
        assert "source_delay_s" in files                      # obs prepad-shift
        assert "source_prepad_s" in files
        assert "observed_delay_s" in files

        assert float(nz["dt_s"].item()) == pytest.approx(dt, abs=1e-9)
        # Measured peak: sample 50 × dt = 0.1 s.
        assert float(nz["source_delay_s"].item()) == pytest.approx(0.1, abs=1e-6)
        # Aliased wavelet must match primary key byte-for-byte.
        np.testing.assert_array_equal(nz["wavelet"], nz["optimized_siren_wavelet"])

    # The npz must round-trip through FWI's ``_get_wavelet_source_delay_s``
    # without further config — that's the whole point of the schema.
    spec = FromSirenPipelineNpzWavelet(path=str(npz_path))
    assert _get_wavelet_source_delay_s(spec) == pytest.approx(0.1, abs=1e-6)


def test_estimate_wavelet_source_delay_tracks_measured_peak(tmp_path):
    """``source_delay_s`` MUST equal the wavelet's measured peak time.

    The FWI runner uses ``source_delay_s`` as the obs left-shift amount
    that lands the direct arrival on syn's main bang at sample
    ``round(source_delay_s / dt)``. If the saver writes the legacy
    formula ``max(0, -rel_t.min()) + prepad_s`` instead, and the SIREN
    inversion drifted the peak away from that frame (no front-mute), the
    obs/syn alignment is off by the drift and FWI cycle-skips.

    ``source_prepad_s`` is purely informational metadata — it reflects
    the analyze-side prepad if available but does NOT override the
    measured peak.
    """
    from sweep_tasks.wavelet.sweep_torch import save_wavelet_outputs

    nt, dt = 256, 0.002

    # Synthetic rank-1 analyze output: rel_t in [-0.05, +0.40], prepad_s=0.05.
    # Legacy formula would say source_delay_s = -(-0.05) + 0.05 = 0.10 s.
    rank1_path = tmp_path / "rank1.npz"
    rel_t = np.arange(-0.05, 0.40 + 1e-9, dt, dtype=np.float64)
    np.savez(
        rank1_path,
        wavelet_relative=np.zeros_like(rel_t, dtype=np.float32),
        time_relative_s=rel_t.astype(np.float32),
        prepad_s=np.float32(0.05),
    )

    wavelet = np.zeros(nt, dtype=np.float32)
    # Put peak at sample 200 (= 0.4 s) — deliberately away from the
    # legacy prepad frame (0.10 s). The saver MUST report the peak time,
    # not the legacy formula.
    wavelet[200] = 1.0
    out_dir = tmp_path / "siren_pipeline"
    save_wavelet_outputs(
        output_dir=out_dir,
        wavelet=wavelet,
        initial_wavelet=None,
        losses=[1.0],
        data=_fake_prepared_data(nt=nt, dt_s=dt),
        config=_fake_config(nt=nt, dt_s=dt, initial_path=rank1_path),
    )
    with np.load(out_dir / "estimated_wavelet.npz") as nz:
        # Measured peak (0.4 s) wins over the legacy formula (0.1 s) when
        # obs was NOT pre-shifted (data.observed_delay_s == 0).
        assert float(nz["source_delay_s"].item()) == pytest.approx(0.40, abs=1e-6)
        # source_prepad_s mirrors the analyze metadata (informational).
        assert float(nz["source_prepad_s"].item()) == pytest.approx(0.05, abs=1e-6)
        # observed_delay_s reflects what prepare actually applied (here: 0).
        assert float(nz["observed_delay_s"].item()) == pytest.approx(0.0, abs=1e-6)


def test_estimate_wavelet_source_delay_follows_obs_delay_when_shifted(tmp_path):
    """When the prepare step right-shifted obs by ``observed_delay_s`` (legacy
    delay_traces alignment), the npz must report that value as
    ``source_delay_s`` — not the wavelet's measured peak. The FWI runner
    then reapplies an identical shift downstream, keeping syn/obs in the
    same prepad frame the SIREN trained in.
    """
    from sweep_tasks.wavelet.sweep_torch import save_wavelet_outputs

    nt, dt = 256, 0.002
    wavelet = np.zeros(nt, dtype=np.float32)
    # Peak at sample 80 (= 0.16 s). Without obs shift the saver would
    # report this. With obs_delay=0.10 s active, the saver should report
    # the obs-delay anchor instead so the FWI runner re-shifts obs to the
    # exact frame the SIREN saw at training time.
    wavelet[80] = 1.0

    out_dir = tmp_path / "siren_pipeline"
    save_wavelet_outputs(
        output_dir=out_dir,
        wavelet=wavelet,
        initial_wavelet=None,
        losses=[1.0],
        data=_fake_prepared_data(nt=nt, dt_s=dt, observed_delay_s=0.10),
        config=_fake_config(nt=nt, dt_s=dt),
    )
    with np.load(out_dir / "estimated_wavelet.npz") as nz:
        assert float(nz["source_delay_s"].item()) == pytest.approx(0.10, abs=1e-6)
        assert float(nz["observed_delay_s"].item()) == pytest.approx(0.10, abs=1e-6)
