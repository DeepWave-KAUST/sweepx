"""Unit tests for ``sweep_tasks.postproc.filter_image``.

Covers:

* :func:`cosine_taper` shape + boundary behaviour (1 near surface, 0 deep,
  cosine ramp in between).
* :func:`gaussian_lowpass_z` removes long-wavelength z drift while
  preserving short-wavelength content.
* :func:`filter_shallow_low_frequency` is identity (within float noise)
  for an already-deep image; subtracts the drift for a shallow ramp.
* :func:`filter_image_file` round-trip: writes the expected npy / png /
  metadata files and the npy contents match the in-memory algorithm.
* CLI: ``sweep-tasks filter-image <input.npy> --dz-m ...`` produces the
  same artefacts as the Python entry point.
* Schema: ``PostFilterImageSpec`` defaults match the legacy Viking recipe
  and round-trip cleanly through the RTM task YAML.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sweep_tasks.postproc.filter_image import (
    cosine_taper,
    filter_image_file,
    filter_shallow_low_frequency,
    gaussian_lowpass_z,
)


# ---------------------------------------------------------------------------
# Pure algorithm — small synthetic arrays
# ---------------------------------------------------------------------------


def test_cosine_taper_shape_and_boundaries():
    dz = 10.0
    nz = 100
    depth, taper = 200.0, 300.0  # full strength 0..20, ramp 20..50, zero below
    w = cosine_taper(nz=nz, dz_m=dz, depth_m=depth, taper_m=taper)
    assert w.shape == (nz,)
    assert w.dtype == np.float32
    # Shallow (z < depth_m): full strength.
    assert np.allclose(w[:21], 1.0)
    # Deep (z >= depth_m + taper_m, i.e. z >= 500): zero.
    assert np.allclose(w[50:], 0.0)
    # Monotone non-increasing across the ramp.
    assert np.all(np.diff(w[20:51]) <= 1.0e-6)
    # Midpoint should be ~0.5 (cosine of pi/2 ramp).
    mid = w[35]  # z = 350 = depth + taper/2 -> phase=0.5 -> 0.5(1+cos(pi/2))=0.5
    assert mid == pytest.approx(0.5, abs=1.0e-3)


def test_cosine_taper_zero_taper_is_hard_step():
    w = cosine_taper(nz=10, dz_m=1.0, depth_m=4.5, taper_m=0.0)
    # Cliff at z > 4.5: indices 0..4 are 1, indices 5..9 are 0.
    assert np.allclose(w[:5], 1.0)
    assert np.allclose(w[5:], 0.0)


def test_gaussian_lowpass_z_removes_long_wavelength_drift():
    # Use 4096 samples × 10 m so the long-wavelength period fits cleanly
    # many times and FFT edge effects are negligible. The short-wavelength
    # signal should be heavily attenuated; the long drift should pass.
    nz, nx = 4096, 4
    dz = 10.0
    z = np.arange(nz, dtype=np.float64) * dz
    long = 5.0 * np.sin(2 * np.pi * z / 4000.0).astype(np.float32)
    short = 1.0 * np.sin(2 * np.pi * z / 80.0).astype(np.float32)
    data = np.broadcast_to((long + short)[:, None], (nz, nx)).astype(np.float32)
    low = gaussian_lowpass_z(data, dz_m=dz, wavelength_m=300.0)
    # Truncate boundary indices when scoring; FFT periodicity gives a tiny
    # ramp at the very edges that's not a property of the LP itself.
    interior = slice(64, -64)
    err_long = np.std(low[interior] - long[interior, None])
    # The short component must be heavily attenuated → low ≈ long, so
    # (low - long) has tiny std even though short has std ≈ 0.71.
    assert err_long < 0.25, f"long drift recovery std={err_long}"
    # Cross check: short-wavelength energy in `low` is much less than in the
    # raw input.
    in_std = np.std(data[interior] - long[interior, None])
    out_std = np.std(low[interior] - long[interior, None])
    assert out_std < 0.5 * in_std, (
        f"LP did not attenuate short component (in_std={in_std} out_std={out_std})"
    )


def test_filter_shallow_low_frequency_passes_deep_image_through():
    # A purely deep signal (only nonzero in the bottom half) shouldn't be
    # disturbed by the shallow filter at all (taper_weight=0 below depth_m).
    nz, nx = 200, 16
    dz = 10.0
    arr = np.zeros((nz, nx), dtype=np.float32)
    arr[150:, :] = np.random.default_rng(0).standard_normal((50, nx)).astype(np.float32)
    filtered, removed, weights = filter_shallow_low_frequency(
        arr, dz_m=dz, wavelength_m=300.0, depth_m=600.0, taper_m=400.0,
    )
    # Below z = depth_m + taper_m = 1000 m -> index 100 onward: weight=0,
    # so 'removed' must be zero there.
    assert np.allclose(removed[100:], 0.0)
    # Deep signal preserved.
    assert np.allclose(filtered[150:], arr[150:], atol=1.0e-5)
    # Shape + dtype sanity.
    assert filtered.shape == arr.shape == removed.shape
    assert filtered.dtype == np.float32
    assert weights.shape == (nz,)


def test_filter_shallow_low_frequency_subtracts_shallow_drift():
    # Slow-varying drift (lambda=3000m, well above cutoff=300m) + a
    # short-wavelength signal (lambda=80m, well below cutoff). The shallow
    # part should be cleaned up; the deep part untouched.
    nz, nx = 4096, 4
    dz = 10.0
    z = np.arange(nz, dtype=np.float64) * dz
    drift = 5.0 * np.sin(2 * np.pi * z / 3000.0).astype(np.float32)
    signal = 1.0 * np.sin(2 * np.pi * z / 80.0).astype(np.float32)
    data = np.broadcast_to((drift + signal)[:, None], (nz, nx)).astype(np.float32)
    filtered, removed, _ = filter_shallow_low_frequency(
        data, dz_m=dz, wavelength_m=300.0, depth_m=600.0, taper_m=400.0,
    )
    # original = filtered + removed by construction.
    assert np.allclose(data, filtered + removed, atol=1.0e-5)
    # Shallow part: weight=1 across depth_m=600 m (idx 0..60); skip the
    # very first samples to avoid FFT edge effects. After filtering, the
    # drift should be gone; the filtered signal should look mostly like
    # the short-wavelength sinusoid (sine of unit amplitude → std ≈ 0.71).
    shallow = slice(20, 55)
    cleaned_std = np.std(filtered[shallow])
    assert 0.5 < cleaned_std < 1.0, (
        f"cleaned shallow looks wrong (std={cleaned_std}, expect ~0.71)"
    )
    # Cleaned shallow should be much closer to the pure short signal than
    # the raw data is.
    cleaned_to_signal = np.std(filtered[shallow] - signal[shallow, None])
    raw_to_signal = np.std(data[shallow] - signal[shallow, None])
    assert cleaned_to_signal < 0.4 * raw_to_signal, (
        f"drift not sufficiently removed (raw error {raw_to_signal}, "
        f"cleaned error {cleaned_to_signal})"
    )
    # Deep part (well below depth_m + taper_m = 1000 m -> idx 100): the
    # taper is zero so the filter is a no-op.
    deep = slice(120, -64)
    assert np.allclose(filtered[deep], data[deep], atol=1.0e-5)


def test_filter_shallow_low_frequency_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D image"):
        filter_shallow_low_frequency(np.zeros((10,)), dz_m=10.0)


# ---------------------------------------------------------------------------
# File-IO round trip
# ---------------------------------------------------------------------------


def _make_synthetic_npy(tmp_path: Path) -> Path:
    nz, nx = 100, 64
    dz = 12.5
    z = np.arange(nz, dtype=np.float32) * dz
    drift = 3.0 * np.sin(2 * np.pi * z / 2000.0)
    rng = np.random.default_rng(7)
    image = (drift[:, None] + 0.1 * rng.standard_normal((nz, nx))).astype(np.float32)
    path = tmp_path / "rtm_image.npy"
    np.save(path, image)
    return path


def test_filter_image_file_writes_expected_artifacts(tmp_path: Path):
    src = _make_synthetic_npy(tmp_path)
    meta = filter_image_file(
        src,
        output_dir=tmp_path,
        dz_m=12.5,
        dx_m=12.5,
        wavelength_m=300.0,
        depth_m=600.0,
        taper_m=400.0,
        save_png=True,
    )
    stem = "rtm_image_shallow_zlowcut"
    expected = {
        f"{stem}.npy",
        f"{stem}_removed.npy",
        f"{stem}_z_taper.npy",
        f"{stem}.png",
        f"{stem}_comparison.png",
        f"{stem}_metadata.json",
        "rtm_image.npy",
    }
    actually_written = {p.name for p in tmp_path.iterdir()}
    missing = expected - actually_written
    assert not missing, f"missing artefacts: {missing}"
    # The npy from filter_image_file must match the in-memory algorithm.
    orig = np.load(src)
    filt_from_disk = np.load(tmp_path / f"{stem}.npy")
    filt_in_mem, removed_in_mem, _ = filter_shallow_low_frequency(
        orig, dz_m=12.5, wavelength_m=300.0, depth_m=600.0, taper_m=400.0,
    )
    assert np.allclose(filt_from_disk, filt_in_mem, atol=1.0e-6)
    removed_from_disk = np.load(tmp_path / f"{stem}_removed.npy")
    assert np.allclose(removed_from_disk, removed_in_mem, atol=1.0e-6)
    # Metadata sanity.
    md = json.loads((tmp_path / f"{stem}_metadata.json").read_text())
    assert md["lowpass_wavelength_m"] == 300.0
    assert md["filter_full_depth_m"] == 600.0
    assert md["taper_m"] == 400.0
    assert md["dz_m"] == 12.5
    assert md["input_shape"] == list(orig.shape)
    # Returned dict mirrors the json.
    assert meta["output"].endswith(f"{stem}.npy")


def test_filter_image_file_no_png(tmp_path: Path):
    src = _make_synthetic_npy(tmp_path)
    filter_image_file(src, output_dir=tmp_path, dz_m=12.5, save_png=False)
    assert (tmp_path / "rtm_image_shallow_zlowcut.npy").is_file()
    assert not (tmp_path / "rtm_image_shallow_zlowcut.png").exists()
    assert not (tmp_path / "rtm_image_shallow_zlowcut_comparison.png").exists()


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_filter_image_smoke(tmp_path: Path):
    src = _make_synthetic_npy(tmp_path)
    cmd = [
        sys.executable, "-m", "sweep_tasks.cli", "filter-image",
        str(src),
        "--dz-m", "12.5",
        "--wavelength-m", "300",
        "--depth-m", "600",
        "--taper-m", "400",
        "--no-png",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"filter-image CLI failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert (tmp_path / "rtm_image_shallow_zlowcut.npy").is_file()
    assert (tmp_path / "rtm_image_shallow_zlowcut_metadata.json").is_file()


def test_cli_filter_image_infers_dh_from_rtm_result(tmp_path: Path):
    src = _make_synthetic_npy(tmp_path)
    # Stash a fake rtm_result.npz next to the input with a sentinel dh.
    np.savez_compressed(tmp_path / "rtm_result.npz", dh=np.float32(12.5))
    cmd = [
        sys.executable, "-m", "sweep_tasks.cli", "filter-image",
        str(src),
        # NOTE: no --dz-m here on purpose.
        "--no-png",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"filter-image CLI failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "inferred dz/dx=12.5" in proc.stdout, proc.stdout
    md = json.loads(
        (tmp_path / "rtm_image_shallow_zlowcut_metadata.json").read_text()
    )
    assert md["dz_m"] == 12.5


def test_cli_filter_image_requires_dz_when_no_rtm_result(tmp_path: Path):
    src = _make_synthetic_npy(tmp_path)
    cmd = [
        sys.executable, "-m", "sweep_tasks.cli", "filter-image",
        str(src), "--no-png",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--dz-m is required" in proc.stdout


# ---------------------------------------------------------------------------
# Schema integration — PostFilterImageSpec on RTMImagingSpec
# ---------------------------------------------------------------------------


def test_post_filter_spec_defaults_match_legacy():
    from sweep_tasks import PostFilterImageSpec
    spec = PostFilterImageSpec()
    assert spec.enabled is True
    assert spec.wavelength_m == 300.0
    assert spec.depth_m == 600.0
    assert spec.taper_m == 400.0
    assert spec.clip_percentile == 1.0
    assert spec.display_scale == 1.0
    assert spec.cmap == "sweep_image"
    assert spec.targets == "all"
    assert spec.save_png is True


def test_rtm_imaging_spec_round_trips_with_post_filter():
    from sweep_tasks import RTMImagingSpec
    spec = RTMImagingSpec.model_validate({
        "shots_per_batch": 2,
        "loss_kind": "mse",
        "post_filter": {
            "enabled": True,
            "wavelength_m": 500.0,
            "depth_m": 800.0,
            "taper_m": 300.0,
            "targets": ["rtm_image_per_shot_normalised"],
        },
    })
    assert spec.post_filter is not None
    assert spec.post_filter.wavelength_m == 500.0
    assert spec.post_filter.targets == ["rtm_image_per_shot_normalised"]
    # Round-trip through dict.
    spec2 = RTMImagingSpec.model_validate(spec.model_dump())
    assert spec2.post_filter == spec.post_filter
