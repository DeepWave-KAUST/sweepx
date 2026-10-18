"""P20: the practical extras — make_synthetic_model, plot_velocity_model,
plot_velocity_slice, plot_wavelet, animate_fwi_evolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


@pytest.mark.parametrize("kind", ["two_layer", "gradient", "smooth", "layered", "anomaly", "fault"])
def test_make_synthetic_model(tmp_path, kind):
    out = tmp_path / f"{kind}.npy"
    r = registry.get("make_synthetic_model").invoke({
        "kind": kind, "nz": 60, "nx": 80, "vp_min": 1800.0, "vp_max": 3000.0,
        "out_path": str(out),
    })
    assert "error" not in r.value, r.value
    assert out.is_file()
    arr = np.load(out)
    assert arr.shape == (60, 80)
    assert 1800.0 <= arr.min() and arr.max() <= 3000.0


def test_smooth_is_a_gradient_alias(tmp_path):
    """'smooth starting model' is the canonical FWI phrasing — it must be a
    valid kind (regression: the LLM passed kind='smooth' and hit a validation
    error, then said 'there was an issue' and retried)."""
    g = tmp_path / "g.npy"; s = tmp_path / "s.npy"
    for k, p in (("gradient", g), ("smooth", s)):
        registry.get("make_synthetic_model").invoke({
            "kind": k, "nz": 60, "nx": 80, "out_path": str(p)})
    assert np.array_equal(np.load(g), np.load(s))  # smooth ≡ gradient


def test_plot_velocity_model_2d_and_3d(tmp_path):
    pytest.importorskip("sweep_tasks.viz")
    m2 = tmp_path / "m2.npy"
    np.save(m2, np.linspace(1800, 3000, 60 * 80).reshape(60, 80).astype(np.float32))
    r = registry.get("plot_velocity_model").invoke({"model_path": str(m2), "dh": 10.0, "out_path": str(tmp_path / "m2.png")})
    assert "error" not in r.value and Path(r.value["image_path"]).is_file()
    assert r.value["shape"] == [60, 80]

    m3 = tmp_path / "m3.npy"
    np.save(m3, np.full((20, 24, 28), 2200.0, dtype=np.float32))
    r = registry.get("plot_velocity_model").invoke({"model_path": str(m3), "dh": 10.0, "out_path": str(tmp_path / "m3.png")})
    assert "error" not in r.value and Path(r.value["image_path"]).is_file()


def test_plot_velocity_slice(tmp_path):
    m = tmp_path / "m.npy"
    arr = np.full((100, 120), 1800.0, dtype=np.float32)
    arr[50:, :] = 2600.0
    np.save(m, arr)
    r = registry.get("plot_velocity_slice").invoke({
        "model_path": str(m), "x_indices": [20, 60, 100], "dh": 10.0,
        "out_path": str(tmp_path / "slice.png"),
    })
    assert "error" not in r.value, r.value
    assert Path(r.value["image_path"]).is_file()
    assert r.value["x_indices"] == [20, 60, 100]


def test_plot_velocity_slice_3d_center(tmp_path):
    m = tmp_path / "m3.npy"
    np.save(m, np.full((30, 20, 40), 2000.0, dtype=np.float32))
    r = registry.get("plot_velocity_slice").invoke({"model_path": str(m), "out_path": str(tmp_path / "s3.png")})
    assert "error" not in r.value and Path(r.value["image_path"]).is_file()


def test_plot_wavelet(tmp_path):
    r = registry.get("plot_wavelet").invoke({"fm": 12.0, "dt": 1e-3, "nt": 512, "out_path": str(tmp_path / "w.png")})
    assert "error" not in r.value, r.value
    assert Path(r.value["image_path"]).is_file()
    assert abs(r.value["peak_frequency_hz"] - 12.0) < 4.0  # spectral peak near fm


def test_animate_fwi_evolution(tmp_path):
    pytest.importorskip("sweep_tasks.viz")
    # Fabricate an FWI-style task dir with per-epoch models that sharpen.
    epochs = tmp_path / "fwi" / "output" / "epochs"
    epochs.mkdir(parents=True)
    np.save(tmp_path / "fwi" / "output" / "inverted_vp.npy", np.full((40, 50), 1800.0, dtype=np.float32))
    for i, e in enumerate((0, 5, 10)):
        m = np.full((40, 50), 1800.0, dtype=np.float32)
        m[20:, :] = 1800.0 + 600.0 * (i + 1) / 3.0  # bottom layer grows in each epoch
        np.save(epochs / f"vp_epoch_{e:04d}.npy", m)
    r = registry.get("animate_fwi_evolution").invoke({"task_dir": str(tmp_path / "fwi"), "dh": 10.0, "fps": 3})
    assert "error" not in r.value, r.value
    assert Path(r.value["gif_path"]).is_file()
    assert r.value["n_frames"] == 3


def test_plot_velocity_model_missing_file(tmp_path):
    r = registry.get("plot_velocity_model").invoke({"model_path": str(tmp_path / "nope.npy")})
    assert "error" in r.value
