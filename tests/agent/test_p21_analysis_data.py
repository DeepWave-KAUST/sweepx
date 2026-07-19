"""P21: QC + data-I/O tools — check_parameters, plot_observed_data,
compare_shot_gathers, plot_segy, and a tiny multiscale FWI chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


def _model(tmp_path):
    m = np.full((100, 120), 1500.0, dtype=np.float32)
    m[50:, :] = 3000.0
    p = tmp_path / "vp.npy"
    np.save(p, m)
    return str(p)


def test_check_parameters_good_and_bad(tmp_path):
    mp = _model(tmp_path)
    good = registry.get("check_parameters").invoke({"model_path": mp, "dh": 10.0, "dt": 1e-3, "fm": 10.0})
    assert good.value["ok"] is True
    assert good.value["courant_number"] <= 0.9
    bad = registry.get("check_parameters").invoke({"model_path": mp, "dh": 40.0, "dt": 5e-3, "fm": 15.0})
    assert bad.value["ok"] is False
    assert bad.value["warnings"]
    assert bad.value["recommended_dt_max"] > 0 and bad.value["recommended_dh_max"] > 0


def test_check_parameters_needs_velocities():
    r = registry.get("check_parameters").invoke({"dh": 10.0, "dt": 1e-3, "fm": 10.0})
    assert "error" in r.value


def test_plot_observed_data(tmp_path):
    pytest.importorskip("sweep_tasks.viz")
    rec = np.random.randn(1, 200, 50, 1).astype(np.float32)
    p = tmp_path / "rec.npy"
    np.save(p, rec)
    r = registry.get("plot_observed_data").invoke({"npy_path": str(p), "dt": 1e-3, "out_path": str(tmp_path / "g.png")})
    assert "error" not in r.value and Path(r.value["image_path"]).is_file()
    assert r.value["shape"] == [200, 50]


def test_compare_shot_gathers(tmp_path):
    pytest.importorskip("sweep_tasks.viz")
    a = np.random.randn(1, 200, 50, 1).astype(np.float32)
    b = a + 0.5 * np.random.randn(*a.shape).astype(np.float32)
    pa, pb = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(pa, a); np.save(pb, b)
    r = registry.get("compare_shot_gathers").invoke({"record_a": str(pa), "record_b": str(pb), "dt": 1e-3, "out_path": str(tmp_path / "c.png")})
    assert "error" not in r.value, r.value
    assert Path(r.value["image_path"]).is_file()
    assert 0.0 < r.value["relative_residual"] < 2.0


def test_plot_segy(tmp_path):
    pytest.importorskip("sweep_tasks.viz")
    segyio = pytest.importorskip("segyio")
    path = str(tmp_path / "t.segy")
    data = (np.random.randn(40, 300) * 100).astype(np.float32)  # (ntraces, nt)
    segyio.tools.from_array2D(path, data, dt=2000)
    r = registry.get("plot_segy").invoke({"segy_path": path, "out_path": str(tmp_path / "s.png")})
    assert "error" not in r.value, r.value
    assert r.value["shape"] == [300, 40] and abs(r.value["dt"] - 0.002) < 1e-6


@pytest.mark.timeout(180)
def test_run_forward_and_plot_uses_its_own_run(tmp_path):
    """The one-call forward+plot must plot the gather from THE RUN IT JUST DID
    (record_length_s honoured), never a stale task_dir."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    pytest.importorskip("sweep_tasks.viz")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((60, 80), 2000.0, dtype=np.float32))
    r = registry.get("run_forward_and_plot").invoke({
        "vp_path": str(vp), "out_path": str(tmp_path / "g.png"),
        "dh": 10.0, "dt": 1e-3, "record_length_s": 0.4, "fm": 12.0,
        "source_step": 40, "receiver_step": 2, "output_dir": str(tmp_path / "runs"),
    })
    assert r.ok and "error" not in r.value, r.value
    assert Path(r.value["image_path"]).is_file()
    # 0.4 s @ dt=1e-3 → 400 samples; the gather is from this run, not a stale one.
    assert r.value["shape"][0] == 400
    rec = np.load(Path(r.value["task_dir"]) / "output" / "record.npy")
    assert rec.shape[1] == 400


@pytest.mark.timeout(300)
def test_run_multiscale_fwi_two_bands(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    pytest.importorskip("sweep_tasks.viz")
    nz, nx = 48, 60
    true = np.full((nz, nx), 1500.0, dtype=np.float32)
    true[nz // 2:, :] = 2200.0
    init = np.full((nz, nx), 1500.0, dtype=np.float32)
    tp, ip = tmp_path / "true.npy", tmp_path / "init.npy"
    np.save(tp, true); np.save(ip, init)
    r = registry.get("run_multiscale_fwi").invoke({
        "init_model_path": str(ip), "synthetic_true_vp_path": str(tp), "out_dir": str(tmp_path / "ms"),
        "frequencies": [6.0, 10.0], "dh": 10.0, "dt": 1.5e-3, "nt": 160,
        "epochs_per_band": 2, "lr": 20.0, "source_step": 40, "receiver_step": 2, "device": "cpu",
    })
    assert r.ok and r.value.get("state") == "success", r.value
    assert len(r.value["bands"]) == 2
    assert Path(r.value["final_inverted_vp"]).is_file()
