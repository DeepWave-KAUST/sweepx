"""P12: plot_model + plot_convergence viz tools and the run_fwi orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


def _models(tmp_path):
    nz, nx = 40, 48
    true = np.full((nz, nx), 1500.0, dtype=np.float32)
    true[nz // 2:, :] = 2200.0
    init = np.full((nz, nx), 1500.0, dtype=np.float32)
    tp, ip = tmp_path / "true.npy", tmp_path / "init.npy"
    np.save(tp, true)
    np.save(ip, init)
    return str(tp), str(ip)


@pytest.mark.timeout(180)
def test_plot_model_and_convergence(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    pytest.importorskip("sweep_tasks.viz")
    true_p, init_p = _models(tmp_path)
    b = registry.get("build_fwi_spec").invoke({
        "init_model_path": init_p, "synthetic_true_vp_path": true_p,
        "dh": 10.0, "dt": 1.5e-3, "nt": 150, "fm": 12.0,
        "optimizer": "adam", "lr": 20.0, "loss": "mse", "epochs": 3,
        "source_step": 40, "source_depth": 2, "receiver_step": 2, "receiver_depth": 4,
        "abcn": 10, "spatial_order": 4, "device": "cpu", "backend_impl": "eager",
        "use_compile": False, "output_dir": str(tmp_path / "runs"), "task_id": "fwi",
    })
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 150})
    assert r.value.get("state") == "success", r.value
    td = r.value["task_dir"]

    m = registry.get("plot_model").invoke({
        "task_dir": td, "true_model_path": true_p, "init_model_path": init_p, "dh": 10.0,
    })
    assert m.ok and "error" not in m.value, m.value
    assert Path(m.value["image_path"]).is_file()
    assert m.value["panels"] == ["initial", "inverted", "true"]

    c = registry.get("plot_convergence").invoke({"task_dir": td})
    assert c.ok and "error" not in c.value, c.value
    assert Path(c.value["image_path"]).is_file()
    assert c.value["n_iter"] >= 1


@pytest.mark.timeout(240)
def test_run_fwi_orchestration(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    pytest.importorskip("sweep_tasks.viz")
    true_p, init_p = _models(tmp_path)
    a = registry.get("run_fwi").invoke({
        "init_model_path": init_p, "synthetic_true_vp_path": true_p, "out_dir": str(tmp_path / "runs"),
        "dh": 10.0, "dt": 1.5e-3, "nt": 150, "fm": 12.0, "lr": 20.0, "epochs": 3,
        "vp_min": 1400.0, "vp_max": 2400.0, "source_step": 40, "receiver_step": 2,
        "device": "cpu",
    })
    assert a.ok and "error" not in a.value, a.value
    assert a.value["state"] == "success"
    assert Path(a.value["model_image"]).is_file()
    assert Path(a.value["convergence_image"]).is_file()
    assert Path(a.value["inverted_vp"]).is_file()


def test_run_fwi_requires_one_obs_source(tmp_path):
    pytest.importorskip("sweep")
    true_p, init_p = _models(tmp_path)
    a = registry.get("run_fwi").invoke({
        "init_model_path": init_p, "out_dir": str(tmp_path / "runs"),
        "dh": 10.0, "dt": 1.5e-3, "nt": 100,
    })
    assert "error" in a.value and "EXACTLY one" in a.value["error"]


def test_plot_model_no_result(tmp_path):
    pytest.importorskip("sweep_tasks.viz")
    res = registry.get("plot_model").invoke({"task_dir": str(tmp_path / "nope")})
    assert "error" in res.value
