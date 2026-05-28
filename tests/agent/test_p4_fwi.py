"""P4: build_fwi_spec flat FWI builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


def _vp(path, value=1800.0, shape=(32, 40)):
    np.save(path, np.full(shape, value, dtype=np.float32))
    return str(path)


def test_build_fwi_synthetic_obs(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    init = _vp(tmp_path / "init.npy", 1800.0)
    true = _vp(tmp_path / "true.npy", 2000.0)
    res = registry.get("build_fwi_spec").invoke({
        "init_model_path": init, "synthetic_true_vp_path": true,
        "dh": 12.5, "dt": 0.001, "nt": 60, "fm": 10.0,
        "optimizer": "adam", "lr": 10.0, "loss": "mse", "epochs": 3,
        "vp_min": 1500.0, "vp_max": 2500.0,
        "output_dir": str(tmp_path / "r"), "task_id": "fwi_build",
    })
    assert res.ok and "error" not in res.value, res.value
    spec = res.value["spec"]
    assert spec["task_type"] == "fwi"
    assert spec["init_model"]["name"] == "vp"
    assert spec["obs"]["synthetic_from"]["path"].endswith("true.npy")
    assert spec["optimizer"]["kind"] == "adam"
    assert spec["model_bounds"]["vp"]["min"] == 1500.0


def test_build_fwi_requires_one_obs(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    init = _vp(tmp_path / "init.npy")
    # no obs source
    res = registry.get("build_fwi_spec").invoke({
        "init_model_path": init, "dh": 12.5, "dt": 0.001, "nt": 60,
        "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value
    assert "obs source" in res.value["error"]


def test_build_fwi_rejects_two_obs(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    init = _vp(tmp_path / "init.npy")
    true = _vp(tmp_path / "true.npy", 2000.0)
    res = registry.get("build_fwi_spec").invoke({
        "init_model_path": init, "synthetic_true_vp_path": true, "obs_npy_path": str(tmp_path / "x.npy"),
        "dh": 12.5, "dt": 0.001, "nt": 60, "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value


def test_build_fwi_lbfgs_and_bounds(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    init = _vp(tmp_path / "init.npy")
    true = _vp(tmp_path / "true.npy", 2000.0)
    res = registry.get("build_fwi_spec").invoke({
        "init_model_path": init, "synthetic_true_vp_path": true,
        "dh": 12.5, "dt": 0.001, "nt": 60, "optimizer": "lbfgs", "lr": 1.0,
        "loss": "trace_cosine", "epochs": 5, "output_dir": str(tmp_path / "r"),
    })
    assert res.ok and "error" not in res.value, res.value
    assert res.value["spec"]["optimizer"]["kind"] == "lbfgs"
    assert res.value["spec"]["loss"]["kind"] == "trace_cosine"


def test_build_fwi_bad_equation(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    init = _vp(tmp_path / "init.npy")
    true = _vp(tmp_path / "true.npy", 2000.0)
    res = registry.get("build_fwi_spec").invoke({
        "init_model_path": init, "synthetic_true_vp_path": true,
        "dh": 12.5, "dt": 0.001, "nt": 60, "equation": "elastic",  # wrong case/name
        "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value
    assert "Elastic" in res.value["error"]


@pytest.mark.timeout(300)
def test_fwi_endtoend_cpu(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    init = _vp(tmp_path / "init.npy", 1800.0)
    true = _vp(tmp_path / "true.npy", 2000.0)
    out = tmp_path / "runs"
    b = registry.get("build_fwi_spec").invoke({
        "init_model_path": init, "synthetic_true_vp_path": true,
        "dh": 12.5, "dt": 0.001, "nt": 60, "fm": 15.0,
        "source_step": 20, "source_depth": 2, "receiver_step": 5, "receiver_depth": 8,
        "optimizer": "adam", "lr": 10.0, "loss": "mse", "epochs": 2,
        "spatial_order": 4, "abcn": 10, "device": "cpu", "backend_impl": "eager",
        "use_compile": False, "output_dir": str(out), "task_id": "fwi_e2e",
    })
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 250})
    assert r.value.get("state") == "success", r.value
