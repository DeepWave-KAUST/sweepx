"""P5: build_wavefield_spec flat tool."""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


def _vp(path, value=1800.0, shape=(32, 40)):
    np.save(path, np.full(shape, value, dtype=np.float32))
    return str(path)


def test_build_wavefield_ok(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _vp(tmp_path / "vp.npy")
    res = registry.get("build_wavefield_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 100,
        "snapshot_times": [30, 60, 90],
        "output_dir": str(tmp_path / "r"), "task_id": "wf",
    })
    assert res.ok and "error" not in res.value, res.value
    assert res.value["spec"]["task_type"] == "wavefield"
    assert res.value["spec"]["snapshot_times"] == [30, 60, 90]


def test_build_wavefield_snaptime_out_of_range(tmp_path):
    pytest.importorskip("sweep_tasks")
    vp = _vp(tmp_path / "vp.npy")
    res = registry.get("build_wavefield_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 50,
        "snapshot_times": [30, 80],  # 80 >= nt
        "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value
    assert "out of range" in res.value["error"]


def test_build_wavefield_bad_equation(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _vp(tmp_path / "vp.npy")
    res = registry.get("build_wavefield_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 100,
        "snapshot_times": [50], "equation": "VTI",
        "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value
    assert "AcousticVTI" in res.value["error"]


@pytest.mark.timeout(180)
def test_wavefield_e2e_cpu(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _vp(tmp_path / "vp.npy")
    b = registry.get("build_wavefield_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 80, "fm": 15.0,
        "snapshot_times": [20, 50],
        "source_step": 20, "source_depth": 2, "receiver_step": 5, "receiver_depth": 8,
        "spatial_order": 4, "abcn": 10, "device": "cpu", "backend_impl": "eager",
        "use_compile": False, "plot": False,
        "output_dir": str(tmp_path / "r"), "task_id": "wf_e2e",
    })
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 120})
    assert r.value.get("state") == "success", r.value
