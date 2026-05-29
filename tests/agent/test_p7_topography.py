"""P7: topography — user-supplied irregular free-surface via curvilinear grid."""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


def _hill_topo(path, nx, base=12, amp=6, width=18):
    x = np.arange(nx)
    surf = np.round(base - amp * np.exp(-((x - nx / 2) ** 2) / (2.0 * width**2))).astype(np.int64)
    np.save(path, np.clip(surf, 0, nx))
    return str(path)


def test_topography_wired_into_spec(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((80, 120), 2500.0, dtype=np.float32))
    topo = _hill_topo(tmp_path / "topo.npy", 120)
    res = registry.get("build_forward_spec").invoke({
        "vp_path": str(vp), "dh": 10.0, "dt": 0.001, "nt": 40,
        "equation": "AcousticCurvilinear", "topography": topo, "free_surface": True,
        "output_dir": str(tmp_path / "r"), "task_id": "topo",
    })
    assert res.ok and "error" not in res.value, res.value
    assert res.value["spec"]["physics"]["topography"].endswith("topo.npy")


@pytest.mark.timeout(200)
def test_topography_forward_endtoend(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    nz, nx = 80, 120
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((nz, nx), 2500.0, dtype=np.float32))
    topo = _hill_topo(tmp_path / "topo.npy", nx, base=14, amp=6, width=20)
    b = registry.get("build_forward_spec").invoke({
        "vp_path": str(vp), "dh": 10.0, "dt": 0.001, "nt": 60, "fm": 10.0,
        "equation": "AcousticCurvilinear", "topography": topo, "free_surface": True,
        "source_step": 60, "source_depth": 3, "receiver_step": 6, "receiver_depth": 2,
        "spatial_order": 4, "abcn": 15, "device": "cpu", "backend_impl": "eager",
        "use_compile": False, "output_dir": str(tmp_path / "runs"), "task_id": "topo_e2e",
    })
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 150})
    assert r.value.get("state") == "success", r.value
