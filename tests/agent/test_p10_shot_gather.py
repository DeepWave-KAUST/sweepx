"""P10: plot_shot_gather + run_task yaml_path fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


def _build_forward(tmp_path):
    vp = tmp_path / "vp.npy"
    v = np.full((40, 56), 1800.0, dtype=np.float32)
    v[20:, :] = 2600.0
    np.save(vp, v)
    b = registry.get("build_forward_spec").invoke({
        "vp_path": str(vp), "dh": 12.5, "dt": 0.001, "nt": 200, "fm": 12.0,
        "source_step": 28, "source_depth": 2, "receiver_step": 2, "receiver_depth": 4,
        "spatial_order": 4, "abcn": 10, "device": "cpu", "backend_impl": "eager",
        "use_compile": False, "output_dir": str(tmp_path / "runs"), "task_id": "fwd",
    })
    assert b.ok and "error" not in b.value, b.value
    return b.value["yaml_path"]


@pytest.mark.timeout(150)
def test_plot_shot_gather(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    yaml_path = _build_forward(tmp_path)
    r = registry.get("run_task").invoke({"yaml_path": yaml_path, "timeout_s": 120})
    assert r.value.get("state") == "success", r.value
    g = registry.get("plot_shot_gather").invoke({
        "task_dir": r.value["task_dir"], "dt": 0.001, "dh": 12.5, "title": "gather",
    })
    assert g.ok and "error" not in g.value, g.value
    assert Path(g.value["image_path"]).is_file()
    assert g.value["shape"] == [200, 28]  # (nt, nrec); receiver_step=2 over nx=56 → 28 traces


@pytest.mark.timeout(150)
def test_run_task_yaml_fallback(tmp_path):
    """A hallucinated yaml_path under the same output dir resolves to the real one."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    yaml_path = _build_forward(tmp_path)
    wrong = str(Path(yaml_path).parent / "fake-timestamp" / "forward.yaml")
    r = registry.get("run_task").invoke({"yaml_path": wrong, "timeout_s": 120})
    assert r.value.get("state") == "success", r.value


def test_plot_shot_gather_no_record(tmp_path):
    pytest.importorskip("sweep_tasks")
    res = registry.get("plot_shot_gather").invoke({"task_dir": str(tmp_path / "nope")})
    assert "error" in res.value
