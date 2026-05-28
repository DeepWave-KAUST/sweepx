"""P3: schema-driven full-coverage layer (describe_task_schema + build_spec)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


# --- describe_task_schema --------------------------------------------------

def test_describe_top_level_fwi():
    pytest.importorskip("sweep_tasks")
    res = registry.get("describe_task_schema").invoke({"task_type": "fwi"})
    assert res.ok, res.error
    fields = res.value["fields"]
    for f in ("optimizer", "loss", "epochs", "obs", "grid", "time", "wavelet", "geometry", "physics"):
        assert f in fields, f
    assert fields["epochs"]["required"] is True


def test_describe_section_optimizer():
    pytest.importorskip("sweep_tasks")
    res = registry.get("describe_task_schema").invoke({"task_type": "fwi", "section": "optimizer"})
    assert res.ok
    variants = res.value["variants"]
    assert any("Adam" in k for k in variants)
    assert any("LBFGS" in k for k in variants)


def test_describe_section_geometry_variants():
    pytest.importorskip("sweep_tasks")
    res = registry.get("describe_task_schema").invoke({"task_type": "forward", "section": "geometry"})
    assert res.ok
    variants = res.value["variants"]
    assert any("Line" in k for k in variants)
    assert len(variants) >= 3  # line + explicit + from_file + segy variants...


def test_describe_unknown_task():
    res = registry.get("describe_task_schema").invoke({"task_type": "nope"})
    assert "error" in res.value


# --- build_spec ------------------------------------------------------------

def _forward_spec_dict(vp_path, nt=50):
    return {
        "grid": {"dh": 12.5},
        "time": {"dt": 0.001, "nt": nt},
        "wavelet": {"kind": "ricker", "fm": 10.0, "delay": 0.1},
        "geometry": {"kind": "line", "sources": {"step": 20, "depth": 1}, "receivers": {"step": 5, "depth": 8}},
        "physics": {"equation": "Acoustic"},
        "backend": {"impl": "eager", "eager_options": {"use_compile": False}},
        "models": [{"name": "vp", "path": str(vp_path)}],
        "device": "cpu",
    }


def test_build_spec_forward_roundtrip(tmp_path):
    pytest.importorskip("sweep_tasks")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((32, 40), 1800.0, dtype=np.float32))
    res = registry.get("build_spec").invoke({
        "task_type": "forward", "spec": _forward_spec_dict(vp),
        "output_dir": str(tmp_path / "r"), "task_id": "gen_fwd",
    })
    assert res.ok and "error" not in res.value, res.value
    assert Path(res.value["yaml_path"]).is_file()
    assert res.value["spec"]["task_type"] == "forward"


def test_build_spec_validation_error(tmp_path):
    pytest.importorskip("sweep_tasks")
    res = registry.get("build_spec").invoke({
        "task_type": "forward", "spec": {"grid": {"dh": 12.5}},  # missing time/wavelet/geometry/physics/models
        "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value
    assert "spec_attempted" in res.value


@pytest.mark.timeout(120)
def test_build_spec_forward_e2e(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((32, 40), 1800.0, dtype=np.float32))
    spec = _forward_spec_dict(vp, nt=60)
    spec["physics"] = {"equation": "Acoustic", "spatial_order": 4, "abcn": 10}
    spec["wavelet"]["fm"] = 15.0
    b = registry.get("build_spec").invoke({
        "task_type": "forward", "spec": spec,
        "output_dir": str(tmp_path / "r"), "task_id": "gen_e2e",
    })
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 90})
    assert r.value.get("state") == "success", r.value
