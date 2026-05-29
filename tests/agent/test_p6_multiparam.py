"""P6: multi-parameter / elastic / anisotropic / free-surface / 2D-3D forward.

Validates that build_forward_spec auto-selects the right source_type / pml_type
per equation and auto-generates 3-D explicit geometry, so the whole capability
matrix runs end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


def _mk(path, shape, val):
    np.save(path, np.full(shape, float(val), dtype=np.float32))
    return str(path)


# --- unit: auto source_type / pml_type / 3D geometry -----------------------

def test_elastic_auto_source_and_pml(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _mk(tmp_path / "vp.npy", (40, 48), 1800)
    vs = _mk(tmp_path / "vs.npy", (40, 48), 1000)
    rho = _mk(tmp_path / "rho.npy", (40, 48), 2000)
    res = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 10.0, "dt": 0.0015, "nt": 30, "equation": "Elastic",
        "extra_models": {"vs": vs, "rho": rho}, "output_dir": str(tmp_path / "r"), "task_id": "el",
    })
    assert res.ok and "error" not in res.value, res.value
    phys = res.value["spec"]["physics"]
    assert phys["source_type"] == ["sxx", "szz"]
    assert phys["receiver_type"] == ["vx", "vz"]
    assert phys["pml_type"] == "cpmls"


def test_3d_auto_explicit_geometry(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _mk(tmp_path / "vp.npy", (16, 16, 16), 1800)
    res = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 10.0, "dt": 0.0015, "nt": 30, "equation": "Acoustic3D",
        "source_step": 40, "receiver_step": 8, "output_dir": str(tmp_path / "r"), "task_id": "a3d",
    })
    assert res.ok and "error" not in res.value, res.value
    geo = res.value["spec"]["geometry"]
    assert geo["kind"] == "explicit"
    assert len(geo["sources"][0]) == 3  # (x, y, z)


# --- end-to-end: representative multi-param / 3D runs -----------------------

@pytest.mark.timeout(300)
@pytest.mark.parametrize("label,equation,shape,extra", [
    ("elastic_2d",  "Elastic",     (40, 48),     {"vs": 1000.0, "rho": 2000.0}),
    ("vti_2d",      "AcousticVTI", (40, 48),     {"epsilon": 0.1, "delta": 0.05}),
    ("acoustic_3d", "Acoustic3D",  (16, 16, 16), {}),
    ("elastic_3d",  "Elastic3D",   (16, 16, 16), {"vs": 1000.0, "rho": 2000.0}),
])
def test_multiparam_forward_endtoend(tmp_path, label, equation, shape, extra):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _mk(tmp_path / f"{label}_vp.npy", shape, 1800)
    extra_models = {n: _mk(tmp_path / f"{label}_{n}.npy", shape, v) for n, v in extra.items()}
    b = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 10.0, "dt": 0.0015, "nt": 30, "fm": 12.0,
        "equation": equation, "extra_models": extra_models or None,
        "source_step": 40, "source_depth": 2, "receiver_step": 8, "receiver_depth": 2,
        "spatial_order": 4, "abcn": 8, "device": "cpu", "backend_impl": "eager",
        "use_compile": False, "output_dir": str(tmp_path / "runs"), "task_id": label,
    })
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 250})
    assert r.value.get("state") == "success", r.value
