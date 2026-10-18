"""P2.5: equation discovery (list_equations) + multi-model build + validation."""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


def _save(path, value, shape=(32, 40)):
    np.save(path, np.full(shape, value, dtype=np.float32))
    return str(path)


# --- list_equations --------------------------------------------------------

def test_list_equations_returns_real_set():
    pytest.importorskip("sweep")
    res = registry.get("list_equations").invoke({})
    assert res.ok, res.error
    val = res.value
    assert val["count"] >= 20
    eqs = val["equations"]
    assert eqs["Acoustic"]["models"] == ["vp"]
    assert eqs["AcousticVTI"]["models"] == ["vp", "epsilon", "delta"]
    assert eqs["Elastic"]["models"] == ["vp", "vs", "rho"]


def test_list_equations_filter():
    pytest.importorskip("sweep")
    res = registry.get("list_equations").invoke({"filter": "vti"})
    assert res.ok
    assert res.value["count"] >= 1
    assert all("vti" in k.lower() for k in res.value["equations"])


# --- build_forward_spec validation ----------------------------------------

def test_build_rejects_unknown_equation(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _save(tmp_path / "vp.npy", 1800.0)
    res = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 50,
        "equation": "VTI", "output_dir": str(tmp_path / "r"),
    })
    assert res.ok  # function returns an error dict, doesn't raise
    assert "error" in res.value
    assert "unknown equation" in res.value["error"].lower()
    assert "AcousticVTI" in res.value["error"]  # difflib suggestion


def test_build_reports_missing_models(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _save(tmp_path / "vp.npy", 1800.0)
    res = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 50,
        "equation": "AcousticVTI", "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value
    assert "epsilon" in res.value["error"]
    assert "delta" in res.value["error"]


def test_build_rejects_unneeded_models(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _save(tmp_path / "vp.npy", 1800.0)
    rho = _save(tmp_path / "rho.npy", 1000.0)
    res = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 50,
        "equation": "Acoustic", "extra_models": {"rho": rho},  # Acoustic needs vp only
        "output_dir": str(tmp_path / "r"),
    })
    assert "error" in res.value
    assert "rho" in res.value["error"]


def test_build_multimodel_vti_ordered(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _save(tmp_path / "vp.npy", 1800.0)
    eps = _save(tmp_path / "eps.npy", 0.05)
    dlt = _save(tmp_path / "dlt.npy", 0.03)
    res = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 50,
        "equation": "AcousticVTI",
        "extra_models": {"delta": dlt, "epsilon": eps},  # deliberately out of order
        "output_dir": str(tmp_path / "r"), "task_id": "vti_build",
    })
    assert res.ok and "error" not in res.value, res.value
    models = res.value["spec"]["models"]
    # Must be reordered to MODEL_SPECS order: vp, epsilon, delta
    assert [m["name"] for m in models] == ["vp", "epsilon", "delta"]


# --- end-to-end multi-model VTI forward (CPU eager) ------------------------

@pytest.mark.timeout(180)
def test_vti_forward_endtoend_cpu(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = _save(tmp_path / "vp.npy", 1800.0)
    eps = _save(tmp_path / "eps.npy", 0.05)
    dlt = _save(tmp_path / "dlt.npy", 0.03)
    out = tmp_path / "runs"

    build = registry.get("build_forward_spec").invoke({
        "vp_path": vp, "dh": 12.5, "dt": 0.001, "nt": 60, "fm": 15.0,
        "equation": "AcousticVTI",
        "extra_models": {"epsilon": eps, "delta": dlt},
        "source_step": 20, "source_depth": 2, "receiver_step": 5, "receiver_depth": 8,
        "spatial_order": 4, "abcn": 10, "device": "cpu", "backend_impl": "eager",
        "use_compile": False, "output_dir": str(out), "task_id": "vti_e2e",
    })
    assert build.ok and "error" not in build.value, build.value

    run = registry.get("run_task").invoke({
        "yaml_path": build.value["yaml_path"], "timeout_s": 150,
    })
    assert run.value.get("state") == "success", run.value
