"""P14: forward-modelling parameter coverage — free surface, source/receiver/pml
overrides, wavelet knobs, abcn, spatial_order, plus tiny end-to-end runs for the
acoustic and elastic paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


def _vp(tmp_path, shape=(48, 64), val=2000.0):
    p = tmp_path / "vp.npy"
    np.save(p, np.full(shape, np.float32(val)))
    return str(p)


def _build(tmp_path, **over):
    args = {
        "vp_path": _vp(tmp_path), "dh": 10.0, "dt": 1e-3, "nt": 60, "fm": 12.0,
        "device": "cpu", "backend_impl": "eager", "use_compile": False,
        "output_dir": str(tmp_path / "runs"),
    }
    args.update(over)
    return registry.get("build_forward_spec").invoke(args)


def test_free_surface_flag(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = _build(tmp_path, free_surface=True, task_id="fs")
    assert r.ok and "error" not in r.value, r.value
    assert r.value["spec"]["physics"]["free_surface"] is True


def test_wavelet_delay_and_scale(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = _build(tmp_path, wavelet_delay=0.08, wavelet_scale=2.5, task_id="wav")
    assert r.ok and "error" not in r.value, r.value
    w = r.value["spec"]["wavelet"]
    assert abs(w["delay"] - 0.08) < 1e-9 and abs(w["scale"] - 2.5) < 1e-9


def test_abcn_and_spatial_order(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = _build(tmp_path, abcn=40, spatial_order=6, task_id="pml")
    assert r.ok and "error" not in r.value, r.value
    assert r.value["spec"]["physics"]["abcn"] == 40
    assert r.value["spec"]["physics"]["spatial_order"] == 6


def test_source_receiver_pml_override_elastic(tmp_path):
    """Elastic with an explicit single-component receiver + explicit pml."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    shape = (48, 64)
    for n, v in (("vs", 1200.0), ("rho", 2200.0)):
        np.save(tmp_path / f"{n}.npy", np.full(shape, np.float32(v)))
    r = _build(
        tmp_path, equation="Elastic",
        extra_models={"vs": str(tmp_path / "vs.npy"), "rho": str(tmp_path / "rho.npy")},
        source_type=["sxx", "szz"], receiver_type=["vz"], pml_type="cpmls", task_id="el_over",
    )
    assert r.ok and "error" not in r.value, r.value
    phys = r.value["spec"]["physics"]
    assert phys["receiver_type"] == ["vz"]
    assert phys["pml_type"] == "cpmls"


def test_equation_default_pml_autofills_elastic(tmp_path):
    """Without an explicit pml_type, Elastic must auto-resolve to its own default
    (cpmls) — the plain-acoustic default would crash the elastic solver."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    shape = (48, 64)
    for n, v in (("vs", 1200.0), ("rho", 2200.0)):
        np.save(tmp_path / f"{n}.npy", np.full(shape, np.float32(v)))
    r = _build(
        tmp_path, equation="Elastic",
        extra_models={"vs": str(tmp_path / "vs.npy"), "rho": str(tmp_path / "rho.npy")},
        task_id="el_def",
    )
    assert r.ok and "error" not in r.value, r.value
    assert r.value["spec"]["physics"]["pml_type"] == "cpmls"


@pytest.mark.timeout(150)
def test_forward_run_acoustic_free_surface(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    b = _build(tmp_path, free_surface=True, source_step=40, receiver_step=2, nt=200, task_id="fsrun")
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 120})
    assert r.value.get("state") == "success", r.value


@pytest.mark.timeout(180)
def test_forward_run_elastic(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    shape = (48, 64)
    for n, v in (("vs", 1200.0), ("rho", 2200.0)):
        np.save(tmp_path / f"{n}.npy", np.full(shape, np.float32(v)))
    b = _build(
        tmp_path, equation="Elastic",
        extra_models={"vs": str(tmp_path / "vs.npy"), "rho": str(tmp_path / "rho.npy")},
        source_step=40, receiver_step=2, nt=200, task_id="elrun",
    )
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 150})
    assert r.value.get("state") == "success", r.value
    assert Path(r.value["task_dir"]).is_dir()
