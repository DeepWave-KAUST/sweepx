"""P11: animate_wavefield orchestration + equations_for_models + vp0 primary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


def test_equations_for_models_ranking():
    pytest.importorskip("sweep")
    from sweep_agent.tools.introspect import equations_for_models

    # Matched on the NON-primary models (vp / vp0 come from vp_path).
    assert equations_for_models({"vs", "rho"}, ndim=2)[0] == "Elastic"
    assert equations_for_models({"vs", "rho"}, ndim=3)[0] == "Elastic3D"
    assert equations_for_models({"epsilon", "delta"})[0] == "AcousticVTI"
    assert equations_for_models({"epsilon", "delta", "theta"})[0] == "AcousticTTI"
    # vp0-primary equation: its non-primary set resolves to ElasticTTI.
    etti = {"vs0", "rho", "epsilon", "delta", "gamma", "theta", "phi"}
    assert equations_for_models(etti)[0] == "ElasticTTI"


def test_build_wavefield_vp0_primary(tmp_path):
    """vp_path supplies the equation's FIRST model even when it's named vp0."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    shape = (60, 60)

    def npy(name, val):
        p = tmp_path / f"{name}.npy"
        np.save(p, np.full(shape, np.float32(val)))
        return str(p)

    vp0 = npy("vp0", 2400.0)
    extra = {
        "vs0": npy("vs0", 1200.0), "rho": npy("rho", 2200.0),
        "epsilon": npy("eps", 0.2), "delta": npy("del", 0.05),
        "gamma": npy("gam", 0.1), "theta": npy("theta", 0.5), "phi": npy("phi", 0.0),
    }
    b = registry.get("build_wavefield_spec").invoke({
        "vp_path": vp0, "equation": "ElasticTTI", "extra_models": extra,
        "dh": 10.0, "dt": 8e-4, "nt": 40, "fm": 18.0, "snapshot_times": [20, 39],
        "abcn": 10, "spatial_order": 4, "source_at_center": True,
        "device": "cpu", "backend_impl": "eager", "use_compile": False,
        "output_dir": str(tmp_path / "runs"), "task_id": "etti",
    })
    assert b.ok and "error" not in b.value, b.value
    models = b.value["spec"]["models"]
    # First model is vp0 and points at the vp_path file.
    assert models[0]["name"] == "vp0" and models[0]["path"] == vp0
    assert [m["name"] for m in models] == ["vp0", "vs0", "rho", "epsilon", "delta", "gamma", "theta", "phi"]


def test_wrong_equation_for_extra_models_hint(tmp_path):
    """Leaving equation=Acoustic but passing vs+rho suggests Elastic, not 'remove'."""
    pytest.importorskip("sweep")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((40, 40), 2500.0, dtype=np.float32))
    for n in ("vs", "rho"):
        np.save(tmp_path / f"{n}.npy", np.full((40, 40), 1500.0, dtype=np.float32))
    r = registry.get("build_wavefield_spec").invoke({
        "vp_path": str(vp), "equation": "Acoustic",
        "extra_models": {"vs": str(tmp_path / "vs.npy"), "rho": str(tmp_path / "rho.npy")},
        "dh": 10.0, "dt": 1e-3, "nt": 20, "fm": 12.0, "snapshot_times": [10],
        "output_dir": str(tmp_path / "runs"),
    })
    assert "error" in r.value
    assert "Elastic" in r.value["error"], r.value["error"]


@pytest.mark.timeout(180)
def test_animate_wavefield_acoustic(tmp_path):
    """End-to-end one-call animation on a tiny acoustic model."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((60, 60), 2000.0, dtype=np.float32))
    a = registry.get("animate_wavefield").invoke({
        "vp_path": str(vp), "out_path": str(tmp_path / "wave.gif"),
        "dh": 10.0, "dt": 1e-3, "nt": 80, "fm": 12.0, "abcn": 10,
        "n_frames": 6, "device": "cpu", "output_dir": str(tmp_path / "runs"),
    })
    assert a.ok and "error" not in a.value, a.value
    assert a.value["equation"] == "Acoustic"
    assert Path(a.value["gif_path"]).is_file()
    assert a.value["n_frames"] >= 2
