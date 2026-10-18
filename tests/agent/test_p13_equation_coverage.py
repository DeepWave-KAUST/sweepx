"""P13: broad equation coverage — build_forward_spec validates a spec for every
representative equation family (2-D, 3-D, 1st-order, VTI/TTI variants, elastic,
DAS), with the right model set resolved from introspection.

These are validation-only (no propagation) so the whole equation zoo is covered
cheaply; a couple of families are also smoke-run in the other P-tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


# Representative equations spanning every family. 3-D ones get 3-D dummy models.
EQUATIONS_2D = [
    "Acoustic", "Acoustic1st", "AcousticVRZ",
    "AcousticVTI", "AcousticVTI1st", "AcousticVTILiang", "AcousticVTIDuveneck",
    "AcousticVTIDefault", "AcousticTTI", "AcousticTTILiang", "AcousticTariq",
    "Elastic", "ElasticAPM", "ElasticTTI", "ElasticTTISG",
    "DASElastic", "DASMu", "DASZhao",
]
EQUATIONS_3D = ["Acoustic3D", "AcousticVTIDuveneck3D", "Elastic3D"]


def _make_model_files(tmp_path, names, shape):
    """Create a dummy .npy for each model name with a plausible constant value."""
    defaults = {
        "vp": 2500.0, "vp0": 2500.0, "vv": 2500.0, "v": 1500.0,
        "vs": 1400.0, "vs0": 1400.0, "rho": 2200.0,
        "epsilon": 0.1, "delta": 0.05, "gamma": 0.1, "eta": 0.1,
        "theta": 0.3, "phi": 0.0,
    }
    paths = {}
    for n in names:
        p = tmp_path / f"{n}.npy"
        np.save(p, np.full(shape, np.float32(defaults.get(n, 1.0))))
        paths[n] = str(p)
    return paths


@pytest.mark.parametrize("equation", EQUATIONS_2D)
def test_build_forward_spec_2d_equations(tmp_path, equation):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    from sweep_agent.tools.introspect import all_equations
    required = all_equations()[equation]
    paths = _make_model_files(tmp_path, required, (48, 64))
    primary = required[0]
    extra = {m: paths[m] for m in required[1:]}
    r = registry.get("build_forward_spec").invoke({
        "vp_path": paths[primary], "equation": equation,
        "extra_models": extra or None,
        "dh": 10.0, "dt": 1e-3, "nt": 50, "fm": 12.0,
        "device": "cpu", "backend_impl": "eager", "use_compile": False,
        "output_dir": str(tmp_path / "runs"), "task_id": f"fwd_{equation}",
    })
    assert r.ok and "error" not in r.value, r.value
    spec = r.value["spec"]
    # The equation's models came out in canonical order with vp_path as primary.
    assert [m["name"] for m in spec["models"]] == required
    assert spec["models"][0]["path"] == paths[primary]
    assert spec["physics"]["equation"] == equation


@pytest.mark.parametrize("equation", EQUATIONS_3D)
def test_build_forward_spec_3d_equations(tmp_path, equation):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    from sweep_agent.tools.introspect import all_equations
    required = all_equations()[equation]
    paths = _make_model_files(tmp_path, required, (16, 18, 20))  # (nz, ny, nx)
    primary = required[0]
    extra = {m: paths[m] for m in required[1:]}
    r = registry.get("build_forward_spec").invoke({
        "vp_path": paths[primary], "equation": equation,
        "extra_models": extra or None,
        "dh": 10.0, "dt": 1e-3, "nt": 40, "fm": 12.0,
        "source_step": 8, "receiver_step": 4,
        "device": "cpu", "backend_impl": "eager", "use_compile": False,
        "output_dir": str(tmp_path / "runs"), "task_id": f"fwd_{equation}",
    })
    assert r.ok and "error" not in r.value, r.value
    geom = r.value["spec"]["geometry"]
    assert geom["kind"] == "explicit"           # 3-D auto-generates explicit geometry
    assert len(geom["sources"][0]) == 3         # (x, y, z)


def test_unknown_equation_suggests(tmp_path):
    pytest.importorskip("sweep")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((40, 40), 2500.0, dtype=np.float32))
    r = registry.get("build_forward_spec").invoke({
        "vp_path": str(vp), "equation": "Elastik",  # typo
        "dh": 10.0, "dt": 1e-3, "nt": 40,
        "output_dir": str(tmp_path / "runs"),
    })
    assert "error" in r.value and "list_equations" in r.value["error"]
