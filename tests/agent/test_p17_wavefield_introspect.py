"""P17: wavefield equation coverage + schema introspection breadth.

build_wavefield_spec across equation families (validation), and
describe_task_schema / list_equations exercised on their nested variants so the
discovery path the LLM relies on stays correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


WAVEFIELD_EQS = ["Acoustic", "AcousticVTI", "AcousticTTI", "Elastic"]


@pytest.mark.parametrize("equation", WAVEFIELD_EQS)
def test_build_wavefield_equation_families(tmp_path, equation):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    from sweep_agent.tools.introspect import all_equations
    required = all_equations()[equation]
    shape = (48, 64)
    defaults = {"vp": 2500.0, "vs": 1300.0, "rho": 2200.0, "epsilon": 0.1, "delta": 0.05, "theta": 0.3}
    paths = {m: str(tmp_path / f"{m}.npy") for m in required}
    for m, p in paths.items():
        np.save(p, np.full(shape, np.float32(defaults.get(m, 1.0))))
    extra = {m: paths[m] for m in required[1:]}
    r = registry.get("build_wavefield_spec").invoke({
        "vp_path": paths[required[0]], "equation": equation, "extra_models": extra or None,
        "dh": 10.0, "dt": 1e-3, "nt": 80, "fm": 12.0, "snapshot_times": [20, 50, 79],
        "device": "cpu", "backend_impl": "eager", "use_compile": False,
        "output_dir": str(tmp_path / "runs"), "task_id": f"wf_{equation}",
    })
    assert r.ok and "error" not in r.value, r.value
    assert r.value["spec"]["snapshot_times"] == [20, 50, 79]
    assert r.value["spec"]["physics"]["equation"] == equation


def test_snapshot_times_out_of_range(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((40, 40), 2500.0, dtype=np.float32))
    r = registry.get("build_wavefield_spec").invoke({
        "vp_path": str(vp), "dh": 10.0, "dt": 1e-3, "nt": 50, "fm": 12.0,
        "snapshot_times": [10, 60], "output_dir": str(tmp_path / "runs"),
    })
    assert "error" in r.value and "out of range" in r.value["error"]


def test_describe_task_schema_sections():
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    # optimizer section should expose the adam/sgd/lbfgs variants.
    r = registry.get("describe_task_schema").invoke({"task_type": "fwi", "section": "optimizer"})
    assert r.ok and "error" not in r.value, r.value
    blob = str(r.value).lower()
    assert "adam" in blob and ("lbfgs" in blob or "sgd" in blob)


def test_describe_task_schema_geometry_variants():
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = registry.get("describe_task_schema").invoke({"task_type": "forward", "section": "geometry"})
    assert r.ok and "error" not in r.value, r.value
    blob = str(r.value).lower()
    assert "line" in blob and "explicit" in blob


def test_list_equations_filter():
    pytest.importorskip("sweep")
    r = registry.get("list_equations").invoke({"filter": "elastic"})
    assert r.ok and "error" not in r.value, r.value
    blob = str(r.value)
    assert "Elastic" in blob
