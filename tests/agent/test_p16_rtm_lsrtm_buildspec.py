"""P16: generic build_spec validates every task_type — including RTM and LSRTM,
which have no flat builder and must go through the schema-driven path.

RTM/LSRTM are not run here: their forward/adjoint imaging needs the CUDA
c-backend + an acquisition plan, so (like sweep-tasks' own smoke tests) we
validate the spec only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sweep_agent.tools import registry


@pytest.mark.parametrize("task_type", ["forward", "wavefield", "fwi", "rtm", "lsrtm"])
def test_build_spec_validates_templates(tmp_path, task_type):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    from sweep_tasks import new_template

    spec = dict(new_template(task_type))
    spec.pop("task_type", None)
    spec["output_dir"] = str(tmp_path / "runs")
    r = registry.get("build_spec").invoke({"task_type": task_type, "spec": spec})
    assert r.ok and "error" not in r.value, r.value
    assert r.value.get("yaml_path") and Path(r.value["yaml_path"]).is_file()
    assert r.value["spec"]["task_type"] == task_type


def test_build_spec_rtm_imaging_block(tmp_path):
    """RTM carries an imaging block distinct from forward/fwi."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    from sweep_tasks import new_template

    spec = dict(new_template("rtm"))
    spec.pop("task_type", None)
    spec["output_dir"] = str(tmp_path / "runs")
    r = registry.get("build_spec").invoke({"task_type": "rtm", "spec": spec})
    assert r.ok and "error" not in r.value, r.value
    assert "imaging" in r.value["spec"]
    assert "velocity_model" in r.value["spec"]


def test_build_spec_invalid_returns_error(tmp_path):
    """A structurally wrong spec comes back as a data error, not an exception."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = registry.get("build_spec").invoke({
        "task_type": "forward",
        "spec": {"grid": {"dh": "not-a-number"}, "output_dir": str(tmp_path)},
    })
    assert "error" in r.value


def test_build_spec_unknown_task_type(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = registry.get("build_spec").invoke({
        "task_type": "frobnicate", "spec": {"output_dir": str(tmp_path)},
    })
    assert "error" in r.value
