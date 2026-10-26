"""``$VAR`` in a task YAML: expanded in every string, refused when not set.

Only the keys of ``RESOLVED_PATH_KEYS`` (and ``water: {seabed | mask}``) are
re-based on the YAML's directory; that set is a contract with sweep-studio
(``test_resolved_keys_are_exactly_the_published_set``). Expansion is not
limited to it: ``geometry.plan_path: ${VIKING_HOME}/run/plan.npz`` used to reach
``SeismicPlan.load`` with the ``${VIKING_HOME}`` still in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sweepx.tasks import load_task, load_task_from_dict
from sweepx.tasks.cli import main as cli_main
from sweepx.tasks.yaml_io import resolve_paths

_UNSET = "SWEEPX_TEST_SURELY_UNSET_VAR"


def _rtm_doc(var: str) -> dict:
    """A field-style RTM whose every path hangs off ``$var`` (both spellings)."""
    return {
        "task_type": "rtm",
        "output_dir": f"${{{var}}}/run/rtm",
        "grid": {"dh": 12.5},
        "time": {"dt": 0.001, "nt": 100},
        "wavelet": {"kind": "ricker", "fm": 8.0},
        "geometry": {"kind": "from_plan", "plan_path": f"${{{var}}}/run/plan.npz"},
        "obs": {"plan": {"plan_path": f"${var}/run/plan.npz"}},
        "physics": {"equation": "Acoustic"},
        "velocity_model": {"name": "vp", "path": f"${{{var}}}/vp.npy"},
    }


def test_every_string_expands_but_only_path_keys_rebase(tmp_path, monkeypatch):
    monkeypatch.setenv("SX_HOME", "/demo/survey")
    doc = {
        "geometry": {"plan_path": "${SX_HOME}/plan.npz", "index_path": "$SX_HOME/i.npz"},
        "obs": {"plan": {"plan_path": "rel/plan.npz"}},
        "loss": {"data_mask_path": "${SX_HOME}/mask.npy"},
        "output_dir": "${SX_HOME}/runs",
        "models": [{"name": "vp", "path": "vp.npy"}],
        "water": {"seabed": "$SX_HOME/sb.npz", "mode": "pin"},
        "notes": ["${SX_HOME}/a", 3],
    }
    out = resolve_paths(doc, tmp_path)
    assert out["geometry"] == {"plan_path": "/demo/survey/plan.npz",
                               "index_path": "/demo/survey/i.npz"}
    assert out["obs"]["plan"]["plan_path"] == "rel/plan.npz"     # not re-based: cwd-relative
    assert out["loss"]["data_mask_path"] == "/demo/survey/mask.npy"
    assert out["output_dir"] == "/demo/survey/runs"
    assert out["models"][0]["path"] == str(tmp_path.resolve() / "vp.npy")
    assert out["water"] == {"seabed": "/demo/survey/sb.npz", "mode": "pin"}
    assert out["notes"] == ["/demo/survey/a", 3]


def test_plan_path_variable_reaches_the_spec_expanded(tmp_path, monkeypatch):
    home = tmp_path / "viking"
    monkeypatch.setenv("SX_HOME", str(home))
    spec = load_task_from_dict(_rtm_doc("SX_HOME"), base_dir=tmp_path)
    assert str(spec.geometry.plan_path) == str(home / "run" / "plan.npz")
    assert str(spec.obs.plan.plan_path) == str(home / "run" / "plan.npz")
    assert str(spec.output_dir) == str(home / "run" / "rtm")


def test_resolve_paths_leaves_an_unset_variable_as_written(tmp_path, monkeypatch):
    """The studio previews raw documents with resolve_paths; it does not raise."""
    monkeypatch.delenv(_UNSET, raising=False)
    out = resolve_paths({"geometry": {"plan_path": f"${{{_UNSET}}}/p.npz"}}, tmp_path)
    assert out["geometry"]["plan_path"] == f"${{{_UNSET}}}/p.npz"


def test_an_unset_variable_is_refused_at_load(tmp_path, monkeypatch):
    monkeypatch.delenv(_UNSET, raising=False)
    with pytest.raises(ValidationError) as err:
        load_task_from_dict(_rtm_doc(_UNSET), base_dir=tmp_path)
    errors = err.value.errors()
    assert {e["type"] for e in errors} == {"env_var_not_set"}
    assert {e["loc"] for e in errors} == {
        ("output_dir",), ("geometry", "plan_path"), ("obs", "plan", "plan_path"),
        ("velocity_model", "path")}
    assert all(_UNSET in e["msg"] for e in errors)


def test_run_with_an_unset_variable_creates_no_literal_directory(tmp_path, monkeypatch):
    """`sweep-tasks run` used to create ``<yaml dir>/${VAR}/...`` and write the
    run metadata there before failing on the first missing input."""
    monkeypatch.delenv(_UNSET, raising=False)
    task = tmp_path / "t.yaml"
    task.write_text(yaml.safe_dump({"task_type": "introspect",
                                    "output_dir": f"${{{_UNSET}}}/runs",
                                    "action": "list_equations"}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError, match=_UNSET):
        cli_main(["run", str(task)])
    with pytest.raises(ValidationError, match=_UNSET):
        cli_main(["run", str(task), "--override", "task_id=x"])
    with pytest.raises(ValidationError, match=_UNSET):
        load_task(task)
    assert not any("$" in p.name for p in tmp_path.rglob("*"))


def test_a_set_variable_runs_into_its_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SX_RUNS", str(tmp_path / "runs_here"))
    task = tmp_path / "t.yaml"
    task.write_text(yaml.safe_dump({"task_type": "introspect", "output_dir": "${SX_RUNS}",
                                    "task_id": "env_ok", "action": "list_equations"}))
    assert cli_main(["run", str(task)]) == 0
    assert (tmp_path / "runs_here" / "env_ok" / "status.json").is_file()


def test_config_with_an_unset_variable_is_refused(tmp_path, monkeypatch, capsys):
    """`--config` pipeline files (`build-index`, `build-plan`, ...) follow the
    same rule: `out: ${VAR}/index.npz` with VAR unset used to write under a
    literal `${VAR}` directory."""
    monkeypatch.delenv(_UNSET, raising=False)
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(yaml.safe_dump({"build_index": {
        "segy_root": f"${{{_UNSET}}}/raw", "out": f"${{{_UNSET}}}/run/index.npz"}}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli_main(["build-index", "--config", str(cfg)])
    assert exc.value.code == 2
    assert _UNSET in capsys.readouterr().out
    assert not any("$" in p.name for p in tmp_path.rglob("*"))


def test_config_expands_a_set_variable(tmp_path, monkeypatch):
    import argparse

    from sweepx.tasks.cli import _merge_yaml_into_args

    monkeypatch.setenv("SX_HOME", str(tmp_path))
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(yaml.safe_dump({"step": {"out": "${SX_HOME}/run/index.npz"}}))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--out")
    args = parser.parse_args(["--config", str(cfg)])
    _merge_yaml_into_args(args, parser, "step")
    assert args.out == f"{tmp_path}/run/index.npz"
