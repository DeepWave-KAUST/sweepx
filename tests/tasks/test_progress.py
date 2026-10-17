"""``<task_dir>/metrics.jsonl``: format of the log and what each task writes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from sweep_tasks import TaskRunner, load_task
from sweep_tasks.progress import METRICS_FILE, ProgressLog


def _strict_loads(line: str) -> dict:
    """``json.loads`` that refuses the non-standard NaN/Infinity tokens."""
    def _reject(token):
        raise AssertionError(f"non-standard JSON token {token!r} in {line!r}")
    return json.loads(line, parse_constant=_reject)


def _events(task_dir: Path) -> list[dict]:
    text = (task_dir / METRICS_FILE).read_text()
    assert text.endswith("\n")
    return [_strict_loads(line) for line in text.splitlines()]


def _write(doc: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


# ---------- ProgressLog --------------------------------------------------


def test_event_line_format(tmp_path):
    log = ProgressLog(tmp_path)
    log.event("iter", stage=0, epoch=3, loss=1.5, lr=np.float32(0.25),
              shape=(2, np.int64(3)), band=None, done=np.bool_(True))
    log.event("end", state="success", error=None)
    ev = _events(tmp_path)
    assert [e["kind"] for e in ev] == ["iter", "end"]
    first = ev[0]
    assert list(first)[:3] == ["t", "elapsed_s", "kind"]
    assert first["t"].endswith("Z")
    datetime.fromisoformat(first["t"].replace("Z", "+00:00"))
    assert isinstance(first["elapsed_s"], float) and first["elapsed_s"] >= 0.0
    assert ev[1]["elapsed_s"] >= first["elapsed_s"]
    assert first["stage"] == 0 and first["epoch"] == 3
    assert first["loss"] == 1.5 and first["lr"] == 0.25
    assert first["shape"] == [2, 3] and first["band"] is None
    assert first["done"] is True
    assert ev[1] == {**ev[1], "state": "success", "error": None}


def test_non_finite_floats_are_strings(tmp_path):
    log = ProgressLog(tmp_path)
    log.event("iter", a=float("nan"), b=float("inf"), c=-float("inf"),
              d=np.float32("nan"), e=np.float64("-inf"), f=[1.0, float("inf")])
    (ev,) = _events(tmp_path)
    assert (ev["a"], ev["b"], ev["c"], ev["d"], ev["e"]) == (
        "nan", "inf", "-inf", "nan", "-inf")
    assert ev["f"] == [1.0, "inf"]


def test_floats_round_trip_exactly(tmp_path):
    values = [0.1, 1.0 / 3.0, 6.02214076e23, 5e-324, float(np.float32(0.7))]
    log = ProgressLog(tmp_path)
    for v in values:
        log.event("iter", loss=v)
    assert [e["loss"] for e in _events(tmp_path)] == values


def test_disabled_log_writes_nothing(tmp_path):
    off = ProgressLog(tmp_path, enabled=False)
    assert not off.enabled
    off.event("start", task_type="fwi")
    off.snapshot(0, tmp_path / "x.npy")
    none_dir = ProgressLog(None)
    assert not none_dir.enabled
    none_dir.event("start", task_type="fwi")
    assert not (tmp_path / METRICS_FILE).exists()


@pytest.mark.parametrize("make_dir", ["missing", "is_a_file"])
def test_unwritable_task_dir_warns_once_and_never_raises(tmp_path, capsys, make_dir):
    if make_dir == "missing":
        task_dir = tmp_path / "does" / "not" / "exist"
    else:
        task_dir = tmp_path / "plain_file"
        task_dir.write_text("")
    log = ProgressLog(task_dir)
    log.event("start", task_type="fwi")
    log.event("iter", epoch=0, loss=1.0)
    log.snapshot(0, task_dir / "a.npy")
    out = capsys.readouterr().out
    assert out.count("[progress]") == 1, out
    assert not log.enabled


def test_bad_values_disable_instead_of_raising(tmp_path, capsys):
    log = ProgressLog(tmp_path)
    log.event("iter", loss=object())
    log.event("iter", loss=1.0)
    assert capsys.readouterr().out.count("[progress]") == 1
    assert not (tmp_path / METRICS_FILE).exists()

    clash = ProgressLog(tmp_path / "x")
    (tmp_path / "x").mkdir()
    clash.event("iter", elapsed_s=1.0)
    assert "[progress]" in capsys.readouterr().out
    assert not clash.enabled


def test_snapshot_path_is_relative_to_task_dir(tmp_path):
    log = ProgressLog(tmp_path)
    log.snapshot(4, tmp_path / "output" / "epochs" / "vp_epoch_0004.npy", name="vp")
    (ev,) = _events(tmp_path)
    assert ev["kind"] == "snapshot"
    assert ev["epoch"] == 4 and ev["name"] == "vp"
    assert ev["path"] == "output/epochs/vp_epoch_0004.npy"


# ---------- tasks --------------------------------------------------------

_SHAPE = (32, 40)


def _common(tmp_path: Path) -> dict:
    return {
        "output_dir": str(tmp_path / "tasks"),
        "device": "cpu",
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.06, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 8, "depth": 2, "start": 4, "stop": 36},
            "receivers": {"step": 2, "depth": 3, "start": 2, "stop": 38},
        },
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 12,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False},
    }


def _true_vp(tmp_path: Path) -> Path:
    z = np.linspace(0.0, 1.0, _SHAPE[0])[:, None]
    vp = (2000.0 + 700.0 * z * np.ones(_SHAPE)).astype(np.float32)
    path = tmp_path / "true_vp.npy"
    np.save(path, vp)
    return path


def _fwi_doc(tmp_path: Path, **extra) -> dict:
    doc = _common(tmp_path)
    doc.update({
        "task_type": "fwi",
        "init_model": {"name": "vp", "shape": list(_SHAPE),
                       "linear_gradient": {"vmin": 2000.0, "vmax": 2500.0}},
        "obs": {"synthetic_from": {"name": "vp", "path": str(_true_vp(tmp_path))}},
        "optimizer": {"kind": "adam", "lr": 10.0, "eps": 1.0e-22},
        "scheduler": {"kind": "step", "step_size": 1, "gamma": 0.5},
        "batchsize": 2, "show_every": 1, "epochs": 4,
        "stages": [{"epochs": 2, "bandpass": {"lo_hz": 2.0, "hi_hz": 12.0}},
                   {"epochs": 2}],
    })
    doc.update(extra)
    return doc


def _run(doc: dict, tmp_path: Path, name: str):
    return TaskRunner().run(load_task(_write(doc, tmp_path / f"{name}.yaml")))


def test_fwi_metrics_match_loss_npy(tmp_path):
    result = _run(_fwi_doc(tmp_path, task_id="fwi_progress"), tmp_path, "fwi")
    assert result.status.state == "success", result.status.error
    ev = _events(result.task_dir)
    kinds = [e["kind"] for e in ev]
    assert kinds[0] == "start" and kinds[-1] == "end"
    assert kinds.count("start") == 1 and kinds.count("end") == 1

    start = ev[0]
    assert start["task_type"] == "fwi" and start["mode"] == "conventional"
    assert start["total_epochs"] == 4 and start["n_stages"] == 2
    assert start["start_epoch"] == 0 and start["resumed_from"] is None

    stages = [e for e in ev if e["kind"] == "stage"]
    assert [s["stage"] for s in stages] == [0, 1]
    assert [s["epochs"] for s in stages] == [2, 2]
    assert [s["first_epoch"] for s in stages] == [0, 2]
    assert stages[0]["band"] == {"lo_hz": 2.0, "hi_hz": 12.0}
    assert stages[1]["band"] is None
    assert stages[0]["shape"] == list(_SHAPE) and stages[0]["dh"] == 10.0

    iters = [e for e in ev if e["kind"] == "iter"]
    assert [i["epoch"] for i in iters] == [0, 1, 2, 3]
    assert [i["stage"] for i in iters] == [0, 0, 1, 1]
    loss_npy = np.load(result.task_dir / "output" / "loss.npy")
    assert np.array_equal(np.array([i["loss"] for i in iters], dtype=np.float64),
                          loss_npy)
    # the lr each step used: one step schedule over the whole run halves it
    # every epoch, across the stage entry (stage base initial_lr * lr_scale)
    assert [i["lr"] for i in iters] == [10.0, 5.0, 2.5, 1.25]

    snaps = [e for e in ev if e["kind"] == "snapshot"]
    assert [s["epoch"] for s in snaps] == [0, 1, 2, 3]
    for s in snaps:
        assert s["name"] == "vp"
        assert (result.task_dir / s["path"]).is_file()

    end = ev[-1]
    assert end["state"] == "success" and end["error"] is None
    status = json.loads((result.task_dir / "status.json").read_text())
    assert status["state"] == end["state"]
    # every kind=stage precedes its iterations
    assert kinds.index("stage") < kinds.index("iter")


def test_fwi_resume_keeps_global_epochs(tmp_path):
    first = _run(_fwi_doc(tmp_path, task_id="first", epochs=2,
                          stages=[{"epochs": 2, "bandpass": {"lo_hz": 2.0, "hi_hz": 12.0}}]),
                 tmp_path, "first")
    assert first.status.state == "success", first.status.error
    second = _run(_fwi_doc(tmp_path, task_id="second", resume_from="first"),
                  tmp_path, "second")
    assert second.status.state == "success", second.status.error
    ev = _events(second.task_dir)
    start = ev[0]
    assert start["kind"] == "start"
    assert start["start_epoch"] == 2 and start["resumed_from"] == "first"
    assert [e["stage"] for e in ev if e["kind"] == "stage"] == [1]
    iters = [e for e in ev if e["kind"] == "iter"]
    assert [i["epoch"] for i in iters] == [2, 3]
    loss_npy = np.load(second.task_dir / "output" / "loss.npy")
    assert loss_npy.size == 4
    assert np.array_equal(np.array([i["loss"] for i in iters]), loss_npy[2:])


def test_lsrtm_iter_losses_match_loss_npy(tmp_path):
    true_path = _true_vp(tmp_path)
    doc = _common(tmp_path)
    doc["physics"].update(equation="AcousticLSRTM", receiver_type=["sh1"])
    doc.update({
        "task_type": "lsrtm", "task_id": "lsrtm_progress",
        "background_model": {"name": "vp", "path": str(true_path),
                             "smooth_sigma_cells": 3.0},
        "true_model": {"name": "vp", "path": str(true_path)},
        "optimizer": {"kind": "adam", "lr": 0.01, "eps": 1.0e-22},
        "batchsize": 2, "show_every": 1, "epochs": 3,
        "stages": [{"epochs": 1}, {"epochs": 2, "lr_scale": 0.5}],
    })
    result = _run(doc, tmp_path, "lsrtm")
    assert result.status.state == "success", result.status.error
    ev = _events(result.task_dir)
    assert ev[0]["kind"] == "start" and ev[0]["task_type"] == "lsrtm"
    assert ev[0]["total_epochs"] == 3 and ev[0]["n_stages"] == 2
    iters = [e for e in ev if e["kind"] == "iter"]
    assert [(i["stage"], i["epoch"]) for i in iters] == [(0, 0), (1, 1), (1, 2)]
    assert [i["lr"] for i in iters] == [0.01, 0.005, 0.005]
    assert np.array_equal(np.array([i["loss"] for i in iters]),
                          np.load(result.task_dir / "output" / "loss.npy"))
    snaps = [e for e in ev if e["kind"] == "snapshot"]
    assert [s["path"] for s in snaps] == [
        f"output/epochs/reflectivity_epoch_{k:04d}.npy" for k in range(3)]
    assert ev[-1]["kind"] == "end" and ev[-1]["state"] == "success"


def test_freqsel_metrics_match_curves(tmp_path):
    doc = _common(tmp_path)
    del doc["wavelet"], doc["geometry"]
    doc.update({
        "task_type": "fwi", "task_id": "freqsel_progress",
        "time": {"dt": 0.001, "nt": 450},
        "init_model": {"name": "vp", "shape": list(_SHAPE),
                       "linear_gradient": {"vmin": 2000.0, "vmax": 2500.0}},
        "optimizer": {"kind": "adam", "lr": 10.0},
        "epochs": 3, "batchsize": 1,
        "source_encoding": {"enabled": True, "mode": "frequency_selection",
                            "frequency": {
                                "synthesize_from_true": True,
                                "true_model_path": str(_true_vp(tmp_path)),
                                "synth_n_nodes": 4, "probe_samples": 200,
                                "k_lo": 2, "k_hi": 3, "steady_samples": 200,
                                "slack_samples": 50, "ramp_s": 0.1,
                                "n_pools": 2}},
        "stages": [{"epochs": 1}, {"epochs": 2}],
    })
    result = _run(doc, tmp_path, "freqsel")
    assert result.status.state == "success", result.status.error
    ev = _events(result.task_dir)
    assert ev[0]["kind"] == "start" and ev[0]["mode"] == "frequency_selection"
    assert ev[0]["total_epochs"] == 3 and ev[0]["n_stages"] == 2
    stages = [e for e in ev if e["kind"] == "stage"]
    assert [(s["stage"], s["first_epoch"]) for s in stages] == [(0, 0), (1, 1)]
    assert stages[0]["band"]["lo_hz"] == 10.0 and stages[0]["band"]["hi_hz"] == 15.0
    iters = [e for e in ev if e["kind"] == "iter"]
    assert [(i["stage"], i["epoch"]) for i in iters] == [(0, 0), (1, 1), (1, 2)]
    with np.load(result.task_dir / "curves.npz") as curves:
        assert np.array_equal(np.array([i["loss"] for i in iters]), curves["losses"])
    for s in (e for e in ev if e["kind"] == "snapshot"):
        assert (result.task_dir / s["path"]).is_file()
    assert ev[-1]["kind"] == "end" and ev[-1]["state"] == "success"


def test_plan_streaming_metrics_match_loss_npy(tmp_path):
    from test_task_fwi_plan_streaming import (
        _build_plan_streaming_spec,
        _build_tiny_plan_streaming_fixture,
    )

    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    doc = _build_plan_streaming_spec(tmp_path, fixture, epochs=3)
    doc.update(device="cpu", task_id="plan_progress",
               stages=[{"epochs": 1, "bandpass": {"lo_hz": 2.0, "hi_hz": 8.0}},
                       {"epochs": 2}])
    result = _run(doc, tmp_path, "plan")
    assert result.status.state == "success", result.status.error
    ev = _events(result.task_dir)
    assert ev[0]["kind"] == "start" and ev[0]["mode"] == "plan_streaming"
    assert ev[0]["total_epochs"] == 3 and ev[0]["n_stages"] == 2
    stages = [e for e in ev if e["kind"] == "stage"]
    assert [s["band"] for s in stages] == [{"lo_hz": 2.0, "hi_hz": 8.0}, None]
    iters = [e for e in ev if e["kind"] == "iter"]
    assert [(i["stage"], i["epoch"]) for i in iters] == [(0, 0), (1, 1), (1, 2)]
    assert np.array_equal(np.array([i["loss"] for i in iters]),
                          np.load(result.task_dir / "output" / "loss.npy"))
    assert [s["path"] for s in ev if s["kind"] == "snapshot"] == [
        f"output/epochs/vp_epoch_{k:04d}.npy" for k in range(3)]
    assert ev[-1]["kind"] == "end" and ev[-1]["state"] == "success"


def _forward_doc(tmp_path: Path, task_type: str = "forward") -> dict:
    doc = _common(tmp_path)
    doc.update({"task_type": task_type, "task_id": task_type,
                "models": [{"name": "vp", "path": str(_true_vp(tmp_path))}]})
    if task_type == "wavefield":
        doc.update(snapshot_times=[50, 150], plot=False)
    return doc


@pytest.mark.parametrize("task_type", ["forward", "wavefield"])
def test_forward_writes_start_progress_end(tmp_path, task_type):
    result = _run(_forward_doc(tmp_path, task_type), tmp_path, task_type)
    assert result.status.state == "success", result.status.error
    ev = _events(result.task_dir)
    assert [e["kind"] for e in ev] == ["start", "progress", "end"]
    assert ev[0]["task_type"] == task_type
    assert ev[0]["total_epochs"] is None and ev[0]["n_stages"] is None
    assert ev[0]["nshots"] == 4
    assert (ev[1]["done"], ev[1]["total"], ev[1]["unit"]) == (4, 4, "shot")
    assert ev[2]["state"] == "success" and ev[2]["error"] is None


def test_failing_task_writes_end_failed(tmp_path, monkeypatch):
    def _boom(self, spec, task_dir):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(TaskRunner, "_run_forward", _boom)
    result = _run(_forward_doc(tmp_path), tmp_path, "boom")
    # unchanged contract: the runner reports the failure, it does not re-raise
    assert result.status.state == "failed"
    assert result.status.error == "RuntimeError: solver exploded"
    ev = _events(result.task_dir)
    assert ev[-1]["kind"] == "end"
    assert ev[-1]["state"] == "failed"
    assert ev[-1]["error"] == "RuntimeError: solver exploded"
    status = json.loads((result.task_dir / "status.json").read_text())
    assert status["state"] == "failed" and status["error"] == ev[-1]["error"]


def test_interrupt_propagates_and_is_recorded(tmp_path, monkeypatch):
    def _abort(self, spec, task_dir):
        raise KeyboardInterrupt("second ctrl-c")

    monkeypatch.setattr(TaskRunner, "_run_forward", _abort)
    with pytest.raises(KeyboardInterrupt):
        _run(_forward_doc(tmp_path), tmp_path, "abort")
    task_dir = tmp_path / "tasks" / "forward"
    ev = _events(task_dir)
    assert ev[-1]["kind"] == "end" and ev[-1]["state"] == "failed"
    assert ev[-1]["error"] == "KeyboardInterrupt: second ctrl-c"
    status = json.loads((task_dir / "status.json").read_text())
    assert status["state"] == "failed"


@pytest.mark.parametrize("code", [0, None, 2, "message"])
def test_system_exit_is_failed_only_with_an_error_code(tmp_path, monkeypatch, code):
    """SystemExit(0) (e.g. DUMP_LOSS_INPUTS_EXIT's "exiting cleanly") is a
    deliberate clean stop; any other code is a failure."""
    def _exit(self, spec, task_dir):
        raise SystemExit(code)

    monkeypatch.setattr(TaskRunner, "_run_forward", _exit)
    with pytest.raises(SystemExit):
        _run(_forward_doc(tmp_path), tmp_path, "exit")
    task_dir = tmp_path / "tasks" / "forward"
    ev = _events(task_dir)
    status = json.loads((task_dir / "status.json").read_text())
    clean = code in (0, None)
    assert ev[-1]["kind"] == "end" and ev[-1]["state"] == status["state"]
    assert status["state"] == ("success" if clean else "failed")
    assert status["error"] == (None if clean else f"SystemExit: {code}") == ev[-1]["error"]
    if clean:
        assert status["summary"] == {"exited_early": f"SystemExit({code})"}

