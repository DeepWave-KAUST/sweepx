"""P8: visualization tools — plot_wavefield / make_wavefield_gif / compare_wavefields."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from sweepx.agent.tools import registry


def _run_wavefield(tmp_path, task_id, snaps=(20, 50, 80)):
    vp = tmp_path / f"{task_id}_vp.npy"
    np.save(vp, np.full((40, 56), 1800.0, dtype=np.float32))
    b = registry.get("build_wavefield_spec").invoke({
        "vp_path": str(vp), "dh": 12.5, "dt": 0.001, "nt": max(snaps) + 1, "fm": 12.0,
        "snapshot_times": list(snaps), "source_step": 20, "source_depth": 2,
        "receiver_step": 5, "receiver_depth": 8, "spatial_order": 4, "abcn": 10,
        "device": "cpu", "backend_impl": "eager", "use_compile": False,
        "output_dir": str(tmp_path / "runs"), "task_id": task_id,
    })
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 120})
    assert r.value.get("state") == "success", r.value
    return r.value["task_dir"]


@pytest.fixture(scope="module")
def wavefield_runs(tmp_path_factory):
    """The two wavefield runs the tests read, built and run once per module.

    Every test used to build and run its own copy of the same wavefield task;
    the tools under test only read ``output/snapshots.npy``, so each test now
    gets a fresh copy of a finished run (see ``_copy_run``) and the runs are
    made once.
    """
    pytest.importorskip("sweep")          # the tests' own skip conditions,
    pytest.importorskip("sweepx.tasks")   # here so a missing tier skips, not errors
    root = tmp_path_factory.mktemp("p8_runs")
    return {task_id: _run_wavefield(root, task_id) for task_id in ("wf1", "wf2")}


def _copy_run(task_dir, tmp_path):
    """A private copy of a finished run, laid out as run_task leaves it
    (``<tmp_path>/runs/<task_id>``), so what one test writes into the run's
    output/ (a PNG, a GIF) is never seen by another."""
    dst = tmp_path / "runs" / Path(task_dir).name
    shutil.copytree(task_dir, dst)
    return str(dst)


@pytest.mark.timeout(180)
def test_plot_wavefield(tmp_path, wavefield_runs):
    pytest.importorskip("sweep")
    pytest.importorskip("sweepx.tasks")
    td = _copy_run(wavefield_runs["wf1"], tmp_path)
    res = registry.get("plot_wavefield").invoke({"task_dir": td, "abcn": 10, "snapshot_index": -1})
    assert res.ok and "error" not in res.value, res.value
    assert Path(res.value["image_path"]).is_file()


@pytest.mark.timeout(180)
def test_make_gif(tmp_path, wavefield_runs):
    pytest.importorskip("sweep")
    pytest.importorskip("sweepx.tasks")
    td = _copy_run(wavefield_runs["wf1"], tmp_path)
    res = registry.get("make_wavefield_gif").invoke({"task_dir": td, "abcn": 10, "fps": 6})
    assert res.ok and "error" not in res.value, res.value
    assert Path(res.value["gif_path"]).is_file()
    assert res.value["n_frames"] == 3


@pytest.mark.timeout(180)
def test_gif_path_fallback(tmp_path, wavefield_runs):
    """Single-task gif tolerates a wrong task_dir by locating the real run."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweepx.tasks")
    td = _copy_run(wavefield_runs["wf1"], tmp_path)
    wrong = str(Path(td).parent / "hallucinated-name")
    res = registry.get("make_wavefield_gif").invoke({"task_dir": wrong, "abcn": 10})
    assert res.ok and "error" not in res.value, res.value
    assert Path(res.value["gif_path"]).is_file()


@pytest.mark.timeout(180)
def test_compare_rejects_missing(tmp_path, wavefield_runs):
    """compare must NOT silently fall back — a wrong/forward task_dir errors."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweepx.tasks")
    td = _copy_run(wavefield_runs["wf1"], tmp_path)
    res = registry.get("compare_wavefields").invoke({
        "task_dirs": [td, str(Path(td).parent / "nonexistent")],
        "labels": ["a", "b"], "abcn": 10, "out_path": str(tmp_path / "cmp.png"),
    })
    assert "error" in res.value


@pytest.mark.timeout(240)
def test_compare_ok(tmp_path, wavefield_runs):
    pytest.importorskip("sweep")
    pytest.importorskip("sweepx.tasks")
    td1 = _copy_run(wavefield_runs["wf1"], tmp_path)
    td2 = _copy_run(wavefield_runs["wf2"], tmp_path)
    res = registry.get("compare_wavefields").invoke({
        "task_dirs": [td1, td2], "labels": ["one", "two"], "abcn": 10,
        "out_path": str(tmp_path / "cmp.png"),
    })
    assert res.ok and "error" not in res.value, res.value
    assert Path(res.value["image_path"]).is_file()
