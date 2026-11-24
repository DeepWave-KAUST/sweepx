"""sweep_tasks (CLI passthrough) and the Slurm tools submit_task / job_status / cancel_job.

Slurm is faked with shell scripts on PATH that log their arguments.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sweepx.tasks")

from sweepx.agent.tools import registry  # noqa: E402


def _call(name: str, **kw):
    r = registry.get(name).invoke(kw)
    assert r.ok, r.error
    return r.value


def _forward_yaml(tmp_path, nt: int = 20, task_id: str | None = "tiny") -> str:
    vp_path = tmp_path / "vp.npy"
    np.save(vp_path, np.full((24, 32), 2000.0, np.float32))
    b = _call("build_forward_spec", vp_path=str(vp_path), dh=10.0, dt=0.001, nt=nt, fm=10.0,
              output_dir=str(tmp_path / "runs"), task_id=task_id, device="cpu")
    assert "error" not in b, b
    return b["yaml_path"]


# --- sweep_tasks ----------------------------------------------------------------

def test_sweep_tasks_help_lists_commands():
    r = _call("sweep_tasks", argv=["--help"])
    assert r["returncode"] == 0 and "error" not in r
    assert "build-plan" in r["stdout"] and "extract-coeff" in r["stdout"]


def test_sweep_tasks_nonzero_exit_is_an_error():
    r = _call("sweep_tasks", argv=["no-such-command"])
    assert r["returncode"] != 0 and "exited with code" in r["error"]


def test_sweep_tasks_timeout_kills_the_run(tmp_path):
    yaml_path = _forward_yaml(tmp_path, nt=200000)
    t0 = time.time()
    r = _call("sweep_tasks", argv=["run", yaml_path], timeout_s=3)
    assert "timed out" in r["error"] and time.time() - t0 < 60


# --- Slurm ------------------------------------------------------------------------

_FAKES = {
    "sbatch": 'echo "$@" >> "$FAKE_SLURM_LOG"; echo "4242;cluster"',
    "squeue": 'echo "squeue $@" >> "$FAKE_SLURM_LOG"; [ -n "$FAKE_SQUEUE" ] && echo "$FAKE_SQUEUE"; true',
    "sacct": 'echo "COMPLETED|00:01:02|0:0"',
    "scancel": 'echo "scancel $@" >> "$FAKE_SLURM_LOG"',
}


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _FAKES.items():
        p = bin_dir / name
        p.write_text(f"#!/bin/bash\n{body}\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "slurm_calls.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log))
    monkeypatch.chdir(tmp_path)
    return log


def test_submit_writes_script_and_record(tmp_path, fake_slurm):
    yaml_path = _forward_yaml(tmp_path)
    r = _call("submit_task", yaml_path=yaml_path, gpus=2, gpu_type="v100", nproc_per_node=2,
              time="01:00:00", partition="batch", mem="64G", setup=["module load cuda/12.2"],
              python="/opt/py/bin/python", overrides=["device=cuda"])
    assert r["job_id"] == "4242"
    assert r["task_dir"] == str((tmp_path / "runs" / "tiny").resolve())
    script = open(r["script"]).read()
    for line in ("#SBATCH --gres=gpu:v100:2", "#SBATCH --time=01:00:00", "#SBATCH --partition=batch",
                 "#SBATCH --mem=64G", f"cd {tmp_path}", "module load cuda/12.2",
                 f"/opt/py/bin/python -m sweepx.tasks run --nproc-per-node 2 --override device=cuda {yaml_path}"):
        assert line in script, line
    assert "--parsable" in fake_slurm.read_text()
    rec = json.loads((tmp_path / "sweep_jobs" / "4242.json").read_text())
    assert rec["log"].endswith("slurm-4242.out") and rec["yaml_path"] == yaml_path


def test_submit_rejects_bad_requests(tmp_path, fake_slurm):
    yaml_path = _forward_yaml(tmp_path)
    assert "exceeds gpus" in _call("submit_task", yaml_path=yaml_path, gpus=1, nproc_per_node=4)["error"]
    assert "not found" in _call("submit_task", yaml_path=str(tmp_path / "missing.yaml"))["error"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("task_type: forward\nno_such_key: 1\n")
    assert "failed to load spec" in _call("submit_task", yaml_path=str(bad))["error"]
    assert "failed to load spec" in _call("submit_task", yaml_path=yaml_path, overrides=["no-equals"])["error"]
    assert "failed to load spec" in _call("submit_task", yaml_path=yaml_path, overrides=["no_such_key=1"])["error"]
    assert not fake_slurm.exists()  # nothing reached sbatch


def test_submit_without_slurm(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert "sbatch not found" in _call("submit_task", yaml_path="x.yaml")["error"]


def test_job_status_running_then_finished(tmp_path, fake_slurm, monkeypatch):
    yaml_path = _forward_yaml(tmp_path)
    r = _call("submit_task", yaml_path=yaml_path)
    log = Path(r["log"])
    log.write_text("\n".join(f"line {i}" for i in range(100)))

    monkeypatch.setenv("FAKE_SQUEUE", "RUNNING|5:00|4:00:00|gpu201-02")
    s = _call("job_status", job_id="4242")
    assert (s["state"], s["elapsed"], s["nodes"]) == ("RUNNING", "5:00", "gpu201-02")
    assert s["log_tail"].splitlines()[-1] == "line 99" and len(s["log_tail"].splitlines()) == 40
    assert "task_status" not in s

    _call("run_task", yaml_path=yaml_path)  # stands in for the job writing the task dir
    monkeypatch.delenv("FAKE_SQUEUE")
    s = _call("job_status", job_id="4242")
    assert (s["state"], s["exit_code"]) == ("COMPLETED", "0:0")
    assert s["task_status"]["state"] == "success"


def test_job_status_pending_reason_and_stale_task_status(tmp_path, fake_slurm, monkeypatch):
    r = _call("submit_task", yaml_path=_forward_yaml(tmp_path))
    monkeypatch.setenv("FAKE_SQUEUE", "PENDING|0:00|4:00:00|(Priority)")
    assert _call("job_status", job_id="4242")["reason"] == "(Priority)"
    # A run killed mid-way leaves status.json at "running".
    task_dir = Path(r["task_dir"])
    task_dir.mkdir(parents=True)
    (task_dir / "status.json").write_text(json.dumps({"task_id": "tiny", "state": "running"}))
    assert "note" not in _call("job_status", job_id="4242")
    monkeypatch.delenv("FAKE_SQUEUE")
    assert "stopped" in _call("job_status", job_id="4242")["note"]


def test_job_status_finds_timestamped_task_dir(tmp_path, fake_slurm):
    yaml_path = _forward_yaml(tmp_path, task_id=None)
    r = _call("submit_task", yaml_path=yaml_path)
    assert r["task_dir"] is None
    _call("run_task", yaml_path=yaml_path)
    s = _call("job_status", job_id="4242")
    assert s["task_dir"].startswith(str((tmp_path / "runs" / "forward-").resolve()))
    assert s["task_status"]["state"] == "success"


def test_cancel_only_recorded_live_jobs(tmp_path, fake_slurm, monkeypatch):
    assert "not submitted with submit_task" in _call("cancel_job", job_id="999")["error"]
    _call("submit_task", yaml_path=_forward_yaml(tmp_path))
    assert "no longer queued" in _call("cancel_job", job_id="4242")["error"]
    monkeypatch.setenv("FAKE_SQUEUE", "RUNNING|5:00|4:00:00|gpu201-02")
    assert _call("cancel_job", job_id="4242") == {"job_id": "4242", "cancelled": True}
    assert "scancel 4242" in fake_slurm.read_text()
