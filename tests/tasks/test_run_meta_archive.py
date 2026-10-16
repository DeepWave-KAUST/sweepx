"""What a run dir keeps so it stays readable once archived: the task file as
written, its labels, its inputs (real path, size, fingerprint), its Slurm job
(batch script, log paths), uncommitted code changes, its logs and a manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sweep_tasks import TaskRunner, load_task
from sweep_tasks._helpers import metadata as M


def test_labels_and_the_environment_of_interest():
    env = {"SWEEP_LABEL_LINE": "exp_line_a", "SWEEP_LABEL_PROJECT": "demo_project",
           "SWEEP_LABEL_": "nameless", "SWEEP_DD_ENABLE": "1", "PYTHONPATH": "/a:/b", "HOME": "/h",
           "SWEEP_AGENT_LLM_API_KEY": "sk-not-for-the-archive"}
    assert M._labels_from_env(env) == {"line": "exp_line_a", "project": "demo_project"}
    got = M._env_of_interest(env)
    assert got["SWEEP_DD_ENABLE"] == "1" and got["PYTHONPATH"] == "/a:/b"
    assert not any(k.startswith("SWEEP_LABEL_") for k in got) and "HOME" not in got
    assert got["SWEEP_AGENT_LLM_API_KEY"] == M.REDACTED                  # set, value not kept


def test_inputs_are_found_with_real_path_size_and_fingerprint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "vp.npy").write_bytes(b"x" * 100)
    link = tmp_path / "vp_link.npy"
    link.symlink_to(data / "vp.npy")
    out = tmp_path / "runs"
    run = out / "this_run"
    (out / "stage1").mkdir(parents=True)
    (out / "stage1" / "vp_final.npy").write_bytes(b"s")
    (run / "output").mkdir(parents=True)
    (run / "output" / "x.npy").write_bytes(b"o")
    cfg = {"output_dir": str(out), "init_model": {"path": str(link)},
           "obs": {"paths": [str(data / "vp.npy"), str(tmp_path / "missing.npy")]},
           "plan": {"dir": str(data)}, "start": {"path": str(out / "stage1" / "vp_final.npy")},
           "mine": {"path": str(run / "output" / "x.npy")},
           "backend": {"boundary": {"disk_dir": str(data)}}, "name": "vp"}
    got = {e["key"]: e for e in M._input_files(cfg, run)}
    # missing / where it writes (output_dir, disk_dir, its own run dir) skipped; a sibling run's output kept
    assert set(got) == {"init_model.path", "obs.paths[0]", "plan.dir", "start.path"}
    assert got["init_model.path"]["realpath"] == str((data / "vp.npy").resolve())
    assert got["init_model.path"]["size"] == 100
    assert got["init_model.path"]["quick_sha256"] == got["obs.paths[0]"]["quick_sha256"]
    assert got["plan.dir"]["is_dir"] is True and "size" not in got["plan.dir"]


def test_quick_hash_tells_files_apart_by_their_tail(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    head = os.urandom(M.QUICK_HASH_BYTES)
    a.write_bytes(head + b"\0" * M.QUICK_HASH_BYTES + b"tail-a")
    b.write_bytes(head + b"\0" * M.QUICK_HASH_BYTES + b"tail-b")
    assert M._quick_hash(a) and M._quick_hash(a) != M._quick_hash(b)


def test_uncommitted_changes_are_kept_as_a_diff(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                       check=True, capture_output=True)

    git("init", "-q")
    (repo / "mod.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "one")
    (repo / "mod.py").write_text("x = 2\n")
    (repo / "scratch.py").write_text("y = 0\n")
    task_dir = tmp_path / "run"
    task_dir.mkdir()
    meta = {"pkg": {"path": str(repo), "sha": "x", "dirty": True}, "clean": {"path": str(repo), "dirty": False}}
    M._write_dirty_diffs(meta, task_dir)
    diff = (task_dir / "provenance" / "pkg.diff").read_text()
    assert "-x = 1" in diff and "+x = 2" in diff and "scratch.py" in diff
    assert meta["pkg"]["diff_file"] == "provenance/pkg.diff" and "diff_file" not in meta["clean"]


def test_the_slurm_job_and_its_batch_script(tmp_path, monkeypatch):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    scontrol = fakebin / "scontrol"
    scontrol.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = write ] && [ "$4" = - ]; then printf "#!/bin/bash\\n#SBATCH --gres=gpu:1\\n'
        'export MY_API_KEY=abc123\\nsweep-tasks run t.yaml\\n"; exit 0; fi\n'
        'echo "JobId=$4 JobName=x StdOut=/w/x.$4.out StdErr=/w/x.$4.err WorkDir=/w Command=/w/job.sbatch"\n')
    scontrol.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}{os.pathsep}{os.environ.get('PATH', '')}")
    task_dir = tmp_path / "run"
    task_dir.mkdir()
    assert M._slurm_job(task_dir, environ={}) is None
    got = M._slurm_job(task_dir, environ={"SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": "/w",
                                          "SLURM_JOB_PARTITION": "gpu"})
    assert got["job_id"] == "123" and got["submit_dir"] == "/w" and got["job_partition"] == "gpu"
    assert got["batch_script"] == "launch/slurm_123.sh" and got["batch_script_redacted"] == 1
    script = (task_dir / "launch" / "slurm_123.sh").read_text()
    assert "--gres=gpu:1" in script and "abc123" not in script and f"export MY_API_KEY={M.REDACTED}" in script
    assert (got["stdout"], got["stderr"], got["workdir"]) == ("/w/x.123.out", "/w/x.123.err", "/w")


def test_a_resumed_run_lists_the_launches_before_it(tmp_path, monkeypatch):
    task_yaml = tmp_path / "t.yaml"
    task_yaml.write_text(yaml.safe_dump({"task_type": "introspect", "action": "list_equations",
                                         "output_dir": str(tmp_path / "tasks")}))
    spec, run = load_task(task_yaml), tmp_path / "tasks" / "r"
    monkeypatch.setattr(M, "_slurm_job", lambda task_dir, environ=None: {"job_id": os.environ["SLURM_JOB_ID"]})
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    M._dump_run_metadata(spec, run, source={"path": str(task_yaml)})
    M._dump_run_metadata(spec, run, extras={"x": 1})                      # same launch, a later dump
    first = json.loads((run / "run_meta.json").read_text())
    assert first["previous_launches"] == [] and first["source"]["copied_to"] == "task_source.yaml"
    M._ONCE.clear()                                                       # a new process: the resume
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    M._dump_run_metadata(spec, run)
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["scheduler"] == {"job_id": "222"} and meta["source"] is None
    assert [x["scheduler"] for x in meta["previous_launches"]] == [{"job_id": "111"}]
    assert meta["previous_launches"][0]["started_at"] == first["started_at"]


def test_a_run_keeps_its_source_logs_labels_and_manifest(tmp_path, monkeypatch):
    def fake_introspect(self, spec, task_dir):
        print("hello from the task")
        print("a warning", file=sys.stderr)
        art = Path(task_dir) / "output" / "x.txt"
        art.write_text("result\n")
        return [art], {"ok": 1}

    monkeypatch.setattr(TaskRunner, "_run_introspect", fake_introspect)
    monkeypatch.setenv("SWEEP_LABEL_LINE", "demo_line")
    task_yaml = tmp_path / "t.yaml"
    text = "# the demo task (this comment is kept)\n" + yaml.safe_dump(
        {"task_type": "introspect", "action": "list_equations", "output_dir": str(tmp_path / "tasks"),
         "task_id": "archive_demo"}, sort_keys=False)
    task_yaml.write_text(text)
    stdout_before = sys.stdout
    result = TaskRunner().run(load_task(task_yaml), source={"path": str(task_yaml), "overrides": ["seed=3"]})
    assert sys.stdout is stdout_before                                   # restored
    assert result.status.state == "success", result.status.error
    d = result.task_dir
    assert (d / "task_source.yaml").read_text() == text
    meta = json.loads((d / "run_meta.json").read_text())
    assert meta["labels"] == {"line": "demo_line"}
    assert meta["source"]["overrides"] == ["seed=3"]
    assert meta["source"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert meta["run_dir"] == str(d.resolve()) and "PYTHONPATH" in meta["env_vars_of_interest"]
    log = (d / "logs" / "stdout.log").read_text()
    assert log.startswith("# ---- ") and "hello from the task" in log
    assert "a warning" in (d / "logs" / "stderr.log").read_text()
    status = json.loads((d / "status.json").read_text())
    assert status["artifacts_rel"] == ["output/x.txt"]
    manifest = json.loads((d / "manifest.json").read_text())
    listed = {f["path"]: f["size"] for f in manifest["files"]}
    assert listed["output/x.txt"] == 7 and {"status.json", "run_meta.json", "task_source.yaml"} <= set(listed)
    assert listed["logs/stdout.log"] == (d / "logs" / "stdout.log").stat().st_size   # taken after the logs closed
    assert manifest["n_files"] == len(manifest["files"]) and not manifest["truncated"]


def test_run_logs_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(TaskRunner, "_run_introspect",
                        lambda self, spec, task_dir: (print("quiet") or [], {}))
    monkeypatch.setenv("SWEEP_RUN_LOG", "0")
    task_yaml = tmp_path / "t.yaml"
    task_yaml.write_text(yaml.safe_dump({"task_type": "introspect", "action": "list_equations",
                                         "output_dir": str(tmp_path / "tasks")}))
    result = TaskRunner().run(load_task(task_yaml))
    assert result.status.state == "success"
    assert not (result.task_dir / "logs" / "stdout.log").exists()
