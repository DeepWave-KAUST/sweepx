"""Slurm tools — submit a task YAML as a batch job, poll it, cancel it.

``run_task`` runs in-process and blocks; production runs (hours, several GPUs,
``--nproc-per-node``) go through ``sbatch`` instead, so they survive the client
disconnecting and never run on the submitting (login) node.

Each submission writes ``./sweep_jobs/<name>-<stamp>/job.sbatch`` plus the Slurm
log, and a record ``./sweep_jobs/<jobid>.json`` that ``job_status`` and
``cancel_job`` read. ``cancel_job`` only cancels jobs recorded there.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sweepx.agent.tools import register, registry

_JOBS_DIR = Path("sweep_jobs")
_LOG_TAIL_LINES = 40


def _slurm(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)


def _queue_entry(job_id: str) -> tuple[str, str, str, str] | None:
    """``(state, elapsed, time_limit, reason_or_nodes)`` while the job is in the
    queue, else None."""
    q = _slurm("squeue", "-h", "-j", job_id, "-o", "%T|%M|%l|%R")
    if q.returncode != 0 or not q.stdout.strip():
        return None
    state, elapsed, limit, where = q.stdout.strip().splitlines()[0].split("|", 3)
    return state, elapsed, limit, where


def _record_path(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.json"


def _load_record(job_id: str) -> dict[str, Any] | None:
    p = _record_path(job_id)
    return json.loads(p.read_text()) if p.is_file() else None


def _resolve_task_dir(rec: dict[str, Any]) -> Path | None:
    """The run's task directory: fixed when the spec names a task_id, else the
    newest ``<task_type>-<timestamp>`` directory created after submission."""
    if rec.get("task_dir"):
        p = Path(rec["task_dir"])
        return p if p.is_dir() else None
    out = Path(rec["output_dir"])
    if not out.is_dir():
        return None
    cands = [d for d in out.glob(f"{rec['task_type']}-*")
             if d.is_dir() and d.stat().st_mtime >= rec["submitted_at"] - 5]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


class SubmitTaskParams(BaseModel):
    yaml_path: str = Field(..., description="Task YAML (from build_*_spec or written by hand).")
    gpus: int = Field(1, ge=0, description="GPUs to request (one node).")
    gpu_type: str | None = Field(None, description="GPU model for --gres=gpu:<type>:<n>, e.g. v100 / a100. Set it on clusters with mixed GPU generations.")
    nproc_per_node: int | None = Field(
        None, ge=1,
        description="Passed to `sweep-tasks run --nproc-per-node` (torchrun ranks). Only for tasks that shard "
                    "shots across ranks; omit for one process. Must not exceed gpus.",
    )
    overrides: list[str] = Field(
        default_factory=list,
        description="KEY=VALUE overrides for `sweep-tasks run --override`, applied in the job, e.g. "
                    "[\"device=cuda\"] for a spec built on a node without GPUs.",
    )
    time: str = Field("04:00:00", description="Wall-clock limit (--time).")
    partition: str | None = Field(None, description="--partition")
    account: str | None = Field(None, description="--account")
    qos: str | None = Field(None, description="--qos")
    mem: str | None = Field(None, description="--mem, e.g. 128G")
    cpus_per_task: int | None = Field(None, ge=1, description="--cpus-per-task")
    constraint: str | None = Field(None, description="--constraint")
    sbatch_args: list[str] = Field(default_factory=list, description="Extra #SBATCH options, e.g. [\"--exclusive\"].")
    setup: list[str] = Field(default_factory=list, description="Shell lines run before the task, e.g. module loads.")
    python: str | None = Field(None, description="Python interpreter for the job (default: the one running these tools).")
    job_name: str | None = Field(None, description="Slurm job name (default: sweepx-<task_type>).")


@register(
    name="submit_task",
    description=(
        "Submit a sweepx.tasks YAML as a Slurm batch job (`sweep-tasks run` under sbatch) and return the job id. "
        "Use it instead of run_task for long or multi-GPU runs. The spec is loaded first, so a broken YAML "
        "fails here, not in the queue. Poll with job_status; stop with cancel_job."
    ),
    params_model=SubmitTaskParams,
)
def submit_task(args: SubmitTaskParams) -> dict[str, Any]:
    if shutil.which("sbatch") is None:
        return {"error": "sbatch not found: start the tools on a Slurm login node to submit jobs"}
    if args.nproc_per_node is not None and args.nproc_per_node > max(args.gpus, 1):
        return {"error": f"nproc_per_node={args.nproc_per_node} exceeds gpus={args.gpus}"}
    try:
        import yaml

        from sweepx.tasks import load_task_from_dict
        from sweepx.tasks.runtime.config import apply_overrides
    except ImportError as exc:
        return {"error": f"sweepx.tasks is not importable: {exc}"}
    yaml_path = Path(args.yaml_path).expanduser().resolve()
    if not yaml_path.is_file():
        return {"error": f"YAML spec not found: {args.yaml_path}"}
    # Validate exactly what the job will run: the YAML with the overrides applied.
    try:
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        apply_overrides(raw, args.overrides)
        spec = load_task_from_dict(raw, base_dir=yaml_path.parent)
    except Exception as exc:
        return {"error": f"failed to load spec: {type(exc).__name__}: {exc}"}

    workdir = Path.cwd()
    name = args.job_name or f"sweepx-{spec.task_type}"
    job_dir = (_JOBS_DIR / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}").resolve()
    job_dir.mkdir(parents=True)
    log = job_dir / "slurm-%j.out"

    opts = [f"--job-name={name}", f"--output={log}", f"--time={args.time}", "--nodes=1"]
    if args.gpus > 0:
        opts.append(f"--gres=gpu:{args.gpu_type}:{args.gpus}" if args.gpu_type else f"--gres=gpu:{args.gpus}")
    for flag, value in (("partition", args.partition), ("account", args.account), ("qos", args.qos),
                        ("mem", args.mem), ("cpus-per-task", args.cpus_per_task), ("constraint", args.constraint)):
        if value is not None:
            opts.append(f"--{flag}={value}")
    opts += args.sbatch_args

    run = [args.python or sys.executable, "-m", "sweepx.tasks", "run"]
    if args.nproc_per_node is not None:
        run += ["--nproc-per-node", str(args.nproc_per_node)]
    for o in args.overrides:
        run += ["--override", o]
    run.append(str(yaml_path))
    script = "\n".join(
        ["#!/bin/bash", *(f"#SBATCH {o}" for o in opts), "set -e", f"cd {shlex.quote(str(workdir))}",
         *args.setup, shlex.join(run), ""]
    )
    script_path = job_dir / "job.sbatch"
    script_path.write_text(script)

    sub = _slurm("sbatch", "--parsable", str(script_path))
    if sub.returncode != 0:
        return {"error": f"sbatch failed: {sub.stderr.strip() or sub.stdout.strip()}", "script": str(script_path)}
    job_id = sub.stdout.strip().split(";")[0]

    output_dir = (workdir / Path(spec.output_dir).expanduser()).resolve()
    rec = {
        "job_id": job_id, "script": str(script_path), "log": str(log).replace("%j", job_id),
        "yaml_path": str(yaml_path), "task_type": spec.task_type, "output_dir": str(output_dir),
        "task_dir": str(output_dir / spec.task_id) if spec.task_id else None,
        "submitted_at": time.time(),
    }
    _record_path(job_id).write_text(json.dumps(rec, indent=2))
    return {"job_id": job_id, "script": rec["script"], "log": rec["log"], "task_dir": rec["task_dir"],
            "output_dir": rec["output_dir"]}


class JobIdParams(BaseModel):
    job_id: str = Field(..., pattern=r"^\d+$", description="Slurm job id returned by submit_task.")


@register(
    name="job_status",
    description=(
        "Status of a Slurm job: queue state, elapsed time, pending reason or exit code, the tail of its log, "
        "and, for jobs from submit_task, the task's status.json once the run has started."
    ),
    params_model=JobIdParams,
)
def job_status(args: JobIdParams) -> dict[str, Any]:
    if shutil.which("squeue") is None:
        return {"error": "squeue not found: this machine has no Slurm client"}
    out: dict[str, Any] = {"job_id": args.job_id}
    queued = _queue_entry(args.job_id)
    if queued is not None:
        state, elapsed, limit, where = queued
        out.update(state=state, elapsed=elapsed, time_limit=limit)
        out["reason" if state == "PENDING" else "nodes"] = where  # squeue %R
    else:
        a = _slurm("sacct", "-X", "-n", "-P", "-j", args.job_id, "-o", "State,Elapsed,ExitCode")
        if a.returncode != 0 or not a.stdout.strip():
            out["state"] = "UNKNOWN"
        else:
            state, elapsed, exit_code = a.stdout.strip().splitlines()[0].split("|")
            out.update(state=state, elapsed=elapsed, exit_code=exit_code)

    rec = _load_record(args.job_id)
    if rec is None:
        return out
    log = Path(rec["log"])
    out["log"] = str(log)
    if log.is_file():
        out["log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-_LOG_TAIL_LINES:])
    task_dir = _resolve_task_dir(rec)
    if task_dir is not None:
        out["task_dir"] = str(task_dir)
        if (task_dir / "status.json").is_file():
            status = registry.get("read_status").invoke({"task_dir": str(task_dir)}).value
            out["task_status"] = status
            if queued is None and isinstance(status, dict) and status.get("state") in ("pending", "running"):
                out["note"] = (f"the job has ended but status.json still says {status['state']!r}: the run was "
                               "stopped (cancelled, out of time or killed) before the runner could record it")
    return out


@register(
    name="cancel_job",
    description="Cancel a Slurm job submitted with submit_task (scancel). Other jobs are refused.",
    params_model=JobIdParams,
)
def cancel_job(args: JobIdParams) -> dict[str, Any]:
    if _load_record(args.job_id) is None:
        return {"error": f"job {args.job_id} was not submitted with submit_task from this directory; not cancelling"}
    if shutil.which("scancel") is None:
        return {"error": "scancel not found: this machine has no Slurm client"}
    if _queue_entry(args.job_id) is None:
        return {"error": f"job {args.job_id} is no longer queued or running; nothing to cancel"}
    r = _slurm("scancel", args.job_id)
    if r.returncode != 0:
        return {"error": f"scancel failed: {r.stderr.strip()}"}
    return {"job_id": args.job_id, "cancelled": True}
