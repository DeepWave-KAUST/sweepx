"""``run_task`` — load a YAML spec, dispatch to TaskRunner, return task_dir + status.

Single entry point for executing any sweep_tasks spec (forward, fwi, lsrtm, rtm,
wavefield, introspect). The LLM hands us the YAML path produced by one of the
``build_*_spec`` tools; we hand back the on-disk artifacts location and a small
JSON summary suitable for the next reasoning step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sweep_agent.tools import register


class RunTaskParams(BaseModel):
    """Run a previously-built spec YAML through sweep_tasks.TaskRunner."""

    yaml_path: str = Field(..., description="Path to a validated sweep_tasks YAML spec (output of a build_*_spec tool).")
    timeout_s: float | None = Field(
        None,
        gt=0,
        description=(
            "Optional wall-clock timeout in seconds. The runner is synchronous; "
            "set this when running tiny smoke tests to avoid hangs. Leave None for production."
        ),
    )


def _summarize_status(status_obj: Any) -> dict[str, Any]:
    """Pull the small interesting subset of a TaskStatus into a dict."""
    # TaskStatus is a dataclass; works whether we have it or just its .to_dict() output.
    to_dict = getattr(status_obj, "to_dict", None)
    raw = to_dict() if callable(to_dict) else dict(status_obj)
    keep = ("task_id", "task_type", "state", "started_at", "finished_at", "error", "summary")
    out = {k: raw.get(k) for k in keep}
    arts = raw.get("artifacts") or []
    out["artifact_count"] = len(arts)
    # Only show the first few artifacts inline; the full list is in status.json.
    out["artifacts_sample"] = arts[:6]
    return out


@register(
    name="run_task",
    description=(
        "Execute a sweep_tasks YAML spec via TaskRunner. Works for forward, FWI, LSRTM, RTM, "
        "wavefield, and introspect tasks (auto-dispatched by `task_type` inside the YAML). "
        "Returns the task_dir on disk, final state ('success' | 'failed'), error message if any, "
        "and a short summary. Always follow up with `read_status` / `list_artifacts` for details "
        "before reporting back to the user."
    ),
    params_model=RunTaskParams,
)
def run_task(args: RunTaskParams) -> dict[str, Any]:
    try:
        from sweep_tasks import TaskRunner, load_task
    except ImportError as exc:
        return {"error": f"sweep_tasks is not importable: {exc}"}

    yaml_path = Path(args.yaml_path).expanduser()
    if not yaml_path.is_file():
        # LLMs often hallucinate the yaml_path (fake timestamps, wrong prefix).
        # Fall back to the most recent .yaml under likely roots — usually the one
        # a build_*_spec just wrote.
        roots = [yaml_path.parent, yaml_path.parent.parent, Path("sweep_runs"), Path.cwd() / "sweep_runs"]
        cands: list[Path] = []
        for root in roots:
            try:
                if root.is_dir():
                    cands += list(root.glob("*.yaml"))
            except OSError:
                pass
        if cands:
            yaml_path = max(cands, key=lambda f: f.stat().st_mtime)
    if not yaml_path.is_file():
        return {"error": f"YAML spec not found: {args.yaml_path}"}

    try:
        spec = load_task(yaml_path)
    except Exception as exc:
        return {"error": f"failed to load spec: {type(exc).__name__}: {exc}"}

    # Optional cooperative timeout via SIGALRM (POSIX only).
    timeout_handler_set = False
    if args.timeout_s is not None and hasattr(os, "fork"):
        import signal

        def _on_timeout(signum, frame):  # pragma: no cover - exercised via integration
            raise TimeoutError(f"run_task exceeded {args.timeout_s}s budget")

        signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, args.timeout_s)
        timeout_handler_set = True

    try:
        result = TaskRunner().run(spec)
    except Exception as exc:
        return {"error": f"TaskRunner raised: {type(exc).__name__}: {exc}"}
    finally:
        if timeout_handler_set:
            import signal
            signal.setitimer(signal.ITIMER_REAL, 0)

    status_dict = _summarize_status(result.status)
    status_dict["task_dir"] = str(result.task_dir)
    status_dict["status_json"] = str(Path(result.task_dir) / "status.json")
    return status_dict
