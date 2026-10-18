"""``read_status`` + ``list_artifacts`` — inspect a finished task_dir.

Both tools are read-only, cheap, and safe to call repeatedly. They let the LLM
report a final answer to the user without us having to plumb the entire
TaskResult through tool-call JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sweep_agent.tools import register


_ART_GROUPS: dict[str, tuple[str, ...]] = {
    "numpy":    (".npy", ".npz"),
    "image":    (".png", ".jpg", ".jpeg", ".pdf"),
    "config":   (".yaml", ".yml", ".json", ".toml"),
    "log":      (".log", ".txt", ".out", ".err"),
    "data":     (".segy", ".sgy", ".h5", ".hdf5"),
    "tensor":   (".pt", ".pth", ".ckpt"),
}


def _classify(ext: str) -> str:
    e = ext.lower()
    for group, exts in _ART_GROUPS.items():
        if e in exts:
            return group
    return "other"


class ReadStatusParams(BaseModel):
    task_dir: str = Field(..., description="Task directory returned by run_task (or its status_json path).")


@register(
    name="read_status",
    description=(
        "Read a task's status.json from a TaskRunner output directory. Returns task_id, state "
        "(pending/running/success/failed), error (if any), summary metrics, and the list of "
        "artifact paths the runner produced. Use this after `run_task` to confirm success and "
        "before listing artifacts."
    ),
    params_model=ReadStatusParams,
)
def read_status(args: ReadStatusParams) -> dict[str, Any]:
    p = Path(args.task_dir).expanduser()
    if p.is_file() and p.name == "status.json":
        status_path = p
    else:
        status_path = p / "status.json"
    if not status_path.is_file():
        return {"error": f"status.json not found at {status_path}"}
    try:
        return json.loads(status_path.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"status.json is not valid JSON: {exc}"}


class ListArtifactsParams(BaseModel):
    task_dir: str = Field(..., description="Task directory returned by run_task.")
    max_per_group: int = Field(20, ge=1, le=200, description="Cap on entries listed per artifact group.")


@register(
    name="list_artifacts",
    description=(
        "Group the files inside a task directory by type (numpy / image / config / log / data / "
        "tensor / other) and return paths + sizes. Use to point the user at the right output: "
        "the synthetic shot record (.npy / .npz), QC images (.png), or status.json."
    ),
    params_model=ListArtifactsParams,
)
def list_artifacts(args: ListArtifactsParams) -> dict[str, Any]:
    root = Path(args.task_dir).expanduser()
    if not root.is_dir():
        return {"error": f"task_dir not found: {root}"}

    groups: dict[str, list[dict[str, Any]]] = {k: [] for k in (*_ART_GROUPS, "other")}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        group = _classify(path.suffix)
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        groups[group].append(
            {
                "path": str(path),
                "size_mb": round(size / 1024 / 1024, 3) if size >= 0 else None,
            }
        )

    # Truncate each group to keep the JSON compact for the LLM.
    truncated: dict[str, Any] = {}
    for grp, entries in groups.items():
        if not entries:
            continue
        if len(entries) > args.max_per_group:
            truncated[grp] = {
                "count": len(entries),
                "shown": entries[: args.max_per_group],
                "truncated": True,
            }
        else:
            truncated[grp] = {"count": len(entries), "shown": entries, "truncated": False}
    return {"task_dir": str(root), "groups": truncated}
