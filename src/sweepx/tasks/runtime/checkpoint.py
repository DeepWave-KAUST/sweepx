"""Atomic checkpoint save/load.

:func:`save_payload` / :func:`load_payload` are the schema-free primitives
(write ``.tmp`` then rename). :func:`save_run_checkpoint` /
:func:`load_run_checkpoint` wrap them for a task directory, which is what
the FWI and LSRTM runners use to checkpoint and resume.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_payload(
    path: str | Path,
    payload: dict[str, Any],
    *,
    weights_only: bool | None = None,
) -> Path:
    """Atomically save an arbitrary dict via ``torch.save`` (write ``.tmp`` then rename).

    Lower-level cousin of :func:`save_state` for callers that build their own
    payload dict (e.g. a YAML-driven runner that wants `model_state`,
    `optimizer_state`, `scheduler_state`, `losses`, RNG state, etc., all keyed
    however they like). ``save_state``'s ``model=``/``optimizer=`` kwargs are
    convenient when those are the only two things you serialize, but force a
    schema (`payload["model"]`, `payload["optimizer"]`); this helper imposes
    no schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if weights_only is None:
        torch.save(payload, tmp)
    else:
        torch.save(payload, tmp, _use_new_zipfile_serialization=True)
    os.replace(tmp, path)
    return path


def load_payload(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    """Load a dict written by :func:`save_payload` (or any ``torch.save`` dict).

    ``weights_only`` defaults to ``False`` because most FWI payloads contain
    arbitrary Python objects (e.g. metadata dicts) that the safe loader
    refuses. Set ``True`` if you control the source and want the
    safer torch>=2.4 pickle-restricted path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path} to resume from.")
    return torch.load(str(path), map_location=map_location, weights_only=weights_only)


def save_run_checkpoint(task_dir: "str | Path", payload: dict[str, Any]) -> Path:
    """Write ``<task_dir>/checkpoint.pt`` atomically."""
    return save_payload(Path(task_dir) / "checkpoint.pt", payload)


def load_run_checkpoint(prev_task_dir: "str | Path") -> dict[str, Any]:
    """Read ``<task_dir>/checkpoint.pt``. Raises FileNotFoundError if absent."""
    return load_payload(Path(prev_task_dir) / "checkpoint.pt", weights_only=False)


__all__ = ["save_payload", "load_payload",
           "save_run_checkpoint", "load_run_checkpoint"]
