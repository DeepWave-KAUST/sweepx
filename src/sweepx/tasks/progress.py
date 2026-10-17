"""Structured live progress: ``<task_dir>/metrics.jsonl``.

stdout is for people; a watcher (sweep-studio, a notebook) needs the same
numbers without parsing log text, while the task is still running. Each
event is one JSON object on its own line, appended and flushed at once::

    {"t": iso8601-utc, "elapsed_s": float, "kind": ..., ...fields}

Kinds written by the runner and the tasks: ``start``, ``stage``, ``iter``,
``snapshot``, ``progress``, ``end``.

Writing the log must not change a run: callers pass values they already
hold as Python numbers (no new ``.item()`` / device syncs, no RNG draws),
and the first failure to write disables the log after one ``[progress]``
line instead of raising into the task.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METRICS_FILE = "metrics.jsonl"
_RESERVED = frozenset({"t", "elapsed_s", "kind"})


def _jsonable(value: Any) -> Any:
    """Map a field value to plain JSON; non-finite floats become strings."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isfinite(f):
            return f
        if math.isnan(f):
            return "nan"
        return "inf" if f > 0 else "-inf"
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    raise TypeError(f"unsupported metrics value type {type(value).__name__}")


def optimizer_lr(optimizer: Any) -> float | None:
    """Learning rate of the first param group, or ``None`` without an optimizer."""
    if optimizer is None:
        return None
    return optimizer.param_groups[0]["lr"]


def band_desc(bandpass: Any) -> dict[str, float] | None:
    """``{lo_hz, hi_hz}`` of a :class:`StageBandpass`, ``None`` when unfiltered."""
    if bandpass is None:
        return None
    return {"lo_hz": float(bandpass.lo_hz), "hi_hz": float(bandpass.hi_hz)}


class ProgressLog:
    """Append-only writer for ``metrics.jsonl``; a no-op when disabled.

    Disabled when ``enabled`` is false (non-root ranks) or ``task_dir`` is
    ``None``. The file is opened per event, so nothing stays open across a
    long run and a line is on disk as soon as :meth:`event` returns.
    """

    def __init__(self, task_dir: Path | None, enabled: bool = True) -> None:
        self.task_dir = Path(task_dir) if task_dir is not None else None
        self.path = (self.task_dir / METRICS_FILE
                     if enabled and self.task_dir is not None else None)
        self._t0 = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def event(self, kind: str, /, **fields: Any) -> None:
        if self.path is None:
            return
        try:
            clash = _RESERVED.intersection(fields)
            if clash:
                raise ValueError(f"reserved field name(s) {sorted(clash)}")
            record: dict[str, Any] = {
                "t": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds").replace("+00:00", "Z"),
                "elapsed_s": round(time.monotonic() - self._t0, 6),
                "kind": str(kind),
            }
            for key, value in fields.items():
                record[key] = _jsonable(value)
            line = json.dumps(record, allow_nan=False) + "\n"
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
        except (OSError, TypeError, ValueError) as err:
            print(f"[progress] cannot write {self.path} "
                  f"({type(err).__name__}: {err}); live metrics disabled "
                  f"for the rest of this run", flush=True)
            self.path = None

    def snapshot(self, epoch: int, path: Path, /, **fields: Any) -> None:
        """``kind=snapshot`` with ``path`` relative to the task directory."""
        if self.path is None:
            return
        try:
            rel = Path(path).relative_to(self.task_dir)
        except ValueError as err:
            print(f"[progress] snapshot {path} is outside {self.task_dir} "
                  f"({err}); live metrics disabled for the rest of this run",
                  flush=True)
            self.path = None
            return
        self.event("snapshot", epoch=epoch, path=rel.as_posix(), **fields)


def progress_for(runner: Any) -> ProgressLog:
    """The run's log, set by ``TaskRunner.run``; disabled when a task method
    is called on its own (tests, notebooks)."""
    log = getattr(runner, "_progress", None)
    return log if log is not None else ProgressLog(None, enabled=False)
