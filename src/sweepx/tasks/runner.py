"""Synchronous TaskRunner: turns a validated TaskSpec into actual work.

Each `_run_*` method mirrors the matching script in `examples/`:

  forward    -> examples/wavefields/free_surface_forward/acoustic_free_surface.py
  wavefield  -> same, with snapshot times + plots
  fwi        -> examples/FWI/2d/acoustic/torch/_fwi_marmousi_common.py
  lsrtm      -> examples/LSRTM/2d/acoustic/torch/lsrtm.py

The runner stays close to those scripts on purpose — they have been
validated end-to-end on Marmousi. Any new physics knob should land in the
schema, not as a runtime kwarg.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



# Canonical filter primitives — used across every FWI / RTM bandpass site
# in this module. ``bandpass`` is scipy ``sosfiltfilt`` on numpy (CPU);
# ``bandpass_torch`` is the FFT |H(z)|² Butterworth on torch (GPU/CPU,
# autograd-friendly). Importing once at module load avoids the ~50 µs
# import-on-call cost previously paid by half a dozen inline imports.
from sweep_tasks.schemas import (
    TaskSpec,
)
from sweep_tasks._helpers.util import (
    _make_task_id,
    _now_iso,
)
from sweep_tasks.tasks.forward import ForwardRunnerMixin
from sweep_tasks.tasks.lsrtm import LSRTMRunnerMixin
from sweep_tasks.tasks.fwi_freqsel import FreqselRunnerMixin
from sweep_tasks.tasks.rtm import RTMRunnerMixin
from sweep_tasks.tasks.fwi import FWIRunnerMixin
from sweep_tasks.tasks.fwi_plan_streaming import PlanStreamingFWIMixin
from sweep_tasks._helpers.metadata import _dump_run_metadata


# ---------- result + status containers -----------------------------------

@dataclass
class TaskStatus:
    task_id: str
    task_type: str
    state: str = "pending"  # pending | running | success | failed
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "artifacts": self.artifacts,
            "summary": self.summary,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


@dataclass
class TaskResult:
    status: TaskStatus
    task_dir: Path


# ---------- helpers shared by the _run_* methods --------------------------


class TaskRunner(
    ForwardRunnerMixin,
    FreqselRunnerMixin,
    FWIRunnerMixin,
    PlanStreamingFWIMixin,
    RTMRunnerMixin,
    LSRTMRunnerMixin,
):
    """Synchronous local task runner. One-shot: instantiate and call .run(spec)."""

    def run(self, spec: TaskSpec) -> TaskResult:
        from sweep_tasks.runtime import distributed as _dist

        dist_info = _dist.init_distributed_if_needed()
        self._dist = dist_info

        # Rank 0 picks the task_id (timestamp-based) and broadcasts so every rank
        # writes to the same task directory under output_dir.
        if dist_info.is_root:
            task_id = _make_task_id(spec.task_type, spec.task_id)
        else:
            task_id = None
        task_id = _dist.broadcast_object(task_id, dist_info, src=0)

        output_dir = Path(spec.output_dir).expanduser()
        task_dir = (output_dir / task_id).resolve()
        if dist_info.is_root:
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "output").mkdir(exist_ok=True)
            (task_dir / "logs").mkdir(exist_ok=True)
        _dist.barrier(dist_info)

        status = TaskStatus(
            task_id=task_id,
            task_type=spec.task_type,
            state="running",
            started_at=_now_iso(),
        )
        if dist_info.is_root:
            status.write(task_dir / "status.json")
            # Persist the resolved config UP-FRONT so every run dir — for ANY
            # task_type (forward/wavefield/fwi/rtm/lsrtm/introspect) — has
            # config_resolved.yaml + run_meta.json, even if the run crashes
            # before the task-specific dump. FWI paths re-dump later with
            # runtime extras (shape/origin/net_params/…), enriching this.
            try:
                _dump_run_metadata(spec, task_dir)
            except Exception as _meta_err:  # noqa: BLE001  (never let meta break the run)
                print(f"[run-meta] early config dump failed: {_meta_err}")

        dispatcher = {
            "introspect": self._run_introspect,
            "forward": self._run_forward,
            "wavefield": self._run_wavefield,
            "fwi": self._run_fwi,
            "lsrtm": self._run_lsrtm,
            "rtm": self._run_rtm,
        }
        runner_fn = dispatcher.get(spec.task_type)
        if runner_fn is None:
            status.state = "failed"
            status.error = f"Unknown task_type '{spec.task_type}'."
            status.finished_at = _now_iso()
            if dist_info.is_root:
                status.write(task_dir / "status.json")
            _dist.cleanup_distributed(dist_info)
            return TaskResult(status=status, task_dir=task_dir)

        try:
            artifacts, summary = runner_fn(spec, task_dir)
            status.state = "success"
            status.artifacts = [str(p) for p in artifacts]
            status.summary = summary
        except Exception as e:  # noqa: BLE001  (we want to surface any failure)
            import traceback as _tb

            status.state = "failed"
            status.error = f"{type(e).__name__}: {e}"
            # Print the full traceback BEFORE the finally-block's status
            # write, in case the write itself errors (e.g. the runner
            # deleted the task dir as part of its own cleanup) and ends
            # up masking the original exception in the stderr.
            print(f"[runner] task failed: {status.error}")
            _tb.print_exc()
        finally:
            status.finished_at = _now_iso()
            if dist_info.is_root:
                try:
                    task_dir.mkdir(parents=True, exist_ok=True)
                    status.write(task_dir / "status.json")
                except Exception as werr:  # noqa: BLE001
                    print(f"[runner] failed to write status.json: {werr!r}")
            _dist.barrier(dist_info)
            _dist.cleanup_distributed(dist_info)

        return TaskResult(status=status, task_dir=task_dir)

    # -- introspect --------------------------------------------------------


    # -- fwi ---------------------------------------------------------------


    # -- lsrtm -------------------------------------------------------------

    # ------------------------------------------------------------------
    # OBN 3-D CRG-plan FWI (source-encoded supershot path)
    # ------------------------------------------------------------------


    # -- rtm ---------------------------------------------------------------




# ---------- equation-name helpers ----------------------------------------


