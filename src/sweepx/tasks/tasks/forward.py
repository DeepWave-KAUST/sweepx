"""Forward / wavefield / introspect task runners (mixin). Verbatim from runner.py."""
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import sweep
import sweep.equations as eq_mod

from sweepx.tasks.schemas import ForwardSpec, IntrospectSpec, WavefieldSpec
from sweepx.tasks._helpers.geometry import _build_geometry_2d
from sweepx.tasks._helpers.model import (
    _get_equation_class,
    _infer_shape,
    _model_array,
    _model_names_for_equation,
    _wavefield_names_for_equation,
)
from sweepx.tasks._helpers.plotting import _plot_wavefield_snapshots
from sweepx.tasks._helpers.solver_build import _build_solver, _solver_models_in_order
from sweepx.tasks._helpers.util import (
    _apply_seed,
    _nonfinite_summary,
    _raise_if_nonfinite,
    _resolve_device,
)
from sweepx.tasks._helpers.wavelet_build import _build_wavelet
from sweepx.tasks.progress import progress_for


class ForwardRunnerMixin:
    def _run_introspect(self, spec: IntrospectSpec, task_dir: Path):
        action = spec.action
        result: Any
        torch_binding_available = sweep.is_torch_binding_available()

        if action == "list_equations":
            rows = []
            for name, cls in sorted(eq_mod._equation_classes().items()):
                supported = eq_mod.supports_torch_binding(cls)
                rows.append({
                    "name": name,
                    "models": _model_names_for_equation(cls),
                    "torch_binding_support": "yes" if supported else "no",
                    "torch_binding_available": "yes" if supported and torch_binding_available else "no",
                })
            result = rows

        elif action == "list_supported_bindings":
            result = eq_mod.torch_binding_supported_equations()

        elif action == "describe_equation":
            if not spec.target:
                raise ValueError("describe_equation requires `target` (equation name).")
            cls = _get_equation_class(spec.target)
            supported = eq_mod.supports_torch_binding(cls)
            result = {
                "name": spec.target,
                "models": _model_names_for_equation(cls),
                "wavefields": _wavefield_names_for_equation(cls),
                "torch_binding_support": "yes" if supported else "no",
                "torch_binding_available": "yes" if supported and torch_binding_available else "no",
            }

        elif action == "describe_field":
            if not spec.target or not spec.field_or_model:
                raise ValueError("describe_field requires `target` and `field_or_model`.")
            cls = _get_equation_class(spec.target)
            result = cls.describe_field(spec.field_or_model)

        elif action == "describe_model":
            if not spec.target or not spec.field_or_model:
                raise ValueError("describe_model requires `target` and `field_or_model`.")
            cls = _get_equation_class(spec.target)
            result = cls.describe_model(spec.field_or_model)

        else:
            raise ValueError(f"Unknown introspect action '{action}'.")

        payload = {"action": action, "target": spec.target,
                   "field_or_model": spec.field_or_model, "result": result}
        out_path = task_dir / "output" / "introspect.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str))

        # also print a human-friendly view to stdout for `sweep run` users
        print(json.dumps(payload, indent=2, default=str))
        summary = {"action": action, "row_count": len(result) if isinstance(result, list) else None}
        return [out_path], summary

    # -- forward -----------------------------------------------------------

    def _run_forward(self, spec: ForwardSpec, task_dir: Path):
        import torch

        _apply_seed(spec.seed)
        dev = _resolve_device(spec.device)
        equation_cls = _get_equation_class(spec.physics.equation)
        shape = _infer_shape(spec.models, spec.grid.shape)

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )
        wavelet = _build_wavelet(spec.wavelet, spec.time)
        sources, receivers = _build_geometry_2d(spec.geometry, shape)
        models = _solver_models_in_order(equation_cls, spec.models, base_dir=None, dev=dev)

        print(f"[forward] equation={spec.physics.equation} shape={shape} "
              f"nshots={sources.shape[0]} nreceivers={receivers.shape[1]} "
              f"nt={spec.time.nt} backend={spec.backend.impl}")
        # All shots go through one solver call, so progress is all-or-nothing.
        progress = progress_for(self)
        nshots = int(sources.shape[0])
        progress.event("start", task_type="forward", total_epochs=None,
                       n_stages=None, nshots=nshots)

        t0 = time.perf_counter()
        with torch.no_grad():
            record = solver(wavelet, sources, receivers, models=models)
        elapsed = (time.perf_counter() - t0) * 1000.0
        progress.event("progress", done=nshots, total=nshots, unit="shot")
        record_np = record.detach().cpu().numpy()

        out_dir = task_dir / "output"
        record_path = out_dir / "record.npy"
        sources_path = out_dir / "sources.npy"
        receivers_path = out_dir / "receivers.npy"
        np.save(record_path, record_np)
        np.save(sources_path, sources)
        np.save(receivers_path, receivers)
        # After the save, so a blown-up run still leaves record.npy to look at,
        # but ends failed instead of reporting success on NaN / Inf.
        _raise_if_nonfinite(_nonfinite_summary(
            record_np, "record.npy", record_path,
            axis_names=("shot", "time step", "receiver")))

        summary = {
            "elapsed_ms": elapsed,
            "record_shape": list(record_np.shape),
            "nshots": int(sources.shape[0]),
            "nreceivers": int(receivers.shape[1]),
            "nt": int(spec.time.nt),
        }
        print(f"[forward] done in {elapsed:.1f} ms, record shape {record_np.shape}")
        return [record_path, sources_path, receivers_path], summary

    # -- wavefield ---------------------------------------------------------

    def _run_wavefield(self, spec: WavefieldSpec, task_dir: Path):

        _apply_seed(spec.seed)
        # Backstop for WavefieldSpec's load-time check (a spec built with
        # model_construct, or edited after validation): on impl='c' the call
        # below dies inside the C propagator with an unexpected-keyword
        # TypeError, and with use_ckpt=True sweep refuses return_wavefield
        # only after the models are built, so refuse both up front.
        no_snapshots = (
            "Snapshot capture is implemented on the eager backend: set "
            "`backend: {impl: eager}` for a wavefield task. (forward / "
            "fwi / rtm tasks are unaffected and keep the compiled path.)"
        )
        if spec.backend.impl != "eager":
            raise NotImplementedError(
                f"backend.impl={spec.backend.impl!r} cannot capture wavefield "
                f"snapshots. {no_snapshots}"
            )
        if spec.backend.use_ckpt:
            raise NotImplementedError(
                "backend.use_ckpt=true cannot return wavefield snapshots "
                "(checkpointing discards the interior field): set "
                "`backend.use_ckpt: false` for a wavefield task."
            )
        dev = _resolve_device(spec.device)
        equation_cls = _get_equation_class(spec.physics.equation)
        shape = _infer_shape(spec.models, spec.grid.shape)

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )
        wavelet = _build_wavelet(spec.wavelet, spec.time)
        sources, receivers = _build_geometry_2d(spec.geometry, shape)
        models = _solver_models_in_order(equation_cls, spec.models, base_dir=None, dev=dev)

        print(f"[wavefield] equation={spec.physics.equation} shape={shape} "
              f"snapshots={spec.snapshot_times}")
        progress = progress_for(self)
        nshots = int(sources.shape[0])
        progress.event("start", task_type="wavefield", total_epochs=None,
                       n_stages=None, nshots=nshots)

        t0 = time.perf_counter()
        result = solver(
            wavelet, sources, receivers, models=models,
            return_wavefield=True, snapshot_times=list(spec.snapshot_times),
        )
        # A backend without snapshot capture ignores ``return_wavefield`` /
        # ``snapshot_times`` and hands back the record alone. Say which knob is
        # unsupported, instead of dying on a tuple unpack one line down.
        if not (isinstance(result, tuple) and len(result) == 2):
            raise NotImplementedError(
                f"backend.impl={spec.backend.impl!r} returned no wavefield "
                f"snapshots — it ignored return_wavefield=True. {no_snapshots}"
            )
        record, snapshots = result
        elapsed = (time.perf_counter() - t0) * 1000.0
        progress.event("progress", done=nshots, total=nshots, unit="shot")
        record_np = record.detach().cpu().numpy()
        snapshots_np = snapshots.detach().cpu().numpy()

        out_dir = task_dir / "output"
        record_path = out_dir / "record.npy"
        snapshots_path = out_dir / "snapshots.npy"
        np.save(record_path, record_np)
        np.save(snapshots_path, snapshots_np)
        # After both saves (see _run_forward). The snapshots are taken inside
        # the grid, PML included, so they also name the field and the time
        # step where the blow-up shows, which the record alone cannot.
        # snapshots.npy is (n_snapshots, n_fields, n_shots, 1, *padded grid).
        field_names = getattr(solver, "wavefield_names", None)
        if field_names is not None and len(field_names) != snapshots_np.shape[1]:
            field_names = None
        _raise_if_nonfinite(
            _nonfinite_summary(
                record_np, "record.npy", record_path,
                axis_names=("shot", "time step", "receiver")),
            _nonfinite_summary(
                snapshots_np, "snapshots.npy", snapshots_path,
                axis_names=("snapshot", "field", "shot"),
                labels={"snapshot": [f"t-step {t}" for t in spec.snapshot_times],
                        "field": field_names}),
        )

        artifacts: list[Path] = [record_path, snapshots_path]
        if spec.plot:
            try:
                # Background for the snapshots: the velocity model the wave is
                # actually travelling through. `vp` by name when the equation
                # has one, otherwise the first model in the list — an elastic
                # or VTI run then shows its leading model rather than nothing.
                bg_ref = next((m for m in spec.models if m.name == "vp"),
                              spec.models[0] if spec.models else None)
                bg = _model_array(bg_ref) if bg_ref is not None else None
                # 3-D is cut three ways; put the inline and crossline through
                # the SOURCE so the wavefront is in frame, and the depth slice
                # at mid-model (the source sits near the surface, where a depth
                # slice would be almost empty).
                slice_xyz = None
                if len(shape) == 3 and len(sources):
                    slice_xyz = (int(sources[0][0]), int(sources[0][1]),
                                 int(shape[0] // 2))
                fig_path = _plot_wavefield_snapshots(
                    snapshots_np, list(spec.snapshot_times), spec.physics.abcn,
                    shape, out_dir / "snapshots.png", spec.physics.free_surface,
                    model=bg,
                    model_label=(f"{bg_ref.name} (m/s)" if bg_ref is not None
                                 else None),
                    dh=spec.grid.dh, slice_xyz=slice_xyz,
                )
                artifacts.append(fig_path)
            except Exception as plot_err:  # noqa: BLE001
                print(f"[wavefield] plotting skipped: {plot_err}")

        summary = {
            "elapsed_ms": elapsed,
            "record_shape": list(record_np.shape),
            "snapshots_shape": list(snapshots_np.shape),
        }
        print(f"[wavefield] done in {elapsed:.1f} ms, snapshots {snapshots_np.shape}")
        return artifacts, summary

    # -- fwi (frequency-selection encoded) ---------------------------------
