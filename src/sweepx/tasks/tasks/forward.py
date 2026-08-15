"""Forward / wavefield / introspect task runners (mixin). Verbatim from runner.py."""
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import sweep
import sweep.equations as eq_mod

from sweep_tasks.schemas import ForwardSpec, IntrospectSpec, WavefieldSpec
from sweep_tasks._helpers.geometry import _build_geometry_2d
from sweep_tasks._helpers.model import (
    _get_equation_class,
    _infer_shape,
    _model_names_for_equation,
    _wavefield_names_for_equation,
)
from sweep_tasks._helpers.plotting import _plot_wavefield_snapshots
from sweep_tasks._helpers.solver_build import _build_solver, _solver_models_in_order
from sweep_tasks._helpers.util import _apply_seed, _resolve_device
from sweep_tasks._helpers.wavelet_build import _build_wavelet


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

        t0 = time.perf_counter()
        with torch.no_grad():
            record = solver(wavelet, sources, receivers, models=models)
        elapsed = (time.perf_counter() - t0) * 1000.0
        record_np = record.detach().cpu().numpy()

        out_dir = task_dir / "output"
        record_path = out_dir / "record.npy"
        sources_path = out_dir / "sources.npy"
        receivers_path = out_dir / "receivers.npy"
        np.save(record_path, record_np)
        np.save(sources_path, sources)
        np.save(receivers_path, receivers)

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

        t0 = time.perf_counter()
        record, snapshots = solver(
            wavelet, sources, receivers, models=models,
            return_wavefield=True, snapshot_times=list(spec.snapshot_times),
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        record_np = record.detach().cpu().numpy()
        snapshots_np = snapshots.detach().cpu().numpy()

        out_dir = task_dir / "output"
        record_path = out_dir / "record.npy"
        snapshots_path = out_dir / "snapshots.npy"
        np.save(record_path, record_np)
        np.save(snapshots_path, snapshots_np)

        artifacts: list[Path] = [record_path, snapshots_path]
        if spec.plot:
            try:
                fig_path = _plot_wavefield_snapshots(
                    snapshots_np, list(spec.snapshot_times), spec.physics.abcn,
                    shape, out_dir / "snapshots.png", spec.physics.free_surface,
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
