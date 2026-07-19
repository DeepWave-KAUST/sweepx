"""``build_wavefield_spec`` — flat shortcut for wavefield-snapshot modelling.

Wavefield modelling is forward modelling that also dumps the full wavefield at
chosen time steps (for movies / QC / paper figures). It shares every forward
parameter, so we inherit BuildForwardParams and reuse its equation/multi-model
validation, adding only snapshot_times + plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from sweep_agent.tools import register
from sweep_agent.tools.build_forward import (
    BuildForwardParams,
    _deep_merge,
    _maybe_upgrade_3d,
    _physics_source_fields,
    _resolve_geometry,
    _resolve_models,
    _resolve_nt,
    _slug,
)


class BuildWavefieldParams(BuildForwardParams):
    snapshot_times: list[int] = Field(
        ...,
        min_length=1,
        description="Time-step indices at which to save full-wavefield snapshots, e.g. [100, 200, 300]. Each must be in [0, nt).",
    )
    plot: bool = Field(True, description="Also render PNG images of each snapshot (set False to only save .npy snapshots).")


def _construct_wavefield_dict(p: BuildWavefieldParams, models: list[dict[str, Any]], geometry: dict[str, Any]) -> dict[str, Any]:
    delay = p.wavelet_delay if p.wavelet_delay is not None else 1.0 / p.fm
    spec: dict[str, Any] = {
        "task_type": "wavefield",
        "output_dir": p.output_dir,
        "seed": p.seed,
        "device": p.device,
        "grid": {"dh": p.dh},
        "time": {"dt": p.dt, "nt": p.nt},
        "wavelet": {"kind": "ricker", "fm": p.fm, "delay": delay, "scale": p.wavelet_scale},
        "geometry": geometry,
        "physics": {
            "equation": p.equation,
            "spatial_order": p.spatial_order,
            "abcn": p.abcn,
            "free_surface": p.free_surface,
            **_physics_source_fields(p.equation, p.source_type, p.receiver_type, p.pml_type),
        },
        "backend": {"impl": p.backend_impl},
        "models": models,
        "snapshot_times": p.snapshot_times,
        "plot": p.plot,
    }
    if p.backend_impl == "eager":
        spec["backend"]["eager_options"] = {"use_compile": p.use_compile}
    if p.topography is not None:
        spec["physics"]["topography"] = p.topography
    if p.task_id is not None:
        spec["task_id"] = p.task_id
    if p.extra:
        spec = _deep_merge(spec, p.extra)
    return spec


@register(
    name="build_wavefield_spec",
    description=(
        "Construct a validated wavefield-snapshot spec: like build_forward_spec but also dumps the "
        "full wavefield at the given snapshot_times (for movies / QC / figures). Same parameters as "
        "build_forward_spec plus snapshot_times (list of time-step indices in [0, nt)) and plot. "
        "Call inspect_file on the model first; for non-Acoustic equations use list_equations + "
        "extra_models. Returns the validated spec + YAML path; then run_task(yaml_path)."
    ),
    params_model=BuildWavefieldParams,
)
def build_wavefield_spec(args: BuildWavefieldParams) -> dict[str, Any]:
    try:
        from sweep_tasks.schemas import WavefieldSpec
        from sweep_tasks import dump_task
    except ImportError as exc:
        return {"error": f"sweep_tasks is not importable: {exc}"}

    nt_eff, nt_err = _resolve_nt(args)
    if nt_err is not None:
        return nt_err

    bad = [t for t in args.snapshot_times if t < 0 or t >= args.nt]
    if bad:
        return {"error": f"snapshot_times {bad} out of range; must be in [0, nt={args.nt})."}

    import numpy as _np
    try:
        vp_shape = tuple(_np.load(args.vp_path, mmap_mode="r").shape)
    except Exception as exc:
        return {"error": f"could not read shape of vp_path '{args.vp_path}': {exc}"}

    # A 3-D model needs a 3-D equation; auto-upgrade Acoustic→Acoustic3D etc.
    args.equation, eq_err = _maybe_upgrade_3d(args.equation, len(vp_shape))
    if eq_err is not None:
        return {"error": eq_err}

    models, err = _resolve_models(args)  # reused from build_forward (duck-typed on equation/extra_models/vp_path)
    if err is not None:
        return err

    try:
        geometry = _resolve_geometry(args, vp_shape)
    except ValueError as exc:
        return {"error": str(exc)}

    spec_dict = _construct_wavefield_dict(args, models, geometry)
    try:
        spec = WavefieldSpec.model_validate(spec_dict)
    except Exception as exc:
        return {"error": f"WavefieldSpec validation failed: {exc}", "spec_attempted": spec_dict}

    eff_id = args.task_id or _slug(Path(args.vp_path).stem) + "-wavefield"
    yaml_path: str | None = None
    if args.save_yaml:
        root = Path(args.output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        yaml_path = str((root / f"{_slug(eff_id)}.yaml").resolve())
        dump_task(spec, yaml_path)

    return {
        "spec": spec.model_dump(mode="json"),
        "yaml_path": yaml_path,
        "task_id_hint": eff_id,
        "summary": (
            f"WavefieldSpec ready: {args.equation}, snapshots at steps {args.snapshot_times}, "
            f"plot={args.plot}. Run with run_task(yaml_path=...)."
        ),
    }
