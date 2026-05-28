"""``build_forward_spec`` — turn flat LLM-friendly arguments into a ForwardSpec.

The full ``sweep_tasks.ForwardSpec`` schema is wide and deeply nested; small
local models hallucinate field names. We expose a flat surface here that covers
~90% of forward-modelling needs and forwards the long-tail via ``extra``
(deep-merged into the spec dict before validation). The result is a real,
Pydantic-validated ``ForwardSpec`` rendered to both dict and YAML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sweep_agent.tools import register


_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text).strip("-_") or "forward"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — values from ``overlay`` win, sub-dicts are merged."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class BuildForwardParams(BaseModel):
    """Flat parameter surface for a 2-D acoustic forward modelling task.

    Everything not covered here can be forwarded through ``extra`` (e.g.
    ``extra={"physics": {"abcn": 40}}`` widens the PML). Validation happens
    inside the tool, so the LLM sees a Pydantic error message if anything is
    off and can retry.
    """

    # --- Required ----------------------------------------------------------
    vp_path: str = Field(..., description="Absolute path to the .npy velocity model (shape (nz, nx) for 2-D, (nz, ny, nx) for 3-D).")
    dh: float = Field(..., gt=0, description="Grid spacing in metres. Same for x and z. Must satisfy CFL with dt and v_max.")
    dt: float = Field(..., gt=0, description="Time step in seconds. CFL bound: dt < dh / (v_max * sqrt(ndim)).")
    nt: int = Field(..., ge=1, description="Number of time samples; total record length is dt*nt seconds.")

    # --- Wavelet -----------------------------------------------------------
    fm: float = Field(8.0, gt=0, description="Ricker centre (peak) frequency in Hz. Must satisfy dh*fm*4 <= v_min for ≥4 pts/wavelength.")
    wavelet_delay: float | None = Field(None, ge=0, description="Ricker peak-time delay in seconds; default 1/fm so the wavelet is causal.")
    wavelet_scale: float = Field(1.0, description="Multiplicative amplitude scale on the wavelet.")

    # --- Geometry (line) ---------------------------------------------------
    source_step: int = Field(50, ge=1, description="Source spacing in grid cells along x.")
    source_depth: int = Field(1, ge=0, description="Source depth in grid cells (0 = top row). Surface acquisition default = 1.")
    receiver_step: int = Field(1, ge=1, description="Receiver spacing in grid cells along x.")
    receiver_depth: int = Field(18, ge=0, description="Receiver depth in grid cells. Marine streamer at ~225 m for dh=12.5 → depth=18.")

    # --- Physics + backend -------------------------------------------------
    equation: str = Field("Acoustic", description="Wave equation class name; common: Acoustic, AcousticVTI, Elastic.")
    spatial_order: int = Field(8, description="Finite-difference spatial order (2/4/6/8). Higher = more accurate, more memory.")
    abcn: int = Field(20, ge=0, description="PML thickness in grid cells. Typical 20-40.")
    free_surface: bool = Field(False, description="Free-surface BC at z=0 (sea surface). False = no free surface.")
    backend_impl: Literal["eager", "c"] = Field("eager", description="`eager` = pure-torch (slow, universal); `c` = sweep CUDA kernels (fast, requires CUDA build).")
    use_compile: bool = Field(
        False,
        description=(
            "Wrap the eager propagator step in torch.compile (inductor). Speeds up long runs on CUDA "
            "but requires the `sweep.torch` plugin and a working inductor toolchain. Default off for "
            "portability; set True only when the user explicitly asks for compile or the run is large."
        ),
    )

    # --- Identity / IO ----------------------------------------------------
    output_dir: str = Field("./sweep_runs", description="Parent directory for the run; <output_dir>/<task_id>/ holds artifacts + status.json.")
    task_id: str | None = Field(None, description="Subdirectory name; auto-generated from timestamp when omitted.")
    device: str = Field("auto", description="`auto` picks cuda if available; or pin to `cpu` / `cuda:0`.")
    seed: int = Field(0, description="RNG seed (numpy + torch).")

    # --- Output of this tool ----------------------------------------------
    save_yaml: bool = Field(True, description="When True, dump the validated spec to <output_dir>/<task_id>.yaml for reproducibility.")

    # --- Escape hatch -----------------------------------------------------
    extra: dict[str, Any] | None = Field(
        None,
        description=(
            "Deep-merged overrides on top of the constructed spec dict. Use to set "
            "ForwardSpec fields not exposed above (e.g. {'physics': {'pml_type': 'cpml'}}). "
            "Be conservative — these bypass the flat surface's defaults."
        ),
    )


def _construct_spec_dict(p: BuildForwardParams) -> dict[str, Any]:
    delay = p.wavelet_delay if p.wavelet_delay is not None else 1.0 / p.fm
    spec: dict[str, Any] = {
        "task_type": "forward",
        "output_dir": p.output_dir,
        "seed": p.seed,
        "device": p.device,
        "grid": {"dh": p.dh},
        "time": {"dt": p.dt, "nt": p.nt},
        "wavelet": {
            "kind": "ricker",
            "fm": p.fm,
            "delay": delay,
            "scale": p.wavelet_scale,
        },
        "geometry": {
            "kind": "line",
            "sources":   {"step": p.source_step,   "depth": p.source_depth},
            "receivers": {"step": p.receiver_step, "depth": p.receiver_depth},
        },
        "physics": {
            "equation": p.equation,
            "spatial_order": p.spatial_order,
            "abcn": p.abcn,
            "free_surface": p.free_surface,
        },
        "backend": {"impl": p.backend_impl},
        "models": [{"name": "vp", "path": p.vp_path}],
    }
    # Wire use_compile through eager_options only when the eager backend is in
    # play; the schema rejects eager_options on impl='c'.
    if p.backend_impl == "eager":
        spec["backend"]["eager_options"] = {"use_compile": p.use_compile}
    if p.task_id is not None:
        spec["task_id"] = p.task_id
    if p.extra:
        spec = _deep_merge(spec, p.extra)
    return spec


@register(
    name="build_forward_spec",
    description=(
        "Construct a validated sweep_tasks.ForwardSpec for a 2-D / 3-D forward modelling run. "
        "Returns the spec as a dict, the YAML path it was written to, and a human summary. "
        "ALWAYS call `inspect_file` on the velocity model first to learn its shape/dtype, then "
        "pick dh/dt/nt/fm consistent with CFL and pts-per-wavelength constraints. The default "
        "geometry is line sources/receivers at small depths (surface acquisition); override "
        "source_depth/receiver_depth for cross-well or interior shots."
    ),
    params_model=BuildForwardParams,
)
def build_forward_spec(args: BuildForwardParams) -> dict[str, Any]:
    # Import lazily so the package keeps importing without sweep_tasks installed.
    try:
        from sweep_tasks.schemas import ForwardSpec
        from sweep_tasks import dump_task
    except ImportError as exc:
        return {"error": f"sweep_tasks is not importable: {exc}"}

    spec_dict = _construct_spec_dict(args)
    try:
        spec = ForwardSpec.model_validate(spec_dict)
    except Exception as exc:
        # Surface validation errors as data, not exceptions — the agent loop
        # feeds them back to the LLM so it can retry.
        return {"error": f"ForwardSpec validation failed: {exc}", "spec_attempted": spec_dict}

    # Pick an effective task_id for the yaml filename even if the spec leaves it None.
    eff_task_id = args.task_id or _slug(Path(args.vp_path).stem) + "-forward"
    yaml_path: str | None = None
    if args.save_yaml:
        out_root = Path(args.output_dir).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)
        yaml_path = str((out_root / f"{eff_task_id}.yaml").resolve())
        dump_task(spec, yaml_path)

    nshots_estimate = "depends on grid.shape; resolved at runtime"
    summary = (
        f"ForwardSpec ready: equation={args.equation}, backend={args.backend_impl}, "
        f"grid dh={args.dh}m, nt={args.nt}, dt={args.dt}s, Ricker fm={args.fm}Hz. "
        f"Line geometry: sources step={args.source_step} depth={args.source_depth}, "
        f"receivers step={args.receiver_step} depth={args.receiver_depth}. "
        f"Output → {args.output_dir}/<task_id>/."
    )
    return {
        "spec": spec.model_dump(mode="json"),
        "yaml_path": yaml_path,
        "task_id_hint": eff_task_id,
        "nshots_estimate": nshots_estimate,
        "summary": summary,
    }
