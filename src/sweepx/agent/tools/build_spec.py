"""``build_spec`` — generic, schema-validated builder for ANY sweep task.

Where ``build_forward_spec`` is a flat shortcut for the common acoustic-forward
case, ``build_spec`` accepts a full spec dict for any task_type (forward /
wavefield / fwi / lsrtm / rtm / introspect) and validates it against the real
``sweepx.tasks`` Pydantic schema. Paired with ``describe_task_schema`` (to learn
the fields) it covers the entire parameter surface — every loss, optimizer,
scheduler, geometry, boundary/checkpoint option, multiscale stage, NN reparam,
data/model plan, etc.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sweepx.agent.tools import register
from sweepx.agent.tools._modelmeta import read_model_meta

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text).strip("-_") or "task"


def _model_paths(node: Any):
    """Every ``"path": "<file>.npy"`` value in a spec dict (models, true models, ...)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "path" and isinstance(v, str) and v.endswith(".npy"):
                yield v
            else:
                yield from _model_paths(v)
    elif isinstance(node, list):
        for v in node:
            yield from _model_paths(v)


def _check_grid_dh(raw: dict[str, Any]) -> str | None:
    """Fill ``grid.dh`` from the models' recorded dh when it is missing; reject a
    spec whose ``grid.dh`` (or whose models) disagree with what the files record."""
    recorded = {p: float(m["dh"]) for p in _model_paths(raw) if (m := read_model_meta(p)) is not None}
    if not recorded:
        return None
    if len(set(recorded.values())) > 1:
        return "the spec's model files record different grid spacings: " + ", ".join(
            f"{p}: {v:g} m" for p, v in recorded.items())
    ref = next(iter(recorded.values()))
    grid = raw.get("grid")
    if grid is None:
        raw["grid"] = {"dh": ref}
        return None
    if isinstance(grid, dict):
        if grid.get("dh") is None:
            grid["dh"] = ref
        elif isinstance(grid["dh"], (int, float)) and abs(float(grid["dh"]) - ref) > 1e-6 * ref:
            return (f"grid.dh={grid['dh']:g} m disagrees with the {ref:g} m recorded for "
                    f"{next(iter(recorded))}. Drop grid.dh or set it to {ref:g}.")
    return None


class BuildSpecParams(BaseModel):
    task_type: str = Field(
        ...,
        description="forward / wavefield / fwi / lsrtm / rtm / introspect.",
    )
    spec: dict[str, Any] = Field(
        ...,
        description=(
            "The full task body as a JSON object (every field except task_type, which is injected). "
            "Call describe_task_schema(task_type[, section]) FIRST to learn the exact fields and "
            "nested 'kind' variants. Nested fields (grid/time/wavelet/geometry/physics/backend/"
            "optimizer/scheduler/loss/obs/stages/...) are themselves objects; discriminated unions "
            "(wavelet/geometry/optimizer/scheduler) select a variant via their 'kind' key."
        ),
    )
    output_dir: str = Field("./sweep_runs", description="Parent dir for the run; <output_dir>/<task_id>/ holds artifacts.")
    task_id: str | None = Field(None, description="Subdir + YAML name; auto if omitted.")
    save_yaml: bool = Field(True, description="Write the validated spec to <output_dir>/<task_id>.yaml.")


@register(
    name="build_spec",
    description=(
        "Build and validate a COMPLETE sweep task spec of any type from a dict. This is the "
        "general-purpose builder covering every sweepx.tasks parameter (all losses, optimizers, "
        "schedulers, geometries, boundary/checkpoint memory options, multiscale stages, NN reparam, "
        "data/model plans, ...). ALWAYS call describe_task_schema(task_type[, section]) first to "
        "learn the field names and nested 'kind' variants — never guess. Returns the validated spec "
        "+ a YAML path; then call run_task(yaml_path). Validation errors are returned as data so you "
        "can fix the dict and retry. For a plain acoustic forward, build_forward_spec is simpler."
    ),
    params_model=BuildSpecParams,
)
def build_spec(args: BuildSpecParams) -> dict[str, Any]:
    try:
        from sweepx.tasks.yaml_io import dump_task, load_task_from_dict
    except ImportError as exc:
        return {"error": f"sweepx.tasks is not importable: {exc}"}

    raw: dict[str, Any] = dict(args.spec)
    raw["task_type"] = args.task_type
    dh_error = _check_grid_dh(raw)
    if dh_error is not None:
        return {"error": dh_error}
    raw.setdefault("output_dir", args.output_dir)
    if args.task_id is not None:
        raw["task_id"] = args.task_id

    try:
        spec = load_task_from_dict(raw)
    except Exception as exc:
        return {
            "error": f"spec validation failed for task_type='{args.task_type}': {exc}",
            "hint": "call describe_task_schema to check field names/types, then fix the dict.",
            "spec_attempted": raw,
        }

    eff_task_id = args.task_id or f"{args.task_type}-spec"
    yaml_path: str | None = None
    if args.save_yaml:
        out_root = Path(args.output_dir).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)
        yaml_path = str((out_root / f"{_slug(eff_task_id)}.yaml").resolve())
        dump_task(spec, yaml_path)

    return {
        "spec": spec.model_dump(mode="json"),
        "yaml_path": yaml_path,
        "task_type": args.task_type,
        "summary": (
            f"Validated {args.task_type} spec ({len(raw)} top-level fields). "
            f"Run it with run_task(yaml_path=...)."
        ),
    }
