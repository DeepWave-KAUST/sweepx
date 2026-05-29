"""High-level orchestration tools that bundle build+run+visualize into ONE call.

Small LLMs (7B) reliably do single-step flows but stumble on multi-run
aggregation — running N wavefields and then feeding the N returned task_dirs to a
compare tool (they hallucinate the dirs, overflow context on retries). This tool
does that whole loop internally, so the model makes a single call and never
juggles task_dirs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from sweep_agent.tools import register


class CompareEquationWavefieldsParams(BaseModel):
    vp_path: str = Field(..., description="Velocity model .npy shared by all equations.")
    equations: list[str] = Field(
        ...,
        min_length=2,
        description="Equations to compare, e.g. ['Acoustic','AcousticVTI','AcousticTTI']. Use list_equations if unsure of names.",
    )
    out_path: str = Field(..., description="Output path for the side-by-side comparison PNG.")
    extra_models: dict[str, str] | None = Field(
        None,
        description=(
            "Shared pool of extra model files {name: path} (epsilon/delta/theta/vs/rho). Each "
            "equation automatically takes the ones it needs; vp comes from vp_path. e.g. "
            "{'epsilon':'/p/e.npy','delta':'/p/d.npy','theta':'/p/t.npy'}."
        ),
    )
    dh: float = Field(10.0, gt=0)
    dt: float = Field(1.0e-3, gt=0)
    nt: int = Field(300, ge=2)
    fm: float = Field(12.0, gt=0)
    abcn: int = Field(20, ge=0)
    spatial_order: int = Field(4)
    source_step: int = Field(50, ge=1)
    source_depth: int = Field(2, ge=0)
    receiver_step: int = Field(5, ge=1)
    receiver_depth: int = Field(2, ge=0)
    free_surface: bool = Field(False)
    device: str = Field("cpu")
    output_dir: str = Field("./sweep_runs")


@register(
    name="compare_equation_wavefields",
    description=(
        "Compare several equations' wavefields side by side in ONE call: runs a wavefield for each "
        "equation (same model + parameters) and draws the comparison figure. Internally builds, runs, "
        "and collects each task_dir, so you don't manage them yourself — ideal for 'isotropic vs VTI "
        "vs TTI wavefront' comparisons. Provide a shared extra_models pool (epsilon/delta/theta/...); "
        "each equation auto-takes what it needs. Returns the comparison image path."
    ),
    params_model=CompareEquationWavefieldsParams,
)
def compare_equation_wavefields(args: CompareEquationWavefieldsParams) -> dict[str, Any]:
    from sweep_agent.tools import registry

    try:
        from sweep_agent.tools.introspect import all_equations
        eqs_models = all_equations()
    except Exception as exc:
        return {"error": f"could not load equation list: {exc}"}

    pool = {"vp": args.vp_path, **(args.extra_models or {})}
    snap_t = args.nt - 1
    task_dirs: list[str] = []
    for eq in args.equations:
        required = eqs_models.get(eq)
        if required is None:
            return {"error": f"unknown equation '{eq}'. call list_equations for valid names."}
        missing = [m for m in required if m not in pool]
        if missing:
            return {"error": f"equation '{eq}' needs models {required}; missing {missing} — add them to extra_models."}
        em = {m: pool[m] for m in required if m != "vp"}
        b = registry.get("build_wavefield_spec").invoke({
            "vp_path": args.vp_path, "equation": eq, "extra_models": em or None,
            "dh": args.dh, "dt": args.dt, "nt": args.nt, "fm": args.fm,
            "snapshot_times": [snap_t], "abcn": args.abcn, "spatial_order": args.spatial_order,
            "source_step": args.source_step, "source_depth": args.source_depth,
            "receiver_step": args.receiver_step, "receiver_depth": args.receiver_depth,
            "free_surface": args.free_surface, "device": args.device,
            "backend_impl": "eager", "use_compile": False,
            "output_dir": args.output_dir, "task_id": f"cmpeq_{eq}",
        })
        if not b.ok or "error" in b.value:
            return {"error": f"build {eq}: {b.value.get('error') if b.ok else b.error}"}
        r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 400})
        if r.value.get("state") != "success":
            return {"error": f"run {eq}: {r.value.get('error')}"}
        task_dirs.append(r.value["task_dir"])

    c = registry.get("compare_wavefields").invoke({
        "task_dirs": task_dirs, "labels": list(args.equations), "abcn": args.abcn,
        "out_path": args.out_path, "snapshot_index": 0, "free_surface": args.free_surface,
    })
    if not c.ok or "error" in c.value:
        return {"error": f"compare: {c.value.get('error') if c.ok else c.error}"}
    return {"image_path": args.out_path, "equations": list(args.equations), "task_dirs": task_dirs}
