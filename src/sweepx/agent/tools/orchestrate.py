"""High-level orchestration tools that bundle build+run+visualize into ONE call.

Small LLMs (7B) reliably do single-step flows but stumble on multi-run
aggregation — running N wavefields and then feeding the N returned task_dirs to a
compare tool (they hallucinate the dirs, overflow context on retries). This tool
does that whole loop internally, so the model makes a single call and never
juggles task_dirs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from sweepx.agent.tools import register
from sweepx.agent.tools.build_forward import _output_file
from sweepx.agent.tools.build_wavefield import _p_and_s_source


def _deliver(produced: str | None, dest: Path) -> str:
    """Copy a figure a tool wrote under its task_dir to where the caller asked."""
    if not produced:
        return str(dest)
    try:
        if Path(produced).resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(produced, dest)
        return str(dest)
    except OSError:
        return produced


class CompareEquationWavefieldsParams(BaseModel):
    vp_path: str = Field(..., description="Velocity model .npy shared by all equations.")
    equations: list[str] = Field(
        ...,
        min_length=2,
        description="Equations to compare, e.g. ['Acoustic','AcousticVTI','AcousticTTI']. Use list_equations if unsure of names.",
    )
    out_path: str = Field(..., description="Output PNG path for the side-by-side comparison (a directory gets wavefield_comparison.png).")
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
    device: str = Field("auto", description="'auto' (the GPU if there is one), 'cuda', 'mps' or 'cpu'.")
    output_dir: str = Field("./sweep_runs")
    source_at_center: bool = Field(
        True,
        description=(
            "Place a SINGLE source at the model centre — the canonical view for "
            "wavefront-shape comparison (Duveneck Fig.1/2 style: full circular wavefront "
            "for isotropic, elliptical for VTI, tilted for TTI). When True (default) this "
            "overrides source_step/source_depth and uses an explicit one-shot geometry. "
            "Set False only if you specifically want surface acquisition (poor view of "
            "anisotropy)."
        ),
    )


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
    from sweepx.agent.tools import registry

    try:
        from sweepx.agent.tools.introspect import all_equations
        eqs_models = all_equations()
    except Exception as exc:
        return {"error": f"could not load equation list: {exc}"}

    pool = {"vp": args.vp_path, **(args.extra_models or {})}
    snap_t = args.nt - 1

    # Build the centred-source geometry override ONCE (same for every equation).
    # ExplicitGeometry uses (x, z) for 2-D and (x, y, z) for 3-D — same convention
    # as build_forward._resolve_geometry. A single source at the model centre with
    # a sparse top-row receiver line keeps the figure focused on wavefront SHAPE.
    extra_override: dict[str, Any] | None = None
    if args.source_at_center:
        import numpy as _np
        try:
            vp_shape = tuple(_np.load(args.vp_path, mmap_mode="r").shape)
        except Exception as exc:
            return {"error": f"could not read shape of vp_path '{args.vp_path}': {exc}"}
        rstep = max(args.receiver_step, 1)
        if len(vp_shape) == 2:
            nz, nx = vp_shape
            src = [nx // 2, nz // 2]
            recs = [[x, max(args.receiver_depth, 0)] for x in range(0, nx, rstep)] or [[nx // 2, 0]]
        elif len(vp_shape) == 3:
            nz, ny, nx = vp_shape
            src = [nx // 2, ny // 2, nz // 2]
            recs = [
                [x, y, max(args.receiver_depth, 0)]
                for y in range(0, ny, rstep)
                for x in range(0, nx, rstep)
            ] or [[nx // 2, ny // 2, 0]]
        else:
            return {"error": f"unsupported model ndim={len(vp_shape)}; expected 2-D or 3-D."}
        extra_override = {"geometry": {"kind": "explicit", "sources": [src], "receivers": recs}}

    task_dirs: list[str] = []
    for eq in args.equations:
        required = eqs_models.get(eq)
        if required is None:
            return {"error": f"unknown equation '{eq}'. call list_equations for valid names."}
        missing = [m for m in required if m not in pool]
        if missing:
            return {"error": f"equation '{eq}' needs models {required}; missing {missing} — add them to extra_models."}
        em = {m: pool[m] for m in required if m != "vp"}
        build_args: dict[str, Any] = {
            "vp_path": args.vp_path, "equation": eq, "extra_models": em or None,
            "dh": args.dh, "dt": args.dt, "nt": args.nt, "fm": args.fm,
            "snapshot_times": [snap_t], "abcn": args.abcn, "spatial_order": args.spatial_order,
            "source_step": args.source_step, "source_depth": args.source_depth,
            "receiver_step": args.receiver_step, "receiver_depth": args.receiver_depth,
            "free_surface": args.free_surface, "device": args.device,
            "backend_impl": "eager", "use_compile": False,
            "output_dir": args.output_dir, "task_id": f"cmpeq_{eq}",
        }
        if extra_override is not None:
            build_args["extra"] = extra_override
        b = registry.get("build_wavefield_spec").invoke(build_args)
        if not b.ok or "error" in b.value:
            return {"error": f"build {eq}: {b.value.get('error') if b.ok else b.error}"}
        r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 400})
        if r.value.get("state") != "success":
            return {"error": f"run {eq}: {r.value.get('error')}"}
        task_dirs.append(r.value["task_dir"])

    out = _output_file(args.out_path, "wavefield_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    c = registry.get("compare_wavefields").invoke({
        "task_dirs": task_dirs, "labels": list(args.equations), "abcn": args.abcn,
        "out_path": str(out), "snapshot_index": 0, "free_surface": args.free_surface,
    })
    if not c.ok or "error" in c.value:
        return {"error": f"compare: {c.value.get('error') if c.ok else c.error}"}
    return {"image_path": str(out), "equations": list(args.equations), "task_dirs": task_dirs}


class AnimateWavefieldParams(BaseModel):
    vp_path: str = Field(..., description="Primary velocity model .npy (vp, or vp0 for elastic-TTI — the equation's FIRST model).")
    out_path: str = Field(..., description="Output GIF path (a directory gets wavefield.gif).")
    equation: str | None = Field(
        None,
        description=(
            "Equation name. Leave None to auto-detect from the model set you provide "
            "(e.g. vp+vs+rho → Elastic). Set explicitly only to disambiguate."
        ),
    )
    extra_models: dict[str, str] | None = Field(
        None,
        description=(
            "Extra model files beyond the primary one, keyed by model name: "
            "{'vs':..,'rho':..} for Elastic; {'epsilon':..,'delta':..,'theta':..} for "
            "AcousticTTI; the full set (vs0/rho/epsilon/delta/gamma/theta/phi) for ElasticTTI. "
            "theta/phi are in RADIANS. vp comes from vp_path."
        ),
    )
    dh: float = Field(10.0, gt=0)
    dt: float = Field(1.0e-3, gt=0)
    nt: int = Field(300, ge=2)
    fm: float = Field(12.0, gt=0)
    abcn: int = Field(20, ge=0)
    spatial_order: int = Field(4)
    n_frames: int = Field(12, ge=2, le=60, description="How many evenly-spaced snapshots to animate.")
    field: int = Field(0, ge=0, description="Wavefield component index to animate (0 = first field, e.g. vx for elastic).")
    source_at_center: bool = Field(True, description="Centre the source so the wavefront radiates in all directions (the usual view).")
    source_type: list[str] | None = Field(
        None,
        description=(
            "Source field components. Leave None: acoustic equations take their own; an elastic "
            "medium gets a vertical force ['vz'], which radiates both P and S. ['sxx','szz'] is an "
            "explosion: P only, no S."
        ),
    )
    free_surface: bool = Field(False)
    topography_path: str | None = Field(
        None,
        description=(
            "1-D surface-row .npy for an irregular free surface; uses a curvilinear equation, puts "
            "the source just below the surface in the middle of the model, and shows the wave on "
            "the physical grid with the air masked."
        ),
    )
    device: str = Field("auto", description="'auto' (the GPU if there is one), 'cuda', 'mps' or 'cpu'.")
    output_dir: str = Field("./sweep_runs")


@register(
    name="animate_wavefield",
    description=(
        "Simulate a wavefield and animate it to a GIF in ONE call — build + run + make_wavefield_gif, "
        "so you don't manage snapshot_times, the task_dir, or the source geometry. Ideal for 'show me "
        "how the wave propagates' on a homogeneous model (elastic P/S rings, anisotropic wavefronts). "
        "Auto-detects the equation from the models you pass (vp_path + extra_models), centres the source, "
        "and picks evenly-spaced frames. Pass topography_path for an irregular free surface. Returns the "
        "gif_path. For comparing SEVERAL equations side by side, use compare_equation_wavefields instead."
    ),
    params_model=AnimateWavefieldParams,
)
def animate_wavefield(args: AnimateWavefieldParams) -> dict[str, Any]:
    from sweepx.agent.tools import registry

    # 1. Resolve the equation from the extra models when not given. vp_path is
    # the primary model; extra_models are everything after it.
    equation = args.equation
    extra_keys = set((args.extra_models or {}).keys())
    if equation is None:
        if not extra_keys:
            # vp only: topography → curvilinear, otherwise plain acoustic.
            equation = "AcousticCurvilinear" if args.topography_path is not None else "Acoustic"
        else:
            try:
                import numpy as _np
                ndim = len(_np.load(args.vp_path, mmap_mode="r").shape)
            except Exception:
                ndim = None
            try:
                from sweepx.agent.tools.introspect import equations_for_models
                matches = equations_for_models(extra_keys, ndim=ndim)
            except Exception as exc:
                return {"error": f"could not load equation list to auto-detect: {exc}"}
            if not matches:
                return {"error": (
                    f"could not auto-detect an equation from extra_models {sorted(extra_keys)}; "
                    f"pass equation= explicitly (see list_equations)."
                )}
            equation = matches[0]

    # 2. Evenly spaced snapshot frames over (0, nt) — skip t=0 (silent).
    n = max(2, min(args.n_frames, args.nt - 1))
    step = args.nt / (n + 1)
    snaps = sorted({min(args.nt - 1, max(1, int(round(step * (i + 1))))) for i in range(n)})

    # 3. Build the wavefield spec.
    source_type = args.source_type if args.source_type is not None else _p_and_s_source(equation)
    build_args: dict[str, Any] = {
        "vp_path": args.vp_path, "equation": equation, "extra_models": args.extra_models or None,
        "dh": args.dh, "dt": args.dt, "nt": args.nt, "fm": args.fm,
        "snapshot_times": snaps, "abcn": args.abcn, "spatial_order": args.spatial_order,
        "source_at_center": args.source_at_center, "free_surface": args.free_surface,
        "source_type": source_type,
        "device": args.device, "backend_impl": "eager", "use_compile": False,
        "output_dir": args.output_dir, "task_id": f"anim_{equation}",
    }
    if args.topography_path is not None:
        # One source, mid-model, two rows below the surface (the depth counts
        # from the surface); a corner source would hardly reach the relief.
        build_args.update(topography=args.topography_path, source_at_center=False,
                          source_x_frac=0.5, source_depth=2)
    b = registry.get("build_wavefield_spec").invoke(build_args)
    if not b.ok or "error" in b.value:
        return {"error": f"build {equation}: {b.value.get('error') if b.ok else b.error}"}

    # 4. Run.
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 600})
    if r.value.get("state") != "success":
        return {"error": f"run {equation}: {r.value.get('error')}"}

    # 5. Animate. make_wavefield_gif writes to <task_dir>/output/<out_name>;
    # copy it to the caller's out_path afterwards so they get it where asked.
    dest = _output_file(args.out_path, "wavefield.gif")
    gif_args: dict[str, Any] = {
        "task_dir": r.value["task_dir"], "abcn": args.abcn, "field": args.field,
        "free_surface": args.free_surface, "out_name": dest.name,
    }
    if args.topography_path is not None:
        gif_args["topography_path"] = args.topography_path
        nx = int(np.load(args.vp_path, mmap_mode="r").shape[-1])
        gif_args["sources"] = [[int(round(0.5 * (nx - 1))), 2]]   # [x, rows below the surface]
    g = registry.get("make_wavefield_gif").invoke(gif_args)
    if not g.ok or "error" in g.value:
        return {"error": f"animate: {g.value.get('error') if g.ok else g.error}"}
    final = _deliver(g.value.get("gif_path"), dest)
    return {
        "gif_path": final,
        "equation": equation,
        "source_type": source_type,
        "n_frames": g.value.get("n_frames", len(snaps)),
        "task_dir": r.value["task_dir"],
    }


class RunFwiParams(BaseModel):
    init_model_path: str = Field(..., description="Initial velocity model .npy the inversion starts from.")
    out_dir: str = Field(..., description="Output directory for the run + result figures.")
    synthetic_true_vp_path: str | None = Field(
        None,
        description="Ground-truth vp .npy for a SYNTHETIC demo/test: observed data is generated from it, and it's shown next to the result. Use this OR an obs_* source.",
    )
    obs_npy_path: str | None = Field(None, description="Observed data tensor .npy (nshots,nt,nrec,nch). Use for real data instead of synthetic_true_vp_path.")
    obs_segy_path: str | None = Field(None, description="Observed-data SEG-Y. Real-data alternative.")
    equation: str = Field("Acoustic", description="Wave equation; Acoustic for standard acoustic FWI.")
    dh: float = Field(..., gt=0)
    dt: float = Field(1.0e-3, gt=0, description="Time step (s).")
    record_length_s: float | None = Field(None, gt=0, description="Record length in seconds (e.g. 0.6). Give this OR nt.")
    nt: int | None = Field(None, ge=1, description="Time samples; give this OR record_length_s.")
    fm: float = Field(8.0, gt=0)
    optimizer: Literal["adam", "sgd", "lbfgs"] = Field("adam")
    lr: float = Field(
        20.0, gt=0,
        description=(
            "Step size in m/s. Adam/SGD: vp moves by about this much per epoch at first. "
            "L-BFGS: the size of its first step; later steps come from the curvature."
        ),
    )
    loss: str = Field("mse")
    epochs: int = Field(40, ge=1, description="Inversion iterations; ~30-60 gives a visible result on a small demo.")
    batchsize: int | None = Field(
        None, ge=1,
        description="Shots per epoch. None = every shot, so each epoch is one full-gradient step and the loss curve compares epoch to epoch.",
    )
    vp_min: float | None = Field(None, description="Lower bound on vp (m/s). None = 0.75 x the initial model's minimum.")
    vp_max: float | None = Field(None, description="Upper bound on vp (m/s). None = 1.75 x the initial model's maximum.")
    source_step: int = Field(20, ge=1)
    source_depth: int = Field(2, ge=0)
    receiver_step: int = Field(2, ge=1)
    receiver_depth: int = Field(4, ge=0)
    abcn: int = Field(20, ge=0)
    spatial_order: int = Field(8)
    device: str = Field("auto", description="'auto' (the GPU if there is one), 'cuda', 'mps' or 'cpu'.")


@register(
    name="run_fwi",
    description=(
        "Run a full FWI inversion and visualise the result in ONE call — build_fwi_spec + run_task + "
        "plot_model (initial vs inverted vs true + residual) + plot_convergence (loss curve) — so you "
        "don't manage the task_dir or the figure steps. For a SYNTHETIC demo pass synthetic_true_vp_path "
        "(observed data is generated from it); for REAL data pass obs_npy_path / obs_segy_path instead. "
        "inspect_file the inputs first. Returns the inverted model + comparison + convergence image paths."
    ),
    params_model=RunFwiParams,
)
def run_fwi(args: RunFwiParams) -> dict[str, Any]:
    from sweepx.agent.tools import registry

    sources = [s for s in (args.synthetic_true_vp_path, args.obs_npy_path, args.obs_segy_path) if s]
    if len(sources) != 1:
        return {"error": "provide EXACTLY one observed-data source: synthetic_true_vp_path (demo) OR obs_npy_path OR obs_segy_path."}
    try:
        init = np.load(args.init_model_path)
    except Exception as exc:
        return {"error": f"could not read init_model_path '{args.init_model_path}': {exc}"}

    build_args: dict[str, Any] = {
        "init_model_path": args.init_model_path, "equation": args.equation,
        "synthetic_true_vp_path": args.synthetic_true_vp_path,
        "obs_npy_path": args.obs_npy_path, "obs_segy_path": args.obs_segy_path,
        "dh": args.dh, "dt": args.dt, "nt": args.nt, "record_length_s": args.record_length_s,
        "fm": args.fm, "optimizer": args.optimizer, "lr": args.lr, "loss": args.loss,
        "epochs": args.epochs, "batchsize": args.batchsize, "vp_min": args.vp_min, "vp_max": args.vp_max,
        "source_step": args.source_step, "source_depth": args.source_depth,
        "receiver_step": args.receiver_step, "receiver_depth": args.receiver_depth,
        "abcn": args.abcn, "spatial_order": args.spatial_order,
        "device": args.device, "backend_impl": "auto", "use_compile": False,
        "output_dir": args.out_dir, "task_id": "fwi",
    }
    b = registry.get("build_fwi_spec").invoke(build_args)
    if not b.ok or "error" in b.value:
        return {"error": f"build_fwi: {b.value.get('error') if b.ok else b.error}"}

    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 1800})
    if r.value.get("state") != "success":
        return {"error": f"run_fwi: {r.value.get('error')}"}
    task_dir = r.value["task_dir"]
    inverted = Path(task_dir) / "output" / "inverted_vp.npy"
    if inverted.is_file() and np.array_equal(np.load(inverted), init):
        return {"error": (
            f"the inversion ran but did not change the model (lr={args.lr}, "
            f"optimizer={args.optimizer}): increase lr and run again."), "task_dir": task_dir}

    spec = b.value["spec"]
    bounds = spec["model_bounds"]["vp"]
    out: dict[str, Any] = {"task_dir": task_dir, "state": "success", "batchsize": spec["batchsize"],
                           "vp_bounds": [bounds["min"], bounds["max"]]}
    m = registry.get("plot_model").invoke({
        "task_dir": task_dir, "true_model_path": args.synthetic_true_vp_path,
        "init_model_path": args.init_model_path, "dh": args.dh, "title": "FWI result",
    })
    if m.ok and "error" not in m.value:
        out["model_image"] = m.value.get("image_path")
    else:
        out["model_image_error"] = m.value.get("error") if m.ok else m.error
    c = registry.get("plot_convergence").invoke({"task_dir": task_dir})
    if c.ok and "error" not in c.value:
        out["convergence_image"] = c.value.get("image_path")
    out["inverted_vp"] = str(inverted)
    return out


class RunForwardPlotParams(BaseModel):
    vp_path: str = Field(..., description="Velocity model .npy.")
    out_path: str = Field(..., description="Output PNG path for the shot gather (a directory gets shot_gather.png).")
    equation: str = Field("Acoustic", description="Wave equation; Acoustic for standard acoustic.")
    extra_models: dict[str, str] | None = Field(None, description="Extra model files for non-acoustic equations (vs/rho/epsilon/...).")
    dh: float = Field(..., gt=0)
    dt: float = Field(1.0e-3, gt=0)
    record_length_s: float | None = Field(None, gt=0, description="Record length in seconds (e.g. 1.5). Give this OR nt.")
    nt: int | None = Field(None, ge=1, description="Time samples; give this OR record_length_s.")
    fm: float = Field(10.0, gt=0)
    source_depth: int = Field(1, ge=0)
    source_x_frac: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Where the one source sits across x (0=left, 0.5=centre, 1=right). Use for 'source on the left/right'.",
    )
    receiver_step: int = Field(2, ge=1)
    receiver_depth: int = Field(4, ge=0)
    abcn: int = Field(20, ge=0)
    spatial_order: int = Field(8)
    free_surface: bool = Field(False)
    wiggle: bool = Field(False, description="Wiggle display instead of image.")
    device: str = Field("auto", description="'auto' (the GPU if there is one), 'cuda', 'mps' or 'cpu'.")
    output_dir: str = Field("./sweep_runs")


@register(
    name="run_forward_and_plot",
    description=(
        "Run forward modelling AND plot the shot gather in ONE call — build_forward_spec + run_task + "
        "plot_shot_gather, managing the task_dir internally so the gather is ALWAYS from the run you just "
        "did (never a stale one). This is the preferred way to do 'forward + shot gather / 正演并画炮记录'. "
        "Give record_length_s (seconds) or nt. Returns the gather image path + the record shape."
    ),
    params_model=RunForwardPlotParams,
)
def run_forward_and_plot(args: RunForwardPlotParams) -> dict[str, Any]:
    from sweepx.agent.tools import registry

    # One shot: the tool draws one gather, so it models one.
    b = registry.get("build_forward_spec").invoke({
        "vp_path": args.vp_path, "equation": args.equation, "extra_models": args.extra_models or None,
        "dh": args.dh, "dt": args.dt, "nt": args.nt, "record_length_s": args.record_length_s, "fm": args.fm,
        "source_depth": args.source_depth, "source_x_frac": args.source_x_frac,
        "receiver_step": args.receiver_step, "receiver_depth": args.receiver_depth,
        "abcn": args.abcn, "spatial_order": args.spatial_order, "free_surface": args.free_surface,
        "device": args.device, "backend_impl": "auto", "use_compile": False,
        "output_dir": args.output_dir, "task_id": "forward",
    })
    if not b.ok or "error" in b.value:
        return {"error": f"build_forward: {b.value.get('error') if b.ok else b.error}"}
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 1200})
    if r.value.get("state") != "success":
        return {"error": f"run: {r.value.get('error')}"}
    task_dir = r.value["task_dir"]  # the REAL task_dir — no hallucination

    dest = _output_file(args.out_path, "shot_gather.png")
    g = registry.get("plot_shot_gather").invoke({
        "task_dir": task_dir, "dt": args.dt, "dh": args.dh, "wiggle": args.wiggle, "out_name": dest.name,
    })
    if not g.ok or "error" in g.value:
        return {"error": f"plot: {g.value.get('error') if g.ok else g.error}", "task_dir": task_dir}
    final = _deliver(g.value.get("image_path"), dest)
    return {"image_path": final, "task_dir": task_dir, "shape": g.value.get("shape"), "equation": b.value["spec"]["physics"]["equation"]}


class RunMultiscaleFwiParams(BaseModel):
    init_model_path: str = Field(..., description="Initial velocity model .npy.")
    synthetic_true_vp_path: str = Field(..., description="Ground-truth vp .npy (observed data is generated from it). Synthetic-demo multiscale.")
    out_dir: str = Field(..., description="Output directory; each frequency band gets a sub-run.")
    frequencies: list[float] = Field(
        ..., min_length=1,
        description="Ricker peak frequencies, LOW→HIGH (e.g. [4, 7, 10]). Each band inverts starting from the previous band's result (frequency continuation).",
    )
    dh: float = Field(..., gt=0)
    dt: float = Field(1.0e-3, gt=0, description="Time step (s).")
    record_length_s: float | None = Field(None, gt=0, description="Record length in seconds. Give this OR nt.")
    nt: int | None = Field(None, ge=1, description="Time samples; give this OR record_length_s.")
    epochs_per_band: int = Field(20, ge=1, description="Inversion iterations within each frequency band.")
    lr: float = Field(20.0, gt=0)
    vp_min: float | None = Field(None)
    vp_max: float | None = Field(None)
    source_step: int = Field(20, ge=1)
    receiver_step: int = Field(2, ge=1)
    device: str = Field("auto", description="'auto' (the GPU if there is one), 'cuda', 'mps' or 'cpu'.")


@register(
    name="run_multiscale_fwi",
    description=(
        "Run MULTI-SCALE (frequency-continuation) FWI: invert from low to high frequency, each band "
        "starting from the previous band's inverted model — the standard recipe to avoid cycle-skipping. "
        "Pass frequencies low→high (e.g. [4,7,10]). Internally chains run_fwi per band. Returns each "
        "band's result + the final inverted model & comparison image. Synthetic demo (uses "
        "synthetic_true_vp_path). Use for '多尺度/分频段反演' / robust FWI."
    ),
    params_model=RunMultiscaleFwiParams,
)
def run_multiscale_fwi(args: RunMultiscaleFwiParams) -> dict[str, Any]:
    from sweepx.agent.tools import registry

    freqs = sorted(args.frequencies)
    current_init = args.init_model_path
    bands: list[dict[str, Any]] = []
    for i, fm in enumerate(freqs):
        band_dir = str(Path(args.out_dir).expanduser() / f"band{i+1}_{fm:g}Hz")
        r = registry.get("run_fwi").invoke({
            "init_model_path": current_init, "synthetic_true_vp_path": args.synthetic_true_vp_path,
            "out_dir": band_dir, "dh": args.dh, "dt": args.dt, "nt": args.nt,
            "record_length_s": args.record_length_s, "fm": fm,
            "lr": args.lr, "epochs": args.epochs_per_band, "vp_min": args.vp_min, "vp_max": args.vp_max,
            "source_step": args.source_step, "receiver_step": args.receiver_step, "device": args.device,
        })
        if not r.ok or r.value.get("state") != "success":
            return {"error": f"band {i+1} ({fm} Hz) failed: {r.value.get('error') if r.ok else r.error}", "bands_done": bands}
        inv = r.value.get("inverted_vp")
        bands.append({"band": i + 1, "fm_hz": fm, "inverted_vp": inv,
                      "model_image": r.value.get("model_image"), "convergence_image": r.value.get("convergence_image")})
        current_init = inv  # frequency continuation: next band starts here

    return {
        "state": "success",
        "frequencies": freqs,
        "bands": bands,
        "final_inverted_vp": bands[-1]["inverted_vp"],
        "final_model_image": bands[-1]["model_image"],
    }
