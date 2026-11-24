"""``run_forward_sweep`` — forward modelling straight through the `sweep` solver.

Every other modelling tool here routes through ``sweepx.tasks``: build a spec,
write YAML, hand it to ``TaskRunner``. This one drives the solver directly —
``sweep.equations`` + ``sweep.propagator.torch.PropTorch`` — with no
``sweepx.tasks``, ``sweepx.io`` or ``sweepx.loss`` anywhere in the path. It needs
only ``sweep`` + torch + numpy + matplotlib, so it runs on a laptop, CPU or
Apple-Silicon MPS.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from sweepx.agent.tools import register


def _require_sweep():
    """Return ``(modules, None)`` or ``(None, error_dict)`` when sweep is absent.

    Same contract as the rest of the package: a missing layer is reported as
    data, never raised, so the agent stays up on a base install.
    """
    try:
        import torch  # noqa: F401
        from sweep.equations import _equation_classes  # noqa: F401
        from sweep.propagator.torch import PropTorch  # noqa: F401
        from sweep.signal import ricker  # noqa: F401
    except ImportError as exc:
        return None, {
            "error": (
                f"the sweep solver (or torch) is not importable: {exc}. "
                "sweepx depends on it: reinstall it with `pip install sweep-solver`."
            )
        }
    import torch
    import sweep.equations as eq_mod
    from sweep.propagator.torch import PropTorch
    from sweep.signal import ricker

    return {"torch": torch, "eq_mod": eq_mod, "PropTorch": PropTorch, "ricker": ricker}, None


def _physics_report(*, nz, nx, dh, dt, nt, fm, vmin, vmax, src, rec_x, rec_z, abcn) -> dict[str, Any]:
    """The run in physical units, so the caller (often an LLM that picked the
    numbers) can see what it actually ran: a wrong dh or nt shows up here as a
    record length or a model width nobody asked for."""
    from sweepx.agent.tools.analysis import _CFL_MAX, _FMAX_FACTOR, _PPW_MARGINAL

    courant = vmax * dt * math.sqrt(2) / dh
    ppw = vmin / (_FMAX_FACTOR * fm * dh)
    warnings = []
    if courant > _CFL_MAX:
        warnings.append(f"Courant number {courant:.2f} > {_CFL_MAX}: likely unstable, reduce dt.")
    if ppw < _PPW_MARGINAL:
        warnings.append(f"{ppw:.1f} points per shortest wavelength (< {_PPW_MARGINAL}): grid dispersion; "
                        f"lower fm or use a finer grid.")
    sx, sz = src
    if sx >= nx or sz >= nz:
        warnings.append(f"source cell ({sx}, {sz}) lies outside the {nx} x {nz} model.")
    first_arrival = abs(sx - rec_x).max() * dh / vmax
    if nt * dt < first_arrival:
        warnings.append(f"record ({nt * dt:.3g} s) ends before the direct wave reaches the farthest "
                        f"receiver (~{first_arrival:.2g} s at vmax).")
    if abcn and dh * abcn < vmax / fm:
        warnings.append(f"PML {abcn} cells = {dh * abcn:g} m, under one dominant wavelength "
                        f"({vmax / fm:.0f} m at vmax): expect boundary reflections.")
    return {
        "record_length_s": round(nt * dt, 6),
        "model_extent_m": [round(nx * dh, 3), round(nz * dh, 3)],
        "dh_m": dh,
        "source_m": [round(sx * dh, 3), round(sz * dh, 3)],
        "receivers_x_m": [round(float(rec_x.min()) * dh, 3), round(float(rec_x.max()) * dh, 3),
                          round(float(rec_x[1] - rec_x[0]) * dh, 3) if len(rec_x) > 1 else None],
        "receiver_depth_m": round(rec_z * dh, 3),
        "vp_range": [round(vmin, 1), round(vmax, 1)],
        "courant": round(courant, 3),
        "points_per_wavelength": round(ppw, 2),
        "warnings": warnings,
    }


class RunForwardSweepParams(BaseModel):
    vp_path: str = Field(..., description="Velocity model .npy, shape (nz, nx) in m/s. Call inspect_file first to learn its shape.")
    dh: float = Field(..., gt=0, description="Grid spacing in metres (isotropic).")
    dt: float | None = Field(None, gt=0, description="Time step in seconds. OPTIONAL — if omitted, a CFL-stable dt = 0.4·dh/vmax is computed from the model, so you only need dh, record_length_s (or nt), and fm.")
    nt: int | None = Field(None, gt=0, description="Number of time samples. Give either nt or record_length_s.")
    record_length_s: float | None = Field(None, gt=0, description="Record length in seconds; nt = round(record_length_s / dt).")
    fm: float = Field(8.0, gt=0, description="Ricker peak frequency in Hz.")
    equation: str = Field("Acoustic", description="Equation class name from list_equations, e.g. 'Acoustic' or 'Elastic'. Multi-parameter equations work too — give their extra models via extra_models.")
    extra_models: dict[str, str] | None = Field(None, description="For equations needing more than vp (e.g. Elastic needs vp+vs+rho): map each EXTRA model name to its .npy path, like {\"vs\": \"vs.npy\", \"rho\": \"rho.npy\"}. vp always comes from vp_path; list_equations shows what each equation needs.")
    source_x_frac: float = Field(0.5, ge=0.0, le=1.0, description="Where the one source sits across the model: 0 = left edge, 0.5 = centre (default), 1 = right edge.")
    source_depth: int = Field(2, ge=0, description="Source depth as a row index (grid cells from the top).")
    receiver_depth: int = Field(1, ge=0, description="Receiver-line depth as a row index (grid cells from the top).")
    receiver_step: int = Field(1, ge=1, description="Place a receiver every Nth column.")
    spatial_order: int = Field(4, description="Finite-difference spatial order.")
    abcn: int = Field(40, ge=0, description="PML half-width in grid cells.")
    free_surface: bool = Field(False, description="Apply a free surface at the top boundary.")
    device: str = Field("auto", description="Compute device: 'auto' (default — picks cuda > mps > cpu automatically), or force 'cpu' / 'cuda' (NVIDIA) / 'mps' (Apple Silicon). The runtime-environment note lists what THIS machine has.")
    backend_impl: Literal["auto", "c", "eager"] = Field("auto", description="'auto': the compiled 'c' kernels on an NVIDIA GPU for equations that have them, else 'eager' (PyTorch).")
    out_dir: str = Field("./sweep_runs", description="Directory for the record .npy and the shot-gather PNG.")
    plot: bool = Field(True, description="Also render the shot gather to a PNG.")


@register(
    name="run_forward_sweep",
    description=(
        "Run 2-D forward modelling directly on the `sweep` solver and return the shot gather — "
        "no sweepx.tasks spec, no TaskRunner. This is the tool to use when only the core solver is "
        "installed. Supports Acoustic AND multi-parameter equations like Elastic (vp+vs+rho) or "
        "VTI/TTI — set `equation` and pass the extra models via `extra_models` (list_equations shows "
        "what each needs). Give vp_path, dh, and either nt or record_length_s (dt auto-computed if "
        "omitted); source at source_x_frac across the top (centre by default), receivers on a full-width surface line. "
        "Returns the record .npy, its shape ((nt, nrec) or (nt, nrec, ncomp)), a shot-gather PNG, and "
        "`physics`: the run in physical units (record length, source and receiver positions in metres, "
        "Courant number, points per wavelength) plus warnings — check it matches what was asked."
    ),
    params_model=RunForwardSweepParams,
)
def run_forward_sweep(args: RunForwardSweepParams) -> dict[str, Any]:
    mods, err = _require_sweep()
    if err is not None:
        return err
    torch = mods["torch"]
    ricker = mods["ricker"]
    # Resolve the device. "auto" (the default) picks the fastest one present —
    # cuda > mps > cpu — so a forward lands on the GPU when there is one without
    # the caller having to name it.
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
            device = "mps"
        else:
            device = "cpu"
    from sweepx.agent.tools.build_forward import _device_unavailable_reason
    dev_err = _device_unavailable_reason(device)
    if dev_err is not None:
        return {"error": dev_err}

    # Cap CPU threads. With use_compile=False the step runs interpreted, and torch
    # defaults to one intra-op thread per core. On a many-core box that badly
    # oversubscribes these small demo grids — every tiny op pays 20+ thread
    # launches — so a 0.6 s shot can take *minutes*. A modest cap is ~100x faster
    # here and still fine on an 8-core laptop. (No effect on CUDA/MPS.)
    if device == "cpu":
        import os
        try:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        except Exception:
            pass
    # 1. np.load the model first — its vmax fixes a stable dt when none was given.
    if not Path(args.vp_path).exists():
        return {"error": f"velocity model not found: {args.vp_path}"}
    vp_np = np.load(args.vp_path)
    if vp_np.ndim != 2:
        return {"error": f"vp must be 2-D (nz, nx); got shape {vp_np.shape}."}
    vp = torch.from_numpy(vp_np.astype(np.float32)).to(device)
    nz, nx = vp_np.shape
    # 2. Resolve dt: use args.dt, else a CFL-stable step (0.4·dh/vmax) so the caller
    #    only needs dh + record_length_s/nt + fm.
    dt = args.dt
    if dt is None:
        vmax = float(np.nanmax(np.abs(vp_np))) or 1500.0
        dt = 0.4 * args.dh / vmax
    # 3. Resolve nt: use args.nt, else round(args.record_length_s / dt).
    nt = args.nt
    if nt is None:
        if args.record_length_s is None:
            return {"error": "give either nt or record_length_s."}
        nt = round(args.record_length_s / dt)
    # 4. Build the Ricker wavelet with mods["ricker"] on a t axis of nt samples
    #    (give it a delay so the wavelet is causal).
    delay = 1.0 / args.fm
    t = np.arange(nt, dtype=np.float32) * dt - delay
    wavelet = torch.tensor((1.0e3 * ricker(t, f=args.fm)).astype(np.float32)).to(device)
    # 4. Build source and receiver index tensors. sources is (nshot, 2) as
    #    (x, z); receivers is (nshot, nrec, 2). Default the source to the middle
    #    column, receivers to a line across the model at receiver_depth.
    src_x = int(round(args.source_x_frac * (nx - 1)))
    # Integer numpy arrays, as sweepx.tasks passes them: the 'c' path requires
    # numpy (it copies and shifts them on the host); eager takes either.
    sources = np.array([[src_x, args.source_depth]], dtype=np.int64)
    rec_x = np.arange(0, nx, args.receiver_step, dtype=np.int64)
    rec_z = np.full_like(rec_x, args.receiver_depth)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]
    # 5. equation = mods["eq_mod"]._equation_classes()[args.equation](
    #        spatial_order=args.spatial_order, device=args.device, backend="torch")
    #    Reject equations needing more than vp — list_equations shows the models
    #    each one wants.
    classes = mods["eq_mod"]._equation_classes()
    if args.equation not in classes:
        return {"error": f"unknown equation '{args.equation}'. See list_equations."}
    equation = classes[args.equation](
        spatial_order=args.spatial_order, device=device, backend="torch"
    )
    # Build the model list the equation asks for, in its exact order. vp comes from
    # vp_path; every other model (vs, rho, epsilon, delta, …) from extra_models. The
    # solver supports Elastic/VTI/TTI directly, so this is all it takes to switch
    # equations on a base install — no sweepx.tasks involved.
    required = list(equation.models)
    extra = args.extra_models or {}
    models = []
    for name in required:
        if name == "vp":
            models.append(vp)
            continue
        path = extra.get(name)
        if path is None:
            return {"error": f"'{args.equation}' needs models {required}; model '{name}' is "
                             f"missing. Pass it via extra_models, e.g. "
                             f"{{\"{name}\": \"/path/to/{name}.npy\"}}."}
        if not Path(path).exists():
            return {"error": f"model '{name}' not found: {path}"}
        arr = np.load(path)
        if arr.shape != vp_np.shape:
            return {"error": f"model '{name}' shape {arr.shape} must match vp shape {vp_np.shape}."}
        models.append(torch.from_numpy(arr.astype(np.float32)).to(device))
    from sweepx.agent.tools.build_forward import _auto_impl
    impl = _auto_impl(args.backend_impl, device, args.equation)
    # 6. prop = mods["PropTorch"](equation, shape=vp.shape, dh=args.dh,
    #        dt=args.dt, abcn=args.abcn, free_surface=args.free_surface,
    #        use_ckpt=False, impl="eager")
    #    record = prop(wavelet, sources, receivers, models=[vp])  under no_grad.
    # use_compile=False keeps the eager path pure-interpreted torch. The default
    # eager step is torch.compile'd for speed, but that invokes the system C++
    # compiler (TorchInductor needs g++>=10 / a modern clang for -std=c++20). On a
    # plain `pip install sweepx` box with an old or absent compiler that raises a
    # CppCompileError, so the robust default for this tool is no compile — small
    # demo shots don't need it. impl='c' runs the shipped compiled kernels.
    impl_kwargs = {"impl": "c"} if impl == "c" else {"impl": "eager", "use_ckpt": False, "use_compile": False}
    prop = mods["PropTorch"](
        equation, shape=(nz, nx), dh=args.dh, dt=dt,
        abcn=args.abcn, free_surface=args.free_surface, **impl_kwargs,
    )
    with torch.no_grad():
        record = prop(wavelet, sources, receivers, models=models)
    # 7. record comes back as (nshot, nt, nrec, nfield) — squeeze to (nt, nrec),
    #    save it as .npy under out_dir.
    rec_np = np.squeeze(record.detach().cpu().numpy())  # (nt, nrec) or (nt, nrec, ncomp)
    if rec_np.ndim not in (2, 3):
        return {"error": f"unexpected record shape {tuple(record.shape)} -> {rec_np.shape}."}
    if not np.isfinite(rec_np).all():
        return {"error": "diverged (non-finite record) - reduce dt; see check_parameters"}
    out_dir = Path(args.out_dir).expanduser().resolve()  # absolute paths back to the caller
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "record.npy"
    np.save(record_path, rec_np)
    
    # 8. If args.plot: draw the gather with matplotlib (see
    #    visualize.py::_require_matplotlib for the guarded-import pattern and
    #    plot_shot_gather for a gather-plotting example) and save a PNG.
    image_path = None
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
        if plt is not None:
            # Multi-component records (e.g. Elastic vx/vz) get one panel per component.
            comps = rec_np if rec_np.ndim == 3 else rec_np[..., None]  # (nt, nrec, ncomp)
            ncomp = comps.shape[-1]
            fig, axes = plt.subplots(1, ncomp, figsize=(6 * ncomp, 5),
                                     constrained_layout=True, squeeze=False)
            for i in range(ncomp):
                panel = comps[..., i]
                pct = np.percentile(np.abs(panel), 99.0) or 1.0
                ax = axes[0, i]
                ax.imshow(panel, cmap="seismic", vmin=-pct, vmax=pct,
                          aspect="auto", extent=[rec_x.min() * args.dh, rec_x.max() * args.dh, nt * dt, 0])
                ax.set_xlabel("receiver x (m)"); ax.set_ylabel("time (s)")
                ax.set_title(f"{args.equation} shot gather" + (f" — comp {i}" if ncomp > 1 else ""))
            png_path = out_dir / "shot_gather.png"
            fig.savefig(png_path, dpi=140); plt.close(fig)
            image_path = str(png_path)




    # 9. Return {"record_path", "image_path", "shape", "device", "nt", "summary"}.
    # Keep every import inside this function or inside _require_sweep, so the
    # module still imports on a base install with no solver present.
    # ------------------------------------------------------------------
    physics = _physics_report(
        nz=nz, nx=nx, dh=args.dh, dt=dt, nt=nt, fm=args.fm,
        vmin=float(np.nanmin(vp_np)), vmax=float(np.nanmax(vp_np)),
        src=(src_x, args.source_depth), rec_x=rec_x, rec_z=args.receiver_depth, abcn=args.abcn,
    )
    return {
        "record_path": str(record_path),
        "image_path": image_path,
        "shape": list(rec_np.shape),
        "device": device,
        "impl": impl,
        "nt": nt,
        "dt": dt,
        "physics": physics,
        "summary": f"Ran {args.equation} forward ({'+'.join(required)}): {rec_np.shape[0]} samples x "
                   f"{rec_np.shape[1]} receivers"
                   + (f" x {rec_np.shape[2]} components" if rec_np.ndim == 3 else "")
                   + f" on {device}/{impl}: {physics['record_length_s']:g} s record, dh={args.dh:g} m, "
                   f"source at x={physics['source_m'][0]:g} m z={physics['source_m'][1]:g} m."
                   + (" WARNINGS: " + " ".join(physics["warnings"]) if physics["warnings"] else ""),
    }