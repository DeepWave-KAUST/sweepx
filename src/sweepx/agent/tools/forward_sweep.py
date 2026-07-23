"""``run_forward_sweep`` — forward modelling straight through the `sweep` solver.

Every other modelling tool here routes through ``sweep_tasks``: build a spec,
write YAML, hand it to ``TaskRunner``. That tier is unreleased, so on a plain
``pip install -e .`` plus ``sweep`` the agent can inspect files and list
equations but cannot actually run a shot.

This tool closes that gap by driving the solver directly —
``sweep.equations`` + ``sweep.propagator.torch.PropTorch`` — with no
``sweep_tasks``, ``sweep_io`` or ``sweep_loss`` anywhere in the path. It needs
only ``sweep`` + torch + numpy + matplotlib, so it runs on a laptop, CPU or
Apple-Silicon MPS.

STATUS: scaffold. The parameter surface, registration and dependency guard are
in place; the modelling body is the exercise — see the TODO in
``run_forward_sweep`` and ``tests/test_forward_sweep.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sweep_agent.tools import register


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
                "Install a PyTorch environment, then `pip install .` from a clone of "
                "https://github.com/DeepWave-KAUST/sweep"
            )
        }
    import torch
    import sweep.equations as eq_mod
    from sweep.propagator.torch import PropTorch
    from sweep.signal import ricker

    return {"torch": torch, "eq_mod": eq_mod, "PropTorch": PropTorch, "ricker": ricker}, None


class RunForwardSweepParams(BaseModel):
    vp_path: str = Field(..., description="Velocity model .npy, shape (nz, nx) in m/s. Call inspect_file first to learn its shape.")
    dh: float = Field(..., gt=0, description="Grid spacing in metres (isotropic).")
    dt: float = Field(..., gt=0, description="Time step in seconds. Call check_parameters first — a too-large dt is unstable.")
    nt: int | None = Field(None, gt=0, description="Number of time samples. Give either nt or record_length_s.")
    record_length_s: float | None = Field(None, gt=0, description="Record length in seconds; nt = round(record_length_s / dt).")
    fm: float = Field(8.0, gt=0, description="Ricker peak frequency in Hz.")
    equation: str = Field("Acoustic", description="Equation class name from list_equations. Only single-model (vp-only) equations work here.")
    source_x: int | None = Field(None, ge=0, description="Source column index; defaults to the middle of the model.")
    source_depth: int = Field(2, ge=0, description="Source row index, in grid cells from the top.")
    receiver_depth: int = Field(1, ge=0, description="Row index of the receiver line, in grid cells from the top.")
    receiver_step: int = Field(1, ge=1, description="Place a receiver every Nth column.")
    spatial_order: int = Field(4, description="Finite-difference spatial order.")
    abcn: int = Field(40, ge=0, description="PML half-width in grid cells.")
    free_surface: bool = Field(False, description="Apply a free surface at the top boundary.")
    device: str = Field("cpu", description="'cpu', 'cuda' (NVIDIA) or 'mps' (Apple Silicon). The runtime-environment note lists what THIS machine has.")
    out_dir: str = Field("./sweep_runs", description="Directory for the record .npy and the shot-gather PNG.")
    plot: bool = Field(True, description="Also render the shot gather to a PNG.")


@register(
    name="run_forward_sweep",
    description=(
        "Run 2-D acoustic forward modelling directly on the `sweep` solver and return the shot "
        "gather — no sweep_tasks spec, no TaskRunner. This is the tool to use when only the core "
        "solver is installed. Give vp_path, dh, dt, and either nt or record_length_s; the source "
        "defaults to the middle of the model and receivers to a full-width surface line. Call "
        "inspect_file on the model and check_parameters on (dh, dt, fm) first. Returns the record "
        ".npy path, its (nt, nrec) shape, the device actually used, and a shot-gather PNG."
    ),
    params_model=RunForwardSweepParams,
)
def run_forward_sweep(args: RunForwardSweepParams) -> dict[str, Any]:
    mods, err = _require_sweep()
    if err is not None:
        return err

    # ------------------------------------------------------------------
    # TODO — implement the forward run. Worked reference (a complete, runnable
    # script using exactly this API):
    #     sweep/examples/wavefields/topography/acoustic2d_hill_demo.py
    torch = mods["torch"]
    ricker = mods["ricker"]
    # 1. Resolve nt: use args.nt, else round(args.record_length_s / args.dt).
    #    Return {"error": ...} if neither was given.
    nt = args.nt
    if nt is None:
        if args.record_length_s is None:
            return {"error": "give either nt or record_length_s."}
        nt = round(args.record_length_s / args.dt)
    # 2. np.load the model, sanity-check it is 2-D, move it to args.device.
    vp_np = np.load(args.vp_path)
    if vp_np.ndim != 2:
        return {"error": f"vp must be 2-D (nz, nx); got shape {vp_np.shape}."}
    vp = torch.from_numpy(vp_np.astype(np.float32)).to(args.device)
    nz, nx = vp_np.shape
    # 3. Build the Ricker wavelet with mods["ricker"] on a t axis of nt samples
    #    (give it a delay so the wavelet is causal).
    delay = 1.0 / args.fm
    t = np.arange(nt, dtype=np.float32) * args.dt - delay
    wavelet = torch.tensor((1.0e3 * ricker(t, f=args.fm)).astype(np.float32)).to(args.device)
    # 4. Build source and receiver index tensors. sources is (nshot, 2) as
    #    (x, z); receivers is (nshot, nrec, 2). Default the source to the middle
    #    column, receivers to a line across the model at receiver_depth.
    src_x = args.source_x if args.source_x is not None else nx // 2
    sources = torch.from_numpy(
        np.array([[src_x, args.source_depth]], dtype=np.int64)
    ).to(args.device)
    rec_x = np.arange(0, nx, args.receiver_step, dtype=np.int64)
    rec_z = np.full_like(rec_x, args.receiver_depth)
    receivers = torch.from_numpy(
        np.stack([rec_x, rec_z], axis=-1)[None, ...]
    ).to(args.device)
    # 5. equation = mods["eq_mod"]._equation_classes()[args.equation](
    #        spatial_order=args.spatial_order, device=args.device, backend="torch")
    #    Reject equations needing more than vp — list_equations shows the models
    #    each one wants.
    classes = mods["eq_mod"]._equation_classes()
    if args.equation not in classes:
        return {"error": f"unknown equation '{args.equation}'. See list_equations."}
    equation = classes[args.equation](
        spatial_order=args.spatial_order, device=args.device, backend="torch"
    )
    if list(equation.models) != ["vp"]:
        return {"error": f"{args.equation} needs models {list(equation.models)}; "
                         "this tool only supports single-vp equations."}
    # 6. prop = mods["PropTorch"](equation, shape=vp.shape, dh=args.dh,
    #        dt=args.dt, abcn=args.abcn, free_surface=args.free_surface,
    #        use_ckpt=False, impl="eager")
    #    record = prop(wavelet, sources, receivers, models=[vp])  under no_grad.
    prop = mods["PropTorch"](
        equation, shape=(nz, nx), dh=args.dh, dt=args.dt,
        abcn=args.abcn, free_surface=args.free_surface, use_ckpt=False, impl="eager",
    )
    with torch.no_grad():
        record = prop(wavelet, sources, receivers, models=[vp])
    # 7. record comes back as (nshot, nt, nrec, nfield) — squeeze to (nt, nrec),
    #    save it as .npy under out_dir.
    rec_np = np.squeeze(record.detach().cpu().numpy())
    if rec_np.ndim != 2:
        return {"error": f"unexpected record shape {tuple(record.shape)} -> {rec_np.shape}."}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "record.npy"
    np.save(record_path, rec_np)
    
    # 8. If args.plot: draw the gather with matplotlib (see
    #    visualize.py::_require_matplotlib for the guarded-import pattern and
    #    plot_observed_data for a gather-plotting example) and save a PNG.
    image_path = None
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
        if plt is not None:
            pct = np.percentile(np.abs(rec_np), 99.0) or 1.0
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            ax.imshow(rec_np, cmap="seismic", vmin=-pct, vmax=pct,
                      aspect="auto", extent=[rec_x.min(), rec_x.max(), nt * args.dt, 0])
            ax.set_xlabel("receiver x (cells)"); ax.set_ylabel("time (s)")
            ax.set_title(f"{args.equation} shot gather")
            png_path = out_dir / "shot_gather.png"
            fig.savefig(png_path, dpi=140); plt.close(fig)
            image_path = str(png_path)




    # 9. Return {"record_path", "image_path", "shape", "device", "nt", "summary"}.
    # Keep every import inside this function or inside _require_sweep, so the
    # module still imports on a base install with no solver present.
    # ------------------------------------------------------------------
    return {
        "record_path": str(record_path),
        "image_path": image_path,
        "shape": list(rec_np.shape),
        "device": args.device,
        "nt": nt,
        "summary": f"Ran {args.equation} forward: {rec_np.shape[0]} samples x "
                   f"{rec_np.shape[1]} receivers on {args.device}.",
    }