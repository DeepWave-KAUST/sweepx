"""Visualization tools — the agent's LLM-callable plotting layer.

Most tools draw with matplotlib here and use ``sweepx.tasks.viz`` for the
seismic-specific pieces (snapshot / model / convergence renderers, colormaps).
Three of them — plot_wavelet, plot_velocity_slice, compare_shot_gathers — need
only matplotlib and work without the sweepx.tasks tier. matplotlib is a hard
dependency; imageio (GIF writing) is optional, see the ``animate`` extra.

These tools also own the agent-side concerns: locating a task's snapshots.npy
and cropping the PML. Pass ``abcn`` (the PML thickness used in the run) so the
absorbing border is cropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sweepx.agent.tools import register


def _out_doc(default_name: str, where: str = "<task_dir>/output") -> str:
    return (f"Where to save the figure. Omit for {where}/{default_name}; a bare file name "
            f"goes under {where}/; any other path is used as given.")


def _out_file(out_path: str | None, default_dir: Path, default_name: str) -> Path:
    """The figure path: ``default_dir/default_name`` when omitted, ``default_dir/<name>``
    for a bare file name, else ``out_path`` (a directory gets ``default_name``)."""
    if not out_path:
        out = default_dir / default_name
    else:
        p = Path(out_path).expanduser()
        out = default_dir / p if (p.parent == Path(".") and not out_path.startswith(".")) else p
        if out_path.endswith("/") or out.is_dir():
            out = out / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.resolve()  # absolute: a relative path invites the model to re-root it


def _exactly_one(model, *names: str):
    given = [n for n in names if getattr(model, n) is not None]
    if len(given) != 1:
        raise ValueError(f"give exactly one of {', '.join(names)} (got {given or 'none'}).")
    return model


def _require_matplotlib():
    """Return ``(pyplot, None)``, or ``(None, error_dict)`` when matplotlib is absent.

    Reporting a missing dependency as data keeps the contract the rest of the
    tools follow: a tool whose layer is missing returns an error, never raises.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return None, {"error": f"matplotlib is not importable: {exc}. Install it with `pip install matplotlib`."}
    return plt, None


def _require_imageio():
    """Return ``(imageio, None)``, or ``(None, error_dict)`` when imageio is absent."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        return None, {
            "error": f"imageio is not importable: {exc}. Install it with `pip install 'sweepx[animate]'`."
        }
    return imageio, None



def resolve_task_dir(task_dir: str, marker: str = "snapshots.npy") -> Path:
    """Return a directory that actually contains output/<marker>.

    Small LLMs frequently mis-pass the task_dir (hallucinated timestamps, wrong
    prefix). If the given path lacks the marker file, fall back to the most
    recently modified output/<marker> under likely roots — almost always the run
    the model just executed. ``marker`` is snapshots.npy for wavefield tools,
    record.npy for shot-gather plotting."""
    p = Path(task_dir)
    if (p / "output" / marker).is_file():
        return p
    roots = [p.parent, Path("sweep_runs"), Path.cwd() / "sweep_runs", p]
    cands: list[Path] = []
    for root in roots:
        try:
            if root.is_dir():
                cands += list(root.glob(f"*/output/{marker}"))
        except OSError:
            pass
    if cands:
        latest = max(cands, key=lambda f: f.stat().st_mtime)
        return latest.parent.parent
    return p


def _load_snaps(task_dir: str, abcn: int, free_surface: bool, shot: int, field: int,
                allow_fallback: bool = True) -> np.ndarray:
    """Load output/snapshots.npy → (n_snap, nz, nx) with PML cropped.

    allow_fallback=True (single-task plot/gif) tolerates a mis-passed task_dir by
    locating the most recent run. Set False for compare — each panel must be a
    distinct exact task, so a wrong / forward task_dir must error rather than
    silently resolve to another run (which would make all panels identical)."""
    rd = resolve_task_dir(task_dir) if allow_fallback else Path(task_dir)
    path = rd / "output" / "snapshots.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"no snapshots.npy under {task_dir}/output — is this a WAVEFIELD task? "
            f"(forward tasks have no snapshots; use build_forward_spec with snapshot_times)"
        )
    snap = np.load(path)  # (n_snap, n_field, nshots, nch, nz_pad, nx_pad)
    arr = snap[:, field, shot, 0]
    nzp, nxp = arr.shape[-2:]
    top = 0 if free_surface else abcn
    return arr[:, top:nzp - abcn, abcn:nxp - abcn]


# ---------------------------------------------------------------------------
# plot_wavefield
# ---------------------------------------------------------------------------

class PlotWavefieldParams(BaseModel):
    task_dir: str = Field(..., description="Task directory from run_task (a wavefield task).")
    abcn: int = Field(..., description="PML thickness used in the run — the same value passed to the builder.")
    free_surface: bool = Field(False, description="Whether the run used a free surface (top row not cropped).")
    snapshot_index: int = Field(-1, description="Which snapshot to plot (index into snapshot_times; -1 = last).")
    shot: int = Field(0, description="Which shot's wavefield (default 0).")
    field: int = Field(0, description="Which wavefield component (default 0 = main field).")
    out_path: str | None = Field(None, description=_out_doc("wavefield.png"))
    title: str | None = Field(None, description="Plot title.")


@register(
    name="plot_wavefield",
    description=(
        "Render one wavefield snapshot from a finished wavefield task to a PNG (via sweepx.tasks.viz). "
        "Crops the PML border using `abcn`. Returns the saved image path. Use after run_task on a "
        "wavefield spec."
    ),
    params_model=PlotWavefieldParams,
)
def plot_wavefield(args: PlotWavefieldParams) -> dict[str, Any]:
    try:
        wf = _load_snaps(args.task_dir, args.abcn, args.free_surface, args.shot, args.field)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err
    from sweepx.tasks.viz import wavefield as viz

    frame = wf[args.snapshot_index]
    rd = resolve_task_dir(args.task_dir)
    out = _out_file(args.out_path, rd / "output", "wavefield.png")
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    viz.plot_snapshot(frame, ax=ax, title=args.title or "wavefield snapshot")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"image_path": str(out), "task_dir_used": str(rd), "shape": list(frame.shape)}


# ---------------------------------------------------------------------------
# compare_wavefields
# ---------------------------------------------------------------------------

class CompareWavefieldsParams(BaseModel):
    task_dirs: list[str] = Field(..., min_length=1, description="Wavefield task directories to compare side by side.")
    labels: list[str] = Field(..., description="Title for each panel (same length as task_dirs).")
    abcn: int = Field(..., description="PML thickness used in the runs (shared).")
    out_path: str = Field(..., description="Full output PNG path for the comparison figure.")
    snapshot_index: int = Field(-1, description="Which snapshot to show (-1 = last).")
    free_surface: bool = Field(False)
    shot: int = Field(0)
    field: int = Field(0)
    suptitle: str | None = Field(None)


@register(
    name="compare_wavefields",
    description=(
        "Plot one snapshot from several wavefield tasks side by side with a shared color scale (via "
        "sweepx.tasks.viz) — e.g. Acoustic vs VTI vs TTI wavefronts. task_dirs must be distinct, real "
        "wavefield runs (no path fallback — a wrong/forward dir errors). Returns the image path."
    ),
    params_model=CompareWavefieldsParams,
)
def compare_wavefields(args: CompareWavefieldsParams) -> dict[str, Any]:
    if len(args.labels) != len(args.task_dirs):
        return {"error": f"labels ({len(args.labels)}) must match task_dirs ({len(args.task_dirs)})."}
    if len(set(args.task_dirs)) != len(args.task_dirs):
        return {"error": "task_dirs must be distinct (one per panel); got duplicates."}
    frames = []
    for td in args.task_dirs:
        try:
            wf = _load_snaps(td, args.abcn, args.free_surface, args.shot, args.field, allow_fallback=False)
        except Exception as exc:
            return {"error": f"{td}: {type(exc).__name__}: {exc}"}
        frames.append(wf[args.snapshot_index])
    from sweepx.tasks.viz import wavefield as viz

    try:
        viz.compare_snapshots(frames, list(args.labels), args.out_path, suptitle=args.suptitle)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"image_path": args.out_path, "n_panels": len(frames)}


# ---------------------------------------------------------------------------
# render_wavefield_gif (animate_wavefield's renderer)
# ---------------------------------------------------------------------------

def render_wavefield_gif(task_dir: str, *, abcn: int, free_surface: bool, shot: int, field: int,
                         out_path: str | None, fps: int, topography_path: str | None = None,
                         curvilinear: bool = False, sources=None, receivers=None) -> dict[str, Any]:
    """Animate every snapshot of a wavefield run into a GIF/MP4 (animate_wavefield's renderer)."""
    # Topography/curvilinear: the un-padded top row IS the free surface, so never
    # crop it — regardless of the free_surface flag the caller passed.
    fs = True if topography_path else free_surface
    try:
        wf = _load_snaps(task_dir, abcn, fs, shot, field)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if wf.shape[0] < 2:
        return {"error": "need >= 2 snapshots for a GIF; re-run with more snapshot_times."}
    from sweepx.tasks.viz import wavefield as viz

    rd = resolve_task_dir(task_dir)
    out = _out_file(out_path, rd / "output", "wavefield.gif")
    try:
        if topography_path:
            topo = np.load(topography_path)
            viz.animate_snapshots_topography(
                list(wf), topo, str(out),
                sources=sources, receivers=receivers,
                curvilinear=curvilinear, fps=fps,
            )
        else:
            viz.animate_snapshots(list(wf), str(out), fps=fps)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"gif_path": str(out), "task_dir_used": str(rd), "n_frames": int(wf.shape[0])}


# ---------------------------------------------------------------------------
# plot_shot_gather
# ---------------------------------------------------------------------------

def _receiver_x(task_dir: Path, shot: int, dh: float | None, nrec: int) -> np.ndarray | None:
    """The receivers' x positions (m) of ``shot``, from the task's receivers.npy
    (grid indices, (nshots, nrec, ndim), x first); None (a trace-number axis)
    without the file or ``dh``, or on a 3-D grid, where the traces are not one
    line."""
    path = task_dir / "output" / "receivers.npy"
    if dh is None or not path.is_file():
        return None
    rec = np.load(path)
    if rec.ndim == 2:            # (nrec, ndim), shared by every shot
        rec = rec[None]
    if rec.ndim != 3 or rec.shape[-1] != 2 or rec.shape[1] != nrec:
        return None
    return rec[min(shot, rec.shape[0] - 1), :, 0] * float(dh)


class PlotShotGatherParams(BaseModel):
    task_dir: str | None = Field(None, description="A run_task forward / FWI task directory (reads its output/record.npy). Give this OR npy_path.")
    npy_path: str | None = Field(None, description="A record .npy file (observed or synthetic): (nshots,nt,nrec,nch), (nshots,nt,nrec) or (nt,nrec). Give this OR task_dir.")
    shot: int = Field(0, description="Which shot to display (default 0).")
    channel: int = Field(0, description="Which channel/component (default 0).")
    dt: float | None = Field(None, description="Time step (s) for the time axis — pass the dt used in the run.")
    dh: float | None = Field(None, description="Receiver spacing (m) for the x axis (with a task_dir, its receivers.npy gives the exact positions).")
    wiggle: bool = Field(False, description="Wiggle display instead of image (good for < ~50 traces).")
    out_path: str | None = Field(None, description=_out_doc("shot_gather.png", "<task_dir>/output (or ./sweep_runs for npy_path)"))
    title: str | None = Field(None, description="Plot title.")

    @model_validator(mode="after")
    def check_one_source(self):
        return _exactly_one(self, "task_dir", "npy_path")


@register(
    name="plot_shot_gather",
    description=(
        "Plot a shot gather — image display by default, or wiggle — from a run_task forward/FWI task_dir "
        "or straight from a record .npy file (observed or synthetic data). Not needed after "
        "run_forward_sweep, which already returns its gather as image_path. Returns the image path "
        "and the displayed (nt, nrec) shape."
    ),
    params_model=PlotShotGatherParams,
)
def plot_shot_gather(args: PlotShotGatherParams) -> dict[str, Any]:
    rd = None
    if args.task_dir is not None:
        rd = resolve_task_dir(args.task_dir, marker="record.npy")
        path = rd / "output" / "record.npy"
        if not path.is_file():
            return {"error": f"no record.npy under {args.task_dir}/output: pass the task_dir run_task returned "
                             f"(run_forward_sweep already saved its gather; see its image_path)"}
        out = _out_file(args.out_path, rd / "output", "shot_gather.png")
    else:
        path = Path(args.npy_path).expanduser()
        if not path.is_file():
            return {"error": f"data file not found: {args.npy_path}"}
        out = _out_file(args.out_path, Path("sweep_runs"), f"{path.stem}_gather.png")
    try:
        arr = _record_to_2d(np.load(path), args.shot, args.channel)
    except ValueError as exc:
        return {"error": str(exc)}

    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err
    from sweepx.tasks.viz import seismic

    x = _receiver_x(rd, args.shot, args.dh, arr.shape[1]) if rd is not None else None
    if x is None and args.dh:
        x = np.arange(arr.shape[1]) * float(args.dh)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if args.wiggle:
        seismic.plot_wiggle(arr, dt=args.dt, ax=ax)
        if args.title:
            ax.set_title(args.title)
    else:
        seismic.plot_shot(arr, dt=args.dt, x=x, ax=ax, title=args.title or "shot gather")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    res = {"image_path": str(out), "shape": list(arr.shape)}
    if rd is not None:
        res["task_dir_used"] = str(rd)
    return res


# ---------------------------------------------------------------------------
# plot_model — FWI / migration result (inverted vs true vs init)
# ---------------------------------------------------------------------------

class PlotModelParams(BaseModel):
    model_path: str | None = Field(None, description="A model .npy to show — (nz,nx), or (nz,ny,nx) drawn as three orthogonal slices. Give this OR task_dir.")
    task_dir: str | None = Field(None, description="An FWI / migration task directory from run_task: shows <task_dir>/output/<model_name> (the inverted model). Give this OR model_path.")
    model_name: str = Field("inverted_vp.npy", description="With task_dir: which model file under <task_dir>/output to read.")
    true_model_path: str | None = Field(None, description="Ground-truth model .npy, shown alongside with a residual panel (2-D). Optional.")
    init_model_path: str | None = Field(None, description="Initial/starting model .npy, shown as the left panel (2-D). Optional.")
    dh: float | None = Field(None, description="Grid spacing (m) for physical axes; samples if omitted.")
    out_path: str | None = Field(None, description=_out_doc("model.png", "<task_dir>/output (or ./sweep_runs for model_path)"))
    title: str | None = Field(None, description="Figure title.")
    cbar_label: str = Field("vp (m/s)", description="Colour-bar label.")

    @model_validator(mode="after")
    def check_one_source(self):
        return _exactly_one(self, "model_path", "task_dir")


@register(
    name="plot_model",
    description=(
        "Plot a velocity/parameter model: a model .npy file (to SEE a model before running anything / "
        "看看这个模型), or the inverted model of an FWI/migration task_dir. Add init_model_path and/or "
        "true_model_path for initial vs inverted vs true panels with an inverted−true residual on a "
        "shared colour scale (反演结果). 3-D models are drawn as three orthogonal slices. Returns the "
        "image path, the shape and the value range."
    ),
    params_model=PlotModelParams,
)
def plot_model(args: PlotModelParams) -> dict[str, Any]:
    rd = None
    if args.task_dir is not None:
        rd = resolve_task_dir(args.task_dir, marker=args.model_name)
        main_path = rd / "output" / args.model_name
        if not main_path.is_file():
            return {"error": f"no {args.model_name} under {args.task_dir}/output (run an FWI/migration task first)"}
        out = _out_file(args.out_path, rd / "output", "model.png")
        main_label = "inverted"
    else:
        main_path = Path(args.model_path).expanduser()
        if not main_path.is_file():
            return {"error": f"model file not found: {args.model_path}"}
        out = _out_file(args.out_path, Path("sweep_runs"), f"{main_path.stem}.png")
        main_label = "model"
    arr = np.load(main_path)
    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err
    res = {"image_path": str(out), "shape": list(arr.shape),
           "vp_min": float(np.nanmin(arr)), "vp_max": float(np.nanmax(arr))}
    if rd is not None:
        res["task_dir_used"] = str(rd)

    if args.init_model_path or args.true_model_path:
        if arr.ndim != 2:
            return {"error": f"the initial/true comparison supports 2-D models; got shape {arr.shape}"}
        panels, labels, true_arr = [], [], None
        try:
            if args.init_model_path:
                panels.append(np.load(args.init_model_path)); labels.append("initial")
            panels.append(arr); labels.append(main_label)
            if args.true_model_path:
                true_arr = np.load(args.true_model_path)
                panels.append(true_arr); labels.append("true")
        except Exception as exc:
            return {"error": f"could not load a comparison model: {exc}"}
        from sweepx.tasks.viz import model as vizmodel

        dh = (args.dh, args.dh) if args.dh else None
        residual = (arr, true_arr) if true_arr is not None else None
        try:
            vizmodel.compare_models(panels, labels, str(out), dh=dh, residual=residual, suptitle=args.title)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {**res, "panels": labels}

    if arr.ndim == 2:
        # Plain matplotlib heat-map — NO sweepx.tasks needed.
        nz, nx = arr.shape
        extent = [0, nx * args.dh, nz * args.dh, 0] if args.dh else [0, nx, nz, 0]
        unit = "m" if args.dh else "samples"
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        im = ax.imshow(arr, aspect="auto", cmap="jet", extent=extent)
        ax.set_title(args.title or f"{main_label} {arr.shape}")
        ax.set_xlabel(f"x ({unit})")
        ax.set_ylabel(f"depth ({unit})")
        fig.colorbar(im, ax=ax, label=args.cbar_label)
    elif arr.ndim == 3:
        try:
            from sweepx.tasks.viz import model as vizmodel
            dh = args.dh or 1.0
            fig, _axes = vizmodel.plot_vp_ortho_slices(
                arr, dh_xyz=(dh, dh, dh), title_prefix=args.title or "vp", cbar_label=args.cbar_label)
        except Exception:
            nz, ny, nx = arr.shape
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
            for ax, (plane, ttl) in zip(axes, (
                    (arr[:, ny // 2, :], f"inline y={ny // 2}"),
                    (arr[:, :, nx // 2], f"crossline x={nx // 2}"),
                    (arr[nz // 2, :, :], f"depth z={nz // 2}"))):
                im = ax.imshow(plane, aspect="auto", cmap="jet")
                ax.set_title(ttl)
            fig.colorbar(im, ax=list(axes), label=args.cbar_label)
            fig.suptitle(args.title or f"{main_label} {arr.shape}")
    else:
        return {"error": f"plot_model supports 2-D / 3-D models; got shape {arr.shape}"}
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# plot_convergence — FWI loss curve
# ---------------------------------------------------------------------------

class PlotConvergenceParams(BaseModel):
    task_dir: str = Field(..., description="FWI task directory from run_task — must contain output/loss.npy.")
    loss_name: str = Field("loss.npy", description="Which loss file under <task_dir>/output to read.")
    logy: bool = Field(True, description="Log-scale y-axis (default True).")
    out_path: str | None = Field(None, description=_out_doc("convergence.png"))
    title: str | None = Field(None, description="Plot title.")


@register(
    name="plot_convergence",
    description=(
        "Plot the FWI loss / misfit convergence curve from a task's loss.npy (via sweepx.tasks.viz). Use after "
        "an FWI run when the user wants to see how the inversion converged / loss 曲线 / 收敛. Returns the "
        "saved image path."
    ),
    params_model=PlotConvergenceParams,
)
def plot_convergence(args: PlotConvergenceParams) -> dict[str, Any]:
    rd = resolve_task_dir(args.task_dir, marker=args.loss_name)
    path = rd / "output" / args.loss_name
    if not path.is_file():
        return {"error": f"no {args.loss_name} under {args.task_dir}/output (run an FWI task first)"}
    loss = np.load(path).reshape(-1)

    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err
    from sweepx.tasks.viz import convergence

    out = _out_file(args.out_path, rd / "output", "convergence.png")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    convergence.plot_loss(loss, ax=ax, logy=args.logy, title=args.title or "FWI convergence")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"image_path": str(out), "task_dir_used": str(rd), "n_iter": int(loss.size)}


# ---------------------------------------------------------------------------
# record helpers
# ---------------------------------------------------------------------------

def _record_to_2d(rec: np.ndarray, shot: int, channel: int) -> np.ndarray:
    if rec.ndim == 4:        # (nshots, nt, nrec, nchannel)
        return rec[shot, :, :, channel]
    if rec.ndim == 3:        # (nshots, nt, nrec)
        return rec[shot]
    if rec.ndim == 2:        # (nt, nrec)
        return rec
    raise ValueError(f"unexpected record shape {rec.shape}")


# ---------------------------------------------------------------------------
# plot_segy — read + display a SEG-Y shot gather
# ---------------------------------------------------------------------------

class PlotSegyParams(BaseModel):
    segy_path: str = Field(..., description="Path to a .segy / .sgy file.")
    max_traces: int = Field(2000, ge=1, description="Cap traces displayed (large files).")
    dt: float | None = Field(None, description="Override time step (s); else read from the SEG-Y header.")
    out_path: str = Field("/tmp/segy_gather.png", description="Full output PNG path.")
    title: str | None = Field(None, description="Plot title.")


@register(
    name="plot_segy",
    description=(
        "Read a SEG-Y file (real field/observed data) and plot it as a shot gather. Use when the user "
        "points at a .segy / .sgy file. Returns the image path, (nt, ntraces) shape and the sample "
        "interval read from the header."
    ),
    params_model=PlotSegyParams,
)
def plot_segy(args: PlotSegyParams) -> dict[str, Any]:
    p = Path(args.segy_path).expanduser()
    if not p.is_file():
        return {"error": f"SEG-Y file not found: {args.segy_path}"}
    try:
        import segyio
    except ImportError as exc:
        return {"error": f"segyio is required to read SEG-Y: {exc}"}
    try:
        with segyio.open(str(p), ignore_geometry=True) as f:
            n = min(f.tracecount, args.max_traces)
            data = np.stack([f.trace[i] for i in range(n)]).T  # (nt, ntraces)
            hdr_dt = segyio.tools.dt(f) / 1.0e6  # microseconds → s
    except Exception as exc:
        return {"error": f"failed to read SEG-Y: {type(exc).__name__}: {exc}"}
    dt = args.dt if args.dt is not None else hdr_dt

    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err
    from sweepx.tasks.viz import seismic

    out = Path(args.out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    seismic.plot_shot(data, dt=dt, ax=ax, title=args.title or f"SEG-Y: {p.name}")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"image_path": str(out), "shape": list(data.shape), "dt": float(dt), "n_traces": int(data.shape[1])}


# ---------------------------------------------------------------------------
# compare_shot_gathers — observed vs synthetic (+ residual), side by side
# ---------------------------------------------------------------------------

class CompareShotGathersParams(BaseModel):
    record_a: str = Field(..., description="First record .npy path (or a task_dir containing output/record.npy) — e.g. observed.")
    record_b: str = Field(..., description="Second record .npy path (or task_dir) — e.g. synthetic.")
    label_a: str = Field("observed", description="Label for the first panel.")
    label_b: str = Field("synthetic", description="Label for the second panel.")
    shot: int = Field(0, description="Which shot to display.")
    channel: int = Field(0, description="Which channel.")
    dt: float | None = Field(None, description="Time step (s) for the time axis.")
    out_path: str = Field("/tmp/gather_comparison.png", description="Full output PNG path.")


def _resolve_record(path_or_dir: str) -> Path | None:
    p = Path(path_or_dir).expanduser()
    if p.is_file():
        return p
    rd = resolve_task_dir(path_or_dir, marker="record.npy")
    cand = rd / "output" / "record.npy"
    return cand if cand.is_file() else None


@register(
    name="compare_shot_gathers",
    description=(
        "Compare two shot gathers side by side with a shared scale plus a residual panel — the core FWI "
        "QC (observed vs synthetic). Each input is a record .npy path OR a task_dir. Use when the user "
        "wants to compare observed vs modelled data / 对比观测与合成 / see the data misfit. Returns the image."
    ),
    params_model=CompareShotGathersParams,
)
def compare_shot_gathers(args: CompareShotGathersParams) -> dict[str, Any]:
    pa, pb = _resolve_record(args.record_a), _resolve_record(args.record_b)
    if pa is None:
        return {"error": f"no record found for record_a={args.record_a}"}
    if pb is None:
        return {"error": f"no record found for record_b={args.record_b}"}
    try:
        a = _record_to_2d(np.load(pa), args.shot, args.channel)
        b = _record_to_2d(np.load(pb), args.shot, args.channel)
    except ValueError as exc:
        return {"error": str(exc)}
    if a.shape != b.shape:
        return {"error": f"record shapes differ: {a.shape} vs {b.shape}"}

    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err

    s = np.percentile(np.abs(np.concatenate([a.ravel(), b.ravel()])), 98) + 1e-30
    diff = a - b
    out = Path(args.out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    nt = a.shape[0]
    extent = [0, a.shape[1], (nt * args.dt) if args.dt else nt, 0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True, sharey=True)
    for ax, arr, lab in ((axes[0], a, args.label_a), (axes[1], b, args.label_b), (axes[2], diff, "residual (a−b)")):
        ax.imshow(arr, cmap="RdBu_r", vmin=-s, vmax=s, aspect="auto", extent=extent)
        ax.set_title(lab); ax.set_xlabel("receiver")
    axes[0].set_ylabel("time (s)" if args.dt else "time (samples)")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-30))
    return {"image_path": str(out), "shape": list(a.shape), "relative_residual": round(rel, 4)}


# ---------------------------------------------------------------------------
# plot_velocity_slice — vertical velocity-vs-depth profile(s) at given x
# ---------------------------------------------------------------------------

class PlotVelocitySliceParams(BaseModel):
    model_path: str = Field(..., description="Velocity-model .npy — (nz,nx) for 2-D or (nz,ny,nx) for 3-D.")
    x_indices: list[int] | None = Field(None, description="Lateral (x) column indices to slice at; default the model centre. Pass several to overlay profiles.")
    y_index: int | None = Field(None, description="For a 3-D model, the crossline (y) index; default the y centre.")
    dh: float | None = Field(None, description="Grid spacing (m) for the depth axis; samples if omitted.")
    out_path: str = Field("/tmp/velocity_slice.png", description="Full output PNG path.")
    title: str | None = Field(None, description="Plot title.")


@register(
    name="plot_velocity_slice",
    description=(
        "Plot the VERTICAL velocity profile(s) — velocity vs depth at a fixed lateral position — from a "
        "model .npy (well-log style: depth increases downward, velocity on the x-axis). Use when the user "
        "wants a 纵向切片 / 速度随深度 / depth profile / a 1-D slice of the model. Pass x_indices to overlay "
        "several lateral positions. Returns the saved image path."
    ),
    params_model=PlotVelocitySliceParams,
)
def plot_velocity_slice(args: PlotVelocitySliceParams) -> dict[str, Any]:
    p = Path(args.model_path).expanduser()
    if not p.is_file():
        return {"error": f"model file not found: {args.model_path}"}
    arr = np.load(p)
    if arr.ndim == 3:
        ny = arr.shape[1]
        yi = args.y_index if args.y_index is not None else ny // 2
        yi = int(np.clip(yi, 0, ny - 1))
        plane = arr[:, yi, :]            # (nz, nx) at fixed crossline y
        ylabel_extra = f" (y={yi})"
    elif arr.ndim == 2:
        plane = arr
        ylabel_extra = ""
    else:
        return {"error": f"plot_velocity_slice supports 2-D / 3-D; got shape {arr.shape}"}

    nz, nx = plane.shape
    xs = args.x_indices if args.x_indices else [nx // 2]
    xs = [int(np.clip(x, 0, nx - 1)) for x in xs]
    depth = np.arange(nz) * (args.dh if args.dh else 1.0)
    depth_label = "depth (m)" if args.dh else "depth (samples)"

    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err

    out = Path(args.out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 7), constrained_layout=True)
    for x in xs:
        xpos = f"x={x * args.dh:.0f} m" if args.dh else f"x={x}"
        ax.plot(plane[:, x], depth, label=xpos)
    ax.invert_yaxis()                    # depth increases downward
    ax.set_xlabel("vp (m/s)")
    ax.set_ylabel(depth_label)
    ax.set_title(args.title or f"vertical velocity profile{ylabel_extra}")
    ax.grid(alpha=0.3)
    if len(xs) > 1:
        ax.legend()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"image_path": str(out), "shape": list(arr.shape), "x_indices": xs}


# ---------------------------------------------------------------------------
# plot_wavelet — Ricker source wavelet waveform + amplitude spectrum
# ---------------------------------------------------------------------------

class PlotWaveletParams(BaseModel):
    fm: float = Field(..., gt=0, description="Ricker centre (peak) frequency in Hz.")
    dt: float = Field(1.0e-3, gt=0, description="Time step (s).")
    nt: int = Field(512, ge=16, description="Number of samples for the wavelet/spectrum.")
    delay: float | None = Field(None, ge=0, description="Peak-time delay (s); default 1/fm.")
    out_path: str = Field("/tmp/wavelet.png", description="Full output PNG path.")


@register(
    name="plot_wavelet",
    description=(
        "Plot the Ricker source wavelet for a given peak frequency: its time-domain waveform AND its "
        "amplitude spectrum (so you can judge whether fm gives enough/too much frequency content). Use "
        "when the user asks to see the source / wavelet / 子波 / 频谱, or to pick a frequency. Returns the "
        "saved image path and the spectral peak frequency."
    ),
    params_model=PlotWaveletParams,
)
def plot_wavelet(args: PlotWaveletParams) -> dict[str, Any]:
    delay = args.delay if args.delay is not None else 1.0 / args.fm
    t = np.arange(args.nt) * args.dt
    try:
        from sweep.signal import ricker
        w = np.asarray(ricker(t - delay, f=args.fm), dtype=np.float64)
    except Exception:
        # Analytic Ricker fallback if sweep.signal isn't importable.
        a = (np.pi * args.fm * (t - delay)) ** 2
        w = (1.0 - 2.0 * a) * np.exp(-a)

    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(args.nt, d=args.dt)
    peak_freq = float(freqs[int(np.argmax(spec))])

    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err

    out = Path(args.out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    a1.plot(t, w, color="C0"); a1.set_xlabel("time (s)"); a1.set_ylabel("amplitude")
    a1.set_title(f"Ricker wavelet (fm = {args.fm:g} Hz)"); a1.grid(alpha=0.3)
    a2.plot(freqs, spec / (spec.max() + 1e-30), color="C3")
    a2.axvline(peak_freq, ls="--", c="k", lw=1, label=f"peak {peak_freq:.1f} Hz")
    a2.set_xlabel("frequency (Hz)"); a2.set_ylabel("normalized amplitude")
    a2.set_xlim(0, min(freqs[-1], 4 * args.fm)); a2.set_title("amplitude spectrum")
    a2.legend(); a2.grid(alpha=0.3)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"image_path": str(out), "peak_frequency_hz": peak_freq, "fm": args.fm}


# ---------------------------------------------------------------------------
# animate_fwi_evolution — GIF of the inverted model improving each epoch
# ---------------------------------------------------------------------------

class AnimateFwiEvolutionParams(BaseModel):
    task_dir: str = Field(..., description="FWI task directory from run_task — must contain output/epochs/vp_epoch_*.npy.")
    true_model_path: str | None = Field(None, description="Optional ground-truth vp .npy to fix the colour scale (else per-run percentile).")
    dh: float | None = Field(None, description="Grid spacing (m) for physical axes.")
    out_path: str | None = Field(None, description=_out_doc("fwi_evolution.gif"))
    fps: int = Field(4, ge=1, le=30, description="Frames per second.")


@register(
    name="animate_fwi_evolution",
    description=(
        "Animate how the FWI-inverted velocity model evolves epoch by epoch into a GIF (reads the "
        "per-epoch models the FWI run saved under output/epochs/). Use after an FWI run when the user "
        "wants to SEE the inversion converge / 反演过程 / how the model improves. Returns the gif path."
    ),
    params_model=AnimateFwiEvolutionParams,
)
def animate_fwi_evolution(args: AnimateFwiEvolutionParams) -> dict[str, Any]:
    rd = resolve_task_dir(args.task_dir, marker="inverted_vp.npy")
    epochs_dir = rd / "output" / "epochs"
    frames_paths = sorted(epochs_dir.glob("vp_epoch_*.npy")) if epochs_dir.is_dir() else []
    if not frames_paths:
        return {"error": f"no per-epoch models under {rd}/output/epochs (run an FWI task that saves epochs)"}
    mats = [np.load(f) for f in frames_paths]
    if mats[0].ndim != 2:
        return {"error": f"animate_fwi_evolution supports 2-D models; got shape {mats[0].shape}"}

    if args.true_model_path and Path(args.true_model_path).is_file():
        ref = np.load(args.true_model_path)
        vmin, vmax = float(np.nanmin(ref)), float(np.nanmax(ref))
    else:
        allv = np.concatenate([m.ravel() for m in mats])
        vmin, vmax = np.percentile(allv, [2, 98])

    plt, _mpl_err = _require_matplotlib()
    if _mpl_err is not None:
        return _mpl_err
    imageio, _io_err = _require_imageio()
    if _io_err is not None:
        return _io_err
    from sweepx.tasks.viz.colormaps import VP_CMAP

    out = _out_file(args.out_path, rd / "output", "fwi_evolution.gif")
    extent = None
    if args.dh:
        nz, nx = mats[0].shape
        extent = (0.0, nx * args.dh, nz * args.dh, 0.0)
    import re
    def _epoch_num(path: Path) -> str:
        mt = re.search(r"(\d+)", path.stem)
        return mt.group(1).lstrip("0") or "0" if mt else "?"

    images = []
    for m, fpath in zip(mats, frames_paths):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        im = ax.imshow(m, cmap=VP_CMAP, vmin=vmin, vmax=vmax, aspect="auto", extent=extent)
        ax.set_title(f"FWI — epoch {_epoch_num(fpath)}")
        if extent:
            ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        fig.colorbar(im, ax=ax, label="vp (m/s)", fraction=0.04, pad=0.02)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out, images, duration=1.0 / args.fps, loop=0)
    return {"gif_path": str(out), "task_dir_used": str(rd), "n_frames": len(images)}
