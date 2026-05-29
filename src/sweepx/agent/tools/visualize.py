"""Visualization tools — let the LLM turn a finished task_dir into figures / GIFs.

Without these, the LLM can only build+run a wavefield (producing snapshots.npy);
it cannot *show* anything. These tools read a task's snapshots and render PNGs /
GIFs, so a natural-language request like "run a VTI wavefield and plot it" can be
completed end-to-end by the model.

All tools take `abcn` (the PML thickness used in the run) so they can crop the
absorbing border — pass the same value you gave the builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sweep_agent.tools import register


def resolve_task_dir(task_dir: str) -> Path:
    """Return a directory that actually contains output/snapshots.npy.

    Small LLMs frequently mis-pass the task_dir (hallucinated timestamps, wrong
    prefix). If the given path has no snapshots, fall back to the most recently
    modified snapshots.npy under likely roots — almost always the run the model
    just executed."""
    p = Path(task_dir)
    if (p / "output" / "snapshots.npy").is_file():
        return p
    roots = [p.parent, Path("sweep_runs"), Path.cwd() / "sweep_runs", p]
    cands: list[Path] = []
    for root in roots:
        try:
            if root.is_dir():
                cands += list(root.glob("*/output/snapshots.npy"))
        except OSError:
            pass
    if cands:
        latest = max(cands, key=lambda f: f.stat().st_mtime)
        return latest.parent.parent
    return p  # unchanged; caller raises a clear error


def _load_snaps(task_dir: str, abcn: int, free_surface: bool, shot: int, field: int,
                allow_fallback: bool = True) -> np.ndarray:
    """Load output/snapshots.npy → (n_snap, nz, nx) with PML cropped.

    allow_fallback=True (single-task plot/gif) tolerates a mis-passed task_dir by
    locating the most recent run. Set False for compare — each panel must be a
    distinct, exact task, so a wrong / forward task_dir must error rather than
    silently resolve to another run (which would make all panels identical)."""
    rd = resolve_task_dir(task_dir) if allow_fallback else Path(task_dir)
    path = rd / "output" / "snapshots.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"no snapshots.npy under {task_dir}/output — is this a WAVEFIELD task? "
            f"(forward tasks have no snapshots; use build_wavefield_spec with snapshot_times)"
        )
    snap = np.load(path)  # (n_snap, n_field, nshots, nch, nz_pad, nx_pad)
    arr = snap[:, field, shot, 0]
    nzp, nxp = arr.shape[-2:]
    top = 0 if free_surface else abcn
    return arr[:, top:nzp - abcn, abcn:nxp - abcn]


class PlotWavefieldParams(BaseModel):
    task_dir: str = Field(..., description="Task directory from run_task (a wavefield task).")
    abcn: int = Field(..., description="PML thickness used in the run (to crop the absorbing border) — the same value passed to the builder.")
    free_surface: bool = Field(False, description="Whether the run used a free surface (top row not cropped).")
    snapshot_index: int = Field(-1, description="Which snapshot to plot (index into snapshot_times; -1 = last).")
    shot: int = Field(0, description="Which shot's wavefield (default 0).")
    field: int = Field(0, description="Which wavefield component (default 0 = main field).")
    out_name: str = Field("wavefield.png", description="Output PNG filename, saved under <task_dir>/output/.")
    title: str | None = Field(None, description="Plot title.")


@register(
    name="plot_wavefield",
    description=(
        "Render one wavefield snapshot from a finished wavefield task to a PNG (seismic colormap, "
        "signed 2-98 percentile clip). Crops the PML border using `abcn`. Returns the saved image "
        "path so you can report it to the user. Use after run_task on a wavefield spec."
    ),
    params_model=PlotWavefieldParams,
)
def plot_wavefield(args: PlotWavefieldParams) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        wf = _load_snaps(args.task_dir, args.abcn, args.free_surface, args.shot, args.field)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    frame = wf[args.snapshot_index]
    vmin, vmax = np.percentile(frame, [2, 98])
    rd = resolve_task_dir(args.task_dir)
    out = rd / "output" / args.out_name
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(frame, cmap="seismic", vmin=float(vmin), vmax=float(vmax), aspect="equal")
    ax.set_title(args.title or "wavefield snapshot")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"image_path": str(out), "task_dir_used": str(rd), "shape": list(frame.shape), "clip": [float(vmin), float(vmax)]}


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
        "Plot one snapshot from several wavefield tasks side by side (shared color scale) — e.g. to "
        "compare Acoustic vs VTI vs TTI wavefronts. Returns the saved comparison image path."
    ),
    params_model=CompareWavefieldsParams,
)
def compare_wavefields(args: CompareWavefieldsParams) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    allv = np.concatenate([f.ravel() for f in frames])
    vmin, vmax = np.percentile(allv, [2, 98])
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, frame, label in zip(axes, frames, args.labels):
        ax.imshow(frame, cmap="seismic", vmin=float(vmin), vmax=float(vmax), aspect="equal")
        ax.set_title(label)
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("z (cells)")
    if args.suptitle:
        fig.suptitle(args.suptitle)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"image_path": str(out), "n_panels": n, "clip": [float(vmin), float(vmax)]}


class MakeGifParams(BaseModel):
    task_dir: str = Field(..., description="Wavefield task directory from run_task.")
    abcn: int = Field(..., description="PML thickness used in the run.")
    free_surface: bool = Field(False, description="Whether the run used a free surface.")
    shot: int = Field(0)
    field: int = Field(0)
    out_name: str = Field("wavefield.gif", description="Output GIF filename, saved under <task_dir>/output/.")
    fps: int = Field(8, ge=1, le=30, description="Frames per second.")
    topography_path: str | None = Field(None, description="Optional path to the topo .npy; overlays the surface line on each frame.")


@register(
    name="make_wavefield_gif",
    description=(
        "Animate all snapshots of a wavefield task into a GIF (seismic colormap, shared color "
        "scale). Optionally overlays a topography line. Returns the saved GIF path. Use after a "
        "wavefield run with several snapshot_times — e.g. to show propagation under topography."
    ),
    params_model=MakeGifParams,
)
def make_wavefield_gif(args: MakeGifParams) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    try:
        wf = _load_snaps(args.task_dir, args.abcn, args.free_surface, args.shot, args.field)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if wf.shape[0] < 2:
        return {"error": "need >= 2 snapshots for a GIF; re-run with more snapshot_times."}
    vmin, vmax = np.percentile(wf, [1, 99])
    topo = None
    if args.topography_path:
        try:
            topo = np.load(args.topography_path)
        except Exception:
            topo = None

    rd = resolve_task_dir(args.task_dir)
    out = rd / "output" / args.out_name
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(wf[0], cmap="seismic", vmin=float(vmin), vmax=float(vmax), aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="amplitude")
    if topo is not None and not args.free_surface:
        ax.plot(np.arange(len(topo)), topo - args.abcn, "k-", lw=1.0)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")

    def update(i):
        im.set_data(wf[i])
        ax.set_title(f"wavefield snapshot {i + 1}/{wf.shape[0]}")
        return [im]

    anim = FuncAnimation(fig, update, frames=wf.shape[0], interval=1000 // args.fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    return {"gif_path": str(out), "task_dir_used": str(rd), "n_frames": int(wf.shape[0])}
