"""Visualization tools — thin LLM-callable wrappers over ``sweep_viz.wavefield``.

The actual plotting lives in sweep-viz (plot_snapshot / animate_snapshots /
compare_snapshots). These tools only handle the agent-side concerns — locating a
task's snapshots.npy and cropping the PML — then delegate the drawing. Pass
``abcn`` (the PML thickness used in the run) so the absorbing border is cropped.
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
            f"(forward tasks have no snapshots; use build_wavefield_spec with snapshot_times)"
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
    out_name: str = Field("wavefield.png", description="Output PNG filename, saved under <task_dir>/output/.")
    title: str | None = Field(None, description="Plot title.")


@register(
    name="plot_wavefield",
    description=(
        "Render one wavefield snapshot from a finished wavefield task to a PNG (via sweep-viz). "
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sweep_viz import wavefield as viz

    frame = wf[args.snapshot_index]
    rd = resolve_task_dir(args.task_dir)
    out = rd / "output" / args.out_name
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
        "sweep-viz) — e.g. Acoustic vs VTI vs TTI wavefronts. task_dirs must be distinct, real "
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
    from sweep_viz import wavefield as viz

    try:
        viz.compare_snapshots(frames, list(args.labels), args.out_path, suptitle=args.suptitle)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"image_path": args.out_path, "n_panels": len(frames)}


# ---------------------------------------------------------------------------
# make_wavefield_gif
# ---------------------------------------------------------------------------

class MakeGifParams(BaseModel):
    task_dir: str = Field(..., description="Wavefield task directory from run_task.")
    abcn: int = Field(..., description="PML thickness used in the run.")
    free_surface: bool = Field(False, description="Whether the run used a free surface.")
    shot: int = Field(0)
    field: int = Field(0)
    out_name: str = Field("wavefield.gif", description="Output GIF (or .mp4) filename, saved under <task_dir>/output/.")
    fps: int = Field(8, ge=1, le=30, description="Frames per second.")
    topography_path: str | None = Field(
        None,
        description=(
            "Path to the 1-D topography .npy used in the run. When set, the wavefield is shown on "
            "the PHYSICAL grid (air above the surface masked white) with the topography line + "
            "source/receiver markers overlaid — like the curvilinear notebook figure."
        ),
    )
    curvilinear: bool = Field(
        True,
        description="True for AcousticCurvilinear (resample computational→physical); False for image-method physical-grid fields. Only used when topography_path is set.",
    )
    sources: list[list[int]] | None = Field(
        None,
        description="Optional source markers as [[x, depth_below_surface], ...] (overlaid on the topo figure).",
    )
    receivers: list[list[int]] | None = Field(
        None,
        description="Optional receiver markers as [[x, depth_below_surface], ...].",
    )


@register(
    name="make_wavefield_gif",
    description=(
        "Animate all snapshots of a wavefield task into a GIF/MP4 (via sweep-viz animate_snapshots). "
        "Returns the saved path. Use after a wavefield run with several snapshot_times — e.g. to show "
        "propagation under topography."
    ),
    params_model=MakeGifParams,
)
def make_wavefield_gif(args: MakeGifParams) -> dict[str, Any]:
    try:
        wf = _load_snaps(args.task_dir, args.abcn, args.free_surface, args.shot, args.field)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if wf.shape[0] < 2:
        return {"error": "need >= 2 snapshots for a GIF; re-run with more snapshot_times."}
    from sweep_viz import wavefield as viz

    rd = resolve_task_dir(args.task_dir)
    out = rd / "output" / args.out_name
    try:
        if args.topography_path:
            topo = np.load(args.topography_path)
            viz.animate_snapshots_topography(
                list(wf), topo, str(out),
                sources=args.sources, receivers=args.receivers,
                curvilinear=args.curvilinear, fps=args.fps,
            )
        else:
            viz.animate_snapshots(list(wf), str(out), fps=args.fps)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"gif_path": str(out), "task_dir_used": str(rd), "n_frames": int(wf.shape[0])}
