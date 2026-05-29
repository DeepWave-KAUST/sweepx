"""Demo: irregular free-surface (topography) acquisition + propagation GIF.

Builds a 2-layer model under a hilly free surface, runs a curvilinear-grid
acoustic wavefield through sweep-agent, and writes:
  - outputs/topo_survey.png    : vp + topography + source/receivers (geometry)
  - outputs/topo_wavefield.gif : pressure snapshots propagating under the hills

The wavefield is shown on the computational (xi, eta) grid where the free
surface is the flat top row — that's the grid the curvilinear solver runs on.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_snapshots, percentile_clip, run_wavefield

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def build_topo(nx: int, base: int = 20, amp: float = 12.0, width: float = 22.0) -> np.ndarray:
    x = np.arange(nx)
    a = amp * np.exp(-((x - nx * 0.35) ** 2) / (2.0 * width**2))
    b = 0.7 * amp * np.exp(-((x - nx * 0.70) ** 2) / (2.0 * (width * 1.3) ** 2))
    return np.clip(np.round(base - (a + b)).astype(np.int64), 0, nx)


def build_vp(nz: int, nx: int, topo: np.ndarray) -> np.ndarray:
    vp = np.full((nz, nx), 3000.0, dtype=np.float32)
    for ix in range(nx):
        s = int(topo[ix])
        vp[s : s + 8, ix] = 1800.0
        vp[s + 8 : s + 28, ix] = 2400.0
    return vp


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="topo_"))
    nz, nx = 120, 220
    topo = build_topo(nx)
    vp = build_vp(nz, nx, topo)
    np.save(tmp / "vp.npy", vp)
    np.save(tmp / "topo.npy", topo)

    nt, stride = 900, 45
    snap_times = list(range(0, nt, stride))
    src_x = nx // 3

    print(f"[topo] running curvilinear wavefield ({len(snap_times)} snapshots) ...")
    td = run_wavefield(
        "AcousticCurvilinear", str(tmp / "vp.npy"), (nz, nx), snap_times,
        output_dir=tmp / "runs", task_id="topo_gif",
        dh=10.0, dt=1.0e-3, nt=nt, fm=9.0, abcn=30,
        source_xz=(src_x, 3), free_surface=True, topography=str(tmp / "topo.npy"),
    )

    # --- acquisition-geometry figure (physical grid) -----------------------
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    vpp = vp.astype(float).copy()
    air = np.arange(nz)[:, None] < topo[None, :]
    vpp[air] = np.nan
    im = ax.imshow(vpp, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="vp (m/s)", shrink=0.85)
    ax.plot(np.arange(nx), topo, "k-", lw=1.3, label="topography")
    ax.plot([src_x], [topo[src_x] + 3], "r*", ms=15, label="source")
    rx = np.arange(2, nx - 2, 3)
    ax.plot(rx, topo[rx] + 2, "yv", ms=3, label="receivers")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Topography acquisition geometry (curvilinear free surface)")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")
    survey = OUT / "topo_survey.png"
    fig.savefig(survey, dpi=130)
    plt.close(fig)

    # --- propagation GIF (computational grid) ------------------------------
    wf = load_snapshots(td, abcn=30, free_surface=True)  # (n_snap, nz, nx)
    vmin, vmax = percentile_clip(wf, 1, 99)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    im = ax.imshow(wf[0], cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("eta (surface = row 0)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="pressure")

    def update(i):
        im.set_data(wf[i])
        ax.set_title(f"Curvilinear wavefield — t-step {snap_times[i]}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(wf), interval=120, blit=False)
    gif = OUT / "topo_wavefield.gif"
    anim.save(gif, writer=PillowWriter(fps=8))
    plt.close(fig)

    print(f"saved: {survey}")
    print(f"saved: {gif}")


if __name__ == "__main__":
    main()
