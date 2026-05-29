"""Demo: generate & plot anisotropic wavefields through sweep-agent.

Same homogeneous vp model, three equations — Acoustic (isotropic), AcousticVTI,
AcousticTTI (tilted 30 deg). A point source in the center; one wavefield
snapshot. The isotropic wavefront is a circle, VTI an ellipse (faster
horizontally), TTI a tilted ellipse. Everything is driven through the same
tools the LLM uses (build_spec / run_task).

Output: outputs/anisotropy_wavefield.png
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_snapshots, percentile_clip, run_wavefield

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="aniso_"))
    nz, nx = 140, 200
    np.save(tmp / "vp.npy", np.full((nz, nx), 2500.0, dtype=np.float32))
    np.save(tmp / "eps.npy", np.full((nz, nx), 0.25, dtype=np.float32))
    np.save(tmp / "delta.npy", np.full((nz, nx), 0.10, dtype=np.float32))
    np.save(tmp / "theta.npy", np.full((nz, nx), np.float32(np.deg2rad(30.0))))

    eps, dlt, th = str(tmp / "eps.npy"), str(tmp / "delta.npy"), str(tmp / "theta.npy")
    snap_t = 260
    cases = [
        ("Acoustic (isotropic)", "Acoustic", None),
        ("AcousticVTI", "AcousticVTI", {"epsilon": eps, "delta": dlt}),
        ("AcousticTTI (30 deg)", "AcousticTTI", {"epsilon": eps, "delta": dlt, "theta": th}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (title, eq, extra) in zip(axes, cases):
        print(f"[aniso] running {eq} ...")
        td = run_wavefield(
            eq, str(tmp / "vp.npy"), (nz, nx), [snap_t], extra_models=extra,
            output_dir=tmp / "runs", task_id=eq.lower(),
            dh=10.0, dt=1.0e-3, nt=snap_t + 1, fm=12.0, abcn=20,
            source_xz=(nx // 2, nz // 2),
        )
        wf = load_snapshots(td, abcn=20)[0]
        vmin, vmax = percentile_clip(wf, 2, 98)
        ax.imshow(wf, cmap="seismic", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("z (cells)")
    fig.suptitle(f"Anisotropic wavefronts at t-step {snap_t}  (homogeneous vp=2500, eps=0.25, delta=0.10)")
    out = OUT / "anisotropy_wavefield.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
