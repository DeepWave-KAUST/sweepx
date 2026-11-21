"""3-D wavefronts for the home-page spotlight: a vertical force in an
isotropic, a VTI and a tilted (TTI) solid, on the 3-D SEM (notebook
60_sem_tti3d's media, at 1.6 times its frequency on a grid 1.6 times as
fine, so the fronts are that much thinner in the picture; twice as fine runs
out of memory on 48 GB in the tilted medium's Bond rotation).

    python sem_aniso3d.py [n_steps]      # n_steps: a short timing run

Writes sem_aniso3d.npz next to this file (the x-z and x-y planes through the
force); sem_aniso3d_plot.py draws the tile from it.
"""
import pathlib
import sys
import time

import numpy as np
import torch

from sweep.propagator import EagerOptions, Full
from sweep.sem import ElasticTTISEM3D, ElasticVSSEM3D, ElasticVTISEM3D, PropSEM

OUT = pathlib.Path(__file__).with_name("sem_aniso3d.npz")
DEV, F32 = "cuda", torch.float32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DH, NO, F0, DELAY = 3.125, 4, 28.8, 0.0375
SHAPE, ABCN, DT = (265, 265, 265), 32, 1.5625e-4        # an 825 m cube
VP, VS, RHO = 2000.0, 1000.0, 2000.0
ANISO = (0.25, 0.05, 0.15)                              # epsilon, delta, gamma
TILT = (np.radians(40.0), np.radians(30.0))             # theta, phi
SNAP = 880                                              # 0.1375 s, 0.1 s after the peak
NT = int(sys.argv[1]) if len(sys.argv) > 1 else SNAP + 1
t = np.arange(NT) * DT
a = (np.pi * F0 * (t - DELAY)) ** 2
wav = torch.from_numpy((1 - 2 * a) * np.exp(-a)).to(DEV, F32)
C0 = 130                                                # the middle of an element
src = np.array([[C0, C0, C0]], dtype=float)             # (ix, iy, iz)
rec = np.array([[[C0, C0, 40.0]]])


def snapshot(eq, extra=()):
    p = PropSEM(eq, shape=SHAPE, dh=DH, dt=DT, n_order=NO, abcn=ABCN, free_surface=False,
                f_dominant=F0, dev=DEV, dtype=F32, impl="eager", memory=Full(),
                eager_options=EagerOptions(use_compile=False), source_type=["vz"],
                receiver_type=["vz"])
    ms = [torch.full(SHAPE, float(v), dtype=F32, device=DEV) for v in (VP, VS, RHO) + extra]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        _, snaps = p(wav, src, rec, models=ms, return_wavefield=True,
                     snapshot_times=[min(SNAP, NT - 1)])
    torch.cuda.synchronize()
    print(f"{eq.__name__}: {NT} steps in {time.time() - t0:.1f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    v = snaps[0, 0, 0, 0].numpy()                       # (nz, ny, nx) with the layer
    pz, py, px = ((n - s) // 2 for n, s in zip(v.shape, SHAPE))
    v = v[pz:pz + SHAPE[0], py:py + SHAPE[1], px:px + SHAPE[2]]
    return v[:, C0, :], v[C0, :, :]


if __name__ == "__main__":
    runs = {"iso": (ElasticVSSEM3D, ()), "vti": (ElasticVTISEM3D, ANISO),
            "tti": (ElasticTTISEM3D, ANISO + TILT)}
    if NT <= SNAP:                                      # timing run: the tilted medium only
        snapshot(*runs["tti"])
        sys.exit()
    out = {}
    for name, (eq, extra) in runs.items():
        out[name + "_xz"], out[name + "_xy"] = snapshot(eq, extra)
        torch.cuda.empty_cache()
    np.savez_compressed(OUT, dh=DH, c0=C0, f0=F0, t_snap=SNAP * DT, **out)
    print("->", OUT)
