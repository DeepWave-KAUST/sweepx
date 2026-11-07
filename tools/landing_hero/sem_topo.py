"""SEM image for the home-page spotlight: elastic waves under a two-hill
surface on the body-fitted Quad9 mesh (the SEM half of the FD-vs-SEM
topography comparison, notebook 41's setup with a wider model).

Writes sem_topo.npz next to this file (vz snapshots, the element nodes and
the surface); sem_topo_plot.py draws the tile from it.
"""
import pathlib
import numpy as np
import torch

from sweep.signal import ricker
from sweep.sem import PropSEM, ElasticVSSEM2DDeformed, suggest_discretization

OUT = pathlib.Path(__file__).with_name("sem_topo.npz")
DEV = "cuda"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

X_MAX, Z_MAX = 2400.0, 1200.0
DH = 5.0
NX, NZ = int(round(X_MAX / DH)) + 1, int(round(Z_MAX / DH)) + 1
F0, DELAY = 12.0, 0.10
SEM_ABCN = 40
SNAP_TS = list(np.round(np.arange(0.20, 0.40 + 1e-9, 0.02), 4))
SRC_DEPTH, REC_DEPTH = 40.0, 8.0
VPE, VSE, RHOE = 2400.0, 1200.0, 2000.0
BASE = 160.0

disc = suggest_discretization(X_MAX, Z_MAX, v_min=VSE, v_max=VPE, f0=F0,
                              ppw=8.0, record_length=SNAP_TS[-1] + 0.01)
SO, DT, NT = disc.n_order, disc.dt, disc.nt


def z_top(x):
    # down-positive surface depth: two hills (rising above BASE) and a valley
    h = (130.0 * np.exp(-((x - 800.0) ** 2) / (2 * 170.0 ** 2))
         + 90.0 * np.exp(-((x - 1650.0) ** 2) / (2 * 140.0 ** 2))
         - 40.0 * np.exp(-((x - 1230.0) ** 2) / (2 * 90.0 ** 2)))
    return BASE - h


xg = np.arange(NX) * DH
topo = z_top(xg) / DH
src_x = 1230.0                                   # valley floor
sc = int(round(src_x / DH))
rec_x = np.arange(100.0, X_MAX - 100.0 + 1e-6, 10.0)
rcol = np.round(rec_x / DH).astype(np.int64)
wav = ricker(np.arange(NT) * DT - DELAY, f=F0).astype(np.float32)
src = torch.tensor([[sc, topo[sc] + SRC_DEPTH / DH]])
rec = torch.tensor(np.stack([rcol, topo[rcol] + REC_DEPTH / DH], -1)[None])
snap_it = [min(int(round(s / DT)), NT - 1) for s in SNAP_TS]

prop = PropSEM(ElasticVSSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=DT,
               n_order=SO, topography=topo, abcn=SEM_ABCN, f_dominant=F0,
               device=DEV, impl="eager", source_type=["sxx", "szz"],
               receiver_type=["vz"])
models = [torch.full((NZ, NX), v, dtype=torch.float32, device=DEV)
          for v in (VPE, VSE, RHOE)]
with torch.no_grad():
    _, snaps = prop(torch.tensor(wav, device=DEV), src, rec, models=models,
                    return_wavefield=True, snapshot_times=snap_it)
k = list(prop.snapshot_fields).index("v_z")
w = snaps[:, k, 0, 0].cpu().numpy()[:, :NZ, SEM_ABCN:SEM_ABCN + NX]
m = prop.mesh
np.savez_compressed(OUT, w=w, x_e=m.x_e.cpu().numpy() - SEM_ABCN * DH,
                    z_e=m.z_e.cpu().numpy(), xg=xg, ztop=z_top(xg),
                    src_x=src_x, src_z=z_top(src_x) + SRC_DEPTH,
                    snap_ts=np.array(SNAP_TS), x_max=X_MAX, dh=DH, so=SO)
print(disc, "\nsnapshots", w.shape, "->", OUT)
