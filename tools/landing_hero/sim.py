"""Simulate the forward / adjoint / gradient loop for the landing animation.

One shot over a layered model.  The "true" model has a slow lens that the
current model lacks; the residual is what the lens adds to the record, and the
adjoint wavefield back-propagates it so the zero-lag cross-correlation with
the forward field (the gradient) outlines the lens.  The cross-correlation is
checked against the autograd gradient of the same misfit.

Writes sim.npz next to this file (about 1 min on one GPU).
"""
import pathlib
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

OUT = pathlib.Path(__file__).with_name("sim.npz")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nz, nx = 160, 480            # 1.6 km x 4.8 km at 10 m
dh, dt, nt = 10.0, 1e-3, 1400
abcn = 50
freq, delay = 9.0, 0.12
snap_every = 8

x = np.arange(nx) * dh
z = np.arange(nz)[:, None] * dh


def interface(z0, amp, wl, phase):
    return z0 + amp * np.sin(2 * np.pi * x / wl + phase)


# layer tops (m), gently undulating
tops = [interface(0.0, 0.0, 1.0, 0.0),
        interface(300.0, 25.0, 3200.0, 0.4),
        interface(620.0, 35.0, 4100.0, 1.6),
        interface(980.0, 30.0, 2900.0, 2.3),
        interface(1300.0, 25.0, 3600.0, 0.9)]
vels = [1800.0, 2100.0, 2350.0, 2550.0, 2700.0]
layer = np.zeros((nz, nx), dtype=np.int64)
for i, top in enumerate(tops):
    layer[z >= top[None, :]] = i
vp0 = np.asarray(vels, dtype=np.float32)[layer]

# the lens the current model is missing
lx, lz, ra, rb = 2650.0, 820.0, 330.0, 95.0
r = np.sqrt(((x[None, :] - lx) / ra) ** 2 + ((z - lz) / rb) ** 2)
lens = 0.5 * (1.0 - np.tanh((r - 1.0) / 0.12))
vp_true = (vp0 - 550.0 * lens).astype(np.float32)

src_ix, src_iz = 225, 2
sources = np.array([[src_ix, src_iz]], dtype=np.int64)
rec_ix = np.arange(8, nx - 8, 4)
receivers = np.array([[[ix, 2] for ix in rec_ix]], dtype=np.int64)

t = np.arange(nt, dtype=np.float32) * dt
wavelet = ricker(t - delay, f=freq).astype(np.float32)[None, :]

eq = Acoustic(spatial_order=8, device=dev, backend="torch")
prop = PropTorch(eq, shape=(nz, nx), dh=dh, dt=dt, nt=nt, abcn=abcn,
                 dev=dev, impl="eager", use_ckpt=False,
                 source_type=["h1"], receiver_type=["h1"])


def T(a):
    return torch.as_tensor(a, device=dev)


def crop(snaps):
    # (nsnap, nfield, B, 1, nz_pad, nx_pad) -> (nsnap, nz, nx)
    s = snaps[:, 0, 0, 0]
    return s[:, abcn:abcn + nz, abcn:abcn + nx].cpu().numpy()


with torch.no_grad():
    d_obs = prop.forward(wavelet, sources, receivers, models=[T(vp_true)])
    d_syn, u = prop.forward(wavelet, sources, receivers, models=[T(vp0)],
                            return_wavefield=True, snapshot_interval=1)
d_obs = d_obs.cpu().numpy()[0, :, :, 0]      # (nt, nrec)
d_syn = d_syn.cpu().numpy()[0, :, :, 0]
res = d_syn - d_obs
u = crop(u)                                  # every step, (nt, nz, nx)

# adjoint: inject the time-reversed residual at the receivers
adj_src = receivers[0][None]                 # (1, nrec, 2)
adj_wav = np.ascontiguousarray(res[::-1].T)        # (nrec, nt)
with torch.no_grad():
    _, lam_rev = prop.forward(adj_wav, adj_src, receivers, models=[T(vp0)],
                              return_wavefield=True, snapshot_interval=1)
lam = crop(lam_rev)[::-1]                    # lam[k] ~ adjoint field at time k

# zero-lag cross-correlation, accumulated backwards in time as the adjoint runs
u_tt = np.zeros_like(u)
u_tt[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dt ** 2
contrib = u_tt * lam                          # (nt, nz, nx)
grad_cum = np.cumsum(contrib[::-1], axis=0)[::-1]   # grad_cum[k] = sum_{j>=k}
grad_xcorr = grad_cum[0] * (2.0 / vp0 ** 3)

# autograd check of the same gradient (sign/shape only)
vp_var = T(vp0).clone().requires_grad_(True)
d = prop.forward(wavelet, sources, receivers, models=[vp_var])
loss = 0.5 * (d - T(d_obs)[None, :, :, None]).pow(2).sum()
loss.backward()
grad_ad = vp_var.grad.cpu().numpy()

mask = np.ones_like(grad_ad, dtype=bool)
mask[:6] = False                              # skip the source/receiver row
cc = np.corrcoef(grad_ad[mask], grad_xcorr[mask])[0, 1]
print(f"corr(autograd, xcorr) = {cc:+.3f}")

k = slice(0, nt, snap_every)
np.savez_compressed(
    OUT, vp0=vp0, vp_true=vp_true, lens=lens.astype(np.float32), layer=layer,
    tops=np.stack(tops), vels=np.asarray(vels), dh=dh, dt=dt, nt=nt,
    snap_every=snap_every, src=sources[0], rec_ix=rec_ix,
    d_obs=d_obs, d_syn=d_syn, res=res,
    u=u[k].astype(np.float16), lam=lam[k].astype(np.float16),
    grad_cum=grad_cum[k].astype(np.float32) * (2.0 / vp0 ** 3),
    grad_ad=grad_ad, corr=cc)
print("wrote", OUT)
