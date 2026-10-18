"""End-to-end FWI on a small 2-layer model with several losses.

We use the **sweep** 2-D acoustic solver with ``impl='c'`` and
``boundary saving`` so the propagation is the same C-extension that
production FWI runs use.  Each loss in ``LOSS_FACTORIES`` is plugged in
as the FWI misfit and we record the per-iteration objective value and
the inverted model.

Model
-----
* ``nz x nx = 60 x 100`` cells, ``dh = 12.5 m``  (12 wavelengths of 100 m
  Ricker at vp=2000 m/s).
* True model:  layered ``vp`` with a step from 2000 to 3000 m/s at
  half-depth and a Gaussian anomaly +400 m/s slightly off-centre.
* Initial model:  a smooth ramp from 2000 to 2500.
* Sources:  5 shots evenly spaced along the surface (``z = 2``).
* Receivers:  surface streamer of 50 receivers (every 2 cells).
* Wavelet:  10 Hz Ricker with delay 0.10 s.

Inversion
---------
* Adam optimiser at lr=20 (the canonical sweep example uses 25 for
  Marmousi, so 20 is conservative for this smaller model).
* 25 epochs, all 5 shots per epoch.
* No frequency multi-scale (we want to see the *raw* basin of attraction
  of each loss on a Ricker source).

Outputs
-------
* ``fwi_layer.json`` — per-loss summary (final RMSE vs. true, loss curve).
* ``figs/fwi_<name>.png`` — true vs. inverted vs. initial model for each
  loss.
* ``figs/fwi_loss_curves.png`` — overlaid loss curves.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sweep_loss as sl
from sweep.equations import Acoustic
from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def make_true_init(nz, nx):
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    vp_true = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 3000.0
    # +400 m/s Gaussian anomaly at (3nz/4, nx/2), sigma 6
    cz, cx, sigma, dvp = 3 * nz // 4, nx // 2, 6.0, 400.0
    vp_true += (dvp * np.exp(-0.5 * (((z - cz) / sigma) ** 2
                                      + ((x - cx) / sigma) ** 2))).astype(np.float32)
    # Initial: smooth ramp 2000 -> 2500 with depth
    zline = np.linspace(2000.0, 2500.0, nz, dtype=np.float32)
    vp_init = np.broadcast_to(zline[:, None], (nz, nx)).copy()
    return vp_true, vp_init


def build_solver(shape, dev, dt, dh, abcn=20):
    eq = Acoustic(spatial_order=4, device=dev, backend="torch")
    return PropTorch(
        eq, shape=shape, dev=dev, dh=dh, dt=dt,
        source_type=["h1"], receiver_type=["h1"],
        abcn=abcn, free_surface=False, pml_type="cpmlr",
        backend="torch", impl="c",
        cuda_options=CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(storage="gpu"),
            ),
        ),
    )


def build_geometry(nz, nx, n_shots=5, rec_step=2):
    sx_pos = np.linspace(8, nx - 8, n_shots, dtype=np.int64)
    sz = np.full_like(sx_pos, 2)
    sources = np.stack([sx_pos, sz], axis=1)
    rx = np.arange(0, nx, rec_step, dtype=np.int64).reshape(-1, 1)
    rz = np.full_like(rx, 2)
    receivers = np.concatenate([rx, rz], axis=1)
    receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)
    return sources, receivers


def build_wavelet(nt, dt, fm=10.0, delay=0.10):
    t = np.arange(0, nt * dt, dt, dtype=np.float32)
    return ricker(t - delay, f=fm).astype(np.float32)


def loss_factories(dt):
    return {
        "L2":        sl.L2Loss(reduction="sum"),
        "L1":        sl.L1Loss(reduction="sum"),
        "Huber":     sl.HuberLoss(delta=0.05, reduction="sum"),
        "NCC":       sl.GlobalCorrelationLoss(reduction="sum"),
        "Envelope":  sl.EnvelopeLoss(p=2, reduction="sum"),
        "FreqL2":    sl.FrequencyDomainL2Loss(reduction="sum"),
        "Laplace":   sl.LaplaceL2Loss(s=3.0, dt=dt, reduction="sum"),
        "AWI":       sl.AWILoss(dt=dt, epsilon=1e-4, reduction="sum"),
        "W1":        sl.Wasserstein1Loss(positive="abs", dt=dt, reduction="sum"),
        "NIM":       sl.NIMLoss(positive="square", dt=dt, reduction="sum"),
    }


def run_fwi(loss_name, loss_fn, true_vp, init_vp, solver, wave, sources, receivers, dev,
            epochs=60, lr=4.0):
    nz, nx = true_vp.shape
    # forward to make observed data with the true model
    true_t = torch.from_numpy(true_vp).to(dev)
    with torch.no_grad():
        obs = solver(wave, sources, receivers, models=[true_t], use_boundary_saving=False)
    obs = obs.detach()

    inv_vp = torch.from_numpy(init_vp).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([inv_vp], lr=lr, eps=1e-22)

    losses = []
    rmses = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        opt.zero_grad()
        syn = solver(wave, sources, receivers, models=[inv_vp], use_boundary_saving=True)
        loss = loss_fn(syn, obs)
        loss.backward()
        opt.step()
        with torch.no_grad():
            cur_rmse = float(torch.sqrt(((inv_vp - true_t) ** 2).mean()).item())
        losses.append(float(loss.detach().cpu()))
        rmses.append(cur_rmse)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  [{loss_name}] ep {ep:03d} loss={losses[-1]:.5e} rmse={cur_rmse:.2f} m/s "
                  f"({time.perf_counter() - t0:.1f}s)")
    elapsed = time.perf_counter() - t0
    return inv_vp.detach().cpu().numpy(), losses, rmses, elapsed


def plot_models(true_vp, init_vp, inv_vp, loss_name, output_dir):
    vmin = true_vp.min(); vmax = true_vp.max()
    nz, nx = true_vp.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, m, t in zip(axes,
                        [true_vp, init_vp, inv_vp],
                        ["True", "Initial", f"Inverted ({loss_name})"]):
        im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.set_title(t)
        ax.set_xlabel("X cell"); ax.set_ylabel("Z cell")
        fig.colorbar(im, ax=ax, shrink=0.85, label="vp (m/s)")
    fig.tight_layout()
    fig.savefig(output_dir / f"fwi_{loss_name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    nz, nx = 60, 100
    dh = 12.5
    dt = 0.0015
    nt = 800
    n_shots = 5
    fm = 10.0
    delay = 0.10

    true_vp, init_vp = make_true_init(nz, nx)
    solver = build_solver((nz, nx), dev, dt, dh)
    sources, receivers = build_geometry(nz, nx, n_shots=n_shots)
    wave = build_wavelet(nt, dt, fm=fm, delay=delay)

    print(f"shape={(nz, nx)} dt={dt} nt={nt} shots={n_shots} recs={receivers.shape[1]}")

    factories = loss_factories(dt)
    summary = []
    all_losses = {}
    all_rmses = {}
    for name, loss_fn in factories.items():
        print(f"=== {name}")
        try:
            inv_vp, losses, rmses, elapsed = run_fwi(
                name, loss_fn, true_vp, init_vp, solver, wave, sources, receivers, dev,
                epochs=60, lr=4.0)
            np.save(HERE / f"inv_{name}.npy", inv_vp)
            plot_models(true_vp, init_vp, inv_vp, name, FIGS)
            summary.append({
                "name": name,
                "final_rmse_m_per_s": rmses[-1],
                "first_rmse_m_per_s": rmses[0],
                "init_rmse_m_per_s": float(np.sqrt(np.mean((init_vp - true_vp) ** 2))),
                "loss_curve": losses,
                "rmse_curve": rmses,
                "elapsed_s": elapsed,
            })
            all_losses[name] = losses
            all_rmses[name] = rmses
        except Exception as e:
            print(f"  FAILED: {e!r}")
            summary.append({"name": name, "error": repr(e)})

    # Overlay normalised loss curves
    if summary:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for name, lc in all_losses.items():
            lc = np.asarray(lc)
            lc_norm = lc / max(abs(lc[0]), 1e-30)
            axes[0].plot(lc_norm, label=name)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss / loss[0]")
        axes[0].set_title("Normalised loss curves")
        axes[0].legend(fontsize=8, loc="best")
        for name, rc in all_rmses.items():
            axes[1].plot(rc, label=name)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("RMSE (m/s)")
        axes[1].set_title("Inverted model RMSE vs true")
        axes[1].legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(FIGS / "fwi_loss_curves.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    out_path = HERE / "fwi_layer.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
