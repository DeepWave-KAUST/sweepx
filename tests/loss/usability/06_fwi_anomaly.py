"""FWI on a small *Gaussian-anomaly* model — designed to actually converge.

The previous 2-layer FWI (`03_fwi_layer.py`) had a 1000 m/s discontinuity
between background and lower layer, plus a +400 m/s anomaly.  Even with
moderate LR none of the losses recovered well in 60 epochs without a
multi-scale strategy.

This script uses a **kinematically gentle** setup that is solvable
without multi-scale:

* Background: constant 2000 m/s.
* Target: a single Gaussian anomaly  (+200 m/s, σ = 8 cell) embedded
  at mid-depth.
* Initial model: constant 2000 m/s (i.e. *no* anomaly).
* Ricker source: **5 Hz peak** (twice the wavelength of the 10 Hz
  source used in `03_fwi_layer.py`) — half-wavelength ≈ 200 m vs.
  anomaly diameter ≈ 100 m, so it is *barely* resolvable but well
  within the basin of attraction of even the L2 loss.
* 11 shots, 50 receivers, 120 epochs.
* **Source-illumination preconditioning** applied to the gradient (the
  standard "diagonal Hessian" trick used in sweep's marmousi example).
* Adam lr=20, eps=1e-22 (matches the sweep marmousi example).

Outputs
-------
* ``fwi_anomaly.json`` — per-loss summary.
* ``figs/fwi_anom_<loss>.png`` — true vs. init vs. inverted.
* ``figs/fwi_anom_loss_curves.png`` — loss + RMSE overlay.
"""

from __future__ import annotations

import json
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


def make_true_init(nz=80, nx=120):
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    vp_true = np.full((nz, nx), 2000.0, dtype=np.float32)
    cz, cx, sigma, dvp = nz // 2, nx // 2, 8.0, 200.0
    vp_true += (dvp * np.exp(-0.5 * (((z - cz) / sigma) ** 2
                                      + ((x - cx) / sigma) ** 2))).astype(np.float32)
    vp_init = np.full((nz, nx), 2000.0, dtype=np.float32)
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


def build_geometry(nz, nx, n_shots=11, rec_step=2):
    sx_pos = np.linspace(6, nx - 6, n_shots, dtype=np.int64)
    sz = np.full_like(sx_pos, 2)
    sources = np.stack([sx_pos, sz], axis=1)
    rx = np.arange(0, nx, rec_step, dtype=np.int64).reshape(-1, 1)
    rz = np.full_like(rx, 2)
    receivers = np.concatenate([rx, rz], axis=1)[None, ...].repeat(sources.shape[0], axis=0)
    return sources, receivers


def loss_factories(dt):
    """Each entry is ``(loss_module, recommended lr)``.

    The lr is tuned per loss because the gradient magnitude depends on the
    misfit formula (L2 is O(residual), L1 is O(sign), NCC is O(1/‖d‖²),
    etc.); a single lr for all is unfair (`lr=2` was good for L2/Huber/NCC
    but blew L1/AWI/W1/ExpPhase up in an earlier run).
    """
    return {
        "L2":       (sl.L2Loss(reduction="sum"),                                       2.0),
        "L1":       (sl.L1Loss(reduction="sum"),                                       0.2),
        "Huber":    (sl.HuberLoss(delta=0.1, reduction="sum"),                         2.0),
        "NCC":      (sl.GlobalCorrelationLoss(reduction="sum"),                        2.0),
        "Envelope": (sl.EnvelopeLoss(p=2, reduction="sum"),                            2.0),
        "FreqL2":   (sl.FrequencyDomainL2Loss(reduction="sum"),                        2.0),
        "AWI":      (sl.AWILoss(dt=dt, epsilon=1e-4, reduction="sum"),                 0.5),
        "W1":       (sl.Wasserstein1Loss(positive="abs", dt=dt, reduction="sum"),      0.3),
        "NIM":      (sl.NIMLoss(positive="square", dt=dt, reduction="sum"),           50.0),
        "ExpPhase": (sl.ExponentiatedPhaseLoss(reduction="sum"),                       0.02),
    }


def run_fwi(name, loss_fn, true_vp, init_vp, solver, wave, sources, receivers, dev,
            epochs=200, lr=2.0, use_illum=True, normalize_loss=False):
    nz, nx = true_vp.shape
    true_t = torch.from_numpy(true_vp).to(dev)
    with torch.no_grad():
        obs = solver(wave, sources, receivers, models=[true_t],
                     use_boundary_saving=False).detach()
    # Per-shot scaling so the gradient magnitude is comparable across losses
    # and not sensitive to the size of the data tensor.
    if normalize_loss:
        norm = float(obs.numel())
    else:
        norm = 1.0

    inv_vp = torch.from_numpy(init_vp).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([inv_vp], lr=lr)

    losses, rmses = [], []
    t0 = time.perf_counter()
    for ep in range(epochs):
        opt.zero_grad()
        syn = solver(wave, sources, receivers, models=[inv_vp],
                     use_boundary_saving=True)
        L = loss_fn(syn, obs) / norm
        L.backward()
        if use_illum:
            illum = getattr(solver, "source_illumination", None)
            if isinstance(illum, torch.Tensor):
                # diagonal-Hessian-like source-illumination preconditioner
                # (Pratt 1999).  eps_floor relative to peak illumination.
                eps_floor = 1e-2 * illum.amax().clamp_min(1e-30)
                inv_vp.grad.div_(illum + eps_floor)
        opt.step()
        with torch.no_grad():
            cur_rmse = float(torch.sqrt(((inv_vp - true_t) ** 2).mean()).item())
        losses.append(float(L.detach().cpu()))
        rmses.append(cur_rmse)
        if ep == 0 or (ep + 1) % 25 == 0:
            print(f"  [{name}] ep {ep:03d} loss={losses[-1]:.4e} rmse={cur_rmse:.2f} m/s "
                  f"({time.perf_counter() - t0:.1f}s)")
    return inv_vp.detach().cpu().numpy(), losses, rmses, time.perf_counter() - t0


def plot_models(true_vp, init_vp, inv_vp, name, output_dir):
    nz, nx = true_vp.shape
    vmin, vmax = true_vp.min(), true_vp.max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, m, t in zip(axes,
                        [true_vp, init_vp, inv_vp],
                        ["True", "Initial", f"Inverted ({name})"]):
        im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
        ax.set_title(t)
        ax.set_xlabel("X cell"); ax.set_ylabel("Z cell")
        fig.colorbar(im, ax=ax, shrink=0.85, label="vp (m/s)")
    fig.tight_layout()
    fig.savefig(output_dir / f"fwi_anom_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    nz, nx = 80, 120
    dh, dt, nt = 12.5, 0.0015, 1200
    fm, delay = 5.0, 0.20
    epochs = 200

    true_vp, init_vp = make_true_init(nz, nx)
    init_rmse = float(np.sqrt(np.mean((init_vp - true_vp) ** 2)))
    print(f"Initial RMSE: {init_rmse:.2f} m/s   (target: Gaussian anomaly +200 m/s)")

    solver = build_solver((nz, nx), dev, dt, dh)
    sources, receivers = build_geometry(nz, nx, n_shots=11)
    t = np.arange(0, nt * dt, dt, dtype=np.float32)
    wave = ricker(t - delay, f=fm).astype(np.float32)

    factories = loss_factories(dt)
    summary = []
    all_losses = {}
    all_rmses = {}
    for name, (loss_fn, lr) in factories.items():
        print(f"=== {name}  (lr={lr})")
        try:
            inv_vp, losses, rmses, elapsed = run_fwi(
                name, loss_fn, true_vp, init_vp, solver, wave, sources, receivers, dev,
                epochs=epochs, lr=lr, use_illum=True)
            np.save(HERE / f"inv_anom_{name}.npy", inv_vp)
            plot_models(true_vp, init_vp, inv_vp, name, FIGS)
            summary.append({
                "name": name,
                "init_rmse_m_per_s": init_rmse,
                "final_rmse_m_per_s": rmses[-1],
                "best_rmse_m_per_s": float(min(rmses)),
                "best_epoch":        int(np.argmin(rmses)),
                "loss_curve": losses,
                "rmse_curve": rmses,
                "elapsed_s": elapsed,
            })
            all_losses[name] = losses
            all_rmses[name] = rmses
        except Exception as e:
            print(f"  FAILED: {e!r}")
            summary.append({"name": name, "error": repr(e)})

    if all_losses:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for name, lc in all_losses.items():
            lc = np.asarray(lc)
            axes[0].plot(lc / max(abs(lc[0]), 1e-30), label=name)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss / loss[0]")
        axes[0].set_title("Normalised loss curves")
        axes[0].legend(fontsize=8, loc="best")
        for name, rc in all_rmses.items():
            axes[1].plot(rc, label=name)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("RMSE (m/s)")
        axes[1].set_title("Inverted-model RMSE vs true")
        axes[1].legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(FIGS / "fwi_anom_loss_curves.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    (HERE / "fwi_anomaly.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote fwi_anomaly.json")


if __name__ == "__main__":
    main()
