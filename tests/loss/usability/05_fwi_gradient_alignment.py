"""Single-shot, single-iteration gradient-alignment diagnostic.

For each loss, we compute the **first-step FWI gradient** with respect to
``vp`` when ``syn`` comes from the initial model and ``obs`` from the
true model.  The "ideal" descent direction is ``-(true_vp − init_vp)``
(steepest descent of ½‖vp − vp_true‖²).  We report the **cosine
similarity** between the loss gradient and that ideal direction:

* cos ≈ +1  → first FWI step moves vp toward the truth (basin shared
  with vp).
* cos ≈ 0  → uninformative direction.
* cos < 0  → cycle-skipping: the loss wants vp to move *away* from the
  truth.

This is a cheap, clear diagnostic that bypasses the LR / multi-scale /
optimizer questions inherent in a 60-epoch FWI loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sweep_loss as sl
from sweep.equations import Acoustic
from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


HERE = Path(__file__).resolve().parent


def make_true_init(nz, nx):
    z, x = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    vp_true = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 3000.0
    cz, cx, sigma, dvp = 3 * nz // 4, nx // 2, 6.0, 400.0
    vp_true += (dvp * np.exp(-0.5 * (((z - cz) / sigma) ** 2
                                      + ((x - cx) / sigma) ** 2))).astype(np.float32)
    zline = np.linspace(2000.0, 2500.0, nz, dtype=np.float32)
    vp_init = np.broadcast_to(zline[:, None], (nz, nx)).copy()
    return vp_true, vp_init


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz, nx = 60, 100
    dh, dt, nt = 12.5, 0.0015, 800
    fm, delay = 10.0, 0.10
    abcn = 20

    vp_true, vp_init = make_true_init(nz, nx)

    eq = Acoustic(spatial_order=4, device=dev, backend="torch")
    solver = PropTorch(
        eq, shape=(nz, nx), dev=dev, dh=dh, dt=dt,
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
    sx_pos = np.linspace(8, nx - 8, 5, dtype=np.int64)
    sz = np.full_like(sx_pos, 2)
    sources = np.stack([sx_pos, sz], axis=1)
    rx = np.arange(0, nx, 2, dtype=np.int64).reshape(-1, 1)
    rz = np.full_like(rx, 2)
    receivers = np.concatenate([rx, rz], axis=1)[None, ...].repeat(sources.shape[0], axis=0)

    t = np.arange(0, nt * dt, dt, dtype=np.float32)
    wave = ricker(t - delay, f=fm).astype(np.float32)

    true_t = torch.from_numpy(vp_true).to(dev)
    with torch.no_grad():
        obs = solver(wave, sources, receivers, models=[true_t],
                     use_boundary_saving=False).detach()

    ideal_dir = (true_t - torch.from_numpy(vp_init).to(dev)).flatten()
    ideal_dir = ideal_dir / ideal_dir.norm().clamp_min(1e-30)

    losses = {
        "L2":            sl.L2Loss(reduction="sum"),
        "L1":            sl.L1Loss(reduction="sum"),
        "Huber":         sl.HuberLoss(delta=0.05, reduction="sum"),
        "NCC":           sl.GlobalCorrelationLoss(reduction="sum"),
        "Envelope(p=2)": sl.EnvelopeLoss(p=2, reduction="sum"),
        "FreqL2":        sl.FrequencyDomainL2Loss(reduction="sum"),
        "Laplace(s=3)":  sl.LaplaceL2Loss(s=3.0, dt=dt, reduction="sum"),
        "AWI":           sl.AWILoss(dt=dt, epsilon=1e-4, reduction="sum"),
        "Deconv":        sl.DeconvolutionLoss(dt=dt, epsilon=1e-4, reduction="sum"),
        "OTMF2":         sl.OTMFLoss(dt=dt, epsilon=1e-4, order=2, reduction="sum"),
        "W1(abs)":       sl.Wasserstein1Loss(positive="abs", dt=dt, reduction="sum"),
        "W2(abs)":       sl.Wasserstein2Loss(positive="abs", dt=dt, n_quantiles=256,
                                              reduction="sum"),
        "NIM(square)":   sl.NIMLoss(positive="square", dt=dt, reduction="sum"),
        "JSD(square)":   sl.JensenShannonLoss(positive="square", reduction="sum"),
        "ExpPhase":      sl.ExponentiatedPhaseLoss(reduction="sum"),
        "CCT":           sl.CrossCorrelationTraveltimeLoss(dt=dt, power=2.0, sigma=80.0,
                                                            reduction="sum"),
        "LocalSim":      sl.LocalSimilarityLoss(sigma_samples=8.0, reduction="sum"),
        "TFPhase":       sl.TimeFrequencyPhaseLoss(alpha=0.5, n_fft=128, hop_length=32,
                                                    reduction="sum"),
    }

    rows = []
    for name, loss_fn in losses.items():
        vp = torch.from_numpy(vp_init).to(dev).requires_grad_(True)
        syn = solver(wave, sources, receivers, models=[vp],
                     use_boundary_saving=True)
        try:
            L = loss_fn(syn, obs)
            L.backward()
        except Exception as e:
            rows.append({"name": name, "error": repr(e)})
            print(f"{name:<14s} ERROR: {e!r}")
            continue
        g = vp.grad.detach()
        # FWI gradient: vp moves toward vp - lr*g.  So the descent direction
        # is -g; compare it with the *ideal* descent direction (truth - init).
        desc = (-g).flatten()
        desc_n = desc / desc.norm().clamp_min(1e-30)
        cos = float((desc_n * ideal_dir).sum().item())
        g_norm = float(g.norm().item())
        row = {
            "name": name,
            "cos_with_ideal": cos,
            "grad_norm": g_norm,
            "loss_value": float(L.detach().cpu().item()),
        }
        rows.append(row)
        print(f"{name:<14s} cos={cos:+.4f}  ‖g‖={g_norm:.3e}  L={row['loss_value']:.3e}")

    (HERE / "fwi_gradient_alignment.json").write_text(json.dumps(rows, indent=2))
    print("\nWrote fwi_gradient_alignment.json")


if __name__ == "__main__":
    main()
