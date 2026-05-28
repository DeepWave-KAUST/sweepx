"""Validate JensenShannonLoss (Yan et al. 2024).

Recipe:
* Build a Ricker pair (syn, obs).
* Reproduce the loss preprocessing: p = positive_transform(syn)/sum,
  q = positive_transform(obs)/sum, m = (p+q)/2.
* Panel (a): densities p, q, m on the time axis.
* Panel (b): per-sample KL integrands p·log(p/m) and q·log(q/m).
* Panel (c): basin scan vs. time-shift (positive='square'); also show
  the upper bound log(2) which JSD respects.
* Panel (d): self-check JSD(p, p) = 0 for a few random realisations
  (must be zero by definition).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    fig_path,
    fractional_shift,
    ricker,
    scan_shift,
    setup_matplotlib,
    write_summary,
)

from sweep_loss import JensenShannonLoss  # noqa: E402
from sweep_loss._utils import normalize_density, positive_transform  # noqa: E402


NAME = "13_jsd"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.020
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt)
    fs = positive_transform(syn, method="square").detach().numpy().ravel()
    fo = positive_transform(obs, method="square").detach().numpy().ravel()
    p = fs / (fs.sum() + 1e-30)
    q = fo / (fo.sum() + 1e-30)
    m = 0.5 * (p + q)
    eps = 1e-30

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) p, q, m
    ax = axes[0, 0]
    ax.plot(t, p, color="C0", lw=1.4, label="p (syn²)")
    ax.plot(t, q, color="C1", lw=1.4, label="q (obs²)")
    ax.plot(t, m, color="C3", lw=1.0, ls="--", label="m = (p+q)/2")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) JSD operates on normalised densities p, q and their mixture m")
    ax.legend(loc="upper right", fontsize=8)

    # (b) KL integrands
    ax = axes[0, 1]
    kl_p = p * (np.log(np.maximum(p, eps)) - np.log(np.maximum(m, eps)))
    kl_q = q * (np.log(np.maximum(q, eps)) - np.log(np.maximum(m, eps)))
    ax.plot(t, kl_p, color="C0", lw=1.2, label="p · log(p/m)  (KL(p‖m) integrand)")
    ax.plot(t, kl_q, color="C1", lw=1.2, label="q · log(q/m)  (KL(q‖m) integrand)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(b) per-sample KL integrands; JSD = ½(ΣKL_p + ΣKL_q)")
    ax.legend(loc="upper right", fontsize=8)

    # (c) basin
    ax = axes[1, 0]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    vals = scan_shift(JensenShannonLoss(positive="square", reduction="mean"),
                      syn_np, dt, tau_grid)
    ax.plot(tau_grid / half_T, vals, color="C2", lw=1.4, label="JSD (per-trace)")
    ax.axhline(math.log(2.0), color="k", lw=0.5, ls=":", label="upper bound log 2")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("JSD (nats)")
    ax.set_title("(c) basin: bounded by log 2; narrow basin → not anti-cycle-skip")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) JSD(p, p) = 0 check on random and Ricker inputs
    ax = axes[1, 1]
    test_cases = []
    rng = np.random.default_rng(0)
    losses_pp = []
    losses_pq_random_pair = []
    for seed in range(8):
        x = rng.standard_normal(nt)
        x_t = torch.tensor(x, dtype=torch.float64).view(1, nt, 1, 1)
        # JSD(x, x) — should be ~0
        v_pp = float(JensenShannonLoss(positive="square", reduction="sum")(x_t, x_t).detach())
        losses_pp.append(v_pp)
        # JSD with another independent gaussian
        y = rng.standard_normal(nt)
        y_t = torch.tensor(y, dtype=torch.float64).view(1, nt, 1, 1)
        v_pq = float(JensenShannonLoss(positive="square", reduction="sum")(x_t, y_t).detach())
        losses_pq_random_pair.append(v_pq)

    width = 0.4
    xs = np.arange(8)
    ax.bar(xs - width / 2, losses_pp, width=width, color="C2", label="JSD(x, x) (should be 0)")
    ax.bar(xs + width / 2, losses_pq_random_pair, width=width, color="C3",
           label="JSD(x, y) (independent gaussians)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(math.log(2.0), color="k", lw=0.5, ls=":", label="log 2 (sup)")
    ax.set_xlabel("seed")
    ax.set_title("(d) self-check JSD(x, x) = 0; vs. independent Gaussian pair")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("JensenShannonLoss — Yan et al. (2024); Endres-Schindelin metric (2003)",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "max_jsd_self_check": float(max(losses_pp)),
        "mean_jsd_random_pair": float(np.mean(losses_pq_random_pair)),
        "log2": math.log(2.0),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
