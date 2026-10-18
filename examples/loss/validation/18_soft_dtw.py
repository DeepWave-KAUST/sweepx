"""Validate SoftDTWLoss (Cuturi-Blondel 2017; Blondel-Mensch-Vert 2020 divergence).

Recipe:
* Build a SHORT Ricker pair (SoftDTW's DP is O(nt²) Python loops; use nt=96).
* Reproduce the same pointwise cost matrix Δ_ij = (s_i − o_j)² the loss uses.
* Panel (a): syn vs. obs traces.
* Panel (b): Δ_ij heatmap with the (shifted) diagonal annotated.
* Panel (c): basin scan vs. time-shift comparing `divergence=True` (the
  Blondel-Mensch-Vert non-negative variant) and `divergence=False` (raw
  Cuturi-Blondel, can be negative for γ > 0).
* Panel (d): SoftDTW(syn, syn) for several seeds at the two settings —
  must be ~0 for `divergence=True`, generally < 0 for `divergence=False`
  (this is the bug the earlier REPORT.md identified and fixed).
"""
from __future__ import annotations

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

from sweep_loss import SoftDTWLoss  # noqa: E402


NAME = "18_soft_dtw"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 96, 4e-3, 4.0, 0.20, 0.040
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    # Cost matrix
    s = torch.tensor(syn_np, dtype=torch.float64)
    o = torch.tensor(obs_np, dtype=torch.float64)
    Delta = ((s.unsqueeze(-1) - o.unsqueeze(-2)) ** 2).detach().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a)
    ax = axes[0, 0]
    ax.plot(t, syn_np, color="C0", lw=1.0, label="syn")
    ax.plot(t, obs_np, color="C1", lw=1.0, label=f"obs (+{int(tau0*1e3)} ms)")
    ax.set_xlim(0.05, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) syn vs. time-shifted obs")
    ax.legend(loc="upper right", fontsize=8)

    # (b) Δ_ij heatmap
    ax = axes[0, 1]
    im = ax.imshow(Delta, origin="lower", extent=[t[0], t[-1], t[0], t[-1]],
                   cmap="viridis")
    # Optimal hard-DTW path lies near t_j = t_i + tau0 (positive shift)
    ax.plot([t[0], t[-1] - tau0], [t[0] + tau0, t[-1]], color="C2", lw=0.8, ls=":",
            label=f"expected warp t_j = t_i + {int(tau0*1e3)} ms")
    ax.set_xlabel("t_j (obs) [s]")
    ax.set_ylabel("t_i (syn) [s]")
    ax.set_title(r"(b) pointwise cost $\Delta_{ij} = (s_i - o_j)^2$")
    ax.legend(loc="upper left", fontsize=7)
    plt.colorbar(im, ax=ax)

    # (c) basin scan
    ax = axes[1, 0]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-2.5 * half_T, 2.5 * half_T, 21)
    for label, fn, color in [
        ("divergence=True (Blondel 2020)",
         SoftDTWLoss(gamma=1.0, divergence=True, reduction="sum"), "C2"),
        ("divergence=False (raw Cuturi-Blondel)",
         SoftDTWLoss(gamma=1.0, divergence=False, reduction="sum"), "C3"),
    ]:
        vals = scan_shift(fn, syn_np, dt, tau_grid)
        ax.plot(tau_grid / half_T, vals, color=color, lw=1.4, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("loss")
    ax.set_title("(c) basin: raw soft-DTW can go negative; divergence stays ≥ 0")
    ax.legend(loc="upper center", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) self-distance test  D(x, x) = ?
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    n_seeds = 6
    self_div, self_raw = [], []
    for seed in range(n_seeds):
        x = rng.standard_normal(nt) * 0.5
        x_t = torch.tensor(x, dtype=torch.float64).view(1, nt, 1, 1)
        v_div = float(SoftDTWLoss(gamma=1.0, divergence=True,  reduction="sum")(x_t, x_t).detach())
        v_raw = float(SoftDTWLoss(gamma=1.0, divergence=False, reduction="sum")(x_t, x_t).detach())
        self_div.append(v_div)
        self_raw.append(v_raw)
    xs = np.arange(n_seeds)
    width = 0.4
    ax.bar(xs - width / 2, self_div, width=width, color="C2", label="divergence=True")
    ax.bar(xs + width / 2, self_raw, width=width, color="C3", label="divergence=False (raw)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("seed")
    ax.set_ylabel("SoftDTW(x, x)")
    ax.set_title("(d) self-distance — only divergence variant is ~0 for x=x")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("SoftDTWLoss — Cuturi-Blondel (2017); Blondel-Mensch-Vert (2020) divergence",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "self_divergence_max_abs": float(np.max(np.abs(self_div))),
        "self_raw_mean":           float(np.mean(self_raw)),
        "self_divergence_all":     self_div,
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
