"""Validate GSOTLoss (Métivier et al. 2018).

Recipe:
* Build a SHORT Ricker pair (Hungarian solver is O(nt³); use nt=96).
* Reproduce the same cost matrix C[i,j] = η(t_i−t_j)² + (s_i−o_j)² the loss
  builds internally, plus its Hungarian assignment.
* Panel (a): the *graph* points (t_i, s_i) and (t_j, o_j) in the time-
  amplitude plane (the figure that gives GSOT its name).
* Panel (b): the optimal permutation drawn as connecting arrows
  syn point → assigned obs point.
* Panel (c): the cost matrix heatmap with the diagonal of the chosen
  permutation overlaid.
* Panel (d): basin scan vs. time shift — Métivier 2018 fig. 3 shows
  GSOT achieves a wide monotone basin (~3–4 half-λ).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    fig_path,
    fractional_shift,
    ricker,
    scan_shift,
    setup_matplotlib,
    write_summary,
)

from sweep_loss import GSOTLoss  # noqa: E402


NAME = "17_gsot"


def run() -> dict:
    plt = setup_matplotlib()

    # nt = 96 keeps the Hungarian solver well under a second
    nt, dt, fc, t0, tau0 = 96, 4e-3, 4.0, 0.20, 0.040
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    # Reproduce the same cost matrix the loss builds (auto-eta).
    d_max = float(max(np.abs(syn_np).max(), np.abs(obs_np).max()))
    Tshift = nt * dt / 4.0
    eta = (d_max / Tshift) ** 2 + 1e-30
    time_cost = (t[:, None] - t[None, :]) ** 2
    amp_cost = (syn_np[:, None] - obs_np[None, :]) ** 2  # i syn, j obs
    cost = eta * time_cost + amp_cost
    row, col = linear_sum_assignment(cost)
    # `col[i]` = obs sample index assigned to syn sample `i`

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) graph-space points
    ax = axes[0, 0]
    ax.scatter(t, syn_np, s=10, color="C0", label="(t_i, s_i)")
    ax.scatter(t, obs_np, s=10, color="C1", label=f"(t_j, o_j)  (+{int(tau0*1e3)} ms)",
               alpha=0.85)
    ax.set_xlim(0.05, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("amplitude")
    ax.set_title("(a) graph-space points used by GSOT")
    ax.legend(loc="upper right", fontsize=8)

    # (b) Hungarian assignment as arrows
    ax = axes[0, 1]
    ax.scatter(t, syn_np, s=10, color="C0", label="syn")
    ax.scatter(t, obs_np, s=10, color="C1", label="obs", alpha=0.85)
    # Only draw arrows in the wavelet support to keep the figure readable
    mask = (np.abs(syn_np) > 0.05 * d_max) | (np.abs(obs_np) > 0.05 * d_max)
    for i in np.where(mask)[0]:
        j = col[i]
        ax.plot([t[i], t[j]], [syn_np[i], obs_np[j]], color="gray", lw=0.5, alpha=0.4)
    ax.set_xlim(0.05, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(b) Hungarian assignment: syn → obs (arrows)")
    ax.legend(loc="upper right", fontsize=8)

    # (c) cost matrix + assignment diagonal
    ax = axes[1, 0]
    im = ax.imshow(cost, origin="lower", extent=[t[0], t[-1], t[0], t[-1]],
                   cmap="viridis")
    ax.plot(t[col], t[row], color="C3", lw=0.8, label="σ* (chosen)")
    ax.set_xlabel("t_j (obs) [s]")
    ax.set_ylabel("t_i (syn) [s]")
    ax.set_title(r"(c) GSOT cost matrix $C_{ij} = \eta(t_i-t_j)^2 + (s_i-o_j)^2$")
    ax.legend(loc="upper left", fontsize=8)
    plt.colorbar(im, ax=ax)

    # (d) basin scan
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-3.5 * half_T, 3.5 * half_T, 31)
    loss_fn = GSOTLoss(dt=dt, reduction="sum")
    vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
    v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
    ax.plot(tau_grid / half_T, v, color="C2", lw=1.6, label="GSOT")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-3.5, 3.5)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin scan — GSOT monotone over wide range (Métivier 2018 fig.3)")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("GSOTLoss — Métivier et al. (2018); Jonker-Volgenant Hungarian solver",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # Sanity: σ is a permutation (each obs sample assigned exactly once)
    return {
        "fig": os.path.basename(out),
        "eta_auto":            float(eta),
        "is_valid_permutation": bool(set(col.tolist()) == set(range(nt))),
        "loss_at_tau0_sum":    float(loss_fn(
            torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1),
            torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1),
        ).detach()),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
