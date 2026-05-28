"""Validate amplitude-normalised correlation misfits (family B).

Covers `GlobalCorrelationLoss` (NCC) and `TraceNormalizedL2Loss`.  The
defining property both losses claim from Choi & Alkhalifah (2012) is
**per-trace amplitude invariance**: scaling ``syn`` by a positive constant
α should leave the misfit unchanged.

Recipe:
* Build a Ricker pair (syn at t0, obs shifted by tau).
* Panel (a): plot syn vs. obs both raw and L2-normalised — visually confirm
  the normalisation is what enters the loss.
* Panel (b): scan α ∈ [0.1, 10] in obs scaling — flat curves for these two
  losses; L2 for comparison shows quadratic growth.
* Panel (c): adjoint sources for syn and α·syn (α=2) — should differ only
  by an O(1/α) rescaling, not in shape.
* Panel (d): equivalence check  J_nL2 = J_NCC + const (per Choi 2012, eq. 9):
  scan time-shift τ and overlay (NCC + offset).
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
    setup_matplotlib,
    write_summary,
)

from sweep_loss import (  # noqa: E402
    GlobalCorrelationLoss,
    L2Loss,
    TraceNormalizedL2Loss,
)
from sweep_loss._utils import l2_normalize  # noqa: E402


NAME = "02_correlation"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.020
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    # Normalised versions (what the loss internally uses)
    syn_n = l2_normalize(syn, dim=-3).detach().numpy().ravel()
    obs_n = l2_normalize(obs, dim=-3).detach().numpy().ravel()

    losses = [
        ("L2 (baseline)",                  L2Loss(reduction="sum")),
        ("GlobalCorrelation (NCC)",        GlobalCorrelationLoss(reduction="sum")),
        ("TraceNormalizedL2",              TraceNormalizedL2Loss(reduction="sum")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) raw + normalised traces
    ax = axes[0, 0]
    ax.plot(t, syn_np, color="C0", lw=1.0, label="syn (raw)")
    ax.plot(t, obs_np, color="C1", lw=1.0, label="obs (raw, +20 ms)")
    ax.plot(t, syn_n,  color="C0", lw=1.4, ls="--", label="syn / ‖syn‖")
    ax.plot(t, obs_n,  color="C1", lw=1.4, ls="--", label="obs / ‖obs‖", alpha=0.8)
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) raw vs. L2-normalised traces (input to NCC / nL2)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # (b) amplitude-invariance scan: loss vs. alpha = scale on obs
    ax = axes[0, 1]
    alpha_grid = np.logspace(-1, 1, 41)
    for label, loss_fn in losses:
        vals = []
        with torch.no_grad():
            for a in alpha_grid:
                vals.append(float(loss_fn(syn, a * obs)))
        ax.plot(alpha_grid, np.array(vals) / (vals[len(alpha_grid)//2] + 1e-30),
                lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"obs amplitude scale $\alpha$")
    ax.set_ylabel("loss / loss(α=1)")
    ax.set_title("(b) amplitude invariance: NCC / nL2 are flat in α; L2 is quadratic")
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (c) adjoint shape for syn vs. 2·syn — shape preserved, magnitude rescaled
    ax = axes[1, 0]
    for label, loss_fn in losses:
        for a, ls in [(1.0, "-"), (2.0, "--")]:
            syn_a = (a * syn).detach().clone().requires_grad_(True)
            obs_a = obs.detach().clone()
            loss_fn.zero_grad() if hasattr(loss_fn, "zero_grad") else None
            v = loss_fn(syn_a, obs_a)
            v.backward()
            g = syn_a.grad.detach().cpu().numpy().ravel()
            # Normalise each gradient to its own peak so we can compare shape
            g_n = g / (np.max(np.abs(g)) + 1e-30)
            ax.plot(t, g_n, ls, lw=1.0, label=f"{label}, α={a}")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("adjoint (peak-normalised)")
    ax.set_title("(c) adjoint shape — α=1 (solid) vs. α=2·syn (dashed)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    # (d) equivalence: J_nL2 == J_NCC + 1 per trace (offset_one=True default)
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 121)
    j_ncc, j_nl2 = [], []
    for tau in tau_grid:
        obs_tau = torch.tensor(
            fractional_shift(syn_np, dt, float(tau)), dtype=torch.float64
        ).view(1, nt, 1, 1)
        with torch.no_grad():
            j_ncc.append(float(GlobalCorrelationLoss(reduction="sum")(syn, obs_tau)))
            j_nl2.append(float(TraceNormalizedL2Loss(reduction="sum")(syn, obs_tau)))
    j_ncc = np.array(j_ncc)
    j_nl2 = np.array(j_nl2)
    ax.plot(tau_grid / half_T, j_ncc, lw=1.6, label="NCC")
    ax.plot(tau_grid / half_T, j_nl2, lw=1.6, ls="--", label="nL2 (= NCC, Choi-Alkhalifah 2012 eq.9)")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_title("(d) NCC and trace-normalised L2 are equivalent (overlay)")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Family B — amplitude-normalised correlation: NCC and trace-normalised L2",
        y=1.002,
    )
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # numerical summary
    summary = {
        "fig": os.path.basename(out),
        "alpha_invariance_max_rel_err": {},
        "equivalence_max_abs_diff": float(np.max(np.abs(j_nl2 - j_ncc))),
    }
    for label, loss_fn in losses:
        with torch.no_grad():
            vals = np.array(
                [float(loss_fn(syn, a * obs)) for a in alpha_grid]
            )
        mid = vals[len(alpha_grid) // 2]
        summary["alpha_invariance_max_rel_err"][label] = float(
            np.max(np.abs(vals - mid) / (mid + 1e-30))
        )
    return summary


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
