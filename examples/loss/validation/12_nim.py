"""Validate NIMLoss (Donno-Chauris-Calandra 2013).

Recipe:
* Build a Ricker pair (syn, obs).
* Reproduce the exact preprocessing the loss uses: f = positive_transform(d),
  p = f / sum(f), F = cumsum(p).
* Panel (a): syn, obs, and their positive transforms f_s, f_o.
* Panel (b): CDFs F_s, F_o — the quantities the loss actually compares.
* Panel (c): (F_s − F_o)² integrand.
* Panel (d): basin scan vs. time-shift for positive ∈ {square, abs}. NIM with
  square should be monotone over ±4 half-wavelengths (anti-cycle-skipping).
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

from sweep_loss import NIMLoss  # noqa: E402
from sweep_loss._utils import normalize_density, positive_transform  # noqa: E402


NAME = "12_nim"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.030
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt)

    # Recreate the loss's pre-processing pipeline
    fs = positive_transform(syn, method="square").detach().numpy().ravel()
    fo = positive_transform(obs, method="square").detach().numpy().ravel()
    ps = fs / (fs.sum() + 1e-30)
    po = fo / (fo.sum() + 1e-30)
    Fs = np.cumsum(ps)
    Fo = np.cumsum(po)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) raw + positive transforms
    ax = axes[0, 0]
    ax.plot(t, syn_np / np.abs(syn_np).max(), color="C0", lw=0.7, label="syn (norm.)")
    ax.plot(t, obs_np / np.abs(obs_np).max(), color="C1", lw=0.7, label="obs (norm.)")
    ax2 = ax.twinx()
    ax2.plot(t, ps, color="C0", lw=1.4, ls="--", label="p_s = syn² / Σsyn²")
    ax2.plot(t, po, color="C1", lw=1.4, ls="--", label="p_o = obs² / Σobs²", alpha=0.8)
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) raw traces and squared-then-normalised densities")
    ax.legend(loc="upper right", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)

    # (b) CDFs
    ax = axes[0, 1]
    ax.plot(t, Fs, color="C0", lw=1.4, label="F_s = cumsum(p_s)")
    ax.plot(t, Fo, color="C1", lw=1.4, label="F_o = cumsum(p_o)", alpha=0.85)
    ax.fill_between(t, Fs, Fo, color="C3", alpha=0.25, label="|F_s − F_o|")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(b) CDFs F_s and F_o (compared in L2 by NIM)")
    ax.legend(loc="lower right", fontsize=8)

    # (c) (Fs - Fo)²
    ax = axes[1, 0]
    diff2 = (Fs - Fo) ** 2
    ax.plot(t, diff2, color="C3", lw=1.4, label=r"$(F_s - F_o)^2$")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(c) per-sample CDF-difference squared")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin scan
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    for label, pos, color in [("positive='square'", "square", "C0"),
                              ("positive='abs'", "abs", "C3")]:
        loss_fn = NIMLoss(positive=pos, dt=dt, reduction="sum")
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, lw=1.4, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin: NIM(square/abs) monotone over ±4 half-λ")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("NIMLoss — Donno-Chauris-Calandra (2013)", y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "Fs_at_t0_s": float(Fs[int(t0 / dt)]),
        "Fo_at_t0_s": float(Fo[int(t0 / dt)]),
        "loss_at_tau_pos_square_sum": float(
            NIMLoss(positive="square", dt=dt, reduction="sum")(
                syn.view(1, nt, 1, 1), obs.view(1, nt, 1, 1)
            ).detach()
        ),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
