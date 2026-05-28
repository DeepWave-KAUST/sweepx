"""Validate Wasserstein1Loss (Métivier 2016; Engquist-Froese 2014).

Recipe:
* Build a Ricker pair (syn, obs).
* Reproduce the preprocessing the loss uses internally: positive_transform →
  normalise → cumsum, for `positive ∈ {'linear', 'abs', 'square'}`.
* Panel (a): densities p_s, p_o for the three positive transforms.
* Panel (b): CDFs F_s, F_o and the area |F_s − F_o| (= W1 by the 1-D formula).
* Panel (c): basin scan vs. time-shift for the three positive transforms.
  Default `positive='linear'` shows a much narrower basin (the
  "Wasserstein degenerates to L2" behaviour our REPORT mentions);
  `'abs'` / `'square'` reach the full ±4 half-λ.
* Panel (d): scaling of W1 with the shift τ should be ≈ |τ| for small τ
  (W1 is metric on shifts of a delta) — overlay the analytic line.
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

from sweep_loss import Wasserstein1Loss  # noqa: E402
from sweep_loss._utils import normalize_density, positive_transform  # noqa: E402


NAME = "14_w1"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.030
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt)

    transforms = [
        ("square", "C0"),
        ("abs",    "C1"),
        ("linear", "C3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) densities for three positive transforms
    ax = axes[0, 0]
    for pos, color in transforms:
        c_param = float(max(np.abs(syn_np).max(), np.abs(obs_np).max())) + 1e-6 if pos == "linear" else None
        fs = positive_transform(syn, method=pos, c=c_param).detach().numpy().ravel()
        ps = fs / (fs.sum() + 1e-30)
        ax.plot(t, ps, color=color, lw=1.4, label=f"p_s ({pos})")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) p_s for three positive transforms — 'linear' is nearly uniform!")
    ax.legend(loc="upper right", fontsize=8)

    # (b) CDFs and |F_s - F_o| for positive='square'
    ax = axes[0, 1]
    pos = "square"
    fs = positive_transform(syn, method=pos).detach().numpy().ravel()
    fo = positive_transform(obs, method=pos).detach().numpy().ravel()
    ps = fs / (fs.sum() + 1e-30)
    po = fo / (fo.sum() + 1e-30)
    Fs = np.cumsum(ps); Fo = np.cumsum(po)
    ax.plot(t, Fs, color="C0", lw=1.4, label="F_s (square)")
    ax.plot(t, Fo, color="C1", lw=1.4, label="F_o (square)", alpha=0.85)
    ax.fill_between(t, Fs, Fo, color="C2", alpha=0.30, label=r"$|F_s - F_o|$  ($W_1$ area)")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(b) 1-D W1 = ∫|F_s − F_o| dt — area between the CDFs")
    ax.legend(loc="lower right", fontsize=8)

    # (c) basin
    ax = axes[1, 0]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    basin_widths = {}
    for pos, color in transforms:
        loss_fn = Wasserstein1Loss(positive=pos, dt=dt, reduction="sum")
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, lw=1.4, label=f"positive='{pos}'")
        # basin half-width: walk out from τ=0 until loss stops increasing
        mid = len(vals) // 2
        right = mid
        while right + 1 < len(vals) and vals[right + 1] > vals[right]:
            right += 1
        basin_widths[pos] = float((right - mid) * (tau_grid[1] - tau_grid[0]) / half_T)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(c) basin: 'square'/'abs' reach ±4; 'linear' degenerates near origin")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) W1 vs |tau| — linear scaling for small shifts
    ax = axes[1, 1]
    tau_lin = np.linspace(-0.05, 0.05, 41)
    w1_vals = {}
    for pos, color in transforms:
        loss_fn = Wasserstein1Loss(positive=pos, dt=dt, reduction="sum")
        vals = scan_shift(loss_fn, syn_np, dt, tau_lin)
        ax.plot(tau_lin * 1e3, vals, color=color, lw=1.2, label=f"positive='{pos}'")
        w1_vals[pos] = vals
    ax.set_xlabel(r"obs shift $\tau$ [ms]")
    ax.set_ylabel(r"$W_1$ [s]")
    ax.set_title("(d) W1 vs τ: linear in |τ| for small shifts (Engquist-Froese 2014)")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Wasserstein1Loss — Métivier et al. (2016); Engquist-Froese (2014) 1-D formula",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "basin_half_width_halfL": basin_widths,
        "w1_at_tau_20ms_square": float(w1_vals["square"][np.argmin(np.abs(tau_lin - 0.020))]),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
