"""Validate LocalSimilarityLoss (Fomel 2007; Zhang-Sirgue-Zhang 2018).

Recipe:
* Build a 'partially-matching' pair: syn = ricker(t0); obs = same Ricker
  in the first half of the trace, but a time-shifted Ricker in the
  second half — so the local similarity attribute should be ≈ 1 on the
  first wavelet and < 1 on the second.
* Reproduce the windowed correlation γ_σ(τ) using the same Gaussian
  kernel the loss builds.
* Panel (a): syn vs. obs (showing the two-region setup).
* Panel (b): the local similarity γ_σ(τ) curve — confirm γ ≈ 1 on the
  matched region, γ << 1 on the shifted region.
* Panel (c): the per-sample misfit ½ w (1−γ)² (the integrand).
* Panel (d): basin scan vs. time-shift for two window sizes σ.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    fig_path,
    fractional_shift,
    ricker,
    scan_shift,
    setup_matplotlib,
    write_summary,
)

from sweep_loss import LocalSimilarityLoss  # noqa: E402
from sweep_loss.local_similarity import _gaussian_kernel, _smooth  # noqa: E402


NAME = "19_local_similarity"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc = 1024, 1e-3, 12.0
    t = np.arange(nt) * dt
    # syn: two Rickers at t1=0.20 s and t2=0.60 s
    syn_np = ricker(nt, dt, fc, 0.20) + ricker(nt, dt, fc, 0.60)
    # obs: same first Ricker, shifted second one
    obs_np = ricker(nt, dt, fc, 0.20) + ricker(nt, dt, fc, 0.60 + 0.020)

    # Reproduce gamma_sigma(tau) using the same helpers as the loss
    sigma_samples = 8.0
    s = torch.tensor(syn_np, dtype=torch.float64).view(1, nt)
    o = torch.tensor(obs_np, dtype=torch.float64).view(1, nt)
    k = _gaussian_kernel(sigma_samples, s.device, s.dtype)
    sx = _smooth(s * o, k); ss = _smooth(s * s, k); oo = _smooth(o * o, k)
    gamma = (sx / torch.sqrt((ss * oo).clamp_min(1e-12))).detach().numpy().ravel()
    energy = (ss * oo).detach().numpy().ravel()
    w = energy / (energy.max() + 1e-12)
    per_sample = 0.5 * w * (1.0 - gamma) ** 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) syn vs obs
    ax = axes[0, 0]
    ax.plot(t, syn_np, color="C0", lw=1.0, label="syn (2 Rickers at 0.20 + 0.60 s)")
    ax.plot(t, obs_np, color="C1", lw=1.0,
            label="obs (1st same; 2nd shifted +20 ms)")
    ax.axvspan(0.10, 0.32, color="C2", alpha=0.10, label="matched region")
    ax.axvspan(0.50, 0.72, color="C3", alpha=0.10, label="mismatched region")
    ax.set_xlim(0.0, 0.8)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) syn vs. partially-mismatched obs")
    ax.legend(loc="upper right", fontsize=7)

    # (b) gamma_sigma(tau)
    ax = axes[0, 1]
    ax.plot(t, gamma, color="C2", lw=1.4, label=r"$\gamma_\sigma(\tau)$ (σ=8 samples)")
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-0.2, 1.1)
    ax.set_xlabel("t [s]")
    ax.set_title("(b) windowed correlation — γ ≈ 1 in matched region, drops in mismatched")
    ax.legend(loc="lower left", fontsize=8)

    # (c) per-sample integrand
    ax = axes[1, 0]
    ax.plot(t, per_sample, color="C3", lw=1.4, label=r"½ w (1−γ)²")
    ax.plot(t, w, color="k", lw=0.6, alpha=0.5, label="w (energy weight)")
    ax.set_xlim(0.0, 0.8)
    ax.set_xlabel("t [s]")
    ax.set_title("(c) per-sample misfit integrand")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin scan vs. global time shift
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    # Use a clean Ricker pair for the basin scan
    pure_syn = ricker(nt, dt, fc, 0.30)
    for label, sig, color in [("σ = 8 samples (local)",   8.0, "C0"),
                              ("σ = 64 samples (near-global)", 64.0, "C3")]:
        loss_fn = LocalSimilarityLoss(sigma_samples=sig, reduction="mean")
        vals = scan_shift(loss_fn, pure_syn, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, lw=1.4, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin: depends on σ; LocalSimilarity is amplitude-invariant")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("LocalSimilarityLoss — Fomel (2007); Zhang-Sirgue-Zhang (2018)",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "gamma_max_matched_region":   float(np.max(gamma[(t >= 0.15) & (t <= 0.30)])),
        "gamma_min_mismatched_region": float(np.min(gamma[(t >= 0.55) & (t <= 0.72)])),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
