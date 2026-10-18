"""Validate Lp / robust M-estimators (family A).

Covers `L2Loss`, `L1Loss`, `HuberLoss`, `PseudoHuberLoss` (= `HybridL1L2Loss`),
`CauchyLoss`, `TukeyLoss`, `GemanMcClureLoss`, `StudentTLoss`.

Recipe:

* Panel (a): ρ(r), the per-sample misfit function as r ∈ [−4, 4] — the
  canonical robust-stats diagnostic figure (Huber 1964; Black & Anandan 1996).
* Panel (b): ψ(r) = ∂ρ/∂r, the *influence function*. The cleanest way to
  see which residuals dominate the gradient: L1 = sign, L2 = identity,
  Tukey clips to zero past c, GemanMcClure & Cauchy redescend, etc.
* Panel (c): a Ricker syn/obs pair with an outlier spike planted in obs.
  Plot the per-sample misfit ρ(r(t)) for each loss — shows visually which
  losses are dominated by the spike (L2) vs. which absorb it (Tukey).
* Panel (d): loss-vs-time-shift basin for each loss. All Lp-family losses
  are expected to have a single basin ~1 half-wavelength wide (no
  anti-cycle-skipping); we report the basin width.
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

from sweep_loss import (  # noqa: E402
    CauchyLoss,
    GemanMcClureLoss,
    HuberLoss,
    HybridL1L2Loss,
    L1Loss,
    L2Loss,
    PseudoHuberLoss,
    StudentTLoss,
    TukeyLoss,
)


NAME = "01_lp_robust"


def _per_sample_rho(LossCls, r: np.ndarray, **kw) -> np.ndarray:
    """Evaluate ρ(r) sample-wise by passing (r, 0) into the loss with reduction=none."""
    syn = torch.tensor(r, dtype=torch.float64)
    obs = torch.zeros_like(syn)
    return LossCls(reduction="none", **kw)(syn, obs).detach().cpu().numpy().ravel()


def _influence_psi(LossCls, r: np.ndarray, **kw) -> np.ndarray:
    """Evaluate ψ(r) = dρ/dr via autograd."""
    syn = torch.tensor(r, dtype=torch.float64).requires_grad_(True)
    obs = torch.zeros_like(syn)
    LossCls(reduction="sum", **kw)(syn, obs).backward()
    return syn.grad.detach().cpu().numpy().ravel()


def run() -> dict:
    plt = setup_matplotlib()

    # Loss zoo + paper-typical knobs.
    losses = [
        ("L2",            L2Loss,           {}),
        ("L1",            L1Loss,           {}),
        ("Huber δ=1",     HuberLoss,        {"delta": 1.0}),
        ("PseudoHuber δ=1", PseudoHuberLoss,{"delta": 1.0}),
        ("HybridL1L2 δ=1",HybridL1L2Loss,   {"delta": 1.0}),
        ("Cauchy c=1",    CauchyLoss,       {"c": 1.0}),
        ("Tukey c=2",     TukeyLoss,        {"c": 2.0}),
        ("Geman-McClure c=1", GemanMcClureLoss, {"c": 1.0}),
        ("Student-t ν=1,σ=1", StudentTLoss, {"nu": 1.0, "sigma": 1.0}),
    ]

    r_grid = np.linspace(-4.0, 4.0, 801)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) ρ(r)
    ax = axes[0, 0]
    for label, cls, kw in losses:
        rho = _per_sample_rho(cls, r_grid, **kw)
        ax.plot(r_grid, rho, lw=1.2, label=label)
    ax.set_title(r"(a) $\rho(r)$ — per-sample misfit")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\rho(r)$")
    ax.set_ylim(-0.1, 5.0)
    ax.axvline(0, color="k", lw=0.4)
    ax.legend(loc="upper center", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # (b) ψ(r) = dρ/dr
    ax = axes[0, 1]
    for label, cls, kw in losses:
        psi = _influence_psi(cls, r_grid, **kw)
        ax.plot(r_grid, psi, lw=1.2, label=label)
    ax.set_title(r"(b) $\psi(r)=\partial\rho/\partial r$ — influence function")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\psi(r)$")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.grid(True, alpha=0.3)

    # (c) per-sample misfit on (shifted syn) vs. (shifted syn + spike). The
    # residual r(t) now has contributions both inside the wavelet (from the
    # 5-ms shift) AND at a single outlier sample, so we can compare which
    # losses are dominated by the outlier vs. the body.
    nt, dt, fc, t0 = 1024, 1e-3, 12.0, 0.30
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, 0.005)  # 5 ms shift → body residual
    spike_idx = int(0.55 / dt)                    # spike well outside wavelet
    obs_np = obs_np.copy()
    obs_np[spike_idx] += 1.5 * syn_np.max()       # moderate outlier

    syn_t = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs_t = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    ax = axes[1, 0]
    ax.plot(t, syn_np, color="C0", lw=0.7, label="syn (clean)")
    ax.plot(t, obs_np, color="C1", lw=0.7, label="obs (+5 ms + spike)")
    ax.set_xlim(0.25, 0.65)
    ax.set_xlabel("t [s]")
    ax.set_title("(c) traces — body residual (5 ms shift) + spike outlier at t≈0.55 s")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) basin of attraction: loss vs. time-shift, in half-wavelengths
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4.0 * half_T, 4.0 * half_T, 121)

    basin_widths = {}
    spike_dominance = {}
    for label, cls, kw in losses:
        loss_fn = cls(reduction="mean", **kw)
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, vals_norm, lw=1.0, label=label)

        # basin half-width: walk out from τ=0 until loss stops increasing
        mid = len(vals) // 2
        right = mid
        while right + 1 < len(vals) and vals[right + 1] > vals[right]:
            right += 1
        basin_widths[label] = float((right - mid) * (tau_grid[1] - tau_grid[0]) / half_T)

        # Spike-dominance: fraction of mean misfit contributed by the spike sample
        per = cls(reduction="none", **kw)(syn_t, obs_t).detach().cpu().numpy().ravel()
        spike_dominance[label] = float(per[spike_idx] / (per.sum() + 1e-30))

    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title(r"(d) basin of attraction (loss-vs-$\tau$, normalised)")
    ax.legend(loc="upper center", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Family A — Lp / robust M-estimators: ρ, ψ, outlier-resistance, and cycle-skip basin",
        y=1.002,
    )
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "basin_half_width_halfL": basin_widths,
        "spike_dominance_fraction": spike_dominance,
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
