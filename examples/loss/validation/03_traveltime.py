"""Validate CrossCorrelationTraveltimeLoss (Luo-Schuster 1991).

Recipe:
* Panel (a): Ricker pair (syn, obs); obs is syn shifted by tau seconds.
* Panel (b): the cross-correlation c(τ) = ⟨syn(t), obs(t+τ)⟩ — peaks at
  +tau (Luo-Schuster 1991, eq. 4). Annotate the centroid that the smooth
  surrogate uses (van Leeuwen-Mulder 2010, our `power=2` weighting).
* Panel (c): loss vs. tau swept — should be a τ² parabola; record the
  measured curvature and the symmetry.
* Panel (d): adjoint source. For traveltime FWI the adjoint is
  proportional to -∂(syn)/∂t weighted by the lag (Marquering-Dahlen-Nolet
  1999); we plot the autograd output and overlay an analytic ∂syn/∂t for
  reference.

Related: 85th EAGE 2024 (Oslo), "Differentiable Traveltime Misfit for
Wave-Equation Tomography" replaces our `power`-weighted centroid with a
softmax-weighted one (`prob = softmax(cc)`, `τ* = Σ prob_i · i`).  Same
family of smooth argmax surrogates; the two limits coincide as
`power → ∞`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    adjoint,
    fig_path,
    fractional_shift,
    ricker,
    setup_matplotlib,
    write_summary,
)

from sweep_loss import CrossCorrelationTraveltimeLoss  # noqa: E402


NAME = "03_traveltime"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 2048, 1e-3, 12.0, 0.50, 0.020
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) traces
    ax = axes[0, 0]
    ax.plot(t, syn_np, color="C0", lw=1.0, label="syn")
    ax.plot(t, obs_np, color="C1", lw=1.0, label=f"obs (+{int(tau0*1e3)} ms)")
    ax.set_xlim(0.35, 0.70)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) syn vs. time-shifted obs")
    ax.legend(loc="upper right", fontsize=8)

    # (b) Cross-correlation c(tau); reproduce the FFT-based cc used inside
    # the loss for visualisation.
    from sweep_loss.traveltime import _cross_correlation

    syn_f = syn.view(1, nt)
    obs_f = obs.view(1, nt)
    c = _cross_correlation(syn_f, obs_f).detach().numpy().ravel()
    tau_samples = np.arange(-(nt - 1), nt)
    tau_seconds = tau_samples * dt

    ax = axes[0, 1]
    ax.plot(tau_seconds * 1e3, c, color="C2", lw=1.0)
    # Theoretical maximum at tau = -tau0 (obs(t+τ) = syn(t) when τ = -tau0)
    peak_idx = int(np.argmax(c))
    measured_peak_ms = tau_seconds[peak_idx] * 1e3
    # With obs(t) = syn(t - tau0), c(τ) = ⟨syn(t), syn(t+τ-tau0)⟩ peaks at +tau0.
    ax.axvline(tau0 * 1e3, color="k", lw=0.4, ls=":", label=f"theory: τ* = +{tau0*1e3:.0f} ms")
    ax.axvline(measured_peak_ms, color="C3", lw=0.6, ls="--",
               label=f"measured argmax: {measured_peak_ms:.1f} ms")
    ax.set_xlim(-80, 80)
    ax.set_xlabel(r"$\tau$ [ms]")
    ax.set_title(r"(b) cross-correlation $c(\tau) = \langle$ syn$(t)$, obs$(t+\tau)\rangle$")
    ax.legend(loc="upper right", fontsize=8)

    # (c) Loss vs. shift — should look like τ²
    ax = axes[1, 0]
    loss_fn = CrossCorrelationTraveltimeLoss(dt=dt, power=2.0, reduction="sum")
    tau_grid = np.linspace(-0.080, 0.080, 81)
    vals = []
    with torch.no_grad():
        for tau in tau_grid:
            obs_tau = torch.tensor(
                fractional_shift(syn_np, dt, float(tau)), dtype=torch.float64
            ).view(1, nt, 1, 1)
            vals.append(float(loss_fn(syn, obs_tau)))
    vals = np.array(vals)
    ax.plot(tau_grid * 1e3, vals, color="C0", lw=1.4, label="CrossCorrelationTraveltimeLoss")
    ax.plot(tau_grid * 1e3, 0.5 * tau_grid ** 2, color="k", lw=1.0, ls=":",
            label=r"theory: $\frac{1}{2}\tau^2$")
    ax.set_xlabel(r"obs shift $\tau$ [ms]")
    ax.set_ylabel("loss [s²]")
    ax.set_title(r"(c) loss vs. $\tau$ — parabolic $\frac{1}{2}\tau^2$ (Luo-Schuster 1991)")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) adjoint source
    ax = axes[1, 1]
    syn_g = syn.detach().clone().requires_grad_(True)
    g = adjoint(CrossCorrelationTraveltimeLoss(dt=dt, power=2.0, reduction="sum"),
                syn_g, obs)
    # analytical reference: ∂(syn)/∂t (the marquering-style banana kernel proxy)
    dsyn_dt = np.gradient(syn_np, dt)
    # normalise each to its own peak for shape comparison
    g_n = g / (np.max(np.abs(g)) + 1e-30)
    d_n = dsyn_dt / (np.max(np.abs(dsyn_dt)) + 1e-30)
    ax.plot(t, g_n, color="C3", lw=1.4, label="autograd adjoint")
    ax.plot(t, d_n, color="k", lw=0.8, ls=":", label=r"$\partial$syn$/\partial t$ (Luo-Schuster eq.7)")
    ax.set_xlim(0.35, 0.65)
    ax.set_xlabel("t [s]")
    ax.set_title("(d) adjoint source: τ-weighted derivative-like kernel via centroid surrogate")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "CrossCorrelationTraveltimeLoss — Luo-Schuster 1991, smooth surrogate van Leeuwen-Mulder 2010",
        y=1.002,
    )
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "cc_peak_at_ms_theory": tau0 * 1e3,
        "cc_peak_at_ms_measured": float(measured_peak_ms),
        "loss_at_tau_eq_tau0_s2": float(vals[np.argmin(np.abs(tau_grid - tau0))]),
        "loss_theory_at_tau_s2": float(0.5 * tau0 ** 2),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
