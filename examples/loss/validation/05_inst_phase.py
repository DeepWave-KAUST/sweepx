"""Validate InstantaneousPhaseLoss and EnvelopePhaseLoss (Bozdağ 2011, Yuan 2016).

Recipe:
* Build a Ricker pair (syn, obs) with a small time shift τ.
* Panel (a): instantaneous phase φ(t) = atan2(H[d], d) for syn and obs.
  Demonstrate that φ jumps at zero-crossings (this is what motivates
  the wrapped-difference formula in the source).
* Panel (b): wrapped phase difference  Δφ = atan2(sin(φ_s-φ_o),
  cos(φ_s-φ_o)).  Overlay the envelope weight w(t) = E_o / max E_o that
  the source code multiplies it by (`envelope_weight=True` default).
* Panel (c): adjoint sources for `InstantaneousPhaseLoss` and
  `EnvelopePhaseLoss` (α = 0.5).
* Panel (d): basin scan — inst-phase basin is narrow (it is just a
  phase residual!), envelope+phase blend recovers some basin width from
  the envelope term.
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
    scan_shift,
    setup_matplotlib,
    write_summary,
)

from sweep_loss import EnvelopePhaseLoss, InstantaneousPhaseLoss  # noqa: E402
from sweep_loss._utils import envelope, hilbert, instantaneous_phase  # noqa: E402


NAME = "05_inst_phase"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.015
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    phi_s = instantaneous_phase(syn, dim=-3).detach().numpy().ravel()
    phi_o = instantaneous_phase(obs, dim=-3).detach().numpy().ravel()
    E_o = envelope(obs, dim=-3).detach().numpy().ravel()
    w = E_o / (E_o.max() + 1e-30)
    dphi = np.arctan2(np.sin(phi_s - phi_o), np.cos(phi_s - phi_o))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) phase of syn / obs
    ax = axes[0, 0]
    ax.plot(t, syn_np / np.abs(syn_np).max(), color="C0", lw=0.7, label="syn (normalised)")
    ax2 = ax.twinx()
    ax2.plot(t, phi_s, color="C0", lw=1.0, ls="--", label="φ[syn]")
    ax2.plot(t, phi_o, color="C1", lw=1.0, ls="--", label="φ[obs]")
    ax2.set_ylabel("φ [rad]")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) instantaneous phase φ = atan2(H[d], d)")
    ax2.legend(loc="upper right", fontsize=8)

    # (b) wrapped phase residual + envelope weight
    ax = axes[0, 1]
    ax.plot(t, dphi, color="C3", lw=1.2, label="Δφ = wrap(φ_s − φ_o)")
    ax.plot(t, w * dphi, color="C2", lw=1.4, label="w · Δφ (what enters the loss)")
    ax.plot(t, w, color="k", lw=0.7, alpha=0.6, label="w = E_o / max E_o")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(b) wrapped phase residual & envelope weight")
    ax.legend(loc="upper right", fontsize=8)

    # (c) adjoint sources
    ax = axes[1, 0]
    g_phi = adjoint(InstantaneousPhaseLoss(reduction="sum"),
                    syn.detach().clone().requires_grad_(True), obs)
    g_ep = adjoint(EnvelopePhaseLoss(alpha=0.5, reduction="sum"),
                   syn.detach().clone().requires_grad_(True), obs)
    ax.plot(t, syn_np / np.abs(syn_np).max(), color="k", lw=0.7, alpha=0.5,
            label="syn (norm.)")
    ax.plot(t, g_phi / (np.max(np.abs(g_phi)) + 1e-30),
            color="C3", lw=1.2, label="InstantaneousPhase adjoint")
    ax.plot(t, g_ep / (np.max(np.abs(g_ep)) + 1e-30),
            color="C2", lw=1.2, label="EnvelopePhase(α=0.5) adjoint")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("adjoint (peak-normalised)")
    ax.set_title("(c) adjoint sources (peak-normalised)")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin scan vs time-shift
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 121)

    for label, loss_fn, color in [
        ("InstantaneousPhase",       InstantaneousPhaseLoss(reduction="sum"), "C3"),
        ("EnvelopePhase (α=0.5)",    EnvelopePhaseLoss(alpha=0.5, reduction="sum"), "C2"),
        ("EnvelopePhase (α=0.1, env-heavy)",
         EnvelopePhaseLoss(alpha=0.1, reduction="sum"), "C0"),
        ("EnvelopePhase (α=0.9, phase-heavy)",
         EnvelopePhaseLoss(alpha=0.9, reduction="sum"), "C1"),
    ]:
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        # Normalise to [0, 1] for visual comparison
        v = vals - vals.min()
        v /= v.max() + 1e-30
        ax.plot(tau_grid / half_T, v, color=color, lw=1.2, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin scan: phase-only is narrow, envelope-blend widens it")
    ax.legend(loc="upper center", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("InstantaneousPhase + EnvelopePhase (Bozdağ 2011 eq.22, Yuan-Simons-Tromp 2016)",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "max_abs_dphi_rad": float(np.max(np.abs(dphi))),
        "max_weighted_dphi": float(np.max(np.abs(w * dphi))),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
