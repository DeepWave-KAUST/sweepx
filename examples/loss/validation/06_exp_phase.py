"""Validate ExponentiatedPhaseLoss (Yuan, Bozdağ et al. 2020, eq. 7).

Recipe:
* Build a Ricker pair (syn, obs).
* Panel (a): the normalised analytic signal tilde_d(t) = a(t) / E(t) lives
  on the **complex unit circle**.  Plot the trajectory (Re tilde, Im tilde)
  for syn and obs in the (Re, Im) plane (this is the figure Yuan 2020 § 2.2
  used to motivate the exponentiated-phase distance).
* Panel (b): per-sample distance |tilde_s − tilde_o|^2.
* Panel (c): adjoint source — smooth & continuous (no phase wrap branch cut).
* Panel (d): basin scan vs. time-shift — wider than instantaneous phase but
  narrower than envelope (Yuan 2020 fig. 2).
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

from sweep_loss import ExponentiatedPhaseLoss  # noqa: E402
from sweep_loss._utils import envelope, hilbert  # noqa: E402


NAME = "06_exp_phase"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.020
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    a_s = hilbert(syn, dim=-3).detach().numpy().squeeze()
    a_o = hilbert(obs, dim=-3).detach().numpy().squeeze()
    E_s = np.sqrt(a_s.real ** 2 + a_s.imag ** 2 + 1e-8)
    E_o = np.sqrt(a_o.real ** 2 + a_o.imag ** 2 + 1e-8)
    tilde_s_R = a_s.real / E_s
    tilde_s_I = a_s.imag / E_s
    tilde_o_R = a_o.real / E_o
    tilde_o_I = a_o.imag / E_o

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) unit-circle trajectories.  Mask out samples where the envelope is too
    # small to be meaningful (otherwise tilde_d at the noise floor scatters
    # everywhere — that's the whole point of EP being noise-sensitive in
    # silent zones).
    mask = (E_s > 0.05 * E_s.max()) & (E_o > 0.05 * E_o.max())
    ax = axes[0, 0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", lw=0.4)
    ax.plot(tilde_s_R[mask], tilde_s_I[mask], color="C0", lw=1.0, label="$\\tilde s$ (syn)")
    ax.plot(tilde_o_R[mask], tilde_o_I[mask], color="C1", lw=1.0,
            label="$\\tilde d$ (obs, +20 ms)", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel(r"$\Re\tilde d$")
    ax.set_ylabel(r"$\Im\tilde d$")
    ax.set_title("(a) normalised analytic signals on the unit circle")
    ax.legend(loc="upper right", fontsize=8)

    # (b) per-sample distance
    dR = tilde_s_R - tilde_o_R
    dI = tilde_s_I - tilde_o_I
    ep_per = 0.5 * (dR * dR + dI * dI)
    ax = axes[0, 1]
    ax.plot(t, ep_per, color="C2", lw=1.0, label="$|\\tilde s - \\tilde d|^2 / 2$")
    ax.plot(t, syn_np / np.abs(syn_np).max() * ep_per.max() * 0.5,
            color="C0", lw=0.6, alpha=0.5, label="syn (scaled)")
    ax.set_xlim(0.15, 0.55)
    ax.set_xlabel("t [s]")
    ax.set_title("(b) per-sample exponentiated-phase misfit")
    ax.legend(loc="upper right", fontsize=8)

    # (c) adjoint
    ax = axes[1, 0]
    g = adjoint(ExponentiatedPhaseLoss(reduction="sum"),
                syn.detach().clone().requires_grad_(True), obs)
    ax.plot(t, syn_np / np.abs(syn_np).max(), color="k", lw=0.6, alpha=0.5,
            label="syn (norm.)")
    ax.plot(t, g / (np.max(np.abs(g)) + 1e-30), color="C3", lw=1.2, label="adjoint")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0.15, 0.55)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("peak-normalised")
    ax.set_title("(c) adjoint source (smooth, no branch cut)")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin scan
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 121)
    vals = scan_shift(ExponentiatedPhaseLoss(reduction="sum"), syn_np, dt, tau_grid)
    v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
    ax.plot(tau_grid / half_T, v, color="C2", lw=1.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin: ExponentiatedPhase smooth, monotone ~1.3 half-λ (Yuan 2020 fig. 2)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("ExponentiatedPhaseLoss — Yuan, Bozdağ, Ciardelli, Gao & Simons (2020)",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # Quick numerical sanity: |tilde| ≈ 1 in high-E samples
    radius_s = np.sqrt(tilde_s_R[mask] ** 2 + tilde_s_I[mask] ** 2)
    radius_o = np.sqrt(tilde_o_R[mask] ** 2 + tilde_o_I[mask] ** 2)
    return {
        "fig": os.path.basename(out),
        "max_dev_from_unit_circle_syn": float(np.max(np.abs(radius_s - 1.0))),
        "max_dev_from_unit_circle_obs": float(np.max(np.abs(radius_o - 1.0))),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
