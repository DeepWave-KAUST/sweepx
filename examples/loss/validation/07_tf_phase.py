"""Validate TimeFrequencyPhaseLoss (Fichtner 2008, Kristeková 2009).

Recipe:
* Build a Ricker pair with a time shift.
* Run the same Gabor STFT the loss uses internally (sweep_loss.tf_phase._gabor_stft).
* Panel (a): STFT magnitude |G_o| (time-frequency picture of obs).
* Panel (b): wrapped time-frequency phase residual w(t,ω) · Δφ.
* Panel (c): adjoint source.
* Panel (d): basin scan — expected wide (~3+ half-λ) due to the envelope term.
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

from sweep_loss import TimeFrequencyPhaseLoss  # noqa: E402
from sweep_loss.tf_phase import _gabor_stft  # noqa: E402


NAME = "07_tf_phase"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.020
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    # STFT params used by the loss (defaults)
    n_fft = 128
    hop_length = n_fft // 4
    sigma_samples = n_fft / 6.0
    Gs = _gabor_stft(syn.view(1, nt), n_fft, hop_length, sigma_samples).detach().numpy()[0]
    Go = _gabor_stft(obs.view(1, nt), n_fft, hop_length, sigma_samples).detach().numpy()[0]

    As = np.sqrt(Gs.real ** 2 + Gs.imag ** 2)
    Ao = np.sqrt(Go.real ** 2 + Go.imag ** 2)
    phi_s = np.angle(Gs)
    phi_o = np.angle(Go)
    dphi = np.arctan2(np.sin(phi_s - phi_o), np.cos(phi_s - phi_o))
    w = Ao / (Ao.max() + 1e-30)

    # Time/freq axes for imshow
    n_frames = Gs.shape[-1]
    t_frames = np.arange(n_frames) * hop_length * dt
    freq_axis = np.fft.rfftfreq(n_fft, d=dt)
    extent = [t_frames[0], t_frames[-1], freq_axis[0], freq_axis[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) |G_o|
    ax = axes[0, 0]
    im = ax.imshow(Ao, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("freq [Hz]")
    ax.set_ylim(0, 4 * fc)
    ax.set_title("(a) |G_o(t,ω)| — Gabor STFT magnitude of obs")
    plt.colorbar(im, ax=ax)

    # (b) w · Δφ
    ax = axes[0, 1]
    im = ax.imshow(w * dphi, origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("freq [Hz]")
    ax.set_ylim(0, 4 * fc)
    ax.set_title("(b) w(t,ω) · Δφ — energy-weighted wrapped TF phase residual")
    plt.colorbar(im, ax=ax)

    # (c) adjoint
    ax = axes[1, 0]
    g = adjoint(TimeFrequencyPhaseLoss(reduction="sum"),
                syn.detach().clone().requires_grad_(True), obs)
    ax.plot(t, syn_np / np.abs(syn_np).max(), color="k", lw=0.6, alpha=0.5,
            label="syn (norm.)")
    ax.plot(t, g / (np.max(np.abs(g)) + 1e-30), color="C3", lw=1.2, label="adjoint")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("peak-normalised")
    ax.set_title("(c) adjoint source (autograd through STFT)")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin scan
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    for label, loss_fn, color in [
        ("alpha=0.0 (env only)", TimeFrequencyPhaseLoss(alpha=0.0, reduction="sum"), "C0"),
        ("alpha=0.5",            TimeFrequencyPhaseLoss(alpha=0.5, reduction="sum"), "C2"),
        ("alpha=1.0 (phase only)", TimeFrequencyPhaseLoss(alpha=1.0, reduction="sum"), "C3"),
    ]:
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, lw=1.4, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin scan: alpha trades off envelope (wide) ↔ phase (narrow)")
    ax.legend(loc="upper center", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("TimeFrequencyPhaseLoss — Fichtner 2008, Kristeková 2006/2009",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "n_fft": n_fft,
        "n_frames": int(n_frames),
        "max_abs_weighted_dphi": float(np.max(np.abs(w * dphi))),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
