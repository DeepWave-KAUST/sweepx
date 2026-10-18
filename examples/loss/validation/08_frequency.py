"""Validate the frequency / Laplace-domain misfits (family E).

Covers `FrequencyDomainL2Loss` (Pratt 1998), `FrequencyAmplitudeLoss`
(Shin-Min 2006), `FrequencyPhaseLoss` (Bednar-Shin-Pyun 2007),
`LogarithmicShinMinLoss` (Shin-Min 2006), `LaplaceL2Loss` (Shin-Cha 2008).

Recipe:
* Build a Ricker pair with a time-shift τ.
* Panel (a): rFFT spectrum |D(ω)| of syn and obs; mark the dominant
  Ricker frequency band.
* Panel (b): per-frequency residuals — |Ds-Do|², (|Ds|-|Do|)²,
  wrapped Δφ², |log Ds - log Do|².
* Panel (c): Laplace damping e^{-s·t} applied to syn − obs; show the
  effective residual at three damping rates.
* Panel (d): basin scan vs. time-shift for all five losses, plus an
  inset showing how restricting `freq_band` to low frequencies widens
  the FreqL2 basin (Bunks 1995 multi-scale principle).
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
    FrequencyAmplitudeLoss,
    FrequencyDomainL2Loss,
    FrequencyPhaseLoss,
    LaplaceL2Loss,
    LogarithmicShinMinLoss,
)


NAME = "08_frequency"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.020
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    # rFFT of the canonical-layout traces (what the loss internally uses)
    Ds = torch.fft.rfft(syn, dim=-3).detach().numpy().squeeze()
    Do = torch.fft.rfft(obs, dim=-3).detach().numpy().squeeze()
    freqs = np.fft.rfftfreq(nt, d=dt)

    As_ = np.abs(Ds);  Ao_ = np.abs(Do)
    Phi_s = np.angle(Ds); Phi_o = np.angle(Do)
    dphi = np.arctan2(np.sin(Phi_s - Phi_o), np.cos(Phi_s - Phi_o))
    eps = 1e-8
    log_amp = np.log(As_ + eps) - np.log(Ao_ + eps)
    L2_per_freq = 0.5 * np.abs(Ds - Do) ** 2
    A_per_freq  = 0.5 * (As_ - Ao_) ** 2
    Ph_per_freq = 0.5 * dphi ** 2
    Log_per_freq = 0.5 * (log_amp ** 2 + dphi ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) amplitude spectra
    ax = axes[0, 0]
    ax.plot(freqs, As_, color="C0", lw=1.2, label="|D_s(f)| (syn)")
    ax.plot(freqs, Ao_, color="C1", lw=1.2, ls="--", label="|D_o(f)| (obs, +20 ms)")
    ax.axvline(fc, color="k", lw=0.4, ls=":", label=f"Ricker f_c = {fc} Hz")
    ax.set_xlim(0, 60)
    ax.set_xlabel("freq [Hz]")
    ax.set_title("(a) rFFT amplitude spectra (the syn/obs input to FreqLosses)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) per-frequency residuals
    ax = axes[0, 1]
    ax.plot(freqs, L2_per_freq, color="C0", lw=1.2, label=r"$\frac{1}{2}|D_s-D_o|^2$ (FreqL2)")
    ax.plot(freqs, A_per_freq,  color="C1", lw=1.2, label=r"$\frac{1}{2}(|D_s|-|D_o|)^2$ (FreqAmp)")
    ax.plot(freqs, Ph_per_freq, color="C2", lw=1.2, label=r"$\frac{1}{2}\Delta\phi^2$ (FreqPhase)")
    ax.plot(freqs, Log_per_freq, color="C3", lw=1.2, label=r"$\frac{1}{2}|\log D_s-\log D_o|^2$ (ShinMin)")
    ax.set_xlim(0, 60)
    ax.set_xlabel("freq [Hz]")
    ax.set_title("(b) per-frequency residuals (note phase carries most of the τ info)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Laplace damping
    ax = axes[1, 0]
    r_t = syn_np - obs_np
    for s_rate, color in zip([1.0, 5.0, 20.0], ["C0", "C2", "C3"]):
        damp = np.exp(-s_rate * t)
        ax.plot(t, damp * r_t, color=color, lw=1.0,
                label=f"e^{{-{s_rate} t}} · (s−o)")
    ax.plot(t, r_t, color="k", lw=0.6, alpha=0.4, label="raw residual (s = 0)")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("t [s]")
    ax.set_title("(c) Laplace damping (Shin-Cha 2008): emphasises early arrivals")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) basin scan: time-shift vs. five losses, plus low-freq band variant
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    nf = nt // 2 + 1
    low_band_k1 = int(np.argmin(np.abs(freqs - 0.6 * fc)))  # cutoff ~ 0.6 fc

    cfg = [
        ("FreqDomain L2 (full band)", FrequencyDomainL2Loss(reduction="sum"), "C0"),
        ("FreqDomain L2 (low band, k1=0.6 fc)",
         FrequencyDomainL2Loss(freq_band=(0, low_band_k1), reduction="sum"), "C0", "--"),
        ("FreqAmplitude",       FrequencyAmplitudeLoss(reduction="sum"), "C1"),
        ("FreqPhase",           FrequencyPhaseLoss(reduction="sum"), "C2"),
        ("ShinMin log",         LogarithmicShinMinLoss(reduction="sum"), "C3"),
        ("LaplaceL2 (s=5, dt=1e-3)", LaplaceL2Loss(s=5.0, dt=dt, reduction="sum"), "C4"),
    ]
    for entry in cfg:
        label, loss_fn, color = entry[0], entry[1], entry[2]
        ls = entry[3] if len(entry) > 3 else "-"
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, ls=ls, lw=1.2, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin scan; restricting to low band widens FreqL2 (Bunks 1995)")
    ax.legend(loc="upper center", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Family E — Frequency / Laplace-domain misfits (Pratt, Shin-Min, Bednar, Shin-Cha)",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "max_log_amp_residual": float(np.max(np.abs(log_amp))),
        "max_phase_residual_rad": float(np.max(np.abs(dphi))),
        "freq_l2_basin_full":  None,
        "low_band_cutoff_index": int(low_band_k1),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
