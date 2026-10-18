"""Validate DeconvolutionLoss (Luo & Sava 2011; Choi-Alkhalifah 2018 variant).

Recipe:
* Build a Ricker pair (syn at t0, obs at t0 + tau).
* Reproduce the deconvolution Ψ(τ) = ifft(O / (S + ε)) the loss computes.
  For a perfect time-shifted pair this is a near-delta at lag = +τ.
* Panel (a): syn vs. obs traces.
* Panel (b): Ψ(τ) on the centred lag axis with the expected delta position.
* Panel (c): τ²·Ψ(τ)² — the integrand whose sum is the Luo-Sava misfit.
* Panel (d): basin scan vs. τ for both `normalize=False` (Luo-Sava, amplitude-
  sensitive) and `normalize=True` (Choi-Alkhalifah 2018, amplitude-invariant).
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

from sweep_loss import DeconvolutionLoss  # noqa: E402


NAME = "10_deconvolution"


def _psi_demo(s: np.ndarray, o: np.ndarray, epsilon: float) -> np.ndarray:
    nt = s.shape[-1]
    n_pad = 2 * nt
    S = np.fft.rfft(s, n=n_pad)
    O = np.fft.rfft(o, n=n_pad)
    Sa = np.abs(S) ** 2
    eps_floor = epsilon * Sa.max()
    Psi_w = np.conj(S) * O / (Sa + eps_floor)
    psi = np.fft.irfft(Psi_w, n=n_pad)
    return np.fft.fftshift(psi)


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.030
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    epsilon = 1e-4
    psi = _psi_demo(syn_np, obs_np, epsilon)
    n_pad = psi.shape[-1]
    lag_s = (np.arange(n_pad) - n_pad // 2) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a)
    ax = axes[0, 0]
    ax.plot(t, syn_np, color="C0", lw=1.0, label="syn")
    ax.plot(t, obs_np, color="C1", lw=1.0, label=f"obs (+{int(tau0*1e3)} ms)")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) syn vs. time-shifted obs")
    ax.legend(loc="upper right", fontsize=8)

    # (b) Ψ(τ)
    ax = axes[0, 1]
    ax.plot(lag_s * 1e3, psi, color="C2", lw=1.4,
            label=r"$\Psi(\tau)$ = ifft($O / (S+\epsilon)$)")
    measured_peak = float(lag_s[int(np.argmax(np.abs(psi)))] * 1e3)
    ax.axvline(tau0 * 1e3, color="k", lw=0.5, ls=":", label=f"expected: +{tau0*1e3:.0f} ms")
    ax.axvline(measured_peak, color="C3", lw=0.6, ls="--",
               label=f"measured: {measured_peak:.1f} ms")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("lag τ [ms]")
    ax.set_title("(b) deconvolution Ψ(τ) — δ-like at +τ_shift")
    ax.legend(loc="upper right", fontsize=8)

    # (c) τ² Ψ²
    ax = axes[1, 0]
    integrand = (lag_s ** 2) * (psi ** 2)
    ax.plot(lag_s * 1e3, integrand, color="C3", lw=1.4, label=r"$\tau^2\,\Psi(\tau)^2$")
    ax.plot(lag_s * 1e3, psi ** 2, color="C2", lw=1.0, alpha=0.6, label=r"$\Psi(\tau)^2$")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("lag τ [ms]")
    ax.set_title("(c) Luo-Sava integrand (the sum is the misfit)")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    luo_sava = scan_shift(DeconvolutionLoss(dt=dt, epsilon=epsilon, reduction="sum"),
                          syn_np, dt, tau_grid)
    choi_alk = scan_shift(DeconvolutionLoss(dt=dt, epsilon=epsilon, normalize=True, reduction="sum"),
                          syn_np, dt, tau_grid)
    ax.plot(tau_grid / half_T, luo_sava / (luo_sava.max() + 1e-30),
            color="C2", lw=1.4, label="DeconvolutionLoss (Luo-Sava)")
    ax.plot(tau_grid / half_T, choi_alk / (choi_alk.max() + 1e-30),
            color="C3", lw=1.4, ls="--", label="DeconvolutionLoss(normalize=True) (Choi-Alkhalifah)")
    ax.plot(tau_grid / half_T, (tau_grid / tau_grid.max()) ** 2,
            color="k", lw=0.6, ls=":", label=r"τ² reference")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin: monotone in τ (anti-cycle-skip), ~τ² convex")
    ax.legend(loc="upper center", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("DeconvolutionLoss — Luo & Sava (2011); normalize=True is Choi-Alkhalifah (2018)",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "expected_psi_peak_ms": tau0 * 1e3,
        "measured_psi_peak_ms": measured_peak,
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
