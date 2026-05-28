"""Validate OTMFLoss (Sun & Alkhalifah 2018, 2019).

Recipe:
* Build a Ricker pair with a time-shift τ in obs.
* Reproduce the same Wiener filter the OTMF loss computes, then apply the
  positive transform (squared / absolute value) and renormalise to a density.
* Panel (a): syn vs. obs.
* Panel (b): Wiener filter w(τ); positive-transformed densities ŵ_sq(τ),
  ŵ_abs(τ).  Density mass concentrates around the kinematic shift.
* Panel (c): order-1 (|τ|·ŵ) and order-2 (τ²·ŵ) integrands.
* Panel (d): basin scan vs. shift for both orders, and for `positive='square'`
  vs `'abs'` — confirm both are convex in τ around the origin.
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

from sweep_loss import OTMFLoss  # noqa: E402


NAME = "11_otmf"


def _wiener_filter(s: np.ndarray, o: np.ndarray, epsilon: float) -> np.ndarray:
    nt = s.shape[-1]
    n_pad = 2 * nt
    S = np.fft.rfft(s, n=n_pad)
    O = np.fft.rfft(o, n=n_pad)
    Sa = np.abs(S) ** 2
    eps_floor = epsilon * Sa.max()
    W = np.conj(S) * O / (Sa + eps_floor)
    w = np.fft.irfft(W, n=n_pad)
    return np.fft.fftshift(w)


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.030
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt, 1, 1)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt, 1, 1)

    epsilon = 1e-4
    w = _wiener_filter(syn_np, obs_np, epsilon)
    n_pad = w.shape[-1]
    lag_s = (np.arange(n_pad) - n_pad // 2) * dt
    w_sq = w ** 2;        w_sq /= w_sq.sum() + 1e-30
    w_abs = np.abs(w);    w_abs /= w_abs.sum() + 1e-30

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a)
    ax = axes[0, 0]
    ax.plot(t, syn_np, color="C0", lw=1.0, label="syn")
    ax.plot(t, obs_np, color="C1", lw=1.0, label=f"obs (+{int(tau0*1e3)} ms)")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) syn vs. time-shifted obs")
    ax.legend(loc="upper right", fontsize=8)

    # (b) w(τ), densities
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(lag_s * 1e3, w, color="C2", lw=1.0, label="Wiener w(τ)")
    ax2.plot(lag_s * 1e3, w_sq, color="C3", lw=1.4, label="ŵ_sq (density, square)")
    ax2.plot(lag_s * 1e3, w_abs, color="C1", lw=1.4, ls="--",
             label="ŵ_abs (density, abs)")
    ax.set_xlim(-150, 150)
    ax.axvline(tau0 * 1e3, color="k", lw=0.5, ls=":", label=f"+τ_shift = {tau0*1e3:.0f} ms")
    ax.set_xlabel("lag τ [ms]")
    ax.set_title("(b) Wiener filter, then positive-transform → density")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # (c) moment integrands
    ax = axes[1, 0]
    ax.plot(lag_s * 1e3, lag_s ** 2 * w_sq, color="C0", lw=1.4,
            label=r"$\tau^2 \hat w_{sq}(\tau)$ (W2² integrand, order=2)")
    ax.plot(lag_s * 1e3, np.abs(lag_s) * w_sq, color="C3", lw=1.4, ls="--",
            label=r"$|\tau| \hat w_{sq}(\tau)$ (W1 integrand, order=1)")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("lag τ [ms]")
    ax.set_title("(c) Wasserstein moment integrands w.r.t. δ_0")
    ax.legend(loc="upper right", fontsize=7)

    # (d) basin
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    for label, loss_fn, color in [
        ("order=2, positive='square'",
         OTMFLoss(dt=dt, epsilon=epsilon, positive="square", order=2, reduction="sum"), "C0"),
        ("order=1, positive='square'",
         OTMFLoss(dt=dt, epsilon=epsilon, positive="square", order=1, reduction="sum"), "C3"),
        ("order=2, positive='abs'",
         OTMFLoss(dt=dt, epsilon=epsilon, positive="abs", order=2, reduction="sum"), "C1"),
    ]:
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, lw=1.4, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin: monotone in τ, convex around 0 (Sun-Alkhalifah 2019 fig.4)")
    ax.legend(loc="upper center", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("OTMFLoss — Sun & Alkhalifah (2018, 2019)", y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    centroid_sq = float((lag_s * w_sq).sum())
    centroid_abs = float((lag_s * w_abs).sum())
    return {
        "fig": os.path.basename(out),
        "centroid_density_sq_ms": centroid_sq * 1e3,
        "centroid_density_abs_ms": centroid_abs * 1e3,
        "expected_centroid_ms":   tau0 * 1e3,
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
