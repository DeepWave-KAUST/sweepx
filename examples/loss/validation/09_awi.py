"""Validate AWILoss (Warner & Guasch 2014, 2016).

Recipe:
* Build a Ricker pair with a time-shift τ in obs.
* Reproduce the same Wiener filter the loss computes internally (W = conj(S)·O/(|S|²+ε)).
* Panel (a): syn and obs traces.
* Panel (b): the Wiener filter w(τ) — for a perfect time-shifted pair it must
  collapse to a delta at lag = +τ; we mark that delta location.
* Panel (c): the AWI numerator integrand (T(τ)·w(τ))² and the denominator
  w(τ)² — visually shows why "spread of w" defines the loss.
* Panel (d): basin scan vs. time-shift — known to be ~τ² for AWI
  (Warner-Guasch 2016 fig. 4).  Compare with L2 for context.
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

from sweep_loss import AWILoss, L2Loss  # noqa: E402


NAME = "09_awi"


def _wiener_filter_demo(s: np.ndarray, o: np.ndarray, epsilon: float) -> np.ndarray:
    """Reproduce the Wiener filter the AWI loss computes (centred lag axis)."""
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
    w_full = _wiener_filter_demo(syn_np, obs_np, epsilon)
    n_pad = w_full.shape[-1]
    lag_idx = np.arange(n_pad) - n_pad // 2
    lag_s = lag_idx * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a)
    ax = axes[0, 0]
    ax.plot(t, syn_np, color="C0", lw=1.0, label="syn")
    ax.plot(t, obs_np, color="C1", lw=1.0, label=f"obs (+{int(tau0*1e3)} ms shift)")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) syn vs. time-shifted obs")
    ax.legend(loc="upper right", fontsize=8)

    # (b) Wiener filter
    ax = axes[0, 1]
    ax.plot(lag_s * 1e3, w_full, color="C2", lw=1.4, label="Wiener filter w(τ)")
    measured_peak_lag = float(lag_s[int(np.argmax(np.abs(w_full)))] * 1e3)
    ax.axvline(tau0 * 1e3, color="k", lw=0.5, ls=":",
               label=f"expected lag = +{tau0*1e3:.0f} ms")
    ax.axvline(measured_peak_lag, color="C3", lw=0.6, ls="--",
               label=f"measured peak = {measured_peak_lag:.1f} ms")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("lag τ [ms]")
    ax.set_title("(b) Wiener filter (ε=1e-4): a near-delta at lag = +τ_shift")
    ax.legend(loc="upper right", fontsize=8)

    # (c) AWI numerator vs. denominator integrand
    ax = axes[1, 0]
    P = lag_s
    Tw_sq = (P * w_full) ** 2
    w_sq = w_full ** 2
    ax.plot(lag_s * 1e3, w_sq, color="C0", lw=1.2, label=r"$w(\tau)^2$ (denominator)")
    ax.plot(lag_s * 1e3, Tw_sq, color="C3", lw=1.2,
            label=r"$(T(\tau) w(\tau))^2 = (\tau\cdot w)^2$ (numerator)")
    ax.set_xlim(-150, 150)
    ax.set_xlabel("lag τ [ms]")
    ax.set_title("(c) AWI numerator/denominator on the lag axis (Warner-Guasch eq.10)")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    awi_vals = scan_shift(AWILoss(dt=dt, epsilon=epsilon, reduction="sum"),
                          syn_np, dt, tau_grid)
    l2_vals  = scan_shift(L2Loss(reduction="sum"), syn_np, dt, tau_grid)
    # normalise each for shape
    ax.plot(tau_grid / half_T, awi_vals / (awi_vals.max() + 1e-30),
            color="C2", lw=1.4, label="AWI")
    ax.plot(tau_grid / half_T, l2_vals / (l2_vals.max() + 1e-30),
            color="C0", lw=1.0, ls="--", label="L2 (baseline)")
    # τ² reference scaled to match
    ax.plot(tau_grid / half_T, (tau_grid / tau_grid.max()) ** 2,
            color="k", lw=0.6, ls=":", label=r"τ² reference")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin scan: AWI is convex in τ (~τ²); L2 cycles (Warner-Guasch 2016 fig.4)")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("AWILoss — Warner & Guasch (2014, 2016)", y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "expected_wiener_peak_ms": tau0 * 1e3,
        "measured_wiener_peak_ms": measured_peak_lag,
        "awi_loss_at_tau0":  float(AWILoss(dt=dt, reduction="sum")(syn, obs).detach()),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
