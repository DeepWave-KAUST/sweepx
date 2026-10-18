"""Verify the SoftDTW divergence-mode fix.

Regression check: the previous default (raw soft-DTW) returned a
**negative** value even when ``syn == obs``, which makes it unusable as
an FWI misfit.  The new ``divergence=True`` default subtracts the
self-terms following Blondel et al. (2020) and is non-negative with
``loss(syn, syn) == 0``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sweep_loss as sl


HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def make_pair(tau, nt=128, dt=4e-3, fc=6.0):
    t = torch.arange(nt) * dt
    t0 = nt * dt * 0.5
    xs = (math.pi * fc * (t - t0)) ** 2
    xo = (math.pi * fc * (t - t0 - tau)) ** 2
    s = ((1 - 2 * xs) * torch.exp(-xs)).view(1, nt, 1, 1)
    o = ((1 - 2 * xo) * torch.exp(-xo)).view(1, nt, 1, 1)
    return s, o


def main():
    nt = 128
    dt = 4e-3
    fc = 6.0
    half_wl = 0.5 / fc
    taus = np.linspace(-4 * half_wl, 4 * half_wl, 41)

    L_div = sl.SoftDTWLoss(gamma=1.0, divergence=True)
    L_raw = sl.SoftDTWLoss(gamma=1.0, divergence=False)

    # Sanity values
    s0, o0 = make_pair(0.0, nt=nt, dt=dt, fc=fc)
    v_div = L_div(s0, o0).item()
    v_raw = L_raw(s0, o0).item()
    print(f"syn==obs:  divergence={v_div:.4e}   raw={v_raw:.4e}")

    # Cycle-skipping curve
    div_curve = np.zeros(len(taus))
    raw_curve = np.zeros(len(taus))
    for i, tau in enumerate(taus):
        s, o = make_pair(tau, nt=nt, dt=dt, fc=fc)
        with torch.no_grad():
            div_curve[i] = float(L_div(s, o).item())
            raw_curve[i] = float(L_raw(s, o).item())

    print(f"divergence: min={div_curve.min():.4e}  argmin tau={taus[np.argmin(div_curve)]*1000:.2f} ms")
    print(f"raw       : min={raw_curve.min():.4e}  argmin tau={taus[np.argmin(raw_curve)]*1000:.2f} ms")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(taus * 1000, div_curve, color="C0", lw=1.5)
    ax[0].axhline(0, color="gray", lw=0.6)
    ax[0].axvline(0, color="gray", lw=0.6)
    ax[0].set_xlabel("imposed lag τ (ms)")
    ax[0].set_ylabel("SoftDTW (divergence)")
    ax[0].set_title("divergence=True  (default after fix)")

    ax[1].plot(taus * 1000, raw_curve, color="C3", lw=1.5)
    ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].axvline(0, color="gray", lw=0.6)
    ax[1].set_xlabel("imposed lag τ (ms)")
    ax[1].set_ylabel("SoftDTW (raw Cuturi-Blondel)")
    ax[1].set_title("divergence=False  (legacy, can be < 0)")
    fig.tight_layout()
    fig.savefig(FIGS / "softdtw_fix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    out = {
        "zero_residual_divergence": v_div,
        "zero_residual_raw":        v_raw,
        "taus_ms":                  (taus * 1000).tolist(),
        "div_curve":                div_curve.tolist(),
        "raw_curve":                raw_curve.tolist(),
    }
    (HERE / "softdtw_fix.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {HERE/'softdtw_fix.json'} and {FIGS/'softdtw_fix.png'}")


if __name__ == "__main__":
    main()
