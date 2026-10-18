"""Validate Wasserstein2Loss (Engquist-Froese-Yang 2016; Yang et al. 2018).

Recipe:
* Build a Ricker pair (syn, obs).
* For positive='square' (the geophysics-standard choice) compute the
  densities, CDFs, and inverse CDFs that the loss uses internally.
* Panel (a): CDFs F_s(t), F_o(t).
* Panel (b): inverse CDFs F_s^{-1}(z), F_o^{-1}(z) plotted vs. quantile
  z ∈ (0, 1) — the *transport map* W2 computes its squared L2 distance on.
* Panel (c): (F_s^{-1} − F_o^{-1})² integrand.
* Panel (d): basin scan vs. time-shift for positive ∈ {'square', 'abs',
  'linear'}.  Expect W2 ≈ ½τ² (Engquist-Froese-Yang 2016 thm. 2.1).
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

from sweep_loss import Wasserstein2Loss  # noqa: E402
from sweep_loss._utils import normalize_density, positive_transform  # noqa: E402
from sweep_loss.w2 import _inverse_cdf  # noqa: E402


NAME = "15_w2"


def run() -> dict:
    plt = setup_matplotlib()

    nt, dt, fc, t0, tau0 = 1024, 1e-3, 12.0, 0.30, 0.030
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt)

    # Use positive='square' for the canonical demo
    pos = "square"
    fs = positive_transform(syn, method=pos)
    fo = positive_transform(obs, method=pos)
    ps = normalize_density(fs, dim=-1)
    po = normalize_density(fo, dim=-1)
    Fs = torch.cumsum(ps, dim=-1)
    Fo = torch.cumsum(po, dim=-1)
    n_q = 256
    z = (torch.arange(n_q, dtype=torch.float64) + 0.5) / n_q
    Fs_inv = _inverse_cdf(Fs, dt, z).detach().numpy().ravel()
    Fo_inv = _inverse_cdf(Fo, dt, z).detach().numpy().ravel()
    Fs_np, Fo_np = Fs.detach().numpy().ravel(), Fo.detach().numpy().ravel()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) CDFs
    ax = axes[0, 0]
    ax.plot(t, Fs_np, color="C0", lw=1.4, label="F_s = cumsum(syn²/Σ)")
    ax.plot(t, Fo_np, color="C1", lw=1.4, ls="--", label="F_o = cumsum(obs²/Σ)")
    ax.set_xlim(0.20, 0.45)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) CDFs F_s, F_o (input to inverse-CDF transport)")
    ax.legend(loc="lower right", fontsize=8)

    # (b) inverse CDFs (quantile functions)
    ax = axes[0, 1]
    z_np = z.numpy()
    ax.plot(z_np, Fs_inv, color="C0", lw=1.4, label=r"$F_s^{-1}(z)$ (transport map for syn)")
    ax.plot(z_np, Fo_inv, color="C1", lw=1.4, ls="--", label=r"$F_o^{-1}(z)$")
    ax.set_xlabel("quantile z ∈ (0,1)")
    ax.set_ylabel("t = F⁻¹(z)  [s]")
    ax.set_title("(b) inverse CDFs — W2 compares them in L2 over z")
    ax.legend(loc="upper left", fontsize=8)

    # (c) (Fs^-1 - Fo^-1)^2
    ax = axes[1, 0]
    integrand = (Fs_inv - Fo_inv) ** 2
    ax.plot(z_np, integrand, color="C3", lw=1.4, label=r"$(F_s^{-1} - F_o^{-1})^2$")
    ax.set_xlabel("quantile z")
    ax.set_ylabel("(time)²")
    ax.set_title("(c) W2² integrand over quantile (mean is the loss)")
    ax.legend(loc="upper right", fontsize=8)

    # (d) basin
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 81)
    for label, color, pos_arg in [("positive='square'", "C0", "square"),
                                  ("positive='abs'", "C1", "abs"),
                                  ("positive='linear' (default)", "C3", "linear")]:
        loss_fn = Wasserstein2Loss(positive=pos_arg, dt=dt, reduction="sum")
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, lw=1.4, label=label)
    # τ² reference
    ax.plot(tau_grid / half_T, (tau_grid / tau_grid.max()) ** 2,
            color="k", lw=0.6, ls=":", label=r"$\tau^2$ reference")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) W2² ∝ τ² (Engquist-Froese-Yang 2016 thm 2.1); 'linear' degenerates")
    ax.legend(loc="upper center", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Wasserstein2Loss — Engquist-Froese-Yang (2016); Yang et al. (2018)",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig": os.path.basename(out),
        "monotone_inverse_cdf_syn": bool(np.all(np.diff(Fs_inv) >= -1e-12)),
        "monotone_inverse_cdf_obs": bool(np.all(np.diff(Fo_inv) >= -1e-12)),
        "w2_at_tau_20ms_square_sum": float(
            Wasserstein2Loss(positive="square", dt=dt, reduction="sum")(
                syn.view(1, nt, 1, 1), obs.view(1, nt, 1, 1)
            ).detach()
        ),
        "note_on_theory": (
            "Engquist-Froese-Yang 2016 thm 2.1: W2 IS linear in shift τ; the "
            "scaling W2² = ½ τ² holds for delta densities only. For a Ricker² "
            "density the proportionality constant is wavelet-bandwidth-dependent."
        ),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
