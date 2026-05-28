"""Validate EnvelopeLoss (Wu 2014 / Bozdağ 2011 / Chi 2014).

Recipe:

* Build a Ricker pair (syn at t0, obs shifted by tau seconds).
* Plot ``E_syn = |hilbert(syn)|`` and ``E_obs = |hilbert(obs)|`` together
  with the raw traces — this is exactly the envelope function that
  ``EnvelopeLoss`` calls internally.
* For each of the four ``EnvelopeLoss`` variants
  ``(p=2)``, ``(p=1)``, ``(log=True)``, ``(squared=True)``:
    - compute the loss value;
    - autograd the adjoint source ``d J / d syn``;
    - scan ``loss(syn, shifted_obs)`` over tau in ±5 half-wavelengths
      to show the wide attraction basin that envelope FWI was invented for.
* Save a 4x4 figure (one row per variant) + a JSON summary.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    Pair,
    adjoint,
    fig_path,
    fractional_shift,
    ricker_pair,
    scan_shift,
    setup_matplotlib,
    write_summary,
)

from sweep_loss import EnvelopeLoss  # noqa: E402
from sweep_loss._utils import envelope as hilbert_envelope  # noqa: E402


NAME = "04_envelope"


def _scan_basin(loss_factory, base_syn: np.ndarray, dt: float, fc: float):
    """Symmetric loss-vs-shift scan in units of [-5, +5] half-wavelengths."""
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-5.0 * half_T, 5.0 * half_T, 121)
    vals = scan_shift(loss_factory(reduction="sum"), base_syn, dt, tau_grid)
    return tau_grid / half_T, vals


def run() -> dict:
    plt = setup_matplotlib()

    pair = ricker_pair(nt=1024, dt=1e-3, fc=12.0, t0=0.30, tau=0.020)
    t = pair.t
    syn_np = pair.syn.detach().numpy().ravel()
    obs_np = pair.obs.detach().numpy().ravel()

    # Envelopes — exactly the helper EnvelopeLoss calls (sweep_loss._utils.envelope)
    E_syn = hilbert_envelope(pair.syn, dim=-3, eps=0.0).detach().numpy().ravel()
    E_obs = hilbert_envelope(pair.obs, dim=-3, eps=0.0).detach().numpy().ravel()

    variants = [
        ("p=2 (Wu 2014)",         lambda **kw: EnvelopeLoss(p=2, **kw)),
        ("p=1",                   lambda **kw: EnvelopeLoss(p=1, **kw)),
        ("log (Bozdağ 2011)",     lambda **kw: EnvelopeLoss(log=True, **kw)),
        ("squared (Chi 2014)",    lambda **kw: EnvelopeLoss(squared=True, **kw)),
    ]

    fig, axes = plt.subplots(
        len(variants), 3, figsize=(13, 2.6 * len(variants)),
        gridspec_kw={"width_ratios": [2.0, 1.6, 1.6]},
    )

    summary = {"variants": []}

    # Shared first column: traces + envelopes
    for row, (label, _factory) in enumerate(variants):
        ax = axes[row, 0]
        ax.plot(t, syn_np, color="C0", lw=1.0, label="syn")
        ax.plot(t, obs_np, color="C1", lw=1.0, label="obs (shift +20 ms)", alpha=0.85)
        ax.plot(t, E_syn, color="C0", lw=1.6, ls="--", label="E[syn]")
        ax.plot(t, E_obs, color="C1", lw=1.6, ls="--", label="E[obs]", alpha=0.85)
        ax.set_xlim(0.20, 0.45)
        ax.set_xlabel("t [s]" if row == len(variants) - 1 else "")
        ax.set_ylabel(label)
        if row == 0:
            ax.set_title("traces (solid) and envelopes (dashed)")
            ax.legend(loc="upper right", ncol=2)

    # Second column: adjoint source per variant
    for row, (label, factory) in enumerate(variants):
        # Fresh syn with grad each row
        syn = pair.syn.detach().clone().requires_grad_(True)
        obs = pair.obs.detach().clone()
        loss_fn = factory(reduction="sum")
        g = adjoint(loss_fn, syn, obs)
        loss_val = float(loss_fn(syn, obs).detach())

        ax = axes[row, 1]
        ax.plot(t, syn_np, color="C0", lw=0.9, alpha=0.5, label="syn")
        ax2 = ax.twinx()
        ax2.plot(t, g, color="C3", lw=1.4, label="adjoint")
        ax2.axhline(0.0, color="k", lw=0.4, alpha=0.5)
        ax.set_xlim(0.20, 0.45)
        ax.set_xlabel("t [s]" if row == len(variants) - 1 else "")
        if row == 0:
            ax.set_title("adjoint source $\\partial J / \\partial \\mathrm{syn}$")
        ax.set_yticks([])
        ax2.tick_params(axis="y", labelcolor="C3")

        summary["variants"].append(
            {
                "label": label,
                "loss_at_tau_20ms": loss_val,
                "adjoint_l2": float(np.linalg.norm(g)),
                "adjoint_max_abs_at_t_s": float(t[int(np.argmax(np.abs(g)))]),
            }
        )

    # Third column: loss-vs-shift basin in half-wavelengths
    for row, (label, factory) in enumerate(variants):
        x, y = _scan_basin(factory, syn_np, pair.dt, pair.fc)
        ax = axes[row, 2]
        ax.plot(x, y, color="C2")
        ax.axvline(0.0, color="k", lw=0.4, alpha=0.5)
        ax.set_xlim(-5, 5)
        ax.set_xlabel(r"shift / half-$\lambda$" if row == len(variants) - 1 else "")
        if row == 0:
            ax.set_title(r"loss vs. obs time-shift $\tau$ (basin)")
        # record the basin shape
        argmin_idx = int(np.argmin(y))
        argmin_x = float(x[argmin_idx])
        summary["variants"][row]["scan_argmin_halfL"] = argmin_x
        summary["variants"][row]["scan_y_at_0"] = float(y[len(y) // 2])

    fig.suptitle(
        "EnvelopeLoss — `sweep_loss.envelope` (=|Hilbert|) and adjoint behave per Wu/Bozdağ/Chi",
        y=1.005,
    )
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    summary["fig"] = os.path.basename(out)
    return summary


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
