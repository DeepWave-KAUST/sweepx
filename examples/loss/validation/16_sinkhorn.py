"""Validate SinkhornLoss (Cuturi 2013; Feydy 2019 debiased).

Recipe:
* Build a SHORT Ricker pair (Sinkhorn's cost matrix is O(nt²); we use
  nt=192 to keep it fast).
* Reproduce the same Sinkhorn iteration the loss uses internally to extract
  the transport plan Π = exp((f_i + g_j − C_ij)/ε).
* Panel (a): syn vs. obs densities p, q on the time axis.
* Panel (b): cost matrix C[i,j] = (t_i − t_j)² heatmap.
* Panel (c): transport plan Π — should be concentrated near the diagonal
  shifted by τ.
* Panel (d): basin scan vs. time-shift for ε=1e-4, 1e-3, 1e-2 (sweep);
  also compare `debiased=True` vs `False`.  Smaller ε ⇒ closer to true W2;
  larger ε ⇒ smoother + biased.
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

from sweep_loss import SinkhornLoss  # noqa: E402
from sweep_loss._utils import normalize_density, positive_transform  # noqa: E402
from sweep_loss.sinkhorn import _sinkhorn_log  # noqa: E402


NAME = "16_sinkhorn"


def run() -> dict:
    plt = setup_matplotlib()

    # Keep nt small — O(nt²) cost matrix.
    nt, dt, fc, t0, tau0 = 192, 2e-3, 6.0, 0.16, 0.040
    t = np.arange(nt) * dt
    syn_np = ricker(nt, dt, fc, t0)
    obs_np = fractional_shift(syn_np, dt, tau0)

    syn = torch.tensor(syn_np, dtype=torch.float64).view(1, nt)
    obs = torch.tensor(obs_np, dtype=torch.float64).view(1, nt)
    fs = positive_transform(syn, method="square")
    fo = positive_transform(obs, method="square")
    p = normalize_density(fs, dim=-1)
    q = normalize_density(fo, dim=-1)
    t_t = torch.arange(nt, dtype=torch.float64) * dt
    C = (t_t.unsqueeze(0) - t_t.unsqueeze(1)) ** 2

    # Manual Sinkhorn to extract the transport plan (with same convention as loss)
    epsilon = 5e-4
    n_iter = 100
    log_p = torch.log(p.clamp_min(1e-30))
    log_q = torch.log(q.clamp_min(1e-30))
    K = -C / epsilon
    f = torch.zeros_like(log_p)
    g = torch.zeros_like(log_q)
    for _ in range(n_iter):
        a = K.unsqueeze(0) + (f / epsilon).unsqueeze(-1)
        # log sum-exp along i (dim=-2)
        m1, _ = a.max(dim=-2, keepdim=True)
        g = epsilon * (log_q - (m1.squeeze(-2) + torch.log((a - m1).exp().sum(dim=-2) + 1e-30)))
        b = K.unsqueeze(0) + (g / epsilon).unsqueeze(-2)
        m2, _ = b.max(dim=-1, keepdim=True)
        f = epsilon * (log_p - (m2.squeeze(-1) + torch.log((b - m2).exp().sum(dim=-1) + 1e-30)))
    # Transport plan
    Pi = torch.exp((f.unsqueeze(-1) + g.unsqueeze(-2) - C.unsqueeze(0)) / epsilon)
    Pi_np = Pi.detach().numpy()[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) p, q
    ax = axes[0, 0]
    ax.plot(t, p.detach().numpy().ravel(), color="C0", lw=1.4, label="p (syn²/Σ)")
    ax.plot(t, q.detach().numpy().ravel(), color="C1", lw=1.4, ls="--",
            label=f"q (obs²/Σ, +{int(tau0*1e3)} ms)")
    ax.set_xlim(0.05, 0.35)
    ax.set_xlabel("t [s]")
    ax.set_title("(a) input densities (positive='square')")
    ax.legend(loc="upper right", fontsize=8)

    # (b) cost matrix C
    ax = axes[0, 1]
    im = ax.imshow(C.numpy(), origin="lower", extent=[t[0], t[-1], t[0], t[-1]],
                   cmap="viridis")
    ax.set_xlabel("t_j [s]"); ax.set_ylabel("t_i [s]")
    ax.set_title(r"(b) cost matrix $C_{ij} = (t_i - t_j)^2$")
    plt.colorbar(im, ax=ax, label="cost (s²)")

    # (c) transport plan Π (log-scale)
    ax = axes[1, 0]
    Pi_disp = np.log(Pi_np + 1e-12)
    im = ax.imshow(Pi_disp, origin="lower", extent=[t[0], t[-1], t[0], t[-1]],
                   cmap="magma", aspect="auto",
                   vmin=np.percentile(Pi_disp, 90), vmax=Pi_disp.max())
    ax.set_xlabel("t_j (obs) [s]")
    ax.set_ylabel("t_i (syn) [s]")
    # The plan should lie close to the line t_j = t_i + tau0 (since obs is +tau0 of syn)
    ax.plot([t0 - 0.05, t0 + 0.05], [t0 - 0.05 - tau0, t0 + 0.05 - tau0],
            color="C2", lw=1.2, ls=":", label=f"diagonal shifted by +{int(tau0*1e3)} ms")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(r"(c) Sinkhorn transport plan $\Pi$ (log) — peaks on the shifted diagonal")
    plt.colorbar(im, ax=ax)

    # (d) basin scan
    ax = axes[1, 1]
    half_T = 1.0 / (2.0 * fc)
    tau_grid = np.linspace(-4 * half_T, 4 * half_T, 41)  # coarser due to cost
    for label, eps_v, debiased, color, ls in [
        ("ε=1e-2, debiased",  1e-2,  True, "C0", "-"),
        ("ε=1e-3, debiased",  1e-3,  True, "C2", "-"),
        ("ε=1e-3, raw",       1e-3, False, "C3", "--"),
    ]:
        loss_fn = SinkhornLoss(epsilon=eps_v, n_iter=80, positive="square",
                               dt=dt, debiased=debiased, reduction="sum")
        vals = scan_shift(loss_fn, syn_np, dt, tau_grid)
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-30)
        ax.plot(tau_grid / half_T, v, color=color, ls=ls, lw=1.4, label=label)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"shift / half-$\lambda$")
    ax.set_ylabel("normalised loss")
    ax.set_title("(d) basin scan: smaller ε ≈ true W2 (convex); raw vs. debiased")
    ax.legend(loc="upper center", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("SinkhornLoss — Cuturi (2013); Feydy et al. (2019) debiased divergence",
                 y=1.002)
    fig.tight_layout()
    out = fig_path(NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # Check that transport plan rows sum to p, columns to q (marginal constraints)
    row_sum = Pi_np.sum(axis=1)
    col_sum = Pi_np.sum(axis=0)
    p_np = p.detach().numpy().ravel(); q_np = q.detach().numpy().ravel()
    return {
        "fig": os.path.basename(out),
        "epsilon": epsilon,
        "max_marginal_err_rows": float(np.max(np.abs(row_sum - p_np))),
        "max_marginal_err_cols": float(np.max(np.abs(col_sum - q_np))),
    }


if __name__ == "__main__":
    payload = run()
    path = write_summary(NAME, payload)
    print(f"wrote {path}")
    print(f"wrote {fig_path(NAME)}")
