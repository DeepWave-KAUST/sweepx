"""Shared helpers for the per-loss validation scripts.

Every script under ``examples/validation/`` follows the same recipe:

1. build a synthetic Ricker trace pair (``obs``, ``syn``);
2. call the loss-specific intermediate quantity (envelope, Wiener filter,
   CDF, STFT, ...) to confirm the loss does what the paper says it does;
3. scan one perturbation (typically the obs time-shift ``tau``) and plot
   loss-vs-tau;
4. compute the adjoint source ``d loss / d syn`` via autograd and overlay
   it on syn.

The scripts write ``figs/<name>.png`` and a tiny ``<name>.json`` summary so
the final ``REPORT.md`` can be regenerated from the JSON artefacts.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch


FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
def ricker(nt: int, dt: float, fc: float, t0: float) -> np.ndarray:
    """Ricker (Mexican hat) wavelet, centred at ``t0`` (seconds)."""
    t = np.arange(nt) * dt
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def time_shift(trace: np.ndarray, n_samples: int) -> np.ndarray:
    """Integer-sample time-shift of a 1-D trace (zero-padded)."""
    out = np.zeros_like(trace)
    if n_samples > 0:
        out[n_samples:] = trace[:-n_samples]
    elif n_samples < 0:
        out[:n_samples] = trace[-n_samples:]
    else:
        out[:] = trace
    return out


def fractional_shift(trace: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Sub-sample shift via FFT (phase-ramp)."""
    nt = trace.shape[-1]
    X = np.fft.rfft(trace)
    f = np.fft.rfftfreq(nt, d=dt)
    X = X * np.exp(-2j * math.pi * f * tau)
    return np.fft.irfft(X, n=nt).astype(trace.dtype)


@dataclass
class Pair:
    """A canonical-layout (1, nt, 1, 1) torch pair with sampling info."""

    syn: torch.Tensor
    obs: torch.Tensor
    dt: float
    fc: float
    t: np.ndarray  # time axis in seconds, shape (nt,)

    @property
    def nt(self) -> int:
        return self.syn.shape[-3]


def ricker_pair(
    nt: int = 1024,
    dt: float = 1e-3,
    fc: float = 12.0,
    t0: float = 0.30,
    tau: float = 0.020,
    requires_grad: bool = True,
    dtype: torch.dtype = torch.float64,
) -> Pair:
    """Build a (syn, obs) pair of canonical layout (1, nt, 1, 1).

    ``syn`` is centred at ``t0``; ``obs`` is the same wavelet shifted by
    ``tau`` seconds (fractional shift via FFT).
    """
    t = np.arange(nt) * dt
    s = ricker(nt, dt, fc, t0)
    o = fractional_shift(s, dt, tau)
    syn = torch.tensor(s, dtype=dtype).view(1, nt, 1, 1)
    obs = torch.tensor(o, dtype=dtype).view(1, nt, 1, 1)
    if requires_grad:
        syn.requires_grad_(True)
    return Pair(syn=syn, obs=obs, dt=dt, fc=fc, t=t)


# ----------------------------------------------------------------------------
# Autograd adjoint
# ----------------------------------------------------------------------------
def adjoint(loss_fn, syn: torch.Tensor, obs: torch.Tensor) -> np.ndarray:
    """Return the adjoint source d loss / d syn for a (1, nt, 1, 1) pair.

    Works for any sweep_loss BaseFWILoss subclass (reduction defaults to mean
    inside the class).  ``syn`` must have ``requires_grad=True``.
    """
    if syn.grad is not None:
        syn.grad.zero_()
    val = loss_fn(syn, obs)
    val.backward()
    g = syn.grad.detach().cpu().numpy().ravel()
    return g


# ----------------------------------------------------------------------------
# Loss-vs-shift scan
# ----------------------------------------------------------------------------
def scan_shift(
    loss_fn,
    base_syn_1d: np.ndarray,
    dt: float,
    tau_grid_s: np.ndarray,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    """Evaluate loss(syn, shifted_obs) over a grid of time shifts.

    ``base_syn_1d`` is a 1-D numpy trace; we hold ``syn = base_syn_1d`` and
    sweep ``obs = fractional_shift(base_syn_1d, tau)``.  Returns a 1-D
    numpy array of loss values.
    """
    nt = base_syn_1d.shape[-1]
    syn = torch.tensor(base_syn_1d, dtype=dtype).view(1, nt, 1, 1)
    out = np.empty(tau_grid_s.shape[0], dtype=float)
    with torch.no_grad():
        for k, tau in enumerate(tau_grid_s):
            obs_np = fractional_shift(base_syn_1d, dt, float(tau))
            obs = torch.tensor(obs_np, dtype=dtype).view(1, nt, 1, 1)
            out[k] = float(loss_fn(syn, obs).detach().cpu())
    return out


# ----------------------------------------------------------------------------
# JSON summary I/O
# ----------------------------------------------------------------------------
def write_summary(name: str, payload: dict) -> str:
    path = os.path.join(FIG_DIR, f"{name}.json")
    safe = json.loads(json.dumps(payload, default=float))
    with open(path, "w") as fh:
        json.dump(safe, fh, indent=2)
    return path


def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, f"{name}.png")


# ----------------------------------------------------------------------------
# Cosmetic
# ----------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
        }
    )
    return plt
