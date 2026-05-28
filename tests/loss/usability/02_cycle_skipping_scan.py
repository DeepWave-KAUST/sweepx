"""Anti-cycle-skipping diagnostic: scan loss vs. time shift of a Ricker pulse.

The plain L2 misfit between a Ricker and a shifted Ricker oscillates with a
period of one wavelength: once the lag exceeds half a wavelength, the
nearest local minimum is no longer the true zero-lag minimum.  This is
*cycle skipping*.

Several losses in this package claim to mitigate cycle skipping by
producing a wider basin of attraction (monotone vs. shift over a wider
range, sometimes even globally).  This script scans the loss as a function
of the imposed lag ``tau`` and plots the curve next to L2 so we can see
whether the basin is genuinely wider.

For each curve we record:
  * the global minimum location (``argmin tau``) — should be 0,
  * the half-width of the convex basin around tau=0 (largest |tau| such
    that the loss is still monotone increasing as we move outward),
  * whether the loss is monotone over the full scanned range.

The reference loss is L2.  L2's basin has a half-width of half a Ricker
wavelength (~ 1/(2*fc)) — that's the cycle-skipping threshold we want to
beat.

Output: ``cycle_skipping_summary.json`` plus PNG figures under ``figs/``.
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


def make_shifted_gather(tau, nt=512, dt=1e-3, fc=12.0):
    """Build a 1-shot, 1-rec Ricker pair where obs is shifted by ``tau`` s.

    Returns (syn, obs) of shape (1, nt, 1, 1).
    """
    t = torch.arange(nt) * dt
    t0_syn = nt * dt * 0.5
    t0_obs = t0_syn + tau
    xs = (math.pi * fc * (t - t0_syn)) ** 2
    xo = (math.pi * fc * (t - t0_obs)) ** 2
    syn = ((1 - 2 * xs) * torch.exp(-xs)).view(1, nt, 1, 1)
    obs = ((1 - 2 * xo) * torch.exp(-xo)).view(1, nt, 1, 1)
    return syn, obs


def scan_loss(loss_fn, taus, nt=512, dt=1e-3, fc=12.0):
    out = np.zeros(len(taus), dtype=np.float64)
    for i, tau in enumerate(taus):
        syn, obs = make_shifted_gather(tau, nt=nt, dt=dt, fc=fc)
        with torch.no_grad():
            out[i] = float(loss_fn(syn, obs).item())
    return out


def basin_half_width(taus, vals):
    """Return the largest |tau| such that the loss is strictly increasing
    when we move outward from tau=0 on each side."""
    i0 = int(np.argmin(np.abs(taus)))            # index nearest tau=0
    # rightward
    r = i0
    while r + 1 < len(taus) and vals[r + 1] > vals[r]:
        r += 1
    # leftward
    l = i0
    while l - 1 >= 0 and vals[l - 1] > vals[l]:
        l -= 1
    return min(abs(taus[r]), abs(taus[l]))


# A representative dial for each loss.  We pass dt where needed.
# Only the losses with realistic anti-cycle-skipping claims are scanned;
# everything else in family A/B is scanned for comparison.
SCAN_LOSSES = {
    "L2Loss":               lambda dt: sl.L2Loss(),
    "L1Loss":               lambda dt: sl.L1Loss(),
    "HuberLoss":            lambda dt: sl.HuberLoss(delta=0.2),
    "GlobalCorrelationLoss": lambda dt: sl.GlobalCorrelationLoss(),
    "TraceNormalizedL2Loss": lambda dt: sl.TraceNormalizedL2Loss(),
    "CrossCorrelationTraveltimeLoss":
        lambda dt: sl.CrossCorrelationTraveltimeLoss(dt=dt, power=2.0, sigma=80.0),
    "EnvelopeLoss_p2":      lambda dt: sl.EnvelopeLoss(p=2),
    "EnvelopeLoss_log":     lambda dt: sl.EnvelopeLoss(log=True),
    "EnvelopeLoss_sq":      lambda dt: sl.EnvelopeLoss(squared=True),
    "InstantaneousPhaseLoss": lambda dt: sl.InstantaneousPhaseLoss(),
    "ExponentiatedPhaseLoss": lambda dt: sl.ExponentiatedPhaseLoss(),
    "EnvelopePhaseLoss":      lambda dt: sl.EnvelopePhaseLoss(alpha=0.5),
    "TimeFrequencyPhaseLoss": lambda dt: sl.TimeFrequencyPhaseLoss(
        alpha=0.5, n_fft=128, hop_length=32),
    "FrequencyDomainL2Loss":  lambda dt: sl.FrequencyDomainL2Loss(),
    "FrequencyPhaseLoss":     lambda dt: sl.FrequencyPhaseLoss(),
    "FrequencyAmplitudeLoss": lambda dt: sl.FrequencyAmplitudeLoss(),
    "LogarithmicShinMinLoss": lambda dt: sl.LogarithmicShinMinLoss(),
    "LaplaceL2Loss":          lambda dt: sl.LaplaceL2Loss(s=5.0, dt=dt),
    "AWILoss":              lambda dt: sl.AWILoss(dt=dt, epsilon=1e-3),
    "DeconvolutionLoss":    lambda dt: sl.DeconvolutionLoss(dt=dt, epsilon=1e-3),
    "OTMFLoss_2":           lambda dt: sl.OTMFLoss(dt=dt, epsilon=1e-3, order=2),
    "OTMFLoss_1":           lambda dt: sl.OTMFLoss(dt=dt, epsilon=1e-3, order=1),
    "NIMLoss":              lambda dt: sl.NIMLoss(positive="square", dt=dt),
    "JensenShannonLoss":    lambda dt: sl.JensenShannonLoss(positive="square"),
    "Wasserstein1Loss":     lambda dt: sl.Wasserstein1Loss(positive="linear", dt=dt),
    "Wasserstein2Loss":     lambda dt: sl.Wasserstein2Loss(
        positive="linear", dt=dt, n_quantiles=128),
    "SinkhornLoss":         lambda dt: sl.SinkhornLoss(
        positive="linear", dt=dt, epsilon=4e-4, n_iter=50),
    "LocalSimilarityLoss":  lambda dt: sl.LocalSimilarityLoss(sigma_samples=8.0),
    # Soft-DTW and GSOT only with shorter trace
    "SoftDTWLoss":          lambda dt: sl.SoftDTWLoss(gamma=1.0),
    "GSOTLoss":             lambda dt: sl.GSOTLoss(dt=dt, max_shift_samples=40),
}


def main():
    nt = 512
    dt = 1e-3
    fc = 12.0          # peak frequency 12 Hz
    half_wavelength = 0.5 / fc                       # ~ 0.0417 s
    max_tau = 4.0 * half_wavelength                  # scan up to 4 half-cycles
    # SoftDTW / GSOT use a separate, shorter trace.
    short_nt = 128
    short_dt = 4e-3            # keeps physical time range about the same
    fc_short = 6.0             # peak frequency lower to avoid sampling issues
    short_half_wl = 0.5 / fc_short

    # Two scan modes: dense around 0 for the convexity test, wider for the
    # overall shape.
    taus = np.linspace(-max_tau, max_tau, 81)
    taus_short = np.linspace(-4 * short_half_wl, 4 * short_half_wl, 41)

    rows = []
    for name, fac in SCAN_LOSSES.items():
        is_short = name in ("SoftDTWLoss", "GSOTLoss")
        t_grid = taus_short if is_short else taus
        params = (short_nt, short_dt, fc_short) if is_short else (nt, dt, fc)
        loss_fn = fac(params[1])
        print(f"=== scanning {name}")
        try:
            vals = scan_loss(loss_fn, t_grid, nt=params[0], dt=params[1], fc=params[2])
        except Exception as e:
            print(f"  failed: {e!r}")
            rows.append({"name": name, "error": repr(e)})
            continue

        argmin_idx = int(np.argmin(vals))
        argmin_tau = float(t_grid[argmin_idx])
        hw = basin_half_width(t_grid, vals)
        # Express the basin half-width in units of "half wavelengths"
        half_wl = short_half_wl if is_short else half_wavelength
        hw_in_halfwl = hw / half_wl
        is_monotone = bool(hw >= float(t_grid.max()) - 1e-9)
        rows.append({
            "name": name,
            "argmin_tau_s": argmin_tau,
            "basin_half_width_s": float(hw),
            "basin_in_half_wavelengths": float(hw_in_halfwl),
            "monotone_over_scan": is_monotone,
            "max_tau_s": float(t_grid.max()),
            "fc_Hz": params[2],
            "vals": vals.tolist(),
            "taus": t_grid.tolist(),
        })
        # save figure
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.plot(t_grid * 1000, vals, color="black", lw=1.5)
        ax.axvline(0.0, color="gray", lw=0.6)
        ax.set_xlabel("imposed lag τ (ms)")
        ax.set_ylabel(name)
        ax.set_title(
            f"{name}\nbasin = {hw_in_halfwl:.2f} half-wavelengths "
            f"(fc={params[2]} Hz)"
        )
        fig.tight_layout()
        fig.savefig(FIGS / f"scan_{name}.png", dpi=120)
        plt.close(fig)

    out_path = HERE / "cycle_skipping_summary.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
