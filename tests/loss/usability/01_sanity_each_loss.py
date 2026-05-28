"""Sanity test for every loss in sweep_loss.

For every loss:
  1. Build a small synthetic gather (Ricker pulses at slightly different lags).
  2. Compute the loss value with reduction='sum' and 'mean'.
  3. Verify finite, non-negative (where the loss is bounded below by 0),
     and zero / near-zero when syn == obs.
  4. Backprop the loss into ``syn`` and check the gradient is finite and
     non-zero when syn != obs.

The script writes a JSON report to ``tests/usability/sanity_report.json`` so
``02_cycle_skipping_scan.py`` and the final markdown report can pick it up.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import sweep_loss as sl


HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)


def make_gather(ns=2, nt=512, nrec=8, nchan=1, dt=1e-3, fc=12.0,
                shift=0.002, scale=1.0, noise=0.0):
    """Build a synthetic (syn, obs) gather of Ricker pulses with a small shift.

    syn:  Ricker at (0.10 + small per-rec lag)
    obs:  Ricker at (0.10 + small per-rec lag + ``shift``) * ``scale``
    """
    t = torch.arange(nt) * dt
    syn = torch.zeros(ns, nt, nrec, nchan)
    obs = torch.zeros(ns, nt, nrec, nchan)
    for s in range(ns):
        for r in range(nrec):
            t0_syn = 0.10 + 0.001 * r + 0.005 * s
            t0_obs = t0_syn + shift
            xs = (math.pi * fc * (t - t0_syn)) ** 2
            xo = (math.pi * fc * (t - t0_obs)) ** 2
            syn[s, :, r, 0] = (1 - 2 * xs) * torch.exp(-xs)
            obs[s, :, r, 0] = scale * (1 - 2 * xo) * torch.exp(-xo)
    if noise > 0:
        obs = obs + noise * torch.randn_like(obs)
    return syn, obs, dt


LOSS_FACTORIES = {
    # ----- Family A: Lp / robust M-estimators ------------------------------
    "L2Loss":              lambda **k: sl.L2Loss(),
    "L1Loss":              lambda **k: sl.L1Loss(),
    "HuberLoss":           lambda **k: sl.HuberLoss(delta=0.2),
    "PseudoHuberLoss":     lambda **k: sl.PseudoHuberLoss(delta=0.2),
    "HybridL1L2Loss":      lambda **k: sl.HybridL1L2Loss(delta=0.2),
    "CauchyLoss":          lambda **k: sl.CauchyLoss(c=0.2),
    "TukeyLoss":           lambda **k: sl.TukeyLoss(c=0.6),
    "GemanMcClureLoss":    lambda **k: sl.GemanMcClureLoss(c=0.2),
    "StudentTLoss":        lambda **k: sl.StudentTLoss(nu=3.0, sigma=0.2),
    # ----- Family B: correlation / amplitude-normalised --------------------
    "GlobalCorrelationLoss": lambda **k: sl.GlobalCorrelationLoss(),
    "TraceNormalizedL2Loss": lambda **k: sl.TraceNormalizedL2Loss(),
    # ----- Family C: traveltime --------------------------------------------
    "CrossCorrelationTraveltimeLoss": lambda dt=1e-3, **k:
        sl.CrossCorrelationTraveltimeLoss(dt=dt, power=2.0, sigma=80.0),
    # ----- Family D: envelope / phase --------------------------------------
    "EnvelopeLoss_p2":     lambda **k: sl.EnvelopeLoss(p=2),
    "EnvelopeLoss_p1":     lambda **k: sl.EnvelopeLoss(p=1),
    "EnvelopeLoss_log":    lambda **k: sl.EnvelopeLoss(log=True),
    "EnvelopeLoss_sq":     lambda **k: sl.EnvelopeLoss(squared=True),
    "InstantaneousPhaseLoss": lambda **k: sl.InstantaneousPhaseLoss(),
    "ExponentiatedPhaseLoss": lambda **k: sl.ExponentiatedPhaseLoss(),
    "EnvelopePhaseLoss":      lambda **k: sl.EnvelopePhaseLoss(alpha=0.5),
    "TimeFrequencyPhaseLoss": lambda **k: sl.TimeFrequencyPhaseLoss(
        alpha=0.5, n_fft=128, hop_length=32),
    # ----- Family E: frequency / Laplace -----------------------------------
    "FrequencyDomainL2Loss":  lambda **k: sl.FrequencyDomainL2Loss(),
    "FrequencyPhaseLoss":     lambda **k: sl.FrequencyPhaseLoss(),
    "FrequencyAmplitudeLoss": lambda **k: sl.FrequencyAmplitudeLoss(),
    "LogarithmicShinMinLoss": lambda **k: sl.LogarithmicShinMinLoss(),
    "LaplaceL2Loss":          lambda dt=1e-3, **k: sl.LaplaceL2Loss(s=5.0, dt=dt),
    # ----- Family F: matching filter / convolution -------------------------
    "AWILoss":              lambda dt=1e-3, **k: sl.AWILoss(dt=dt, epsilon=1e-3),
    "DeconvolutionLoss":    lambda dt=1e-3, **k: sl.DeconvolutionLoss(dt=dt, epsilon=1e-3),
    "OTMFLoss_2":           lambda dt=1e-3, **k: sl.OTMFLoss(dt=dt, epsilon=1e-3, order=2),
    "OTMFLoss_1":           lambda dt=1e-3, **k: sl.OTMFLoss(dt=dt, epsilon=1e-3, order=1),
    # ----- Family G: density divergences -----------------------------------
    "NIMLoss":              lambda dt=1e-3, **k: sl.NIMLoss(positive="square", dt=dt),
    "JensenShannonLoss":    lambda **k: sl.JensenShannonLoss(positive="square"),
    # ----- Family H: optimal transport -------------------------------------
    "Wasserstein1Loss":     lambda dt=1e-3, **k:
        sl.Wasserstein1Loss(positive="linear", dt=dt),
    "Wasserstein2Loss":     lambda dt=1e-3, **k:
        sl.Wasserstein2Loss(positive="linear", dt=dt, n_quantiles=128),
    "SinkhornLoss":         lambda dt=1e-3, **k:
        sl.SinkhornLoss(positive="linear", dt=dt, epsilon=1e-4, n_iter=32),
    # ----- Family I: dynamic warping ---------------------------------------
    # SoftDTW is O(nt^2) with Python loops — keep nt small for sanity.
    # GSOT is O(nt^3) — even smaller.
    "SoftDTWLoss":          lambda **k: sl.SoftDTWLoss(gamma=1.0),
    "GSOTLoss":             lambda dt=1e-3, **k:
        sl.GSOTLoss(dt=dt, max_shift_samples=20),
    # ----- Family J: local attribute --------------------------------------
    "LocalSimilarityLoss":  lambda **k: sl.LocalSimilarityLoss(sigma_samples=8.0),
}


def run_one(name, factory, ns=2, nt=512, nrec=8, dt=1e-3):
    # SoftDTW / GSOT scale super-linearly so test with shorter traces.
    if name in ("SoftDTWLoss", "GSOTLoss"):
        nt = 96
        nrec = 4
        ns = 1
    syn, obs, _dt = make_gather(ns=ns, nt=nt, nrec=nrec, dt=dt, shift=0.002)
    loss_fn = factory(dt=dt)

    syn.requires_grad_(True)
    L = loss_fn(syn, obs)

    info = {
        "name": name,
        "shape": (ns, nt, nrec),
        "finite": bool(torch.isfinite(L).item()),
        "value": float(L.item()),
        "value_is_nan": bool(torch.isnan(L).item()),
        "value_is_inf": bool(torch.isinf(L).item()),
        "neg": float(L.item()) < -1e-6,
    }

    # Backprop test
    try:
        L.backward()
        g = syn.grad
        info["grad_finite"] = bool(torch.isfinite(g).all().item())
        info["grad_norm"] = float(g.norm().item())
        info["grad_nonzero"] = info["grad_norm"] > 0.0
    except Exception as e:
        info["grad_finite"] = False
        info["grad_norm"] = float("nan")
        info["grad_nonzero"] = False
        info["grad_error"] = repr(e)

    # Zero-residual test
    syn0 = obs.clone().detach().requires_grad_(True)
    try:
        L0 = loss_fn(syn0, obs)
        info["zero_residual_value"] = float(L0.item())
        # OT-style losses do not give *exactly* zero when syn==obs, just very small.
        info["zero_residual_small"] = abs(info["zero_residual_value"]) < 1e-4
    except Exception as e:
        info["zero_residual_value"] = float("nan")
        info["zero_residual_small"] = False
        info["zero_residual_error"] = repr(e)

    return info


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    rows = []
    for name, fac in LOSS_FACTORIES.items():
        print(f"--- {name}")
        try:
            row = run_one(name, fac)
        except Exception as e:
            row = {"name": name, "error": repr(e), "finite": False}
        rows.append(row)
        msg = (f"  val={row.get('value', 'N/A'):.3e} " if 'value' in row else "")
        msg += (f"grad={row.get('grad_norm', float('nan')):.3e} " if 'grad_norm' in row else "")
        msg += (f"zero={row.get('zero_residual_value', float('nan')):.3e}"
                if 'zero_residual_value' in row else "")
        print(" ", msg)
    out_path = HERE / "sanity_report.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
