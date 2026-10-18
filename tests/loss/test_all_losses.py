"""Uniform smoke test across *every* loss class.

Per-loss test files cover the maths in depth; this file guards the
cross-cutting invariants that every ``BaseFWILoss`` must satisfy, so a
regression in any single loss (a NaN gradient, a shape bug, a sign error
that makes ``loss(x, x)`` blow up) is caught here regardless of which
module introduced it.

Invariants checked for each loss, on a tiny single trace:

1. **forward** returns a finite scalar;
2. **backward** produces a finite gradient with non-zero norm
   (the misfit actually depends on ``syn``);
3. **identical inputs** give a near-zero, non-negative misfit
   (catches e.g. the raw soft-DTW negativity bug; AWI/OTMF/Deconvolution
   carry an O(epsilon) Tikhonov residual, still well under the bound).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep_loss import (
    AWILoss,
    CauchyLoss,
    CrossCorrelationTraveltimeLoss,
    DeconvolutionLoss,
    EnvelopeLoss,
    EnvelopePhaseLoss,
    ExponentiatedPhaseLoss,
    FrequencyAmplitudeLoss,
    FrequencyDomainL2Loss,
    FrequencyPhaseLoss,
    GemanMcClureLoss,
    GlobalCorrelationLoss,
    HuberLoss,
    HybridL1L2Loss,
    InstantaneousPhaseLoss,
    JensenShannonLoss,
    L1Loss,
    L2Loss,
    LaplaceL2Loss,
    LocalSimilarityLoss,
    LogarithmicShinMinLoss,
    NIMLoss,
    OTMFLoss,
    PseudoHuberLoss,
    SinkhornLoss,
    SoftDTWLoss,
    StudentTLoss,
    TimeFrequencyPhaseLoss,
    TraceNormalizedL2Loss,
    TukeyLoss,
    Wasserstein1Loss,
    Wasserstein2Loss,
)

try:
    import scipy  # noqa: F401
    from sweep_loss import GSOTLoss
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


NT = 48  # tiny so the O(nt^2)/O(nt^3) losses (soft-DTW, GSOT) stay fast


def _make_losses():
    """One representative instance per loss class (id, instance)."""
    losses = [
        ("L2", L2Loss()),
        ("L1", L1Loss()),
        ("Huber", HuberLoss(delta=0.5)),
        ("PseudoHuber", PseudoHuberLoss(delta=0.5)),
        ("HybridL1L2", HybridL1L2Loss(delta=0.5)),
        ("Cauchy", CauchyLoss(c=0.5)),
        ("Tukey", TukeyLoss(c=1.0)),
        ("GemanMcClure", GemanMcClureLoss(c=0.5)),
        ("StudentT", StudentTLoss(nu=2.0, sigma=0.5)),
        ("GlobalCorrelation", GlobalCorrelationLoss()),
        ("TraceNormalizedL2", TraceNormalizedL2Loss()),
        ("CrossCorrelationTraveltime", CrossCorrelationTraveltimeLoss(dt=1e-3)),
        ("Envelope_p2", EnvelopeLoss(p=2)),
        ("Envelope_log", EnvelopeLoss(log=True)),
        ("Envelope_sq", EnvelopeLoss(squared=True)),
        ("InstantaneousPhase", InstantaneousPhaseLoss()),
        ("EnvelopePhase", EnvelopePhaseLoss(alpha=0.5)),
        ("ExponentiatedPhase", ExponentiatedPhaseLoss()),
        # nt=48 < default n_fft=128, so use a small window here.
        ("TimeFrequencyPhase", TimeFrequencyPhaseLoss(n_fft=16, hop_length=4)),
        ("FrequencyDomainL2", FrequencyDomainL2Loss()),
        ("FrequencyAmplitude", FrequencyAmplitudeLoss()),
        ("FrequencyPhase", FrequencyPhaseLoss()),
        ("LogarithmicShinMin", LogarithmicShinMinLoss()),
        ("LaplaceL2", LaplaceL2Loss(s=5.0, dt=1e-3)),
        ("AWI", AWILoss(dt=1e-3)),
        ("Deconvolution", DeconvolutionLoss(dt=1e-3)),
        ("OTMF_2", OTMFLoss(dt=1e-3, order=2)),
        ("OTMF_1", OTMFLoss(dt=1e-3, order=1)),
        ("NIM", NIMLoss(dt=1e-3)),
        ("JensenShannon", JensenShannonLoss()),
        ("Wasserstein1", Wasserstein1Loss(dt=1e-3)),
        ("Wasserstein2", Wasserstein2Loss(dt=1e-3, n_quantiles=64)),
        ("Sinkhorn", SinkhornLoss(epsilon=1e-2, n_iter=20, dt=1e-3)),
        ("SoftDTW", SoftDTWLoss(gamma=1.0)),
        ("LocalSimilarity", LocalSimilarityLoss(sigma_samples=4.0)),
    ]
    if HAS_SCIPY:
        losses.append(("GSOT", GSOTLoss(dt=1e-3)))
    return losses


def _ricker(t0: float) -> np.ndarray:
    t = np.arange(NT) * 1e-3
    x = (np.pi * 20.0 * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def _pair(requires_grad: bool):
    syn = torch.tensor(_ricker(0.020), dtype=torch.float64).view(1, NT, 1, 1)
    obs = torch.tensor(_ricker(0.024), dtype=torch.float64).view(1, NT, 1, 1)
    if requires_grad:
        syn = syn.detach().clone().requires_grad_(True)
    return syn, obs


_LOSSES = _make_losses()
_IDS = [name for name, _ in _LOSSES]


@pytest.mark.parametrize("loss", [fn for _, fn in _LOSSES], ids=_IDS)
def test_forward_finite_scalar(loss):
    syn, obs = _pair(requires_grad=False)
    val = loss(syn, obs)
    assert val.ndim == 0, f"{loss} did not reduce to a scalar"
    assert torch.isfinite(val).all(), f"{loss} produced a non-finite value"


@pytest.mark.parametrize("loss", [fn for _, fn in _LOSSES], ids=_IDS)
def test_backward_finite_nonzero_grad(loss):
    syn, obs = _pair(requires_grad=True)
    loss(syn, obs).backward()
    assert syn.grad is not None, f"{loss} produced no gradient"
    assert torch.isfinite(syn.grad).all(), f"{loss} produced a non-finite gradient"
    assert syn.grad.abs().sum() > 0, f"{loss} produced an all-zero gradient"


@pytest.mark.parametrize("loss", [fn for _, fn in _LOSSES], ids=_IDS)
def test_identical_inputs_near_zero(loss):
    """loss(x, x) must be finite, non-negative, and small.

    Exactly zero for most losses; AWI/OTMF/Deconvolution leave an
    O(epsilon) Tikhonov residual (~1e-4 with the default epsilon), still
    far below the 0.1 bound. This is the invariant that the raw soft-DTW
    (which returned -1.70 at x==x) violated.
    """
    x = torch.tensor(_ricker(0.020), dtype=torch.float64).view(1, NT, 1, 1)
    val = float(loss(x, x.clone()))
    assert np.isfinite(val), f"{loss}: loss(x, x) is not finite"
    assert val >= -1e-6, f"{loss}: loss(x, x) = {val} is negative"
    assert val < 0.1, f"{loss}: loss(x, x) = {val} is not near zero"
