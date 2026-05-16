"""Tests for trace-normalized L2 and global-correlation misfits."""

from __future__ import annotations

import math

import torch

from sweep_loss import (
    GlobalCorrelationLoss,
    TraceNormalizedL2Loss,
    global_correlation_loss,
    trace_normalized_l2_loss,
)


# -------------------------------------------------------------------------
# Global correlation (NCC)
# -------------------------------------------------------------------------
def test_ncc_invariant_to_per_trace_amplitude():
    syn = torch.randn(2, 256, 4, 1)
    obs = torch.randn_like(syn)
    a = GlobalCorrelationLoss(reduction="mean")(syn, obs)
    # Scale every trace independently by random positive numbers
    scale = torch.rand(1, 1, syn.shape[2], syn.shape[3]) * 10 + 0.1
    b = GlobalCorrelationLoss(reduction="mean")(syn * scale, obs * scale * 7.3)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)


def test_ncc_perfect_match_is_zero_with_offset():
    syn = torch.randn(1, 64, 3, 1)
    assert torch.allclose(
        GlobalCorrelationLoss(offset_one=True, reduction="sum")(syn, syn * 4.2),
        torch.tensor(0.0), atol=1e-6
    )


def test_ncc_anti_correlation():
    """Perfectly anti-correlated traces give 1 - (-1) = 2 per trace."""
    syn = torch.randn(1, 128, 3, 1)
    obs = -syn
    # offset_one=True: 1 - corr; corr = -1; so per_trace = 2.  mean = 2.
    loss = GlobalCorrelationLoss(offset_one=True, reduction="mean")(syn, obs)
    assert math.isclose(float(loss), 2.0, abs_tol=1e-5)


def test_ncc_no_offset_returns_neg_one_for_match():
    syn = torch.randn(1, 64, 2, 1)
    obs = syn.clone() * 3.0
    loss = GlobalCorrelationLoss(offset_one=False, reduction="mean")(syn, obs)
    assert math.isclose(float(loss), -1.0, abs_tol=1e-5)


def test_ncc_gradient_flow():
    syn = torch.randn(1, 64, 3, 1, requires_grad=True)
    obs = torch.randn_like(syn)
    GlobalCorrelationLoss(reduction="mean")(syn, obs).backward()
    assert syn.grad is not None and torch.isfinite(syn.grad).all()


# -------------------------------------------------------------------------
# Trace-normalised L2
# -------------------------------------------------------------------------
def test_tnl2_invariant_to_per_trace_amplitude():
    syn = torch.randn(2, 128, 4, 1)
    obs = torch.randn_like(syn)
    a = TraceNormalizedL2Loss(reduction="sum")(syn, obs)
    scale = torch.rand(1, 1, syn.shape[2], syn.shape[3]) * 5 + 0.5
    b = TraceNormalizedL2Loss(reduction="sum")(syn * scale, obs * scale * 2.0)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)


def test_tnl2_perfect_match_zero():
    syn = torch.randn(1, 64, 3, 1)
    assert torch.allclose(
        TraceNormalizedL2Loss(reduction="sum")(syn, syn * 1.5),
        torch.tensor(0.0), atol=1e-6
    )


def test_tnl2_equivalence_with_ncc():
    """For canonical inputs:  J_nL2 = #traces * J_NCC(offset_one=True) / #traces

    More precisely, per-trace: nL2 = 1/2 * sum_t (a_t - b_t)^2 where each
    trace has unit norm -> = 1/2 (||a||^2 + ||b||^2 - 2<a,b>) = 1 - <a,b>.
    So NCC(offset_one=True) per-trace == nL2 per-trace, identically.
    """
    syn = torch.randn(2, 128, 3, 1)
    obs = torch.randn_like(syn)
    a = TraceNormalizedL2Loss(reduction="sum")(syn, obs)
    b = GlobalCorrelationLoss(offset_one=True, reduction="sum")(syn, obs)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_functional_aliases():
    syn = torch.randn(1, 32, 2, 1)
    obs = torch.randn_like(syn)
    assert torch.allclose(
        global_correlation_loss(syn, obs),
        GlobalCorrelationLoss()(syn, obs),
    )
    assert torch.allclose(
        trace_normalized_l2_loss(syn, obs),
        TraceNormalizedL2Loss()(syn, obs),
    )
