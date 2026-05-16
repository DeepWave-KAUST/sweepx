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


# -------------------------------------------------------------------------
# demean option (trace_cosine semantics)
# -------------------------------------------------------------------------
def test_ncc_demean_removes_dc_sensitivity():
    """With demean=True the misfit is invariant to a DC offset; without
    demean, a DC offset changes the cosine (since it shifts the trace
    direction in time)."""
    torch.manual_seed(0)
    syn = torch.randn(1, 128, 3, 1)
    obs = syn.clone() + 0.5  # add DC

    # Without demean: DC pollutes the cosine
    loss_no = GlobalCorrelationLoss(demean=False, reduction="sum")(syn, obs)
    # With demean: DC removed → perfect match
    loss_yes = GlobalCorrelationLoss(demean=True, reduction="sum")(syn, obs)
    assert float(loss_no) > 1e-3
    assert torch.allclose(loss_yes, torch.tensor(0.0), atol=1e-5)


def test_ncc_demean_matches_inline_implementation():
    """GlobalCorrelationLoss(demean=True) must equal the inline trace_cosine
    formula from fwi_workflow-dev (demean → unit-norm → 1 - cosine)."""
    torch.manual_seed(1)
    syn = torch.randn(2, 256, 4, 1)
    obs = torch.randn_like(syn)

    # Reference inline impl (mirrors fwi_workflow-dev trace_cosine_loss).
    s_c = syn - syn.mean(dim=-3, keepdim=True)
    o_c = obs - obs.mean(dim=-3, keepdim=True)
    s_n = s_c / s_c.norm(dim=-3, keepdim=True).clamp_min(1e-12)
    o_n = o_c / o_c.norm(dim=-3, keepdim=True).clamp_min(1e-12)
    ref_per_trace = 1.0 - (s_n * o_n).sum(dim=-3)   # (ns, nrec, 1)
    expected_mean = ref_per_trace.mean()

    got = GlobalCorrelationLoss(offset_one=True, demean=True, reduction="mean")(syn, obs)
    assert torch.allclose(got, expected_mean, rtol=1e-5, atol=1e-6)


def test_tnl2_demean_matches_ncc_demean():
    """With demean=True, TraceNormalizedL2Loss and GlobalCorrelationLoss
    remain equivalent per trace (the demean step is identical for both)."""
    torch.manual_seed(2)
    syn = torch.randn(2, 128, 3, 1) + 0.3  # add DC to verify demean kicks in
    obs = torch.randn_like(syn) - 0.2
    a = TraceNormalizedL2Loss(demean=True, reduction="sum")(syn, obs)
    b = GlobalCorrelationLoss(offset_one=True, demean=True, reduction="sum")(syn, obs)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)
