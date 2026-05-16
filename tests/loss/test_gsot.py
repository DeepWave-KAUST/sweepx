"""Tests for the Graph-space Optimal Transport (GSOT) FWI misfit."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sweep_loss import GSOTLoss, L2Loss, gsot_loss


def _ricker(t, t0, fc=15.0):
    x = (math.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * np.exp(-x)


def test_zero_for_identical():
    nt, dt = 128, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.05)).view(1, nt, 1, 1).double()
    loss = GSOTLoss(eta=1.0, dt=dt, reduction="sum")(o, o)
    assert float(loss) < 1e-10


def test_grows_with_small_shift():
    nt, dt = 256, 1e-3
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.1)).view(1, nt, 1, 1).double()
    losses = []
    for ms in (0.0, 0.002, 0.005, 0.010):
        s = torch.tensor(_ricker(t, 0.1 + ms)).view(1, nt, 1, 1).double()
        losses.append(float(GSOTLoss(dt=dt, reduction="sum")(s, o)))
    assert all(b > a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_gsot_monotone_across_l2_cycle_skipping():
    """Métivier et al. (2018) main result: GSOT removes the cycle-skipping
    plateau."""
    nt, dt = 256, 1e-3   # short trace for cheap O(n^3) Hungarian
    t = np.arange(nt) * dt
    o = torch.tensor(_ricker(t, 0.13)).view(1, nt, 1, 1).double()
    shifts = [0.0, 0.020, 0.030, 0.040, 0.060, 0.080]
    l2_vals, g_vals = [], []
    for ms in shifts:
        s = torch.tensor(_ricker(t, 0.13 + ms)).view(1, nt, 1, 1).double()
        l2_vals.append(float(L2Loss(reduction="sum")(s, o)))
        g_vals.append(float(GSOTLoss(dt=dt, reduction="sum")(s, o)))
    pairs = list(zip(l2_vals, l2_vals[1:]))
    assert any(b < a for a, b in pairs), f"L2 unexpectedly monotone: {l2_vals}"
    pairs = list(zip(g_vals, g_vals[1:]))
    assert all(b > a for a, b in pairs), f"GSOT not monotone: {g_vals}"


def test_banded_assignment_matches_unrestricted_when_band_is_wide():
    """If max_shift_samples >= nt, banded GSOT must equal the unrestricted one."""
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.02)).view(1, nt, 1, 1).double()
    o = torch.tensor(_ricker(t, 0.025)).view(1, nt, 1, 1).double()
    unbanded = float(GSOTLoss(dt=dt, max_shift_samples=None, reduction="sum")(s, o))
    banded = float(GSOTLoss(dt=dt, max_shift_samples=nt, reduction="sum")(s, o))
    assert math.isclose(banded, unbanded, rel_tol=1e-6, abs_tol=1e-9)


def test_banded_assignment_limits_shift():
    """A very narrow band forces the assignment to stay close to the diagonal."""
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.02)).view(1, nt, 1, 1).double()
    o = torch.tensor(_ricker(t, 0.035)).view(1, nt, 1, 1).double()
    wide = float(GSOTLoss(dt=dt, max_shift_samples=20, reduction="sum")(s, o))
    narrow = float(GSOTLoss(dt=dt, max_shift_samples=2, reduction="sum")(s, o))
    # Narrow band gives a larger misfit because the optimal assignment is
    # not allowed to slide as much.
    assert narrow >= wide - 1e-9


def test_gradient_flow():
    nt, dt = 64, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.02)).view(1, nt, 1, 1).float().requires_grad_(True)
    o = torch.tensor(_ricker(t, 0.025)).view(1, nt, 1, 1).float()
    GSOTLoss(dt=dt, reduction="sum")(s, o).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_invalid_params():
    with pytest.raises(ValueError):
        GSOTLoss(dt=0)
    with pytest.raises(ValueError):
        GSOTLoss(eta=-1.0)
    with pytest.raises(ValueError):
        GSOTLoss(max_shift_samples=0)


def test_functional_alias():
    nt, dt = 32, 1e-3
    t = np.arange(nt) * dt
    s = torch.tensor(_ricker(t, 0.01)).view(1, nt, 1, 1)
    o = torch.tensor(_ricker(t, 0.012)).view(1, nt, 1, 1)
    a = GSOTLoss(dt=dt, eta=1e3)(s, o)
    b = gsot_loss(s, o, dt=dt, eta=1e3)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)
