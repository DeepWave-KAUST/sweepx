"""Tests for :class:`sweep_nn.TVPrior` + :class:`sweep_nn.SeabedFreezeMask`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep_nn import SeabedFreezeMask, TVPrior


def test_tv_prior_zero_on_constant_2d():
    v = torch.full((16, 16), 2500.0)
    prior = TVPrior(order="first")
    assert float(prior(v)) == pytest.approx(0.0, abs=1e-12)


def test_tv_prior_zero_on_constant_3d():
    v = torch.full((8, 8, 8), 2500.0)
    prior = TVPrior(order="both")
    assert float(prior(v)) == pytest.approx(0.0, abs=1e-12)


def test_tv_prior_positive_on_step_2d():
    v = torch.full((16, 16), 2500.0)
    v[8:] = 3500.0  # step in z
    prior_first = TVPrior(order="first", x_weight=0.0, z_weight=1.0)
    val = float(prior_first(v))
    # One column has a |Δ| = 1000/scale at the step row; mean over the
    # difference tensor is non-trivially positive.
    assert val > 0.0


def test_tv_prior_curvature_zero_on_linear_ramp():
    """A linear vp(z) has zero second derivative (curvature = 0)."""
    nz, nx = 32, 8
    v = (torch.arange(nz, dtype=torch.float32)[:, None] * 100.0
         + 1500.0) * torch.ones(nz, nx)
    prior = TVPrior(order="second", x_weight=0.0, z_weight=1.0)
    assert float(prior(v)) == pytest.approx(0.0, abs=1e-9)


def test_tv_prior_3d_y_weight_zero_ignored():
    v = torch.randn(8, 8, 8)
    prior_no_y = TVPrior(order="first", x_weight=1.0, y_weight=0.0, z_weight=1.0)
    # Should not error and produce a finite value.
    val = float(prior_no_y(v))
    assert np.isfinite(val) and val >= 0.0


def test_tv_prior_rejects_4d():
    with pytest.raises(ValueError, match="2-D"):
        TVPrior()(torch.zeros(2, 4, 8, 8))


def test_seabed_freeze_mask_2d_zeros_water_column():
    nz, nx = 32, 16
    dz_m = 25.0
    # Flat seabed at z=10*dz=250m.
    sb = np.full(nx, 250.0, dtype=np.float64)
    sf = SeabedFreezeMask(sb, dz_m=dz_m)
    grad = torch.ones((nz, nx))
    sf.apply_to(grad)
    # Rows 0..9 should be zero; row 10 onward keep grad=1.
    assert torch.all(grad[:10] == 0.0)
    assert torch.all(grad[10:] == 1.0)


def test_seabed_freeze_mask_3d_zeros_water_column():
    nz, ny, nx = 16, 8, 8
    dz_m = 50.0
    # Per (y, x) seabed depth: half of x has shallower seabed.
    sb = np.full((ny, nx), 100.0, dtype=np.float64)  # 2 cells deep
    sb[:, :4] = 250.0                                # 5 cells deep
    sf = SeabedFreezeMask(sb, dz_m=dz_m)
    grad = torch.ones((nz, ny, nx))
    sf.apply_to(grad)
    # Row 2 should be (0 for x<4, 1 for x>=4).
    np.testing.assert_array_equal(
        grad[2].numpy(),
        np.concatenate(
            [np.zeros((ny, 4), dtype=np.float32),
             np.ones((ny, 4), dtype=np.float32)],
            axis=-1,
        ),
    )
    # Row 5 onwards should be all-ones everywhere.
    assert torch.all(grad[5:] == 1.0)


def test_seabed_freeze_buffer_cells_extend_frozen_zone():
    nz, nx = 32, 16
    sb = np.full(nx, 250.0)  # row 10 cutoff
    sf = SeabedFreezeMask(sb, dz_m=25.0, buffer_cells=3)
    grad = torch.ones((nz, nx))
    sf.apply_to(grad)
    # Now rows 0..12 should be frozen (10 + 3 buffer).
    assert torch.all(grad[:13] == 0.0)
    assert torch.all(grad[13:] == 1.0)


def test_seabed_freeze_shape_mismatch_raises():
    sb = np.full(8, 100.0)  # nx=8
    sf = SeabedFreezeMask(sb, dz_m=10.0)
    bad_grad = torch.ones((16, 12))  # nx=12 != 8
    with pytest.raises(ValueError, match="1-D seabed"):
        sf.apply_to(bad_grad)
