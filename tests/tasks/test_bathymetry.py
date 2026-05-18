"""Tests for the bathymetry / seabed-depth helper module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_tasks.bathymetry import (
    derive_seabed_depth_from_vp,
    load_seabed_depth_npz,
    save_seabed_depth_npz,
    water_mask_from_seabed_depth,
)


def _synth_vp_3d(nz=20, ny=8, nx=12, depths_cells=None, water_vp=1500.0):
    """Synthetic vp with varying water depth per (y, x) column."""
    if depths_cells is None:
        rng = np.random.default_rng(0)
        depths_cells = rng.integers(2, 6, size=(ny, nx))
    vp = np.empty((nz, ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            d = int(depths_cells[j, i])
            vp[:d, j, i] = water_vp
            vp[d:, j, i] = 1600.0 + 30.0 * np.arange(nz - d, dtype=np.float32)
    return vp, depths_cells


def test_derive_seabed_depth_3d_round_trip():
    """Round-trip: synth vp w/ known depths_cells → derive → match."""
    vp, depths_cells = _synth_vp_3d()
    dh = 75.0
    out = derive_seabed_depth_from_vp(vp, dh_z_m=dh, water_vp=1500.0)
    assert out.shape == depths_cells.shape
    np.testing.assert_array_equal(
        out, (depths_cells * dh).astype(np.float32),
    )


def test_derive_seabed_depth_2d_round_trip():
    """2-D vp gives 1-D seabed_depth."""
    nz, nx = 20, 12
    depths_cells = np.array([3, 5, 2, 4, 6, 3, 3, 5, 2, 7, 4, 3], dtype=np.int64)
    vp = np.empty((nz, nx), dtype=np.float32)
    for i in range(nx):
        d = int(depths_cells[i])
        vp[:d, i] = 1500.0
        vp[d:, i] = 1700.0
    out = derive_seabed_depth_from_vp(vp, dh_z_m=50.0)
    assert out.shape == (nx,)
    np.testing.assert_array_equal(out, depths_cells * 50.0)


def test_derive_seabed_depth_all_water_column_gets_nz_depth():
    """Columns that are 1500 m/s all the way down → depth = nz * dh."""
    nz, ny, nx = 10, 3, 3
    vp = np.full((nz, ny, nx), 1500.0, dtype=np.float32)
    vp[5:, 0, 0] = 1700.0  # one column has rock
    out = derive_seabed_depth_from_vp(vp, dh_z_m=10.0)
    assert out[0, 0] == 5 * 10.0
    # All-water columns get nz * dh.
    assert np.all(out[0, 1:] == nz * 10.0)
    assert np.all(out[1:] == nz * 10.0)


def test_derive_seabed_depth_smoothed_init_via_atol():
    """A slightly-perturbed 1500 (e.g. from smoothing) still classifies
    as water when ``atol`` allows it."""
    vp, depths_cells = _synth_vp_3d()
    vp[:2] += 0.4  # bump water rows by 0.4 m/s
    # atol=1e-3 misses the perturbation -> wrong depth.
    out_tight = derive_seabed_depth_from_vp(vp, dh_z_m=10.0, atol=1e-3)
    assert not np.array_equal(out_tight, depths_cells * 10.0)
    # atol=1.0 catches it.
    out_loose = derive_seabed_depth_from_vp(vp, dh_z_m=10.0, atol=1.0)
    np.testing.assert_array_equal(out_loose, (depths_cells * 10.0).astype(np.float32))


def test_save_load_round_trip(tmp_path):
    depth = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
    p = save_seabed_depth_npz(tmp_path / "x.npz", depth)
    loaded = load_seabed_depth_npz(p)
    np.testing.assert_array_equal(loaded, depth)
    assert loaded.dtype == np.float32


def test_load_rejects_wrong_key(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, wrong_key=np.zeros(5))
    with pytest.raises(KeyError, match="seabed_depth"):
        load_seabed_depth_npz(p)


def test_water_mask_3d_broadcasts_per_column():
    """Each (y, x) column gets its own water-depth threshold."""
    depth = np.array([[150.0, 50.0], [250.0, 100.0]], dtype=np.float32)
    # nz=4, dh=100 -> z_centers = 50, 150, 250, 350
    mask = water_mask_from_seabed_depth(depth, nz=4, dh_z_m=100.0)
    assert mask.shape == (4, 2, 2)
    # depth=150 -> z_center<150 => z=0 (50) only.
    assert mask[0, 0, 0] and not mask[1, 0, 0]
    # depth=50 -> nothing below.
    assert not mask[0, 0, 1]
    # depth=250 -> z=0, z=1 (150).
    assert mask[0, 1, 0] and mask[1, 1, 0] and not mask[2, 1, 0]
    # depth=100 -> z=0 (50) only.
    assert mask[0, 1, 1] and not mask[1, 1, 1]


def test_water_mask_buffer_cells_extends_below():
    depth = np.array([[150.0]], dtype=np.float32)
    # nz=5, dh=100 -> z_centers = 50, 150, 250, 350, 450
    # buffer=0: only z=0 (center 50 < 150)
    m0 = water_mask_from_seabed_depth(depth, nz=5, dh_z_m=100.0, buffer_cells=0)
    assert int(m0[:, 0, 0].sum()) == 1
    # buffer=1: threshold 150+100=250 -> z=0 (50), z=1 (150) both < 250
    m1 = water_mask_from_seabed_depth(depth, nz=5, dh_z_m=100.0, buffer_cells=1)
    assert int(m1[:, 0, 0].sum()) == 2


def test_water_mask_1d():
    depth = np.array([100.0, 250.0, 50.0], dtype=np.float32)
    mask = water_mask_from_seabed_depth(depth, nz=4, dh_z_m=100.0)
    assert mask.shape == (4, 3)
    # First column: depth=100 -> z=0 (50 < 100).
    assert mask[0, 0] and not mask[1, 0]
    # Second column: depth=250 -> z=0 (50), z=1 (150).
    assert mask[0, 1] and mask[1, 1] and not mask[2, 1]
    # Third column: depth=50 -> z=0 not water.
    assert not mask[0, 2]


def test_water_mask_rejects_bad_shape():
    bad = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="seabed_depth must be"):
        water_mask_from_seabed_depth(bad, nz=5, dh_z_m=10.0)


def test_round_trip_equivalence_to_init_vp_equality():
    """Bathymetry-derived mask must equal the (init_vp == 1500) mask
    on every voxel for any clean init."""
    vp, _ = _synth_vp_3d(nz=15, ny=6, nx=8)
    dh = 25.0
    depth = derive_seabed_depth_from_vp(vp, dh_z_m=dh)
    mask = water_mask_from_seabed_depth(depth, nz=15, dh_z_m=dh)
    mask_from_vp = (vp == 1500.0)
    np.testing.assert_array_equal(mask, mask_from_vp)
