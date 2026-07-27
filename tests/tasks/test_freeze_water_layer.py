"""FreezeWaterLayerSpec water-column gradient mask (grid-mode freqsel path)."""
import numpy as np
import pytest
import torch

from sweep_tasks.bathymetry import water_mask_from_seabed_depth
from sweep_tasks.schemas import FreezeWaterLayerSpec


def test_water_mask_covers_water_column():
    ny, nx, nz, dh = 4, 5, 10, 75.0
    seabed = np.full((ny, nx), 225.0, np.float32)   # ~3 cells of water
    m = water_mask_from_seabed_depth(seabed, nz=nz, dh_z_m=dh)
    assert m.shape == (nz, ny, nx)
    assert m[:2].all()          # shallow rows are water
    assert not m[6:].any()      # deep rows are not


def test_freeze_zeros_water_gradient():
    ny, nx, nz, dh = 4, 5, 10, 75.0
    seabed = np.full((ny, nx), 225.0, np.float32)
    m = torch.from_numpy(
        water_mask_from_seabed_depth(seabed, nz=nz, dh_z_m=dh)).bool()
    g = torch.randn(nz, ny, nx)
    g[m] = 0.0
    assert (g[:2] == 0).all()
    assert (g[-1] != 0).any()


def test_spec_requires_path_when_enabled():
    with pytest.raises(Exception):
        FreezeWaterLayerSpec(enabled=True)          # missing seabed_depth_path
    FreezeWaterLayerSpec(enabled=False)             # fine
    FreezeWaterLayerSpec(enabled=True, seabed_depth_path="/tmp/x.npz")
