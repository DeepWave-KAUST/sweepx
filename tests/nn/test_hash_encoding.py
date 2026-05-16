"""Tests for the multi-resolution hash-grid encoder."""

import pytest
import torch

from sweep_nn import MultiResHashGrid


def test_2d_output_shape():
    enc = MultiResHashGrid(dim=2, n_levels=4, n_features_per_level=2,
                           base_resolution=2, finest_resolution=16, log2_hashmap_size=10)
    coords = torch.rand(7, 11, 2)  # (..., 2) in [0, 1)
    out = enc(coords)
    assert out.shape == (7, 11, enc.output_dim)
    assert enc.output_dim == 4 * 2


def test_3d_output_shape():
    enc = MultiResHashGrid(dim=3, n_levels=3, n_features_per_level=2,
                           base_resolution=2, finest_resolution=8, log2_hashmap_size=10)
    coords = torch.rand(5, 3)
    out = enc(coords)
    assert out.shape == (5, enc.output_dim)
    assert enc.output_dim == 3 * 2


def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        MultiResHashGrid(dim=4)


def test_backprop_through_latents():
    enc = MultiResHashGrid(dim=2, n_levels=4, n_features_per_level=2,
                           base_resolution=2, finest_resolution=16, log2_hashmap_size=10)
    coords = torch.rand(20, 2)
    out = enc(coords)
    out.sum().backward()
    assert enc.latents.grad is not None
    assert enc.latents.grad.abs().sum() > 0


def test_first_hash_level_split():
    """Low-res levels use tiled indexing; high-res hit the hash table."""
    enc = MultiResHashGrid(dim=2, n_levels=8, n_features_per_level=2,
                           base_resolution=2, finest_resolution=128, log2_hashmap_size=8)
    # T = 256. The first few resolutions (2, ~3, ~5, ~8) have <= 256 cells
    # so they should be tiled; the higher levels hash.
    assert 0 < enc.first_hash_level < enc.L


def test_device_cuda_optional():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    enc = MultiResHashGrid(dim=2, n_levels=3, n_features_per_level=2,
                           base_resolution=2, finest_resolution=8, log2_hashmap_size=10).cuda()
    coords = torch.rand(5, 2, device="cuda")
    out = enc(coords)
    assert out.device.type == "cuda"
    assert out.shape == (5, enc.output_dim)


def test_coords_in_unit_range_stable():
    """Output should not NaN on coords at unit-interval boundaries."""
    enc = MultiResHashGrid(dim=2, n_levels=4, n_features_per_level=2,
                           base_resolution=2, finest_resolution=16, log2_hashmap_size=10)
    coords = torch.tensor([[0.0, 0.0], [0.999, 0.999], [0.5, 0.5]])
    out = enc(coords)
    assert torch.isfinite(out).all()
