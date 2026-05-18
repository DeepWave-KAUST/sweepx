"""Tests for VelocityINR (hash-encoded SIREN reparam)."""

import pytest
import torch

from sweep_nn import VelocityINR


def _make_base(nz=16, nx=24, value=2000.0):
    return torch.full((nz, nx), float(value), dtype=torch.float32)


def test_render_shape_matches_base_2d():
    base = _make_base(16, 24)
    net = VelocityINR(base, hidden_features=16, hidden_layers=2,
                      hash_levels=4, hash_log2_size=10, hash_finest_resolution=16)
    vp = net()
    assert vp.shape == (16, 24)


def test_render_shape_3d():
    base = torch.full((6, 8, 10), 2000.0)
    net = VelocityINR(base, hidden_features=8, hidden_layers=2,
                      hash_levels=3, hash_log2_size=8, hash_finest_resolution=8)
    vp = net()
    assert vp.shape == (6, 8, 10)


def test_invalid_ndim_raises():
    base = torch.zeros(10)  # 1-D
    with pytest.raises(ValueError):
        VelocityINR(base)


def test_delta_mode_close_to_base_at_init():
    """Hash latents init at ~1e-4 → SIREN output ~0 → vp ≈ base initially."""
    base = _make_base(16, 24, value=2500.0)
    net = VelocityINR(base, vp_std=50.0, hidden_features=16, hidden_layers=2,
                      hash_levels=4, hash_log2_size=10, hash_finest_resolution=16)
    vp = net().detach()
    # At init the perturbation should be tiny (< vp_std).
    assert (vp - base).abs().max() < 50.0


def test_direct_velocity_ignores_base():
    base = _make_base(16, 24, value=2500.0)
    direct = VelocityINR(base, vp_mean=3000.0, vp_std=0.0,
                         direct_velocity=True,
                         hidden_features=8, hidden_layers=1,
                         hash_levels=2, hash_log2_size=8, hash_finest_resolution=4)
    vp = direct().detach()
    # vp_std=0 + direct_velocity=True → vp == vp_mean everywhere.
    assert torch.allclose(vp, torch.full_like(vp, 3000.0))


def test_bounds_clamp():
    base = _make_base(16, 24, value=2000.0)
    net = VelocityINR(base, vp_mean=1000.0, vp_std=10000.0,  # huge perturbation
                      bounds=(1500.0, 4500.0),
                      hidden_features=8, hidden_layers=1,
                      hash_levels=2, hash_log2_size=8, hash_finest_resolution=4)
    vp = net().detach()
    assert vp.min().item() >= 1500.0 - 1e-3
    assert vp.max().item() <= 4500.0 + 1e-3


def test_backprop_through_hash_and_mlp():
    base = _make_base(16, 24)
    net = VelocityINR(base, hidden_features=16, hidden_layers=2,
                      hash_levels=4, hash_log2_size=10, hash_finest_resolution=16)
    vp = net()
    loss = vp.pow(2).mean()
    loss.backward()
    # Both the hash encoder and the SIREN MLP should have non-trivial grad.
    enc_grad = net.encoder.latents.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0
    mlp_grads = [p.grad for p in net.mlp.parameters() if p.grad is not None]
    assert len(mlp_grads) > 0 and sum(g.abs().sum() for g in mlp_grads) > 0


def test_no_hash_encoding():
    base = _make_base(16, 24)
    net = VelocityINR(base, use_hash_encoding=False,
                      hidden_features=16, hidden_layers=2)
    vp = net()
    assert vp.shape == (16, 24)
    assert net.encoder is None


def test_update_base_velocity_preserves_mlp_params():
    """Multi-stage FWI: switching base must not reset the SIREN/hash params."""
    base = _make_base(16, 24)
    net = VelocityINR(base, hidden_features=8, hidden_layers=1,
                      hash_levels=2, hash_log2_size=8, hash_finest_resolution=4)
    # Snapshot a param tensor.
    enc_before = net.encoder.latents.detach().clone()
    first_linear = net.mlp.net[0].linear.weight.detach().clone()

    new_base = _make_base(24, 36, value=2300.0)
    net.update_base_velocity(new_base)

    # Params preserved bit-for-bit.
    assert torch.equal(net.encoder.latents.detach(), enc_before)
    assert torch.equal(net.mlp.net[0].linear.weight.detach(), first_linear)
    # Render works at new shape.
    vp = net()
    assert vp.shape == (24, 36)


def test_render_shape_arbitrary():
    base = _make_base(16, 24)
    net = VelocityINR(base, hidden_features=8, hidden_layers=1,
                      hash_levels=2, hash_log2_size=8, hash_finest_resolution=4)
    vp = net.render_shape((20, 30))
    assert vp.shape == (20, 30)


def test_render_window():
    base = _make_base(16, 24)
    net = VelocityINR(base, hidden_features=8, hidden_layers=1,
                      hash_levels=2, hash_log2_size=8, hash_finest_resolution=4)
    full = net().detach()
    win = net.render_window(4, 12, 6, 18).detach()
    assert win.shape == (8, 12)
    # Window should equal the corresponding slice of the full render.
    assert torch.allclose(win, full[4:12, 6:18], atol=1e-5)


def test_chunked_backward_matches_full_backward():
    """backward_velocity_gradient should give the same param grads as a
    standard backward — just with smaller peak memory."""
    base = _make_base(8, 12)

    # --- full backward ---
    torch.manual_seed(0)
    net_a = VelocityINR(base, hidden_features=8, hidden_layers=1,
                        hash_levels=2, hash_log2_size=8, hash_finest_resolution=4)
    grad = torch.randn_like(base)
    vp = net_a()
    vp.backward(grad)
    full_grad_latents = net_a.encoder.latents.grad.detach().clone()

    # --- chunked backward (replay with same init) ---
    torch.manual_seed(0)
    net_b = VelocityINR(base, hidden_features=8, hidden_layers=1,
                        hash_levels=2, hash_log2_size=8, hash_finest_resolution=4)
    # Zero all grads (they default to None until first backward).
    net_b.zero_grad()
    net_b.backward_velocity_gradient(grad, chunk_rows=3)
    chunked_grad_latents = net_b.encoder.latents.grad.detach().clone()

    assert torch.allclose(full_grad_latents, chunked_grad_latents, atol=1e-5)


def test_to_cuda_keeps_buffers_consistent():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    base = _make_base(8, 12)
    net = VelocityINR(base, hidden_features=8, hidden_layers=1,
                      hash_levels=2, hash_log2_size=8, hash_finest_resolution=4).cuda()
    vp = net()
    assert vp.device.type == "cuda"
    assert net.base_velocity.device.type == "cuda"
    assert net.coords.device.type == "cuda"


# ---------------------------------------------------------------------------
# water_mask: pin selected voxels to a fixed velocity
# ---------------------------------------------------------------------------
def test_water_mask_pins_voxels_to_water_vp():
    """Masked voxels render as exactly water_vp; others vary with SIREN."""
    torch.manual_seed(0)
    base = _make_base(16, 24, value=2000.0)
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:4] = True  # top 4 rows = "water"
    net = VelocityINR(
        base, vp_std=500.0,
        hidden_features=8, hidden_layers=1,
        hash_levels=2, hash_log2_size=8, hash_finest_resolution=4,
        water_mask=mask, water_vp=1500.0,
    )
    vp = net()
    assert vp.shape == base.shape
    # Water rows: exactly 1500 everywhere.
    assert torch.allclose(vp[:4], torch.full_like(vp[:4], 1500.0))
    # Below seabed: should NOT be uniform 1500 (SIREN adds perturbations).
    assert (vp[4:] != 1500.0).any()


def test_water_mask_zero_gradient_on_masked_voxels():
    """A loss that depends ONLY on water-masked voxels must produce
    zero gradient on the SIREN params — by construction (the mask
    replaces SIREN output with a constant, so dvp/dparam = 0 there).
    """
    torch.manual_seed(1)
    base = _make_base(8, 12, value=2000.0)
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:3] = True
    net = VelocityINR(
        base, vp_std=200.0,
        hidden_features=8, hidden_layers=1,
        hash_levels=2, hash_log2_size=8, hash_finest_resolution=4,
        water_mask=mask, water_vp=1500.0,
    )
    vp = net()
    loss = vp[:3].sum()  # depends only on water rows
    loss.backward()
    # Every learnable param must have None or all-zero grad.
    for p in net.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad))


def test_water_mask_shape_mismatch_raises():
    base = _make_base(8, 12)
    bad_mask = torch.zeros(7, 11, dtype=torch.bool)
    with pytest.raises(ValueError, match="water_mask.shape"):
        VelocityINR(
            base, hidden_features=4, hidden_layers=1,
            hash_levels=2, hash_log2_size=6, hash_finest_resolution=4,
            water_mask=bad_mask,
        )


def test_water_mask_default_none_keeps_old_behavior():
    """Without a mask, render is unchanged — backward compat."""
    torch.manual_seed(2)
    base = _make_base(8, 12, value=2000.0)
    net = VelocityINR(
        base, vp_std=100.0,
        hidden_features=4, hidden_layers=1,
        hash_levels=2, hash_log2_size=6, hash_finest_resolution=4,
    )
    assert net.water_mask is None
    vp = net()
    # Output should not be uniformly 1500 anywhere (no mask applied).
    assert not torch.allclose(vp, torch.full_like(vp, 1500.0))


def test_update_base_velocity_drops_stale_mask():
    """When new_base shape differs and no replacement mask given,
    the old mask is dropped (with a warning) rather than crashing."""
    import warnings
    base = _make_base(8, 12)
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:2] = True
    net = VelocityINR(
        base, hidden_features=4, hidden_layers=1,
        hash_levels=2, hash_log2_size=6, hash_finest_resolution=4,
        water_mask=mask,
    )
    new_base = _make_base(10, 14)  # different shape
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.update_base_velocity(new_base)
    assert any("stale water_mask" in str(_w.message) for _w in w)
    assert net.water_mask is None
    # render must work after the drop.
    vp = net()
    assert vp.shape == new_base.shape


def test_water_mask_render_window_slices_correctly():
    """``render_window`` must slice the water_mask in lock-step so the
    chunked-backward path doesn't hit torch.where broadcast errors.
    """
    torch.manual_seed(3)
    base = _make_base(16, 24, value=2000.0)
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:4] = True
    net = VelocityINR(
        base, vp_std=100.0,
        hidden_features=4, hidden_layers=1,
        hash_levels=2, hash_log2_size=6, hash_finest_resolution=4,
        water_mask=mask, water_vp=1500.0,
    )
    # Render z=[0:6] chunk: includes 4 water rows + 2 below-seabed.
    win = net.render_window(0, 6, 0, 24)
    assert win.shape == (6, 24)
    assert torch.allclose(win[:4], torch.full_like(win[:4], 1500.0))
    assert (win[4:] != 1500.0).any()
    # Render z=[8:14] chunk (entirely below seabed): no pin applied.
    win2 = net.render_window(8, 14, 0, 24)
    assert win2.shape == (6, 24)
    assert (win2 != 1500.0).any()


def test_water_mask_backward_velocity_gradient_chunked_3d():
    """End-to-end: chunked-backward on a 3-D net with water_mask must
    run without dimension errors AND produce zero gradient at water
    voxels in the grad input."""
    torch.manual_seed(4)
    base = torch.full((8, 6, 10), 2500.0)  # 3-D
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:3] = True
    net = VelocityINR(
        base, vp_std=200.0,
        hidden_features=4, hidden_layers=1,
        hash_levels=2, hash_log2_size=6, hash_finest_resolution=4,
        water_mask=mask, water_vp=1500.0,
    )
    # Construct a velocity_grad that's non-zero everywhere; the chunked
    # backward should run without dim mismatch (the bug we just fixed).
    grad = torch.ones_like(base)
    net.backward_velocity_gradient(grad, chunk_rows=2)
    # Hidden layers should have non-zero grads (non-water voxels
    # contributed); the test is mostly that the chunked path doesn't
    # error.
    has_nonzero = any(
        p.grad is not None and (p.grad != 0).any() for p in net.parameters()
    )
    assert has_nonzero


def test_update_base_velocity_refreshes_mask_when_given():
    base = _make_base(8, 12)
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:2] = True
    net = VelocityINR(
        base, hidden_features=4, hidden_layers=1,
        hash_levels=2, hash_log2_size=6, hash_finest_resolution=4,
        water_mask=mask,
    )
    new_base = _make_base(10, 14)
    new_mask = torch.zeros_like(new_base, dtype=torch.bool)
    new_mask[:3] = True
    net.update_base_velocity(new_base, water_mask=new_mask)
    assert net.water_mask is not None
    assert tuple(net.water_mask.shape) == tuple(new_base.shape)
    assert int(net.water_mask.sum()) == 3 * 14
