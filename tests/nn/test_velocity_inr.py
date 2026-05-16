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
