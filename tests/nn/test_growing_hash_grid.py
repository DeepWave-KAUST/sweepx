"""Tests for the on-demand GrowingHashGrid encoder."""

import pytest
import torch

from sweep_nn import GrowingHashGrid, MultiResHashGrid

CFG = dict(n_features_per_level=2, log2_hashmap_size=12,
           base_resolution=8, finest_resolution=128)


def test_fixed_output_width_2d():
    g = GrowingHashGrid(dim=2, max_levels=8, initial_levels=2, **CFG)
    out = g(torch.rand(7, 11, 2))
    assert out.shape == (7, 11, 8 * 2)      # width = max_levels*F even with 2 grown
    assert g.output_dim == 16 and g.n_active == 2


def test_fixed_output_width_3d():
    g = GrowingHashGrid(dim=3, max_levels=6, initial_levels=3, **CFG)
    out = g(torch.rand(5, 3))
    assert out.shape == (5, 6 * 2) and g.n_active == 3


def test_ungrown_levels_emit_zero():
    g = GrowingHashGrid(dim=2, max_levels=8, initial_levels=2, **CFG)
    per_lvl = g(torch.rand(64, 2)).reshape(-1, 8, 2).abs().sum(0).sum(-1)
    assert (per_lvl[:2] > 0).all()          # grown levels active
    assert torch.allclose(per_lvl[2:], torch.zeros(6))  # ungrown -> exactly zero


def test_grow_adds_levels_and_params_width_constant():
    g = GrowingHashGrid(dim=3, max_levels=8, initial_levels=2, **CFG)
    p0 = g.n_params()
    new = g.grow(2)
    assert len(new) == 2 and g.n_active == 4 and g.n_params() > p0
    g.grow(4)
    assert g.n_active == 8
    assert g.grow(1) == [] and g.n_active == 8          # capped at max_levels
    assert g(torch.rand(10, 3)).shape[-1] == 8 * 2      # head width never changes


@pytest.mark.parametrize("dim,base,fin", [(2, 8, 128), (3, 4, 64),
                                          (3, [4, 6, 5], [64, 96, 80])])  # incl. anisotropic
def test_numerical_equivalence_when_fully_grown(dim, base, fin):
    L = 6
    cfg = dict(n_features_per_level=2, log2_hashmap_size=12,
               base_resolution=base, finest_resolution=fin)
    torch.manual_seed(0)
    g = GrowingHashGrid(dim=dim, max_levels=L, initial_levels=L, **cfg)
    ref = MultiResHashGrid(dim=dim, n_levels=L, **cfg)
    off = ref.offsets.tolist()
    with torch.no_grad():                    # copy the flat table into per-level params
        for i in range(L):
            seg = ref.latents[off[i]:off[i + 1]]
            assert seg.shape == g.latents[i].shape
            g.latents[i].copy_(seg)
    x = torch.rand(500, dim)
    assert (g(x) - ref(x)).abs().max().item() < 1e-6


def test_gradient_only_to_allocated_levels():
    g = GrowingHashGrid(dim=2, max_levels=8, initial_levels=4, **CFG)
    g(torch.rand(200, 2)).pow(2).sum().backward()
    assert len(g.latents) == 4
    for i in range(4):
        assert g.latents[i].grad is not None and g.latents[i].grad.abs().sum() > 0


def test_grow_then_optimizer_trains_new_level():
    torch.manual_seed(0)
    g = GrowingHashGrid(dim=2, max_levels=8, initial_levels=2, **CFG)
    opt = torch.optim.Adam(g.parameters(), lr=1e-2)
    x = torch.rand(500, 2)
    tgt = torch.rand(500, 8 * 2)
    for _ in range(10):
        opt.zero_grad(); (g(x) - tgt).pow(2).mean().backward(); opt.step()
    new = g.grow(1)                              # allocate level 2
    opt.add_param_group({"params": new})         # register with the live optimizer
    before = g.latents[2].detach().clone()
    for _ in range(10):
        opt.zero_grad(); (g(x) - tgt).pow(2).mean().backward(); opt.step()
    assert (g.latents[2] - before).abs().max().item() > 0  # new level actually trained


def test_grow_to_progress_schedule():
    N = 11
    g = GrowingHashGrid(dim=3, max_levels=8, initial_levels=2, **CFG)
    counts = []
    for e in range(N):
        g.grow_to_progress(e / (N - 1), base_levels=2, final_levels=8, warmup=0.0, ramp_end=1.0)
        counts.append(g.n_active)
    assert counts[0] == 2 and counts[-1] == 8                # base at start, final at end
    assert all(b >= a for a, b in zip(counts, counts[1:]))   # monotone (never ungrows)
    # final_levels cap is respected
    g2 = GrowingHashGrid(dim=3, max_levels=8, initial_levels=2, **CFG)
    for e in range(N):
        g2.grow_to_progress(e / (N - 1), base_levels=2, final_levels=5, warmup=0.0, ramp_end=1.0)
    assert g2.n_active == 5


def test_velocity_inr_hash_growing():
    from sweep_nn.velocity_inr import VelocityINR
    base = torch.full((8, 10, 12), 1500.0)
    net = VelocityINR(base, hash_growing=True, hash_levels=6, hash_c2f_base_levels=2,
                      hash_base_resolution=4, hash_finest_resolution=64, hash_log2_size=12,
                      hidden_features=16, hidden_layers=1, bounds=(1450.0, 5500.0))
    assert isinstance(net.encoder, GrowingHashGrid) and net.encoder.n_active == 2
    assert net.render().shape == base.shape                  # renders at the fixed width


def test_cross_run_resume_auto_grows_on_load():
    """Warm-start a higher band from a saved lower-frequency run: a freshly
    built grid loads a checkpoint that grew to more levels by auto-growing the
    ParameterList on load, then keeps growing for the new band."""
    g1 = GrowingHashGrid(dim=3, max_levels=8, initial_levels=7, **CFG)  # run1 grew to 7
    g2 = GrowingHashGrid(dim=3, max_levels=8, initial_levels=2, **CFG)  # run2 fresh
    g2.load_state_dict(g1.state_dict())                                  # auto-grows 2 -> 7
    assert g2.n_active == 7
    assert all(torch.equal(g1.latents[i], g2.latents[i]) for i in range(7))
    g2.grow(1)                                                           # continue for the next band
    assert g2.n_active == 8


def test_cross_run_resume_via_velocity_inr():
    from sweep_nn.velocity_inr import VelocityINR
    base = torch.full((8, 10, 12), 1500.0)
    mk = lambda init: VelocityINR(base, hash_growing=True, hash_levels=8,
                                  hash_c2f_base_levels=init, hash_base_resolution=4,
                                  hash_finest_resolution=64, hash_log2_size=12,
                                  hidden_features=16, hidden_layers=1, bounds=(1450.0, 5500.0))
    net1, net2 = mk(7), mk(2)
    net2.load_state_dict(net1.state_dict())      # recursive load auto-grows encoder to 7
    assert net2.encoder.n_active == 7
    assert net2.render().shape == base.shape


def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        GrowingHashGrid(dim=4)


def test_resolution_length_mismatch_raises():
    with pytest.raises(ValueError):
        GrowingHashGrid(dim=3, base_resolution=[4, 6])  # 2 entries for dim=3
