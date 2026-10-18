"""Triton hash-encoding backend: numerical parity with the pure-PyTorch path.

Skipped unless triton is importable AND a CUDA device is present. Parity is checked by
toggling ``encoder.backend`` on a single instance (shared latents), so the comparison
isolates the kernel arithmetic. fp32 reduction-order differs, so we assert cos≈1 and a
loose rel-l2, not bit-equality.
"""
import pytest
import torch

from sweep_nn.hash_encoding import MultiResHashGrid
from sweep_nn.coarse_to_fine import CoarseToFineHashGrid
from sweep_nn.triton_hash_encoding import have_triton

pytestmark = pytest.mark.skipif(
    not (have_triton() and torch.cuda.is_available()),
    reason="requires triton and a CUDA device",
)

DEV = "cuda"


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()


def _rel(a, b):
    return (a - b).norm().item() / b.norm().clamp_min(1e-30).item()


CONFIGS = [
    # (dim, base, finest, log2) — spans equal & different per-axis growth, 2D/3D, large table
    (3, [16, 16, 16], [256, 256, 4096], 15),   # different per-axis growth (b_z != b_x)
    (3, [16, 16, 64], [512, 512, 2048], 15),   # equal growth, different base
    (2, [16, 32], [512, 1024], 15),            # anisotropic 2D
    (3, 16, 512, 15),                          # isotropic
    (3, [13, 10, 5], [420, 320, 160], 22),     # user's config: big table, log2=22
]


@pytest.mark.parametrize("dim,base,finest,log2", CONFIGS)
def test_forward_backward_parity(dim, base, finest, log2):
    torch.manual_seed(0)
    enc = MultiResHashGrid(dim=dim, n_levels=16, n_features_per_level=2,
                           log2_hashmap_size=log2, base_resolution=base,
                           finest_resolution=finest, backend="pytorch").to(DEV)
    pos = torch.rand(20000, dim, device=DEV)

    # forward parity
    enc.backend = "pytorch"
    out_pt = enc(pos)
    enc.backend = "triton"
    out_tri = enc(pos)
    assert _cos(out_tri, out_pt) > 0.999999
    assert _rel(out_tri, out_pt) < 1e-4

    # backward (dL/dlatent) parity
    g = torch.randn_like(out_pt)
    enc.backend = "pytorch"
    enc.latents.grad = None
    enc(pos).backward(g)
    grad_pt = enc.latents.grad.detach().clone()
    enc.backend = "triton"
    enc.latents.grad = None
    enc(pos).backward(g)
    grad_tri = enc.latents.grad.detach().clone()
    assert _cos(grad_tri, grad_pt) > 0.999999
    assert _rel(grad_tri, grad_pt) < 1e-4


def test_coarse_to_fine_parity_across_progress():
    """c2f mask must compose with the triton backend at every active-level count."""
    torch.manual_seed(0)
    enc = CoarseToFineHashGrid(dim=3, n_levels=16, n_features_per_level=2,
                               log2_hashmap_size=22, base_resolution=[13, 10, 5],
                               finest_resolution=[420, 320, 160], base_levels=2,
                               ramp="hard", backend="pytorch").to(DEV)
    pos = torch.rand(15000, 3, device=DEV)
    for alpha in (2, 6, 11, 16):
        enc.set_active_levels(float(alpha))
        enc.backend = "pytorch"
        out_pt = enc(pos)
        enc.backend = "triton"
        out_tri = enc(pos)
        assert _cos(out_tri, out_pt) > 0.999999, f"alpha={alpha}"
        assert _rel(out_tri, out_pt) < 1e-4, f"alpha={alpha}"


def test_velocity_inr_triton_backend_runs():
    from sweep_nn.velocity_inr import VelocityINR

    base = torch.linspace(1500, 4500, 16, device=DEV)[:, None, None].expand(16, 16, 16).contiguous()
    inr = VelocityINR(base_velocity=base, hidden_features=64, hidden_layers=2,
                      use_hash_encoding=True, hash_c2f=True, hash_levels=8,
                      hash_log2_size=18, hash_base_resolution=[8, 8, 4],
                      hash_finest_resolution=[128, 128, 256], hash_backend="triton").to(DEV)
    assert inr.encoder.backend == "triton"
    vp = inr.render()
    assert vp.shape == base.shape
    assert torch.isfinite(vp).all()


def test_growing_plus_triton_rejected():
    from sweep_nn.velocity_inr import VelocityINR

    base = torch.zeros(8, 8, 8, device=DEV)
    with pytest.raises(ValueError, match="not supported with hash_growing"):
        VelocityINR(base_velocity=base, use_hash_encoding=True, hash_growing=True,
                    hash_backend="triton").to(DEV)
