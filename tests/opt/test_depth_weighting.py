"""Tests for sweepx.opt.precond.DepthWeighting and Compose."""
import torch

from sweepx.opt.precond import (
    Compose,
    DepthWeighting,
    PseudoHessianPrecond,
)


def test_depth_weighting_increases_with_depth():
    """p=1.0 weights should equal (z / (nz-1))."""
    nz = 5
    grad = torch.ones(3, nz)
    param = torch.zeros(3, nz, requires_grad=True)
    pc = DepthWeighting(depth_axis=-1, p=1.0)
    pc.update([param])
    out = pc(grad, param)
    expected_row = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    for i in range(3):
        assert torch.allclose(out[i], expected_row, atol=1e-6)


def test_depth_weighting_negative_p_boosts_shallow():
    """p=-1 with z_max_strategy='unit' makes shallow weights larger."""
    nz = 4
    grad = torch.ones(nz)
    param = torch.zeros(nz, requires_grad=True)
    pc = DepthWeighting(depth_axis=-1, p=-1.0, z_max_strategy="unit")
    pc.update([param])
    out = pc(grad, param)
    # z=0 -> 0**(-1) = inf; expect a divide-by-zero-style explosion only at z=0.
    # Subsequent values: 1, 0.5, 0.333 — strictly decreasing for p=-1.
    assert torch.isinf(out[0]) or out[0].item() > 1e6
    for i in range(1, nz - 1):
        assert out[i] > out[i + 1]


def test_compose_chains_preconditioners():
    """Compose(PH, Depth) returns Depth(PH(grad)).

    Compose.update forwards every kwarg to its children, so the caller
    only needs to invoke composed.update(...) once with the full kwargs.
    """
    nz = 4
    s = torch.tensor([1.0, 1.0, 1.0, 1.0])
    r = torch.tensor([1.0, 2.0, 4.0, 8.0])
    grad = torch.ones(nz)
    param = torch.zeros(nz, requires_grad=True)

    ph = PseudoHessianPrecond(eps=1e-6, normalize=False)
    dw = DepthWeighting(p=1.0)
    composed = Compose(ph, dw)
    composed.update([param], source_illum=s, receiver_illum=r)

    expected = (grad / (s * r + ph.eps)) * torch.tensor([0.0, 1.0/3.0, 2.0/3.0, 1.0])
    assert torch.allclose(composed(grad, param), expected, atol=1e-6)


def test_depth_weighting_2d_param():
    """For (nx, nz) shape, weight broadcasts only over the depth axis."""
    nx, nz = 7, 4
    grad = torch.ones(nx, nz)
    param = torch.zeros(nx, nz, requires_grad=True)
    pc = DepthWeighting(depth_axis=-1, p=1.0)
    pc.update([param])
    out = pc(grad, param)
    # Every "trace" must look the same.
    for i in range(nx):
        assert torch.allclose(out[0], out[i])
    assert out[0, 0].item() == 0.0
    assert out[0, -1].item() == 1.0
