"""Tests for sweepx.opt.optim.AdamFWI."""
import pytest
import torch

from sweepx.opt.optim import AdamFWI


def test_adam_matches_torch_adam_when_no_extras():
    """variant='adam' with no bounds / mask / weight decay must match torch.optim.Adam."""
    torch.manual_seed(0)
    x_a = torch.tensor([1.0, -2.0, 0.5, 3.0], requires_grad=True)
    x_b = x_a.clone().detach().requires_grad_(True)
    opt_a = AdamFWI([x_a], lr=1e-2, variant="adam", eps=1e-8)
    opt_b = torch.optim.Adam([x_b], lr=1e-2, eps=1e-8)
    for _ in range(100):
        opt_a.zero_grad(); (x_a ** 2).sum().backward(); opt_a.step()
        opt_b.zero_grad(); (x_b ** 2).sum().backward(); opt_b.step()
    assert torch.allclose(x_a.detach(), x_b.detach(), atol=1e-6)


@pytest.mark.parametrize("variant", ["adam", "adamw", "radam", "nadam"])
def test_adam_variants_converge_on_quadratic(variant):
    """All variants must drive the quadratic loss toward zero."""
    x = torch.tensor([3.0, -2.0], requires_grad=True)
    opt = AdamFWI([x], lr=1e-1, variant=variant, eps=1e-8)
    for _ in range(500):
        opt.zero_grad()
        (x ** 2).sum().backward()
        opt.step()
    assert (x ** 2).sum().item() < 1e-6


def test_adam_bounds_clamp(bounded_quadratic_2d):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = AdamFWI([x], lr=1e-2, bounds=(lo, hi), eps=1e-8)
    for _ in range(2000):
        opt.zero_grad(); f(x).backward(); opt.step()
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()
    assert torch.allclose(x.detach(), x_star, atol=1e-2)


def test_adam_mask_zeros_gradient_updates():
    """Masked components must not move regardless of their gradient."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mask = torch.tensor([0.0, 1.0, 0.0])  # only middle component updates
    opt = AdamFWI([x], lr=0.5, mask=mask, eps=1e-8)
    initial = x.detach().clone()
    for _ in range(20):
        opt.zero_grad()
        (x ** 2).sum().backward()
        opt.step()
    assert x[0].item() == initial[0].item()
    assert x[2].item() == initial[2].item()
    assert x[1].item() != initial[1].item()


def test_adamw_decouples_weight_decay():
    """AdamW with weight_decay > 0 must shrink params even at zero gradient."""
    x = torch.tensor([2.0, -1.5], requires_grad=True)
    opt = AdamFWI([x], lr=0.1, variant="adamw", weight_decay=0.5, eps=1e-8)
    initial_norm = float(x.detach().norm())
    for _ in range(5):
        # Zero gradient: x.grad is set to zero, so AdamW should still shrink x via decoupled WD.
        opt.zero_grad()
        x.grad = torch.zeros_like(x)
        opt.step()
    assert float(x.detach().norm()) < initial_norm


def test_amsgrad_uses_max_v():
    """AMSGrad should produce a different trajectory than vanilla Adam with the same seed."""
    torch.manual_seed(42)
    x_a = torch.tensor([1.0, 2.0], requires_grad=True)
    x_b = x_a.clone().detach().requires_grad_(True)
    opt_a = AdamFWI([x_a], lr=0.1, eps=1e-8, amsgrad=False)
    opt_b = AdamFWI([x_b], lr=0.1, eps=1e-8, amsgrad=True)
    for _ in range(20):
        opt_a.zero_grad(); ((x_a - 1.0) ** 2).sum().backward(); opt_a.step()
        opt_b.zero_grad(); ((x_b - 1.0) ** 2).sum().backward(); opt_b.step()
    # The two trajectories should differ; AMSGrad is only equivalent if v never decreased.
    assert not torch.allclose(x_a.detach(), x_b.detach(), atol=1e-8)
