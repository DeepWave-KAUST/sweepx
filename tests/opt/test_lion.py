"""Tests for sweepx.opt.optim.Lion."""
import torch

from sweepx.opt.optim import Lion


def test_lion_converges_on_quadratic():
    x = torch.tensor([3.0, -2.0], requires_grad=True)
    opt = Lion([x], lr=1e-2)
    for _ in range(2000):
        opt.zero_grad()
        (x ** 2).sum().backward()
        opt.step()
    assert (x ** 2).sum().item() < 1e-3


def test_lion_state_uses_only_one_buffer():
    """Lion should hold only `m` per parameter (no v / v_max)."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    opt = Lion([x], lr=1e-2)
    opt.zero_grad(); (x ** 2).sum().backward(); opt.step()
    state = opt.state[x]
    assert set(state.keys()) == {"m"}


def test_lion_bounds_clamp(bounded_quadratic_2d):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = Lion([x], lr=1e-3, bounds=(lo, hi))
    for _ in range(5000):
        opt.zero_grad(); f(x).backward(); opt.step()
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()
    assert torch.allclose(x.detach(), x_star, atol=5e-2)


def test_lion_step_uses_sign():
    """Setting beta1 = 0 turns Lion into pure sign(grad), which is easy to verify."""
    x = torch.tensor([1.5, -3.0], requires_grad=True)
    opt = Lion([x], lr=0.1, betas=(0.0, 0.0))
    opt.zero_grad()
    x.grad = torch.tensor([1.0, -1.0])  # explicit gradient
    opt.step()
    # Update was -0.1 * sign(grad) = (-0.1, +0.1)
    assert torch.allclose(x.detach(), torch.tensor([1.4, -2.9]), atol=1e-6)
