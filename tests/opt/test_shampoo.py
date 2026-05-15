"""Tests for sweep_opt.optim.Shampoo."""
import torch

from sweep_opt.optim import Shampoo


def test_shampoo_converges_on_quadratic():
    x = torch.tensor([3.0, -2.0, 1.5], requires_grad=True)
    opt = Shampoo([x], lr=1e-1, update_freq=5)
    for _ in range(500):
        opt.zero_grad()
        (x ** 2).sum().backward()
        opt.step()
    assert (x ** 2).sum().item() < 1e-3


def test_shampoo_2d_parameter_loss_decreases_substantially():
    """Shampoo's accumulated preconditioner has a sqrt(t) implicit lr decay; we
    verify substantial loss reduction rather than near-zero convergence."""
    torch.manual_seed(0)
    M = torch.randn(4, 5, requires_grad=True)
    M_target = torch.randn(4, 5)
    opt = Shampoo([M], lr=1e-1, update_freq=5)
    initial_loss = float(((M - M_target) ** 2).sum().detach())
    for _ in range(500):
        opt.zero_grad()
        ((M - M_target) ** 2).sum().backward()
        opt.step()
    final_loss = float(((M - M_target) ** 2).sum().detach())
    assert final_loss < 0.2 * initial_loss


def test_shampoo_bounds_clamp(bounded_quadratic_2d):
    f, x_init, x_star, (lo, hi) = bounded_quadratic_2d
    x = x_init.clone().detach().requires_grad_(True)
    opt = Shampoo([x], lr=1e-2, bounds=(lo, hi), update_freq=10)
    for _ in range(2000):
        opt.zero_grad(); f(x).backward(); opt.step()
    assert (x.detach() >= lo - 1e-12).all()
    assert (x.detach() <= hi + 1e-12).all()
