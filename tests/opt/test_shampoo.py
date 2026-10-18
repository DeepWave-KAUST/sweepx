"""Tests for sweep_opt.optim.Shampoo."""
import torch

from sweep_opt.optim import Shampoo


def test_shampoo_converges_on_quadratic():
    """A vector is full-matrix AdaGrad (exponent -1/2), whose step norm is ~lr."""
    x = torch.tensor([3.0, -2.0, 1.5], requires_grad=True)
    opt = Shampoo([x], lr=0.5)
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


def _one_step(opt_cls, p0, grad, **kw):
    p = p0.clone().requires_grad_(True)
    opt = opt_cls([p], **kw)
    p.grad = grad.clone()
    opt.step()
    return (p.detach() - p0)


def test_shampoo_per_group_lr_and_scheduler():
    torch.manual_seed(0)
    a, b = torch.zeros(4, 5, requires_grad=True), torch.zeros(4, 5, requires_grad=True)
    opt = Shampoo([{"params": [a], "lr": 1e-2}, {"params": [b], "lr": 2e-2}], update_freq=1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.0 + 1e-30)
    a0, b0 = a.detach().clone(), b.detach().clone()
    g = torch.randn(4, 5)
    a.grad, b.grad = g.clone(), g.clone()
    opt.step()
    da, db = a.detach() - a0, b.detach() - b0
    assert torch.allclose(db, 2 * da, rtol=1e-5, atol=0)
    sched.step()                                   # lr -> ~0: the next step must not move
    a1 = a.detach().clone()
    a.grad = g.clone()
    opt.step()
    assert torch.allclose(a.detach(), a1, atol=1e-20)


def test_shampoo_eps_is_relative_to_gradient_scale():
    """FWI gradients have |G G^T| ~ 1e-10; the update must not depend on that scale."""
    torch.manual_seed(1)
    p0, g = torch.randn(6, 5), torch.randn(6, 5)
    d_big = _one_step(Shampoo, p0, g, lr=1e-2, update_freq=1)
    d_small = _one_step(Shampoo, p0, g * 1e-6, lr=1e-2, update_freq=1)
    assert torch.allclose(d_big, d_small, rtol=1e-4, atol=1e-7)


def test_shampoo_large_vector_falls_back_to_diagonal():
    """A (1e5, 1) hash table must not allocate a 1e5 x 1e5 factor."""
    n = 100_000
    p = torch.zeros(n, 1, requires_grad=True)
    opt = Shampoo([p], lr=1e-2, eps=1e-6, max_precond_dim=4096)
    g = torch.randn(n, 1)
    p.grad = g.clone()
    opt.step()
    st = opt.state[p]
    assert st["L"] is None and st["R"] is None
    expect = -1e-2 * g / (g * g + 1e-6 * (g * g).max()).sqrt()
    assert torch.allclose(p.detach(), expect, rtol=1e-5, atol=1e-9)


def test_shampoo_vector_uses_square_root():
    """With one factored axis the exponent is -1/2: a single step is g / |g|."""
    g = torch.tensor([[3.0], [4.0]])
    d = _one_step(Shampoo, torch.zeros(2, 1), g, lr=1.0)
    assert torch.allclose(d, -g / 5.0, rtol=1e-3)


def test_shampoo_survives_eigh_failure(monkeypatch):
    torch.manual_seed(2)
    p = torch.randn(4, 5, requires_grad=True)
    opt = Shampoo([p], lr=1e-2, update_freq=1)
    p.grad = torch.randn(4, 5)
    opt.step()
    root = opt.state[p]["L_inv_root"].clone()

    def boom(*a, **k):
        raise torch.linalg.LinAlgError("forced")
    monkeypatch.setattr(torch.linalg, "eigh", boom)
    p.grad = torch.randn(4, 5)
    opt.step()                                     # must not raise
    assert torch.equal(opt.state[p]["L_inv_root"], root)
    assert torch.isfinite(p).all()


def test_shampoo_lazy_refresh_is_stable_with_larger_eps():
    torch.manual_seed(0)
    M = torch.randn(4, 5, requires_grad=True)
    T = torch.randn(4, 5)
    opt = Shampoo([M], lr=1.0, update_freq=5, eps=1e-2)
    for _ in range(500):
        opt.zero_grad()
        ((M - T) ** 2).sum().backward()
        opt.step()
    assert ((M - T) ** 2).sum().item() < 1e-6
