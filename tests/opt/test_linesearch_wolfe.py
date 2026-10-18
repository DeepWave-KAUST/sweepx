"""Tests for the strong-Wolfe line search."""
import pytest

from sweepx.opt.linesearch.wolfe import strong_wolfe


def test_wolfe_finds_minimum_of_simple_quadratic():
    """phi(alpha) = (alpha - 1)^2 -> minimum at alpha = 1."""

    def eval_fn(alpha):
        return (alpha - 1.0) ** 2, 2.0 * (alpha - 1.0)

    init_loss, init_dphi = eval_fn(0.0)
    alpha, phi_alpha = strong_wolfe(
        eval_fn, init_loss, init_dphi, alpha_init=0.5, c1=1e-4, c2=0.1
    )
    # Strong-Wolfe with c2=0.1 demands tight curvature -> alpha must be very close to 1.
    assert abs(alpha - 1.0) < 1e-2
    # And the loss must decrease.
    assert phi_alpha < init_loss


def test_wolfe_satisfies_armijo_and_curvature():
    """phi(alpha) = (alpha - 2)^2 -> minimum at alpha = 2."""

    def eval_fn(alpha):
        return (alpha - 2.0) ** 2, 2.0 * (alpha - 2.0)

    init_loss, init_dphi = eval_fn(0.0)
    c1, c2 = 1e-4, 0.9
    alpha, phi_alpha = strong_wolfe(
        eval_fn, init_loss, init_dphi, alpha_init=0.5, c1=c1, c2=c2
    )
    # Armijo (sufficient decrease).
    assert phi_alpha <= init_loss + c1 * alpha * init_dphi + 1e-12
    # Curvature.
    _, dphi_alpha = eval_fn(alpha)
    assert abs(dphi_alpha) <= -c2 * init_dphi + 1e-12


def test_wolfe_rejects_non_descent():
    def eval_fn(alpha):
        return alpha**2, 2.0 * alpha  # phi'(0) = 0

    with pytest.raises(ValueError):
        strong_wolfe(eval_fn, 0.0, 0.0)


def test_wolfe_rejects_bad_constants():
    def eval_fn(alpha):
        return -alpha, -1.0

    with pytest.raises(ValueError):
        strong_wolfe(eval_fn, 0.0, -1.0, c1=0.5, c2=0.1)


def test_wolfe_caps_at_alpha_max():
    """When the minimum lies beyond alpha_max, the search returns alpha_max."""

    def eval_fn(alpha):
        # Minimum at alpha = 100.
        return (alpha - 100.0) ** 2, 2.0 * (alpha - 100.0)

    init_loss, init_dphi = eval_fn(0.0)
    alpha, _ = strong_wolfe(
        eval_fn, init_loss, init_dphi, alpha_init=1.0, alpha_max=5.0
    )
    assert alpha <= 5.0
