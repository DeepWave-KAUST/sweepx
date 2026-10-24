"""Tests for the strong-Wolfe line search."""
import math

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


def test_wolfe_never_accepts_a_nonfinite_trial():
    """phi is NaN past alpha = 0.3 (an unstable solver); the search must shrink."""

    def eval_fn(alpha):
        if alpha > 0.3:
            return float("nan"), float("nan")
        return (alpha - 2.0) ** 2, 2.0 * (alpha - 2.0)

    init_loss, init_dphi = eval_fn(0.0)
    alpha, phi_alpha = strong_wolfe(eval_fn, init_loss, init_dphi)
    assert 0.0 < alpha <= 0.3
    assert math.isfinite(phi_alpha) and phi_alpha < init_loss


def test_wolfe_returns_zero_when_every_trial_is_nonfinite():
    def eval_fn(alpha):
        if alpha > 0.0:
            return float("inf"), float("nan")
        return 1.0, -1.0

    alpha, phi_alpha = strong_wolfe(eval_fn, 1.0, -1.0)
    assert alpha == 0.0 and phi_alpha == 1.0


def test_wolfe_rejects_nonfinite_start():
    def eval_fn(alpha):
        return 0.0, -1.0

    with pytest.raises(ValueError, match="finite"):
        strong_wolfe(eval_fn, float("nan"), -1.0)
    with pytest.raises(ValueError, match="finite"):
        strong_wolfe(eval_fn, 1.0, float("nan"))


def test_wolfe_out_of_iterations_returns_an_evaluated_step():
    """phi = -alpha never meets the curvature condition; with max_iter=3 the
    search evaluates alpha = 1, 2, 4 and must return 4, not the unevaluated 8."""
    seen = []

    def eval_fn(alpha):
        seen.append(alpha)
        return -alpha, -1.0

    alpha, phi_alpha = strong_wolfe(eval_fn, 0.0, -1.0, alpha_max=100.0, max_iter=3)
    assert alpha in seen and alpha == 4.0 and phi_alpha == -4.0


def test_wolfe_zoom_resolves_tiny_step_scales():
    """alpha* = 1e-20: the bracket width must be judged relative to alpha."""

    def eval_fn(alpha):
        t = alpha * 1e20 - 1.0
        return t * t, 2e20 * t

    init_loss, init_dphi = eval_fn(0.0)
    alpha, _ = strong_wolfe(eval_fn, init_loss, init_dphi, alpha_init=1e-21, c2=0.1)
    assert abs(alpha * 1e20 - 1.0) <= 0.1
