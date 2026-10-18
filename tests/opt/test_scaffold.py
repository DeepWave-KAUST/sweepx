"""Smoke test that the package imports and fixtures load."""
import sweep_opt


def test_package_importable():
    assert sweep_opt.__version__ == "0.1.0"
    assert hasattr(sweep_opt, "optim")
    assert hasattr(sweep_opt, "precond")
    assert hasattr(sweep_opt, "parameterization")
    assert hasattr(sweep_opt, "linalg")
    assert hasattr(sweep_opt, "linesearch")


def test_fixtures_quadratic_2d_exist(quadratic_2d):
    f, x_init, x_star = quadratic_2d
    assert x_init.shape == (2,)
    assert x_star.shape == (2,)
    # The fixture's gradient at x_star should be ~0.
    x = x_star.clone().requires_grad_(True)
    loss = f(x)
    loss.backward()
    assert x.grad.abs().max().item() < 1e-6


def test_fixtures_rosenbrock_min(rosenbrock_2d):
    f, _, x_star = rosenbrock_2d
    assert f(x_star).item() < 1e-12
