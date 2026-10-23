"""Line-search safety shared by LBFGS, LBFGSB, NCG, SteepestDescent, TN and TGN:
non-finite trials are rejected, trials are projected onto the bounds, and the
parameters end at the step the search returned."""
import math

import pytest
import torch

from sweepx.opt.optim import (
    LBFGS,
    LBFGSB,
    NCG,
    SteepestDescent,
    TruncatedGaussNewton,
    TruncatedNewton,
)


def _cliff(x):
    """(x - 2)^2, NaN (value and gradient) for x > 1.5 - like a solver past its CFL limit."""
    return ((x - 2.0) ** 2).sum() + (0.0 * torch.sqrt(1.5 - x)).sum()


def _closure(f, x, opt, seen=None):
    def closure():
        opt.zero_grad()
        if seen is not None:
            seen.append(x.detach().clone())
        loss = f(x)
        loss.backward()
        return loss
    return closure


CLOSURE_OPTS = {
    "lbfgs": lambda x: LBFGS([x]),
    "ncg": lambda x: NCG([x]),
    "sd": lambda x: SteepestDescent([x], line_search="strong_wolfe"),
}


@pytest.mark.parametrize("kind", sorted(CLOSURE_OPTS))
def test_nonfinite_trial_is_rejected(kind):
    """From x = 0 the first trial (alpha = 1 along -g = 4) lands at x = 4: NaN."""
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = CLOSURE_OPTS[kind](x)
    for _ in range(5):
        opt.step(_closure(_cliff, x, opt))
        assert math.isfinite(float(x)) and float(x) <= 1.5
    assert float(_cliff(x.detach())) < 4.0


def test_tn_nonfinite_trial_is_rejected():
    """The Newton step from 0 lands at the NaN side (x = 2)."""
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = TruncatedNewton([x], forcing="constant", forcing_constant=1e-10)
    opt.step(lambda: _cliff(x))
    assert math.isfinite(float(x)) and 0.0 < float(x) <= 1.5


def test_tgn_nonfinite_trial_is_rejected():
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = TruncatedGaussNewton([x], forcing="constant", forcing_constant=1e-10)

    def residual_fn():
        return (x - 2.0) + 0.0 * torch.sqrt(1.5 - x)

    opt.step(residual_fn)
    assert math.isfinite(float(x)) and 0.0 < float(x) <= 1.5


@pytest.mark.parametrize("kind", ["lbfgs", "lbfgsb", "ncg", "sd"])
def test_trial_points_are_projected_onto_bounds(kind):
    """min (x - 3)^2 on [0, 1]: no trial may be evaluated outside the box."""
    x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    opt = {
        "lbfgs": lambda: LBFGS([x], bounds=(0.0, 1.0)),
        "lbfgsb": lambda: LBFGSB([x], bounds=(0.0, 1.0)),
        "ncg": lambda: NCG([x], bounds=(0.0, 1.0)),
        "sd": lambda: SteepestDescent([x], line_search="strong_wolfe", bounds=(0.0, 1.0)),
    }[kind]()
    seen = []
    for _ in range(3):
        opt.step(_closure(lambda y: ((y - 3.0) ** 2).sum(), x, opt, seen))
    assert all(0.0 <= float(v) <= 1.0 for v in seen)
    assert float(x) == 1.0


def test_tn_trial_points_are_projected_onto_bounds():
    x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    opt = TruncatedNewton([x], bounds=(0.0, 1.0))
    seen = []

    def loss_fn():
        seen.append(float(x))
        return ((x - 3.0) ** 2).sum()

    opt.step(loss_fn)
    assert all(0.0 <= v <= 1.0 for v in seen) and float(x) == 1.0


def test_parameters_end_at_returned_step_not_last_trial():
    """f = (x - 1)^2 from 0 with alpha_init = 10 and 2 search iterations: every
    trial (x = 20, 10, 5) is rejected, so x must stay at 0 (f = 1), not at the
    last rejected trial x = 5 (f = 16)."""
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = LBFGS([x], alpha_init=10.0, alpha_max=100.0, max_ls_iter=2)
    with pytest.warns(RuntimeWarning, match="unchanged"):
        opt.step(_closure(lambda y: ((y - 1.0) ** 2).sum(), x, opt))
    assert float(x) == 0.0


def test_nonfinite_start_raises():
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)  # on the NaN side
    opt = LBFGS([x])
    with pytest.raises(ValueError, match="not finite"):
        opt.step(_closure(_cliff, x, opt))
    tn = TruncatedNewton([x])
    with pytest.raises(RuntimeError, match="not finite"):
        tn.step(lambda: _cliff(x))
