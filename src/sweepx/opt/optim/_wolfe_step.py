"""Strong-Wolfe step along a search direction, shared by the line-search optimizers.

Every trial point is projected onto the optional ``bounds`` before it is
evaluated, the parameters are left at the projected ``x0 + alpha* d`` the line
search returns (not at its last trial), and a step that leaves the parameters
unchanged is reported instead of passing silently.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from sweepx.opt._utils import write_flat_to_params
from sweepx.opt.linesearch.wolfe import strong_wolfe

# Appended to the warnings of the optimizers that take ``initial_step``, when
# it is not set.
INITIAL_STEP_HINT = (
    "alpha multiplies a direction built from the raw gradient; for parameters in "
    "physical units with tiny gradients (e.g. a velocity model in m/s) set "
    "initial_step to the size of the first trial step in parameter units, e.g. "
    "initial_step=10.0 for 10 m/s."
)
# ... and to the alpha_max warning when it is set.
BRACKET_HINT = "Raise alpha_max (or initial_step) to allow longer trial steps."


def check_initial_step(initial_step: Optional[float]) -> Optional[float]:
    if initial_step is None:
        return None
    initial_step = float(initial_step)
    if not (math.isfinite(initial_step) and initial_step > 0.0):
        raise ValueError(f"initial_step must be a positive number or None, got {initial_step}")
    return initial_step


def scaled_bracket(initial_step: float, d: Tensor, alpha_init: float,
                   alpha_max: float) -> Tuple[float, float]:
    """``(alpha_0, alpha_max)`` with ``max|alpha_0 d| == initial_step``.

    The expansion cap keeps the caller's ``alpha_max / alpha_init`` ratio.
    """
    d_max = float(d.abs().max())
    if not (math.isfinite(d_max) and d_max > 0.0):
        return alpha_init, alpha_max
    alpha0 = initial_step / d_max
    return alpha0, alpha0 * (alpha_max / alpha_init)


def wolfe_step(
    param_groups,
    evaluate: Callable[[], Tuple[float, Tensor]],
    x0: Tensor,
    d: Tensor,
    init_loss: float,
    init_dphi: float,
    *,
    bounds: Optional[Tuple[float, float]],
    alpha_init: float,
    alpha_max: float,
    c1: float,
    c2: float,
    max_iter: int,
    owner: str,
    step_hints: bool = False,
    initial_step: Optional[float] = None,
) -> float:
    """Run the strong-Wolfe search along ``d`` from ``x0``; return ``alpha*``.

    ``evaluate()`` computes the loss and the flat gradient at the parameters
    currently written into ``param_groups``. ``step_hints`` adds advice on
    ``initial_step`` / ``alpha_max`` to the warnings (for the optimizers that
    take ``initial_step``).
    """

    def trial(alpha: float) -> Tensor:
        x = torch.add(x0, d, alpha=alpha)
        if bounds is not None:
            x.clamp_(bounds[0], bounds[1])
        return x

    def eval_alpha(alpha: float):
        write_flat_to_params(param_groups, trial(alpha))
        loss, grad = evaluate()
        return loss, float(torch.dot(grad, d))

    alpha_star, _ = strong_wolfe(
        eval_alpha, init_loss, init_dphi,
        alpha_init=alpha_init, alpha_max=alpha_max,
        c1=c1, c2=c2, max_iter=max_iter,
    )
    x_star = trial(alpha_star)
    write_flat_to_params(param_groups, x_star)
    d_max = float(d.abs().max())
    if torch.equal(x_star, x0):
        first = f"alpha={alpha_init:.3g}, changes them by at most {alpha_init * d_max:.3g}"
        if torch.equal(torch.add(x0, d, alpha=alpha_init), x0):
            reason = f"the first trial step ({first}) is below their floating-point resolution"
        elif bounds is not None and torch.equal(trial(alpha_init), x0):
            reason = None  # the box blocks every moving component: nothing to report
        else:
            reason = f"no trial step decreased the loss (the first trial, {first})"
        if reason is not None:
            hint = INITIAL_STEP_HINT if step_hints and initial_step is None else ""
            warnings.warn(
                f"{owner}: the line search left the parameters unchanged: {reason}. "
                f"{hint}".rstrip(),
                RuntimeWarning,
                stacklevel=3,
            )
    elif alpha_star >= alpha_max * (1.0 - 1e-12):
        hint = (INITIAL_STEP_HINT if initial_step is None else BRACKET_HINT) if step_hints else ""
        warnings.warn(
            f"{owner}: the line search stopped at alpha_max={alpha_max:.3g} with the loss "
            f"still decreasing steeply, so the step (at most {alpha_star * d_max:.3g} per "
            f"parameter) is limited by the step bracket. {hint}".rstrip(),
            RuntimeWarning,
            stacklevel=3,
        )
    return alpha_star
