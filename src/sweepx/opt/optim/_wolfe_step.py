"""Strong-Wolfe step along a search direction, shared by the line-search optimizers.

Every trial point is projected onto the optional ``bounds`` before it is
evaluated, the parameters are left at the projected ``x0 + alpha* d`` the line
search returns (not at its last trial), and a step that leaves the parameters
unchanged is reported instead of passing silently.
"""
from __future__ import annotations

import math
import types
import warnings
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor

from sweepx.opt._utils import flat_grads, flat_params, iter_params, write_flat_to_params
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


def closure_evaluate(param_groups, closure) -> Callable[[], Tuple[Tensor, Tensor, tuple]]:
    """``evaluate()`` for :func:`wolfe_step` from a closure that fills ``.grad``.

    Returns ``(loss, flat_grad, grad_is_none)``; the last item records which
    parameters were left without a ``.grad``, so :class:`AcceptedGradCache`
    can restore ``.grad`` exactly as the closure left it.
    """

    def evaluate():
        with torch.enable_grad():
            loss = closure()
        none = tuple(p.grad is None for p in iter_params(param_groups))
        return loss.detach(), flat_grads(param_groups), none

    return evaluate


def _same_closure(a, b) -> bool:
    # Bound methods are rebuilt on every attribute access; they compare equal
    # when they bind the same function to the same object.
    return a is b or (isinstance(a, types.MethodType) and a == b)


class AcceptedGradCache:
    """Loss and gradient at the point the last line search accepted.

    The line search has already evaluated the closure at the point it accepts,
    so the next ``step()`` need not evaluate it again there. :meth:`load`
    reuses the stored result only when the parameters are bitwise equal to the
    stored point and the same closure is passed; anything else clears it.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._x: Optional[Tensor] = None
        self._loss: Optional[Tensor] = None
        self._grad: Optional[Tensor] = None
        self._none: Optional[tuple] = None
        self._closure = None

    def store(self, param_groups, closure, accepted) -> None:
        """Keep ``accepted`` (``closure_evaluate``'s result at the current
        parameters, or ``None``)."""
        if accepted is None:
            self.clear()
            return
        self._loss, self._grad, self._none = accepted
        self._x = flat_params(param_groups)
        self._closure = closure

    def load(self, param_groups, closure) -> Optional[Tensor]:
        """Write the stored gradient into ``.grad`` and return the stored loss,
        or return ``None`` (and clear) when it does not apply."""
        if self._x is None:
            return None
        if not (_same_closure(closure, self._closure)
                and torch.equal(flat_params(param_groups), self._x)):
            self.clear()
            return None
        offset = 0
        for p, none in zip(iter_params(param_groups), self._none):
            n = p.numel()
            if none:
                p.grad = None
            else:
                g = self._grad[offset : offset + n].view_as(p)
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)
            offset += n
        return self._loss


class AcceptedGradMixin:
    """``reuse_accepted_grad`` for the closure-based line-search optimizers.

    The line search evaluates the closure at the point it accepts, so with
    ``reuse_accepted_grad=True`` (the default) that loss and gradient become
    the next ``step()``'s starting loss and gradient instead of a second
    closure call there. This saves one closure call (a full forward and
    adjoint in FWI) per step. They are reused only when the next ``step()``
    gets the same closure object and the parameters are bitwise equal to the
    accepted point, and only when the accepted point was the line search's
    last trial; otherwise the closure is called as before. The closure's last
    call before a step is therefore always at the point the step starts from,
    so whatever the closure updates as a side effect (e.g. a preconditioner
    refreshed from the illumination of each backward) is the same as without
    reuse. With a deterministic closure the iterates are bitwise identical.

    The closure must also be a fixed function of the parameters while it
    is reused. A closure that draws a new shot batch on each call, or reads a
    frequency band or other setting that changes between steps, needs
    ``reuse_accepted_grad=False``, a new closure object per change, or
    :meth:`invalidate_grad_cache` after each change. The cache costs one extra
    copy of the parameters and the gradient.
    """

    def _init_grad_cache(self, reuse_accepted_grad: bool) -> None:
        self.reuse_accepted_grad = bool(reuse_accepted_grad)
        self._grad_cache = AcceptedGradCache()

    def invalidate_grad_cache(self) -> None:
        """Drop the stored loss and gradient, so the next ``step()`` calls the
        closure at its starting point. Call it after changing what the closure
        computes (shot batch, frequency band, data, loss) without changing the
        closure object or the parameters."""
        self._grad_cache.clear()

    def _start_loss(self, closure):
        """Loss at the current parameters with their gradient in ``.grad``."""
        if self.reuse_accepted_grad:
            loss = self._grad_cache.load(self.param_groups, closure)
            if loss is not None:
                return loss
        with torch.enable_grad():
            return closure()

    def _keep_accepted(self, closure, accepted) -> None:
        if self.reuse_accepted_grad:
            self._grad_cache.store(self.param_groups, closure, accepted)


def wolfe_step(
    param_groups,
    evaluate: Callable[[], Tuple[Any, ...]],
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
    keep_accepted: bool = False,
) -> Tuple[float, Optional[tuple]]:
    """Run the strong-Wolfe search along ``d`` from ``x0``.

    ``evaluate()`` computes the loss and the flat gradient at the parameters
    currently written into ``param_groups`` and returns them as the first two
    items of a tuple. ``step_hints`` adds advice on ``initial_step`` /
    ``alpha_max`` to the warnings (for the optimizers that take
    ``initial_step``).

    Returns ``(alpha*, accepted)``. With ``keep_accepted``, ``accepted`` is
    the tuple ``evaluate()`` returned at ``x0 + alpha* d`` when that was the
    last trial (the usual case), so the caller can reuse it as the next
    iterate's loss and gradient; it is ``None`` otherwise (``alpha* == 0``, or
    the search fell back to an earlier trial) and always without
    ``keep_accepted``.
    """

    def trial(alpha: float) -> Tensor:
        x = torch.add(x0, d, alpha=alpha)
        if bounds is not None:
            x.clamp_(bounds[0], bounds[1])
        return x

    last = None

    def eval_alpha(alpha: float):
        nonlocal last
        write_flat_to_params(param_groups, trial(alpha))
        out = evaluate()
        if keep_accepted:
            last = (alpha, out)
        return float(out[0]), float(torch.dot(out[1], d))

    alpha_star, _ = strong_wolfe(
        eval_alpha, init_loss, init_dphi,
        alpha_init=alpha_init, alpha_max=alpha_max,
        c1=c1, c2=c2, max_iter=max_iter,
    )
    # Only the last trial: then the closure's last call was at the accepted
    # point, as it would be if the caller called it there again.
    accepted = last[1] if last is not None and last[0] == alpha_star else None
    last = None
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
    return alpha_star, accepted
