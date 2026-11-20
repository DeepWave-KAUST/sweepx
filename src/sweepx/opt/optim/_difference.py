"""Curvature products from first derivatives, for losses autograd cannot differentiate twice.

The ``'autograd'`` routes of the second-order optimizers differentiate a
gradient again (H v) or a vector-Jacobian product (Gauss-Newton's J v), so they
need a loss that autograd can differentiate twice -- with sweep, the uncompiled
eager propagator on the full tape.  The ``'difference'`` routes take central
differences of first-order evaluations at ``theta +- e v`` instead, and run
with everything a gradient runs with (sweep's ``impl='c'``, boundary saving,
the compiled eager step):

    H v ~ (g(theta + e v) - g(theta - e v)) / (2 e)     two gradients
    J v ~ (r(theta + e v) - r(theta - e v)) / (2 e)     two residuals, no gradient

``e`` changes no parameter by more than ``rel_step`` of its own largest
magnitude; the error is an O(e^2) truncation plus a float round-off of O(1/e).

A probe may leave ``bounds`` (an iterate on or next to a bound); a loss that is
defined there, as an FWI misfit is, keeps the central difference.  Where the
central product is not finite and one side stays inside the box, the product
is taken one-sided on that side instead, e.g. ``(g(theta + e v) - g(theta)) / e``
(O(e) accurate, one evaluation more).

The routes need a gradient (a residual for J v) that is a smooth function of
the parameters on the scale of ``e``.  sweep's qP equations (``AcousticVTI``,
``AcousticTTI`` and their Born pair) are not where epsilon differs from delta --
the difference of their gradients grows as the step shrinks, and
``sweep.autodiff`` refuses it -- so these routes do not serve them.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

ROUTES = ("autograd", "difference")


def check_route(name: str, value: str) -> str:
    if value not in ROUTES:
        raise ValueError(f"{name} must be one of {ROUTES}, got {value!r}")
    return value


def check_rel_step(value: float) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"rel_step must be positive, got {value!r}")
    return value


def difference_step(params: Sequence[Tensor], v_list: Sequence[Tensor], rel_step: float) -> Optional[float]:
    """The largest ``e`` that changes no parameter by more than ``rel_step`` of
    its own largest magnitude (a parameter that is all zero counts as 1);
    ``None`` for a zero direction."""
    steps = []
    for p, v in zip(params, v_list):
        vmax = float(v.detach().abs().max()) if v.numel() else 0.0
        if vmax == 0.0:
            continue
        pmax = float(p.detach().abs().max()) if p.numel() else 0.0
        steps.append(rel_step * (pmax if pmax > 0.0 else 1.0) / vmax)
    return min(steps) if steps else None


def _flat(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def _room(params: Sequence[Tensor], v_list: Sequence[Tensor], bounds, sign: float) -> float:
    """The largest ``t >= 0`` that keeps ``theta + sign * t * v`` inside
    ``bounds`` (scalars, or one value per coordinate of the flattened
    parameters, as the optimizers project them); 0 when ``theta`` is outside."""
    x, d = _flat(params), sign * _flat(v_list)
    room = math.inf
    for bound, out, below in ((bounds[1], d > 0, False), (bounds[0], d < 0, True)):
        if bound is None:
            continue
        b = torch.as_tensor(bound, dtype=x.dtype, device=x.device).expand_as(x)
        if bool(((x < b) if below else (x > b)).any()):
            return 0.0
        if bool(out.any()):
            room = min(room, float(((b - x)[out] / d[out]).min()))
    return max(room, 0.0)


def _finite(tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


@contextmanager
def shifted(params: Sequence[Tensor], shifts: Sequence[Tensor]):
    """The parameters at ``theta + shift`` inside the block, and restored
    bit for bit afterwards (from a copy, not by subtracting the shift)."""
    saved = [p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            for p, s in zip(params, shifts):
                p.add_(s)
        yield
    finally:
        with torch.no_grad():
            for p, x in zip(params, saved):
                p.copy_(x)


def gradient_difference(loss_fn: Callable[[], Tensor], params: Sequence[Tensor],
                        v_list: Sequence[Tensor], rel_step: float,
                        bounds: Optional[Tuple] = None) -> List[Tensor]:
    """``H v`` from two gradients of ``loss_fn``, one tensor per parameter."""
    e = difference_step(params, v_list, rel_step)
    if e is None:
        return [torch.zeros_like(p) for p in params]

    def gradient(sign):
        with shifted(params, [sign * e * v for v in v_list]):
            with torch.enable_grad():
                loss = loss_fn()
                return [g.detach() for g in torch.autograd.grad(loss, params)]

    grads = {sign: gradient(sign) for sign in (1.0, -1.0)}
    hv = [(a - b) / (2.0 * e) for a, b in zip(grads[1.0], grads[-1.0])]
    if bounds is None or _finite(hv):
        return hv
    for sign in (1.0, -1.0):                 # a probe outside the box where the loss is not finite
        if _finite(grads[sign]) and _room(params, v_list, bounds, sign) >= e:
            return [sign * (a - b) / e for a, b in zip(grads[sign], gradient(0.0))]
    return hv


def residual_difference(residual_fn: Callable[[], Tensor], params: Sequence[Tensor],
                        v_list: Sequence[Tensor], rel_step: float,
                        bounds: Optional[Tuple] = None) -> Optional[Tensor]:
    """``J v`` from two residuals (no gradient taken); ``None`` for a zero direction."""
    e = difference_step(params, v_list, rel_step)
    if e is None:
        return None

    def residual(sign):
        with shifted(params, [sign * e * v for v in v_list]):
            with torch.no_grad():
                return residual_fn().detach()

    residuals = {sign: residual(sign) for sign in (1.0, -1.0)}
    jv = (residuals[1.0] - residuals[-1.0]) / (2.0 * e)
    if bounds is None or _finite([jv]):
        return jv
    for sign in (1.0, -1.0):                 # a probe outside the box where the residual is not finite
        if _finite([residuals[sign]]) and _room(params, v_list, bounds, sign) >= e:
            return sign * (residuals[sign] - residual(0.0)) / e
    return jv
