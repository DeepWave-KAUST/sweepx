"""Sharpness-Aware Minimization (SAM) wrapper.

Wraps any base optimizer in two-step "ascent then descent" updates:

    1. e_w = rho * g / ||g||
    2. theta_temp = theta + e_w     (perturb to the worst nearby point)
    3. g_temp = grad(f) at theta_temp
    4. theta_new = base_optimizer step using g_temp at the original theta

The result is a step that minimizes the *sharpness* of the loss surface
in addition to the loss itself, which empirically finds flatter minima.

Reference:
    Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. (2021).
    "Sharpness-aware minimization for efficiently improving generalization."
    ICLR 2021. arXiv:2010.01412. https://arxiv.org/abs/2010.01412.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.optim.optimizer import Optimizer

from sweepx.opt._utils import flat_grads, iter_params


class SAM(Optimizer):
    """SAM wrapper around an arbitrary base optimizer.

    Args:
        base_optimizer: an instance of any optimizer (torch or sweep-opt).
            The same param-groups will be used; SAM intercepts the
            gradient before calling ``base_optimizer.step()``.
        rho: ascent radius. ``rho=0`` reduces SAM to the base optimizer.
        eps: numerical stability for the gradient normalization.

    Calling convention:
        ``opt.step(closure)`` where ``closure`` calls
        ``optimizer.zero_grad(); forward; loss.backward(); return loss``.
        SAM calls the closure **twice** per step (once for ``g``, once
        for ``g_temp``).
    """

    def __init__(self, base_optimizer: Optimizer, *, rho: float = 0.05,
                 eps: float = 1e-12) -> None:
        if rho < 0.0:
            raise ValueError("rho must be non-negative")
        # Borrow the base optimizer's param-groups so torch.optim's
        # introspection still works on us.
        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.eps = eps

    @torch.no_grad()
    def _ascent_step(self) -> dict:
        """Perturb params by e_w = rho * g / ||g||. Returns the saved e_w per param."""
        g_flat = flat_grads(self.param_groups)
        norm = float(g_flat.norm())
        if norm < self.eps:
            return {}
        scale = self.rho / (norm + self.eps)
        e_w = {}
        for p in iter_params(self.param_groups):
            if p.grad is None:
                continue
            ew_p = scale * p.grad
            p.data.add_(ew_p)
            e_w[id(p)] = ew_p
        return e_w

    @torch.no_grad()
    def _restore(self, e_w: dict) -> None:
        for p in iter_params(self.param_groups):
            ew_p = e_w.get(id(p))
            if ew_p is None:
                continue
            p.data.sub_(ew_p)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is None:
            raise RuntimeError("SAM requires a closure (called twice per step).")

        with torch.enable_grad():
            loss_orig = closure()
        if self.rho == 0.0:
            self.base_optimizer.step()
            return loss_orig

        e_w = self._ascent_step()
        if not e_w:
            self.base_optimizer.step()
            return loss_orig

        with torch.enable_grad():
            closure()  # Re-populate .grad at the perturbed point.
        self._restore(e_w)
        self.base_optimizer.step()
        return loss_orig
