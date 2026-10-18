"""Steepest descent with optional bounds, preconditioner, and Wolfe line search."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweepx.opt._utils import (
    flat_grads,
    flat_params,
    iter_params,
    write_flat_to_params,
)
from sweepx.opt.linesearch.wolfe import strong_wolfe


class SteepestDescent(Optimizer):
    """Plain steepest descent (gradient descent) with FWI-friendly extras.

    Two operating modes share the same class:

    1. **No line search** (``line_search=None``): the standard
       ``x_{k+1} = x_k - lr * P g_k`` step. Closure is optional - if absent,
       the gradient already populated by the caller's external ``backward()``
       is consumed.
    2. **Strong-Wolfe line search** (``line_search="strong_wolfe"``): the step
       length is chosen by Nocedal & Wright Algorithms 3.5-3.6 along
       ``d = -P g_k``. A closure is required.

    Bounds, when supplied as ``bounds=(lo, hi)``, are enforced by an in-place
    ``clamp_(lo, hi)`` on every parameter after each update (projected gradient).

    A preconditioner, when supplied, must follow the
    :class:`sweepx.opt.precond.base.Preconditioner` contract: it is called
    per-parameter before the descent direction is formed.

    Args:
        params: iterable of parameter tensors (or torch param-groups).
        lr: step size used when ``line_search`` is ``None``.
        bounds: optional ``(lo, hi)`` box-projection enforced after each step.
        preconditioner: optional callable applied to the gradient before the
            step direction is computed.
        line_search: ``None`` for fixed-lr SD, or ``"strong_wolfe"`` for
            Wolfe-controlled step length.
        c1, c2, alpha_init, alpha_max, max_ls_iter: Wolfe parameters; ignored
            when ``line_search`` is ``None``.

    References:
        Nocedal, J., and Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
        Springer. DOI:10.1007/978-0-387-40065-5.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        *,
        bounds: Optional[Tuple[float, float]] = None,
        preconditioner=None,
        line_search: Optional[str] = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_init: float = 1.0,
        alpha_max: float = 10.0,
        max_ls_iter: int = 25,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if line_search not in (None, "strong_wolfe"):
            raise ValueError(f"unknown line_search: {line_search!r}")
        super().__init__(params, dict(lr=lr))
        self.bounds = bounds
        self.preconditioner = preconditioner
        self.line_search = line_search
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.max_ls_iter = max_ls_iter

    # ------------------------------------------------------------------ utils
    def _project(self) -> None:
        if self.bounds is None:
            return
        lo, hi = self.bounds
        for p in iter_params(self.param_groups):
            p.data.clamp_(lo, hi)

    def _flat_direction(self, total_numel: int) -> Tensor:
        """Return the flat descent direction d = -P g (or -g without precond)."""
        d = torch.empty(total_numel, dtype=next(iter_params(self.param_groups)).dtype,
                        device=next(iter_params(self.param_groups)).device)
        offset = 0
        for p in iter_params(self.param_groups):
            n = p.numel()
            if p.grad is None:
                d[offset : offset + n] = 0.0
            else:
                g = p.grad.detach()
                if self.preconditioner is not None:
                    g = self.preconditioner(g, p)
                d[offset : offset + n] = (-g).reshape(-1)
            offset += n
        return d

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform one optimization step."""
        if self.line_search is None:
            return self._fixed_lr_step(closure)
        return self._wolfe_step(closure)

    def _fixed_lr_step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lr = self.defaults["lr"]
        for p in iter_params(self.param_groups):
            if p.grad is None:
                continue
            g = p.grad
            if self.preconditioner is not None:
                g = self.preconditioner(g, p)
            p.data.add_(g, alpha=-lr)
        self._project()
        return loss

    def _wolfe_step(self, closure):
        if closure is None:
            raise RuntimeError(
                "SteepestDescent with line_search='strong_wolfe' requires a closure."
            )
        with torch.enable_grad():
            loss = closure()

        x0 = flat_params(self.param_groups).clone()
        n = x0.numel()
        d = self._flat_direction(n)
        g = flat_grads(self.param_groups)

        init_loss = float(loss.detach())
        init_dphi = float(torch.dot(g, d))
        if init_dphi >= 0.0:
            # Preconditioner produced an ascent direction; fall back to -g.
            d = -g
            init_dphi = float(torch.dot(g, d))
            if init_dphi >= 0.0:
                # Even -g is not descent (numerical zero gradient); skip step.
                return loss

        def eval_alpha(alpha: float):
            write_flat_to_params(self.param_groups, x0 + alpha * d)
            with torch.enable_grad():
                ll = closure()
            new_g = flat_grads(self.param_groups)
            return float(ll.detach()), float(torch.dot(new_g, d))

        alpha_star, _ = strong_wolfe(
            eval_alpha, init_loss, init_dphi,
            alpha_init=self.alpha_init,
            alpha_max=self.alpha_max,
            c1=self.c1, c2=self.c2,
            max_iter=self.max_ls_iter,
        )
        # eval_alpha already wrote params at the final alpha; just project.
        self._project()
        return loss
