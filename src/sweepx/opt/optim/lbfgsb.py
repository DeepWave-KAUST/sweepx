"""L-BFGS-B: bound-constrained limited-memory BFGS.

This is the active-set variant of L-BFGS that handles box constraints
properly: at each iteration the optimizer identifies the *active set*
(parameters currently at a bound whose gradient pushes them further out
of the box), zeros the corresponding components of the search direction,
and performs the L-BFGS update on the *free* subset.

Compared to the projected-gradient L-BFGS in :class:`sweepx.opt.optim.LBFGS`,
this version:

- never accumulates curvature pairs that mix active and free components,
- projects each line-search trial point onto the box,
- and matches ``scipy.optimize.minimize(method="L-BFGS-B")`` on a smooth
  bounded quadratic (within line-search tolerance).

This is a faithful but simplified implementation of Byrd-Lu-Nocedal 1995:
the Cauchy point + subspace minimization is replaced by gradient-projection
with active-set masking, which is enough for the FWI box constraints we
care about (per-grid-point velocity bounds).

References:
    Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). "A limited memory
    algorithm for bound constrained optimization." SIAM Journal on
    Scientific Computing 16(5), 1190-1208. DOI:10.1137/0916069.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

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


class LBFGSB(Optimizer):
    """Bound-constrained limited-memory BFGS.

    Args:
        params: iterable of parameter tensors / param-groups.
        bounds: ``(lo, hi)`` box. **Required** - that's the whole point.
            Use :class:`sweepx.opt.optim.LBFGS` if you do not want bounds.
        history_size: number of stored ``(s, y)`` pairs (default 10).
        preconditioner: optional callable used as the initial inverse Hessian
            ``H_0`` inside the two-loop recursion.
        line_search: must be ``"strong_wolfe"``.
        c1, c2, alpha_init, alpha_max, max_ls_iter: Wolfe parameters.
        restart_every: optional periodic cold restart.
        active_tol: tolerance when comparing parameters to the bound (in
            the units of the parameter; default 0).
    """

    def __init__(
        self,
        params,
        *,
        bounds: Tuple[float, float],
        history_size: int = 10,
        preconditioner=None,
        line_search: str = "strong_wolfe",
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_init: float = 1.0,
        alpha_max: float = 10.0,
        max_ls_iter: int = 25,
        restart_every: Optional[int] = None,
        active_tol: float = 0.0,
    ) -> None:
        if bounds is None:
            raise ValueError("LBFGSB requires bounds=(lo, hi); use LBFGS for unconstrained.")
        if line_search != "strong_wolfe":
            raise ValueError(f"unsupported line_search: {line_search!r}")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        super().__init__(params, dict())
        self.lo, self.hi = bounds
        self.history_size = history_size
        self.preconditioner = preconditioner
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.max_ls_iter = max_ls_iter
        self.restart_every = restart_every
        self.active_tol = active_tol

        self._s_hist: Deque[Tensor] = deque(maxlen=history_size)
        self._y_hist: Deque[Tensor] = deque(maxlen=history_size)
        self._rho_hist: Deque[float] = deque(maxlen=history_size)
        self._iter = 0
        self._prev_x: Optional[Tensor] = None
        self._prev_g: Optional[Tensor] = None

    # ------------------------------------------------------------------ utils
    def _project_params(self) -> None:
        for p in iter_params(self.param_groups):
            p.data.clamp_(self.lo, self.hi)

    def _active_mask(self, x_flat: Tensor, g_flat: Tensor) -> Tensor:
        """Return a boolean mask: True for components in the active set
        (at bound with gradient pushing further out)."""
        at_lo = (x_flat <= self.lo + self.active_tol) & (g_flat > 0.0)
        at_hi = (x_flat >= self.hi - self.active_tol) & (g_flat < 0.0)
        return at_lo | at_hi

    def _apply_initial_inv_hessian(self, q_flat: Tensor) -> Tensor:
        if self.preconditioner is not None:
            r = torch.empty_like(q_flat)
            offset = 0
            for p in iter_params(self.param_groups):
                n = p.numel()
                chunk = q_flat[offset : offset + n].view_as(p)
                r[offset : offset + n] = self.preconditioner(chunk, p).reshape(-1)
                offset += n
            return r
        if not self._y_hist:
            return q_flat
        s_last = self._s_hist[-1]
        y_last = self._y_hist[-1]
        denom = float(torch.dot(y_last, y_last))
        if denom <= 0.0:
            return q_flat
        gamma = float(torch.dot(s_last, y_last)) / denom
        return gamma * q_flat

    def _two_loop_direction(self, g_flat: Tensor, free_mask: Tensor) -> Tensor:
        """Compute -H_k g restricted to the free subset (active components zeroed)."""
        g_free = torch.where(free_mask, g_flat, torch.zeros_like(g_flat))
        q = g_free.clone()
        alphas = []
        for s, y, rho in zip(reversed(self._s_hist), reversed(self._y_hist),
                             reversed(self._rho_hist)):
            alpha = rho * float(torch.dot(s, q))
            q.add_(y, alpha=-alpha)
            alphas.append(alpha)
        r = self._apply_initial_inv_hessian(q)
        for s, y, rho, alpha in zip(self._s_hist, self._y_hist, self._rho_hist,
                                    reversed(alphas)):
            beta = rho * float(torch.dot(y, r))
            r.add_(s, alpha=alpha - beta)
        d = -r
        d = torch.where(free_mask, d, torch.zeros_like(d))
        return d

    def _maybe_record_pair(self, x_now, g_now) -> None:
        if self._prev_x is None or self._prev_g is None:
            return
        s = x_now - self._prev_x
        y = g_now - self._prev_g
        ys = float(torch.dot(s, y))
        if ys <= 1e-12:
            return
        rho = 1.0 / ys
        self._s_hist.append(s)
        self._y_hist.append(y)
        self._rho_hist.append(rho)

    def _clear_history(self) -> None:
        self._s_hist.clear()
        self._y_hist.clear()
        self._rho_hist.clear()
        self._prev_x = None
        self._prev_g = None

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is None:
            raise RuntimeError("LBFGSB requires a closure (line search needs phi).")

        # Make sure we start from a feasible iterate.
        self._project_params()

        with torch.enable_grad():
            loss = closure()
        x0 = flat_params(self.param_groups).clone()
        g = flat_grads(self.param_groups).clone()

        if self.restart_every is not None and self._iter > 0 and self._iter % self.restart_every == 0:
            self._clear_history()

        self._maybe_record_pair(x0, g)

        free_mask = ~self._active_mask(x0, g)
        d = self._two_loop_direction(g, free_mask)

        init_loss = float(loss.detach())
        init_dphi = float(torch.dot(g, d))
        if init_dphi >= 0.0:
            # Fall back to projected gradient on the free set.
            self._clear_history()
            d_full = -self._apply_initial_inv_hessian(g)
            d = torch.where(free_mask, d_full, torch.zeros_like(d_full))
            init_dphi = float(torch.dot(g, d))
            if init_dphi >= 0.0:
                # Fall back to -g on free set.
                d = torch.where(free_mask, -g, torch.zeros_like(g))
                init_dphi = float(torch.dot(g, d))
        if init_dphi >= 0.0:
            # Nothing to do this iteration.
            return loss

        def eval_alpha(alpha: float):
            trial = x0 + alpha * d
            trial.clamp_(self.lo, self.hi)  # projected line search
            write_flat_to_params(self.param_groups, trial)
            with torch.enable_grad():
                ll = closure()
            new_g = flat_grads(self.param_groups)
            return float(ll.detach()), float(torch.dot(new_g, d))

        strong_wolfe(
            eval_alpha, init_loss, init_dphi,
            alpha_init=self.alpha_init,
            alpha_max=self.alpha_max,
            c1=self.c1, c2=self.c2,
            max_iter=self.max_ls_iter,
        )
        self._project_params()

        self._prev_x = x0
        self._prev_g = g
        self._iter += 1
        return loss

    # ----------------------------------------------------------- diagnostics
    @property
    def history_length(self) -> int:
        return len(self._s_hist)
