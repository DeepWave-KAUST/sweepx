"""L-BFGS with projected bounds, restart, and an optional preconditioner H_0.

This is a from-scratch reimplementation of the limited-memory BFGS two-loop
recursion (Nocedal 1980; Nocedal & Wright 2006, Algorithm 7.4) that adds
three FWI-friendly knobs ``torch.optim.LBFGS`` does not expose:

1. ``bounds=(lo, hi)`` - projected gradient enforcement of velocity (or any)
   box constraints. The active-set Byrd-Lu-Nocedal L-BFGS-B variant is
   :class:`sweepx.opt.optim.LBFGSB`.
2. ``preconditioner`` - an optional preconditioner used as the initial
   inverse Hessian ``H_0`` inside the two-loop recursion. Pairs naturally
   with :class:`sweepx.opt.precond.PseudoHessianPrecond`.
3. ``restart_every`` - hard cold restart every ``N`` iterations (clear the
   stored ``(s, y)`` history). Useful for stochastic / source-encoded FWI
   loops where the curvature pairs become stale faster than the iterate.

References:
    Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
    Mathematics of Computation 35(151), 773-782.
    DOI:10.1090/S0025-5718-1980-0572855-7.

    Nocedal, J., and Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
    Springer, Section 7.2 and Algorithm 7.4.
    DOI:10.1007/978-0-387-40065-5.
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


class LBFGS(Optimizer):
    """Limited-memory BFGS with bounds, restart, and optional preconditioning.

    ``step()`` always requires a closure (matches :class:`torch.optim.LBFGS`).

    Args:
        params: iterable of parameter tensors / param-groups.
        history_size: number of ``(s, y)`` pairs to retain (m in N&W). Defaults
            to 10, which matches torch's default.
        bounds: optional ``(lo, hi)`` box-projection enforced after each step.
        preconditioner: optional callable ``(grad, param) -> grad`` used as
            the initial inverse Hessian ``H_0`` inside the two-loop
            recursion. Defaults to the standard Liu-Nocedal scalar
            ``gamma_k = s_{k-1} . y_{k-1} / y_{k-1} . y_{k-1}``.
        line_search: must be ``"strong_wolfe"``.
        c1, c2, alpha_init, alpha_max, max_ls_iter: Wolfe parameters
            (defaults are the standard quasi-Newton choice).
        restart_every: optional periodic cold restart.
        strict_descent: if True (default), reject steps whose initial
            ``phi'(0) >= 0`` and fall back to ``-g`` for that iteration.
    """

    def __init__(
        self,
        params,
        *,
        history_size: int = 10,
        bounds: Optional[Tuple[float, float]] = None,
        preconditioner=None,
        line_search: str = "strong_wolfe",
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_init: float = 1.0,
        alpha_max: float = 10.0,
        max_ls_iter: int = 25,
        restart_every: Optional[int] = None,
        strict_descent: bool = True,
    ) -> None:
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        if line_search != "strong_wolfe":
            raise ValueError(f"unsupported line_search: {line_search!r}")
        if restart_every is not None and restart_every <= 0:
            raise ValueError("restart_every must be positive or None")
        super().__init__(params, dict())
        self.history_size = history_size
        self.bounds = bounds
        self.preconditioner = preconditioner
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.max_ls_iter = max_ls_iter
        self.restart_every = restart_every
        self.strict_descent = strict_descent

        self._s_hist: Deque[Tensor] = deque(maxlen=history_size)
        self._y_hist: Deque[Tensor] = deque(maxlen=history_size)
        self._rho_hist: Deque[float] = deque(maxlen=history_size)
        self._iter: int = 0
        # Cached previous-iteration state used to form the next (s, y) pair.
        self._prev_x: Optional[Tensor] = None
        self._prev_g: Optional[Tensor] = None

    # ------------------------------------------------------------------ utils
    def _project(self) -> None:
        if self.bounds is None:
            return
        lo, hi = self.bounds
        for p in iter_params(self.param_groups):
            p.data.clamp_(lo, hi)

    def _apply_initial_inv_hessian(self, q_flat: Tensor) -> Tensor:
        """Return H_0 q. Either the preconditioner or the scalar gamma."""
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

    def _two_loop_direction(self, g_flat: Tensor) -> Tensor:
        """Return ``d = -H_k g`` via the L-BFGS two-loop recursion."""
        q = g_flat.clone()
        alphas = []
        # First loop: most-recent to oldest.
        for s, y, rho in zip(reversed(self._s_hist), reversed(self._y_hist),
                             reversed(self._rho_hist)):
            alpha = rho * float(torch.dot(s, q))
            q.add_(y, alpha=-alpha)
            alphas.append(alpha)
        r = self._apply_initial_inv_hessian(q)
        # Second loop: oldest to most-recent.
        for s, y, rho, alpha in zip(self._s_hist, self._y_hist, self._rho_hist,
                                    reversed(alphas)):
            beta = rho * float(torch.dot(y, r))
            r.add_(s, alpha=alpha - beta)
        return -r

    def _maybe_record_pair(self, x_now: Tensor, g_now: Tensor) -> None:
        """Form (s, y) from the previous iterate / gradient and append to history."""
        if self._prev_x is None or self._prev_g is None:
            return
        s = x_now - self._prev_x
        y = g_now - self._prev_g
        ys = float(torch.dot(s, y))
        if ys <= 1e-12:
            # Curvature condition violated; skip pair (BFGS standard practice).
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
            raise RuntimeError("LBFGS requires a closure (line search needs to evaluate phi).")

        with torch.enable_grad():
            loss = closure()

        x0 = flat_params(self.param_groups).clone()
        g = flat_grads(self.param_groups).clone()

        # Periodic cold restart: clear history, recompute direction as -H_0 g.
        if self.restart_every is not None and self._iter > 0 and self._iter % self.restart_every == 0:
            self._clear_history()

        # Form (s, y) from the previous iteration, if available.
        self._maybe_record_pair(x0, g)

        d = self._two_loop_direction(g)

        init_loss = float(loss.detach())
        init_dphi = float(torch.dot(g, d))

        if self.strict_descent and init_dphi >= 0.0:
            # Fall back to steepest-descent direction (preconditioned if available).
            self._clear_history()
            d = -self._apply_initial_inv_hessian(g)
            init_dphi = float(torch.dot(g, d))
            if init_dphi >= 0.0:
                d = -g
                init_dphi = float(torch.dot(g, d))
        if init_dphi >= 0.0:
            return loss

        def eval_alpha(alpha: float):
            write_flat_to_params(self.param_groups, x0 + alpha * d)
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
        self._project()

        # Cache iteration-k state so iteration-(k+1) can form (s_k, y_k).
        self._prev_x = x0
        self._prev_g = g
        self._iter += 1
        return loss

    # ----------------------------------------------------------- diagnostics
    @property
    def history_length(self) -> int:
        """Number of (s, y) pairs currently stored."""
        return len(self._s_hist)
