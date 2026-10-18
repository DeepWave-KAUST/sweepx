"""Truncated Newton (Hessian-Free Newton) optimizer.

The Newton step ``H d = -g`` is solved approximately by inner PCG. Two
forcing sequences control how tightly the inner CG must converge:

- ``"eisenstat_walker"`` (default): ``eta_k = min(0.5, sqrt(||g_k||))``
  adapts the tolerance to gradient magnitude (Eisenstat & Walker 1996).
- ``"constant"``: a single user-supplied tolerance.

The outer step length is then chosen by strong-Wolfe.

Unlike the rest of sweepx.opt's optimizers, ``TruncatedNewton.step()`` takes
a ``loss_fn`` that **must not call backward** - the optimizer needs the
intact autograd graph to build Hessian-vector products. A standard
``zero_grad / forward / backward / return loss`` closure would discard the
graph too early.

References:
    Nash, S. G. (2000). "A survey of truncated-Newton methods."
    Journal of Computational and Applied Mathematics 124(1-2), 45-59.
    DOI:10.1016/S0377-0427(00)00426-X.

    Eisenstat, S. C., and Walker, H. F. (1996). "Choosing the forcing terms
    in an inexact Newton method." SIAM Journal on Scientific Computing
    17(1), 16-32. DOI:10.1137/0917003.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweepx.opt._utils import (
    flat_params,
    iter_params,
    write_flat_to_params,
)
from sweepx.opt.linalg.cg import cg
from sweepx.opt.linesearch.wolfe import strong_wolfe


class TruncatedNewton(Optimizer):
    """Truncated (Hessian-Free) Newton with PCG inner solver and Wolfe line search.

    Args:
        params: iterable of parameter tensors / param-groups.
        bounds: optional ``(lo, hi)`` box-projection enforced after each step.
        preconditioner: optional callable ``(grad, param) -> grad`` used as
            the inner-CG preconditioner. Same contract as in
            :mod:`sweepx.opt.precond`.
        forcing: ``"eisenstat_walker"`` or ``"constant"``.
        forcing_constant: tolerance used when ``forcing="constant"``.
        max_inner_iter: cap on PCG iterations per Newton step.
        c1, c2, alpha_init, alpha_max, max_ls_iter: Wolfe parameters.

    Calling convention:
        ``opt.step(loss_fn)`` where ``loss_fn`` takes no arguments and returns
        the loss as a tensor that requires_grad. **Do not call ``backward``
        inside ``loss_fn``** - the optimizer needs the graph intact.
    """

    def __init__(
        self,
        params,
        *,
        bounds: Optional[Tuple[float, float]] = None,
        preconditioner=None,
        forcing: str = "eisenstat_walker",
        forcing_constant: float = 1e-2,
        max_inner_iter: int = 50,
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_init: float = 1.0,
        alpha_max: float = 10.0,
        max_ls_iter: int = 25,
    ) -> None:
        if forcing not in ("eisenstat_walker", "constant"):
            raise ValueError(f"unknown forcing: {forcing!r}")
        if max_inner_iter < 1:
            raise ValueError("max_inner_iter must be >= 1")
        super().__init__(params, dict())
        self.bounds = bounds
        self.preconditioner = preconditioner
        self.forcing = forcing
        self.forcing_constant = forcing_constant
        self.max_inner_iter = max_inner_iter
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.max_ls_iter = max_ls_iter
        self._last_inner_iters: int = 0
        self._last_forcing: float = 0.0

    # ------------------------------------------------------------------ utils
    def _project(self) -> None:
        if self.bounds is None:
            return
        lo, hi = self.bounds
        for p in iter_params(self.param_groups):
            p.data.clamp_(lo, hi)

    def _flatten_list(self, tensors) -> Tensor:
        return torch.cat([t.reshape(-1) for t in tensors])

    def _unflatten(self, flat: Tensor, like) -> list:
        out = []
        offset = 0
        for t in like:
            n = t.numel()
            out.append(flat[offset : offset + n].view_as(t))
            offset += n
        return out

    def _flat_precond(self, r_flat: Tensor, params) -> Tensor:
        if self.preconditioner is None:
            return r_flat
        out = torch.empty_like(r_flat)
        offset = 0
        for p in params:
            n = p.numel()
            chunk = r_flat[offset : offset + n].view_as(p)
            out[offset : offset + n] = self.preconditioner(chunk, p).reshape(-1)
            offset += n
        return out

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, loss_fn: Callable[[], Tensor] = None):  # type: ignore[override]
        if loss_fn is None:
            raise RuntimeError(
                "TruncatedNewton.step requires a loss_fn that returns the loss "
                "WITHOUT calling backward (the optimizer needs the autograd graph)."
            )

        params = list(iter_params(self.param_groups))
        if not all(p.requires_grad for p in params):
            raise RuntimeError("All optimized parameters must require grad for HVP.")

        # 1. Loss + first gradient with create_graph=True for HVP.
        with torch.enable_grad():
            loss = loss_fn()
            grads_list = torch.autograd.grad(
                loss, params, create_graph=True, allow_unused=False,
            )
        g_flat = self._flatten_list([g.detach() for g in grads_list]).clone()
        g_norm = float(g_flat.norm())
        if g_norm == 0.0:
            return loss

        # 2. Build matvec for the inner CG: H @ v.
        def matvec(v_flat: Tensor) -> Tensor:
            v_list = self._unflatten(v_flat, params)
            with torch.enable_grad():
                inner = sum((g * v).sum() for g, v in zip(grads_list, v_list))
                hv = torch.autograd.grad(inner, params, retain_graph=True)
            return self._flatten_list([h.detach() for h in hv])

        # 3. Forcing tolerance.
        if self.forcing == "eisenstat_walker":
            tol = min(0.5, g_norm ** 0.5)
        else:
            tol = self.forcing_constant
        self._last_forcing = tol

        # 4. Solve H d = -g via PCG.
        precond_callable = (
            (lambda r: self._flat_precond(r, params)) if self.preconditioner else None
        )
        d, info = cg(
            matvec, -g_flat,
            tol=tol, max_iter=self.max_inner_iter,
            preconditioner=precond_callable,
        )
        self._last_inner_iters = info.iters

        if info.negative_curvature and float(d.norm()) == 0.0:
            # Could not produce any descent step at all; bail out.
            return loss

        # Drop the first-grad graph before doing the line search to free memory.
        del grads_list

        init_loss = float(loss.detach())
        init_dphi = float(torch.dot(g_flat, d))
        if init_dphi >= 0.0:
            d = -g_flat
            init_dphi = float(torch.dot(g_flat, d))
        if init_dphi >= 0.0:
            return loss

        # 5. Line search.
        x0 = flat_params(self.param_groups).clone()

        def eval_alpha(alpha: float):
            write_flat_to_params(self.param_groups, x0 + alpha * d)
            with torch.enable_grad():
                ll = loss_fn()
                grads_a = torch.autograd.grad(ll, params)
            g_a_flat = self._flatten_list([g.detach() for g in grads_a])
            return float(ll.detach()), float(torch.dot(g_a_flat, d))

        strong_wolfe(
            eval_alpha, init_loss, init_dphi,
            alpha_init=self.alpha_init,
            alpha_max=self.alpha_max,
            c1=self.c1, c2=self.c2,
            max_iter=self.max_ls_iter,
        )
        self._project()
        return loss

    # ----------------------------------------------------------- diagnostics
    @property
    def last_inner_iters(self) -> int:
        return self._last_inner_iters

    @property
    def last_forcing(self) -> float:
        return self._last_forcing
