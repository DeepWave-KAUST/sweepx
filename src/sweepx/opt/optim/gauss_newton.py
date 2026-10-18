"""Truncated Gauss-Newton optimizer for non-linear least squares.

For ``L(theta) = 0.5 ||r(theta)||^2`` with residual ``r`` and Jacobian
``J = dr/dtheta``, the Gauss-Newton matrix ``J^T J`` approximates the
Hessian by dropping the second-derivative-of-residual term ``sum r_i H_i``.
``J^T J`` is positive semidefinite, which is convenient for inner CG.

Truncated GN is the standard Newton-type method for FWI's least-squares
data misfit (Metivier et al. 2013). It is matrix-free: each inner CG iter
needs one ``J v`` (a JVP) and one ``J^T u`` (a VJP), both available via
PyTorch autograd.

References:
    Metivier, L., Brossier, R., Virieux, J., and Operto, S. (2013).
    "Full waveform inversion and the truncated Newton method." SIAM Journal
    on Scientific Computing 35(2), B401-B437.
    DOI:10.1137/120877854.
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


class TruncatedGaussNewton(Optimizer):
    """Truncated Gauss-Newton optimizer with PCG inner solver and Wolfe line search.

    Args:
        params: iterable of parameter tensors / param-groups.
        bounds: optional ``(lo, hi)`` box-projection enforced after each step.
        preconditioner: optional callable used inside the inner CG.
        damping: Levenberg-Marquardt damping added to ``J^T J``: solves
            ``(J^T J + damping I) d = -g``. Defaults to 0.0 (pure GN).
        forcing: ``"eisenstat_walker"`` or ``"constant"``.
        forcing_constant: tolerance used when ``forcing="constant"``.
        max_inner_iter: PCG iteration cap per outer step.
        c1, c2, alpha_init, alpha_max, max_ls_iter: Wolfe parameters.

    Calling convention:
        ``opt.step(residual_fn)`` where ``residual_fn`` returns the residual
        tensor ``r(theta)`` (NOT a scalar loss). The optimizer forms
        ``L = 0.5 ||r||^2`` and ``g = J^T r`` internally and uses ``J^T J``
        as the curvature matrix. **Do not call ``backward`` inside
        ``residual_fn``** - the autograd graph must remain intact for the
        JVP / VJP machinery.
    """

    def __init__(
        self,
        params,
        *,
        bounds: Optional[Tuple[float, float]] = None,
        preconditioner=None,
        damping: float = 0.0,
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
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        super().__init__(params, dict())
        self.bounds = bounds
        self.preconditioner = preconditioner
        self.damping = damping
        self.forcing = forcing
        self.forcing_constant = forcing_constant
        self.max_inner_iter = max_inner_iter
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.max_ls_iter = max_ls_iter
        self._last_inner_iters = 0
        self._last_forcing = 0.0

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
    def step(self, residual_fn: Callable[[], Tensor] = None):  # type: ignore[override]
        if residual_fn is None:
            raise RuntimeError(
                "TruncatedGaussNewton.step requires a residual_fn returning r(theta) "
                "WITHOUT calling backward."
            )

        params = list(iter_params(self.param_groups))
        if not all(p.requires_grad for p in params):
            raise RuntimeError("All optimized parameters must require grad.")

        with torch.enable_grad():
            residual = residual_fn()
            if not residual.requires_grad:
                raise RuntimeError("residual must depend on params with requires_grad=True")
            loss = 0.5 * (residual ** 2).sum()
            # Gradient g = J^T r computed via standard backward on the loss.
            grads_list = torch.autograd.grad(
                loss, params, create_graph=False, retain_graph=True,
            )
        g_flat = self._flatten_list([g.detach() for g in grads_list]).clone()
        g_norm = float(g_flat.norm())
        if g_norm == 0.0:
            return loss

        # Build matvec: J^T J v + damping * v.
        # JVP J v computed as a double-backward through the residual (Pearlmutter trick),
        # then the VJP J^T (J v) brings us back to parameter space.
        upstream = torch.zeros_like(residual, requires_grad=True)
        adjoint_grads = torch.autograd.grad(
            residual, params, grad_outputs=upstream,
            create_graph=True, retain_graph=True,
        )

        def matvec(v_flat: Tensor) -> Tensor:
            v_list = self._unflatten(v_flat, params)
            with torch.enable_grad():
                inner_v = sum((g * v).sum() for g, v in zip(adjoint_grads, v_list))
                jvp = torch.autograd.grad(inner_v, upstream, retain_graph=True)[0]
                jt_jv = torch.autograd.grad(residual, params, grad_outputs=jvp,
                                            retain_graph=True)
            out = self._flatten_list([t.detach() for t in jt_jv])
            if self.damping > 0.0:
                out = out + self.damping * v_flat
            return out

        # Forcing tolerance.
        if self.forcing == "eisenstat_walker":
            tol = min(0.5, g_norm ** 0.5)
        else:
            tol = self.forcing_constant
        self._last_forcing = tol

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
            return loss

        # Drop the GN graph references.
        del adjoint_grads, upstream, residual

        init_loss = float(loss.detach())
        init_dphi = float(torch.dot(g_flat, d))
        if init_dphi >= 0.0:
            d = -g_flat
            init_dphi = float(torch.dot(g_flat, d))
        if init_dphi >= 0.0:
            return loss

        x0 = flat_params(self.param_groups).clone()

        def eval_alpha(alpha: float):
            write_flat_to_params(self.param_groups, x0 + alpha * d)
            with torch.enable_grad():
                r_a = residual_fn()
                ll_a = 0.5 * (r_a ** 2).sum()
                grads_a = torch.autograd.grad(ll_a, params)
            g_a_flat = self._flatten_list([g.detach() for g in grads_a])
            return float(ll_a.detach()), float(torch.dot(g_a_flat, d))

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
