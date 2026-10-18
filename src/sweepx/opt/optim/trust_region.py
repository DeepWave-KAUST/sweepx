"""Trust-region optimizer with Steihaug-CG (TTR) and adaptive subspace (ASTR) variants.

Two flavours share one outer driver:

- ``variant="ttr"``: solves the trust-region subproblem
  ``min 0.5 d^T H d + g^T d  subject to ||d|| <= delta``
  by Steihaug truncated CG. Default; what most "trust-region Newton"
  papers mean.
- ``variant="astr"``: solves the same subproblem on the 2-D subspace
  ``span(g, d_prev)`` (Wang & Rao 2015), where ``d_prev`` is the previous
  accepted step. Cheap (2-D analytic solve) and surprisingly competitive
  on FWI.

Both require Hessian-vector products through autograd (loss_fn must NOT
call backward), and both adjust the trust radius according to the standard
``rho = actual / predicted`` recipe (Nocedal & Wright Algorithm 4.1).

References:
    Steihaug, T. (1983). DOI:10.1137/0720042.
    Wang, Y., and Rao, Y. (2015). "Reflection seismic waveform tomography in
    transversely isotropic media." Geophysical Journal International 202(2),
    1362-1373. DOI:10.1093/gji/ggv226.
    Nocedal, J., and Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
    Algorithms 4.1 and 7.2. DOI:10.1007/978-0-387-40065-5.
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
from sweepx.opt.linalg.cg import steihaug_cg


class TrustRegion(Optimizer):
    """Trust-region Newton-type optimizer (TTR / ASTR).

    Args:
        params: iterable of parameter tensors / param-groups.
        variant: ``"ttr"`` (Steihaug-CG) or ``"astr"`` (2-D subspace).
        bounds: optional ``(lo, hi)`` projected box.
        preconditioner: optional callable used inside Steihaug-CG (TTR only).
        initial_radius: initial trust radius.
        max_radius: cap on trust radius.
        eta_acceptable: rho threshold above which the step is accepted.
        eta_increase: rho threshold above which the radius grows.
        radius_shrink: factor < 1 applied to delta on rejection.
        radius_grow: factor > 1 applied to delta on a "very good" step
            that hit the trust boundary.
        max_inner_iter: cap on Steihaug-CG iterations (TTR only).
        cg_tol: relative residual tolerance for Steihaug-CG (TTR only).

    Calling convention:
        ``opt.step(loss_fn)`` where ``loss_fn`` returns the loss tensor
        WITHOUT calling backward.
    """

    def __init__(
        self,
        params,
        *,
        variant: str = "ttr",
        bounds: Optional[Tuple[float, float]] = None,
        preconditioner=None,
        initial_radius: float = 1.0,
        max_radius: float = 100.0,
        eta_acceptable: float = 0.1,
        eta_increase: float = 0.75,
        radius_shrink: float = 0.25,
        radius_grow: float = 2.0,
        max_inner_iter: int = 50,
        cg_tol: float = 1e-2,
    ) -> None:
        if variant not in ("ttr", "astr"):
            raise ValueError(f"unknown variant: {variant!r}")
        if not (0.0 < initial_radius <= max_radius):
            raise ValueError("require 0 < initial_radius <= max_radius")
        super().__init__(params, dict())
        self.variant = variant
        self.bounds = bounds
        self.preconditioner = preconditioner
        self.radius = float(initial_radius)
        self.max_radius = float(max_radius)
        self.eta_acceptable = eta_acceptable
        self.eta_increase = eta_increase
        self.radius_shrink = radius_shrink
        self.radius_grow = radius_grow
        self.max_inner_iter = max_inner_iter
        self.cg_tol = cg_tol

        self._prev_step: Optional[Tensor] = None
        self._last_rho: float = 0.0
        self._last_accepted: bool = False
        self._last_boundary: bool = False

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

    # ------------------------------------------------------- subproblem solves
    def _ttr_step(self, matvec, g_flat, params) -> Tensor:
        precond_callable = (
            (lambda r: self._flat_precond(r, params)) if self.preconditioner else None
        )
        d, info = steihaug_cg(
            matvec, -g_flat, trust_radius=self.radius,
            tol=self.cg_tol, max_iter=self.max_inner_iter,
            preconditioner=precond_callable,
        )
        self._last_boundary = info.boundary
        return d

    def _astr_step(self, matvec, g_flat) -> Tensor:
        """Solve the 2-D subproblem on span(g, prev_step) (Wang & Rao 2015)."""
        # Build orthonormal basis (u, v).
        u = -g_flat / (g_flat.norm() + 1e-30)
        if self._prev_step is None:
            basis = [u]
        else:
            v = self._prev_step
            v = v - (v @ u) * u
            v_norm = float(v.norm())
            if v_norm < 1e-12:
                basis = [u]
            else:
                v = v / v_norm
                basis = [u, v]

        # Project g and assemble small Hessian H_S.
        g_S = torch.tensor([float(b @ g_flat) for b in basis])
        Hb = [matvec(b) for b in basis]
        m = len(basis)
        H_S = torch.zeros((m, m))
        for i in range(m):
            for j in range(m):
                H_S[i, j] = float(basis[i] @ Hb[j])
        H_S = 0.5 * (H_S + H_S.T)  # symmetrize

        # Solve constrained 2-D problem analytically: try unconstrained first.
        try:
            c = torch.linalg.solve(H_S, -g_S)
        except RuntimeError:
            # Singular Hessian projection - fall back to negative gradient direction.
            c = -g_S * (self.radius / (g_S.norm() + 1e-30))

        c_norm = float(c.norm())
        if c_norm > self.radius:
            c = c * (self.radius / c_norm)
            self._last_boundary = True
        else:
            self._last_boundary = False

        # Reconstruct d in full space.
        d = sum(float(c[i]) * basis[i] for i in range(m))
        return d

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, loss_fn: Callable[[], Tensor] = None):  # type: ignore[override]
        if loss_fn is None:
            raise RuntimeError(
                "TrustRegion.step requires a loss_fn returning the loss tensor "
                "WITHOUT calling backward."
            )
        params = list(iter_params(self.param_groups))
        if not all(p.requires_grad for p in params):
            raise RuntimeError("All optimized parameters must require grad.")

        # 1. Loss + grad with graph for HVP.
        with torch.enable_grad():
            loss = loss_fn()
            grads_list = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = self._flatten_list([g.detach() for g in grads_list]).clone()
        if float(g_flat.norm()) == 0.0:
            return loss

        def matvec(v_flat: Tensor) -> Tensor:
            v_list = self._unflatten(v_flat, params)
            with torch.enable_grad():
                inner = sum((g * v).sum() for g, v in zip(grads_list, v_list))
                hv = torch.autograd.grad(inner, params, retain_graph=True)
            return self._flatten_list([h.detach() for h in hv])

        # 2. Solve trust-region subproblem.
        if self.variant == "ttr":
            d = self._ttr_step(matvec, g_flat, params)
        else:
            d = self._astr_step(matvec, g_flat)

        # Predicted reduction from the quadratic model.
        Hd = matvec(d)
        predicted = -float(g_flat @ d) - 0.5 * float(d @ Hd)

        # Drop graph references before evaluating actual reduction.
        del grads_list

        # 3. Trial step.
        x0 = flat_params(self.param_groups).clone()
        write_flat_to_params(self.param_groups, x0 + d)
        with torch.enable_grad():
            new_loss = loss_fn()
        actual = float(loss.detach()) - float(new_loss.detach())

        # 4. rho and radius update.
        if predicted <= 0.0:
            rho = -1.0
        else:
            rho = actual / predicted
        self._last_rho = rho

        if rho < 0.25:
            self.radius = max(self.radius * self.radius_shrink, 1e-12)
        elif rho > self.eta_increase and self._last_boundary:
            self.radius = min(self.radius * self.radius_grow, self.max_radius)

        if rho > self.eta_acceptable:
            # Accept the step; current params are at x0 + d.
            self._project()
            self._prev_step = d.clone()
            self._last_accepted = True
            return new_loss
        # Reject: restore.
        write_flat_to_params(self.param_groups, x0)
        self._last_accepted = False
        return loss

    # ----------------------------------------------------------- diagnostics
    @property
    def last_rho(self) -> float:
        return self._last_rho

    @property
    def last_accepted(self) -> bool:
        return self._last_accepted

    @property
    def last_boundary(self) -> bool:
        return self._last_boundary

    @property
    def trust_radius(self) -> float:
        return self.radius
