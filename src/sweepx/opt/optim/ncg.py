"""Nonlinear Conjugate Gradient with Polak-Ribiere+ updates and Wolfe line search.

The Polak-Ribiere+ (PR+) variant uses ``beta_k = max(0, beta_PR)`` which
automatically restarts when the standard PR formula goes negative, fixing the
classical "jamming" failure of Fletcher-Reeves and the lack of guaranteed
descent of plain Polak-Ribiere. PR+ is the workhorse choice for nonlinear CG
in FWI (Powell 1977; Nocedal & Wright 2006, Section 5.2).

References:
    Powell, M. J. D. (1977). "Restart procedures for the conjugate gradient
    method." Mathematical Programming 12, 241-254.
    DOI:10.1007/BF01593790.

    Nocedal, J., and Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
    Springer, Section 5.2. DOI:10.1007/978-0-387-40065-5.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweepx.opt._utils import (
    flat_grads,
    flat_params,
    iter_params,
)
from sweepx.opt.optim._wolfe_step import (
    check_initial_step,
    scaled_bracket,
    wolfe_step,
)


class NCG(Optimizer):
    """Nonlinear Conjugate Gradient with Polak-Ribiere+ and a strong-Wolfe line search.

    Args:
        params: iterable of parameter tensors / param-groups.
        bounds: optional ``(lo, hi)`` box; every line-search trial point is
            projected onto it.
        preconditioner: optional callable ``(grad, param) -> grad``. The PR+
            update is computed on the *preconditioned* gradient
            (Hager-Zhang style preconditioned NCG).
        line_search: must be ``"strong_wolfe"`` (other choices may be added
            later). NCG without a line search is unsound, so this argument
            exists only for forward compatibility.
        c1, c2: Wolfe constants. Defaults match the values recommended for
            CG (``c2=0.1``) in Nocedal & Wright Section 3.5.
        alpha_init, alpha_max, max_ls_iter: Wolfe step bracket / iteration cap.
            ``alpha`` multiplies the search direction ``d``, whose size follows
            the gradient.
        initial_step: size of the first line-search trial of the first step,
            in parameter units (``max |alpha_0 d| = initial_step``). Later
            searches start from the previous step,
            ``alpha_0 = alpha_{k-1} g_{k-1}.d_{k-1} / g_k.d_k`` (Nocedal &
            Wright eq. 3.60), which needs no units. No trial changes a
            parameter by more than ``alpha_max / alpha_init`` times
            ``initial_step``. ``None`` (default) starts every search at
            ``alpha_init``, which does not move parameters with tiny
            gradients, such as a velocity model in m/s.
        restart_every: if set, force ``d = -P g`` every ``N`` iterations as
            an extra safety reset on top of PR+'s automatic restart.

    Notes:
        ``step()`` always requires a closure. Without a closure the line
        search has nothing to query.
    """

    def __init__(
        self,
        params,
        *,
        bounds: Optional[Tuple[float, float]] = None,
        preconditioner=None,
        line_search: str = "strong_wolfe",
        c1: float = 1e-4,
        c2: float = 0.1,
        alpha_init: float = 1.0,
        alpha_max: float = 10.0,
        max_ls_iter: int = 25,
        initial_step: Optional[float] = None,
        restart_every: Optional[int] = None,
    ) -> None:
        if line_search != "strong_wolfe":
            raise ValueError(f"unsupported line_search: {line_search!r}")
        if restart_every is not None and restart_every <= 0:
            raise ValueError("restart_every must be positive or None")
        super().__init__(params, dict())
        self.bounds = bounds
        self.preconditioner = preconditioner
        self.line_search = line_search
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.max_ls_iter = max_ls_iter
        self.initial_step = check_initial_step(initial_step)
        self.restart_every = restart_every

        # Inter-iteration state.
        self._prev_alpha_dphi: Optional[float] = None  # alpha_{k-1} g_{k-1}.d_{k-1}
        self._prev_g: Optional[Tensor] = None
        self._prev_z: Optional[Tensor] = None
        self._prev_d: Optional[Tensor] = None
        self._iter: int = 0
        self._last_beta: float = 0.0
        self._last_restart: bool = True  # first iteration always uses -z.

    # ----------------------------------------------------------------- utils
    def _precondition(self, g_flat: Tensor) -> Tensor:
        """Apply the preconditioner per parameter, returning a flat tensor."""
        if self.preconditioner is None:
            return g_flat
        z = torch.empty_like(g_flat)
        offset = 0
        for p in iter_params(self.param_groups):
            n = p.numel()
            chunk = g_flat[offset : offset + n].view_as(p)
            z[offset : offset + n] = self.preconditioner(chunk, p).reshape(-1)
            offset += n
        return z

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is None:
            raise RuntimeError("NCG requires a closure (line search needs to evaluate phi).")

        with torch.enable_grad():
            loss = closure()

        x0 = flat_params(self.param_groups).clone()
        g = flat_grads(self.param_groups).clone()
        z = self._precondition(g).clone()

        force_restart = (
            self._prev_d is None
            or (self.restart_every is not None and self._iter % self.restart_every == 0)
        )

        if force_restart:
            d = -z
            beta = 0.0
            self._last_restart = True
        else:
            denom = float(torch.dot(self._prev_g, self._prev_z))
            num = float(torch.dot(g - self._prev_g, z))
            beta_pr = num / denom if abs(denom) > 1e-30 else 0.0
            beta = max(0.0, beta_pr)
            self._last_restart = beta == 0.0
            d = -z + beta * self._prev_d
        self._last_beta = beta

        init_loss = float(loss.detach())
        init_dphi = float(torch.dot(g, d))

        # Safety: if d is not a descent direction (numerical or restart corner case),
        # fall back to -z, then -g.
        if init_dphi >= 0.0:
            d = -z
            init_dphi = float(torch.dot(g, d))
            self._last_restart = True
            beta = 0.0
        if init_dphi >= 0.0:
            d = -g
            init_dphi = float(torch.dot(g, d))
        if init_dphi >= 0.0:
            return loss

        def evaluate():
            with torch.enable_grad():
                ll = closure()
            return float(ll.detach()), flat_grads(self.param_groups)

        if self.initial_step is None:
            alpha0, alpha_max = self.alpha_init, self.alpha_max
        else:
            alpha0, alpha_max = scaled_bracket(self.initial_step, d, self.alpha_init, self.alpha_max)
            if self._prev_alpha_dphi is not None:
                # Nocedal & Wright eq. 3.60, inside the same bracket.
                alpha0 = min(self._prev_alpha_dphi / init_dphi, alpha_max)
        alpha_star = wolfe_step(
            self.param_groups, evaluate, x0, d, init_loss, init_dphi,
            bounds=self.bounds, alpha_init=alpha0, alpha_max=alpha_max,
            c1=self.c1, c2=self.c2, max_iter=self.max_ls_iter,
            owner="NCG", step_hints=True, initial_step=self.initial_step,
        )
        self._prev_alpha_dphi = alpha_star * init_dphi if alpha_star > 0.0 else None

        # Cache the gradient at the *start* of this step, so the next step
        # can compute PR+ as (g_new - g_old)^T z_new / (g_old^T z_old).
        self._prev_g = g
        self._prev_z = z
        self._prev_d = d.clone()
        self._iter += 1
        return loss

    # ----------------------------------------------------------- diagnostics
    @property
    def last_beta(self) -> float:
        """Last PR+ beta used (0.0 means restart). Useful for unit tests / logging."""
        return self._last_beta

    @property
    def last_restart(self) -> bool:
        """True iff the last step was a restart (d = -P g)."""
        return self._last_restart
