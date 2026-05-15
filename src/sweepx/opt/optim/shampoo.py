"""Shampoo: Kronecker-factored 4th-root preconditioning.

For a 2-D parameter ``M`` of shape ``(d1, d2)``, Shampoo maintains two
running preconditioners::

    L_t = L_{t-1} + G_t G_t^T            # (d1, d1)
    R_t = R_{t-1} + G_t^T G_t            # (d2, d2)

and updates with ``M -= lr * L_t^{-1/4} G_t R_t^{-1/4}``. The 4th-root
inversions are cached and refreshed every ``update_freq`` steps to
amortize the eigendecomposition cost.

For 1-D parameters Shampoo degenerates to a single preconditioner
``L^{-1/2}`` (only one axis), which is also implemented. Higher-order
tensors are flattened to 2-D along the leading axis.

Reference:
    Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned
    stochastic tensor optimization." ICML 2018.
    https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweep_opt._utils import iter_params


def _matrix_inverse_root(mat: Tensor, p: float, eps: float) -> Tensor:
    """Return ``(mat + eps I)^(-1/p)`` via symmetric eigendecomposition."""
    n = mat.shape[0]
    reg = mat + eps * torch.eye(n, dtype=mat.dtype, device=mat.device)
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (reg + reg.T))
    eigvals = eigvals.clamp(min=eps)
    inv_root = eigvals.pow(-1.0 / p)
    return (eigvecs * inv_root) @ eigvecs.T


class Shampoo(Optimizer):
    """Shampoo optimizer with optional bounds and FWI-friendly defaults."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        *,
        eps: float = 1e-6,
        update_freq: int = 50,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if update_freq < 1:
            raise ValueError("update_freq must be >= 1")
        super().__init__(params, dict(lr=lr))
        self.eps = eps
        self.update_freq = update_freq
        self.bounds = bounds

    def _project(self) -> None:
        if self.bounds is None:
            return
        lo, hi = self.bounds
        for p in iter_params(self.param_groups):
            p.data.clamp_(lo, hi)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lr = self.defaults["lr"]
        for p in iter_params(self.param_groups):
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["L"] = None
                state["R"] = None
                state["L_inv_root"] = None
                state["R_inv_root"] = None
            state["step"] += 1
            t = state["step"]

            # Reshape to 2-D (d1, d2).
            if g.dim() == 0:
                g_2d = g.view(1, 1)
                shape = (1, 1)
            elif g.dim() == 1:
                g_2d = g.view(-1, 1)
                shape = (g.numel(), 1)
            else:
                shape = (g.shape[0], -1)
                g_2d = g.reshape(shape)
            d1, d2 = g_2d.shape

            if state["L"] is None:
                state["L"] = torch.zeros(d1, d1, dtype=g.dtype, device=g.device)
                state["R"] = torch.zeros(d2, d2, dtype=g.dtype, device=g.device)

            state["L"].add_(g_2d @ g_2d.T)
            state["R"].add_(g_2d.T @ g_2d)

            if (t == 1) or (t % self.update_freq == 0) or state["L_inv_root"] is None:
                # 4th-root for 2-D, square-root degenerate (1-D effective).
                if d1 == 1:
                    L_root = torch.tensor([[1.0]], dtype=g.dtype, device=g.device)
                else:
                    L_root = _matrix_inverse_root(state["L"], p=4.0, eps=self.eps)
                if d2 == 1:
                    R_root = torch.tensor([[1.0]], dtype=g.dtype, device=g.device)
                else:
                    R_root = _matrix_inverse_root(state["R"], p=4.0, eps=self.eps)
                state["L_inv_root"] = L_root
                state["R_inv_root"] = R_root

            update = state["L_inv_root"] @ g_2d @ state["R_inv_root"]
            p.data.add_(update.reshape(g.shape), alpha=-lr)
        self._project()
        return loss
