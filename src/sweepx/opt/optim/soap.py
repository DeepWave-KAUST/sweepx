"""SOAP: Shampoo's eigenbasis + Adam updates.

Maintain Shampoo-style preconditioners ``L`` and ``R`` for a 2-D parameter,
diagonalize them once every ``update_freq`` steps to obtain rotations
``Q_L, Q_R``, and run Adam updates in the rotated eigenbasis::

    G_rot      = Q_L^T G Q_R
    Adam state in rotated basis  (m, v)
    update_rot = m_hat / (sqrt(v_hat) + eps)
    update     = Q_L update_rot Q_R^T

Compared to Shampoo, SOAP couples the second-order rotation with
Adam-style first-moment momentum; in practice it is competitive with both.

Reference:
    Vyas, N., Morwani, D., Zhao, R., Kaplun, G., Kakade, S., and
    Shanker, S. (2024). "SOAP: Improving and stabilizing Shampoo using
    Adam." arXiv:2409.11321. https://arxiv.org/abs/2409.11321.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweep_opt._utils import iter_params


class SOAP(Optimizer):
    """SOAP optimizer (Shampoo eigenbasis + Adam) for 2-D parameters."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        *,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        update_freq: int = 50,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        super().__init__(params, dict(lr=lr))
        self.b1, self.b2 = betas
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
                state["QL"] = None
                state["QR"] = None
                state["m"] = None
                state["v"] = None
            state["step"] += 1
            t = state["step"]

            if g.dim() == 0:
                g_2d = g.view(1, 1)
            elif g.dim() == 1:
                g_2d = g.view(-1, 1)
            else:
                g_2d = g.reshape(g.shape[0], -1)
            d1, d2 = g_2d.shape

            if state["L"] is None:
                state["L"] = torch.zeros(d1, d1, dtype=g.dtype, device=g.device)
                state["R"] = torch.zeros(d2, d2, dtype=g.dtype, device=g.device)

            state["L"].mul_(self.b2).add_(g_2d @ g_2d.T, alpha=1.0 - self.b2)
            state["R"].mul_(self.b2).add_(g_2d.T @ g_2d, alpha=1.0 - self.b2)

            if state["QL"] is None or t % self.update_freq == 0:
                _, QL = torch.linalg.eigh(0.5 * (state["L"] + state["L"].T))
                _, QR = torch.linalg.eigh(0.5 * (state["R"] + state["R"].T))
                state["QL"] = QL
                state["QR"] = QR

            QL = state["QL"]
            QR = state["QR"]
            g_rot = QL.T @ g_2d @ QR

            if state["m"] is None:
                state["m"] = torch.zeros_like(g_rot)
                state["v"] = torch.zeros_like(g_rot)
            m = state["m"]
            v = state["v"]
            m.mul_(self.b1).add_(g_rot, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(g_rot, g_rot, value=1.0 - self.b2)
            bias1 = 1.0 - self.b1 ** t
            bias2 = 1.0 - self.b2 ** t
            update_rot = (m / bias1) / ((v / bias2).sqrt() + self.eps)
            update = QL @ update_rot @ QR.T
            p.data.add_(update.reshape(g.shape), alpha=-lr)
        self._project()
        return loss
