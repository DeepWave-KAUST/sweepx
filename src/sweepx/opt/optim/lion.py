"""Lion optimizer (Chen et al. 2023).

Sign-of-interpolated-momentum step. The update uses ``sign`` of a
momentum-blended gradient, which makes Lion roughly half the memory of
Adam (no second-moment buffer) while remaining competitive on many
benchmarks.

Update rule, with ``beta1, beta2 = (0.9, 0.99)`` defaults:

    c_t = sign(beta1 * m_{t-1} + (1 - beta1) * g_t)
    theta_t = theta_{t-1} - lr * (c_t + weight_decay * theta_{t-1})
    m_t = beta2 * m_{t-1} + (1 - beta2) * g_t

Reference:
    Chen, X., et al. (2023). "Symbolic discovery of optimization
    algorithms." arXiv:2302.06675. https://arxiv.org/abs/2302.06675.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweep_opt._utils import iter_params


class Lion(Optimizer):
    """Lion (sign-momentum) optimizer with FWI-friendly bounds."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        *,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(params, dict(lr=lr))
        self.b1, self.b2 = betas
        self.weight_decay = weight_decay
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
                state["m"] = torch.zeros_like(p.data)
            m = state["m"]
            update = torch.sign(self.b1 * m + (1.0 - self.b1) * g)
            if self.weight_decay > 0.0:
                update = update + self.weight_decay * p.data
            p.data.add_(update, alpha=-lr)
            m.mul_(self.b2).add_(g, alpha=1.0 - self.b2)
        self._project()
        return loss
