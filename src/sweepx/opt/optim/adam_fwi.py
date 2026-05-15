"""Adam-FWI: Adam-family optimizers with FWI-friendly extras.

Single class, four update rules selected by the ``variant`` flag:

- ``"adam"``    - Kingma & Ba 2014 (arXiv:1412.6980).
- ``"adamw"``   - Loshchilov & Hutter 2017 (arXiv:1711.05101) with
                  decoupled weight decay.
- ``"radam"``   - Liu et al. 2019 Rectified Adam (arXiv:1908.03265).
- ``"nadam"``   - Dozat 2016, Nesterov-accelerated Adam.

On top of the stock formulas, all variants accept the FWI-flavoured knobs:

- ``bounds=(lo, hi)`` projected after each step.
- ``mask`` tensor (or callable returning one) to zero gradient updates in
  pinned regions (e.g. water-layer freeze).
- ``amsgrad=True`` to use the running max of the second moment.
- ``eps=1e-22`` default matches sweep's existing FWI examples.

References:
    Kingma, D. P., and Ba, J. (2015). "Adam: A method for stochastic
    optimization." arXiv:1412.6980. https://arxiv.org/abs/1412.6980.

    Loshchilov, I., and Hutter, F. (2017). "Decoupled weight decay
    regularization." arXiv:1711.05101. https://arxiv.org/abs/1711.05101.

    Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J.
    (2019). "On the variance of the adaptive learning rate and beyond."
    arXiv:1908.03265. https://arxiv.org/abs/1908.03265.

    Dozat, T. (2016). "Incorporating Nesterov momentum into Adam."
    https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweep_opt._utils import iter_params


_VARIANTS = ("adam", "adamw", "radam", "nadam")


class AdamFWI(Optimizer):
    """Adam family with bounds, masking, and AMSGrad. See module docstring."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        *,
        variant: str = "adam",
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-22,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        bounds: Optional[Tuple[float, float]] = None,
        mask: Optional[Union[Tensor, dict]] = None,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        super().__init__(params, dict(lr=lr))
        self.variant = variant
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.bounds = bounds
        self.mask_arg = mask

    def _resolve_mask(self, p: Tensor) -> Optional[Tensor]:
        m = self.mask_arg
        if m is None:
            return None
        if isinstance(m, dict):
            return m.get(id(p))
        if callable(m):
            return m(p)
        return m

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
            grad = p.grad
            mask = self._resolve_mask(p)
            if mask is not None:
                grad = grad * mask
            if self.weight_decay > 0.0:
                if self.variant == "adamw":
                    # Decoupled weight decay (applied to the parameter directly).
                    p.data.mul_(1.0 - lr * self.weight_decay)
                else:
                    grad = grad + self.weight_decay * p.data

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["m"] = torch.zeros_like(p.data)
                state["v"] = torch.zeros_like(p.data)
                if self.amsgrad:
                    state["v_max"] = torch.zeros_like(p.data)

            state["step"] += 1
            t = state["step"]
            m = state["m"]
            v = state["v"]
            m.mul_(self.b1).add_(grad, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(grad, grad, value=1.0 - self.b2)

            if self.amsgrad:
                state["v_max"] = torch.max(state["v_max"], v)
                v_hat_t = state["v_max"]
            else:
                v_hat_t = v

            if self.variant in ("adam", "adamw"):
                bias1 = 1.0 - self.b1 ** t
                bias2 = 1.0 - self.b2 ** t
                m_hat = m / bias1
                v_hat = v_hat_t / bias2
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-lr)
            elif self.variant == "nadam":
                bias1 = 1.0 - self.b1 ** t
                bias2 = 1.0 - self.b2 ** t
                m_hat = m / bias1
                v_hat = v_hat_t / bias2
                # Nesterov: replace m_hat with b1 * m_hat + (1-b1)*grad/bias1.
                m_nest = self.b1 * m_hat + (1.0 - self.b1) * grad / bias1
                p.data.addcdiv_(m_nest, v_hat.sqrt().add_(self.eps), value=-lr)
            elif self.variant == "radam":
                rho_inf = 2.0 / (1.0 - self.b2) - 1.0
                rho_t = rho_inf - 2.0 * t * (self.b2 ** t) / (1.0 - self.b2 ** t)
                bias1 = 1.0 - self.b1 ** t
                m_hat = m / bias1
                if rho_t > 4.0:
                    bias2 = 1.0 - self.b2 ** t
                    v_hat = v_hat_t / bias2
                    r_t = (
                        ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    ) ** 0.5
                    p.data.addcdiv_(m_hat * r_t, v_hat.sqrt().add_(self.eps),
                                    value=-lr)
                else:
                    p.data.add_(m_hat, alpha=-lr)
        self._project()
        return loss
