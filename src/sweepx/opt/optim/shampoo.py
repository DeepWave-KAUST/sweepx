"""Shampoo: Kronecker-factored inverse-root preconditioning.

For a 2-D parameter ``M`` of shape ``(d1, d2)``, Shampoo maintains two
running preconditioners::

    L_t = L_{t-1} + G_t G_t^T            # (d1, d1)
    R_t = R_{t-1} + G_t^T G_t            # (d2, d2)

and updates with ``M -= lr * L_t^{-1/4} G_t R_t^{-1/4}``. The inverse roots
are cached and refreshed every ``update_freq`` steps to amortize the
eigendecomposition cost.

With ``k`` factored axes the exponent is ``-1/(2k)`` (Gupta et al. 2018):
vectors ``(n, 1)`` use ``L^{-1/2}``. An axis longer than ``max_precond_dim``
is not factored; a parameter with no factored axis at all falls back to
the diagonal (AdaGrad) form of the same preconditioner. Higher-order
tensors are flattened to 2-D along the leading axis.

Reference:
    Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned
    stochastic tensor optimization." ICML 2018.
    https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

from sweepx.opt._utils import iter_params
from sweepx.opt.optim._kron import as_matrix, factored_axes, inverse_root


class Shampoo(Optimizer):
    """Shampoo optimizer with optional bounds and FWI-friendly defaults.

    ``lr``, ``eps``, ``update_freq`` and ``max_precond_dim`` are per
    parameter group, so groups and ``torch.optim.lr_scheduler`` work as for
    any torch optimizer. ``eps`` is relative to the largest eigenvalue of
    each factor.

    A cached root (``update_freq > 1``) can amplify a gradient direction that
    appeared after the last refresh by up to ``eps^(-1/(2k))``; with a small
    ``eps`` that diverges. Raise ``eps`` (e.g. ``1e-2``) when refreshing lazily.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        *,
        eps: float = 1e-4,
        update_freq: int = 1,
        max_precond_dim: int = 4096,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if update_freq < 1:
            raise ValueError("update_freq must be >= 1")
        if max_precond_dim < 1:
            raise ValueError("max_precond_dim must be >= 1")
        super().__init__(params, dict(lr=lr, eps=eps, update_freq=update_freq,
                                      max_precond_dim=max_precond_dim))
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
        for group in self.param_groups:
            lr, eps, freq = group["lr"], group["eps"], group["update_freq"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = as_matrix(p.grad)
                d1, d2 = g.shape
                fl, fr = factored_axes(d1, d2, group["max_precond_dim"])
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["L"] = torch.zeros(d1, d1, dtype=g.dtype, device=g.device) if fl else None
                    state["R"] = torch.zeros(d2, d2, dtype=g.dtype, device=g.device) if fr else None
                    state["diag"] = None if (fl or fr) else torch.zeros_like(g)
                    state["L_inv_root"] = None
                    state["R_inv_root"] = None
                state["step"] += 1

                if state["diag"] is not None:          # no factored axis: AdaGrad
                    acc = state["diag"].addcmul_(g, g)
                    top = float(acc.max())
                    if top > 0.0:
                        upd = g / (acc + eps * top).sqrt()
                        p.data.add_(upd.reshape(p.shape), alpha=-lr)
                    continue

                refresh = state["step"] % freq == 0 or state["step"] == 1
                k = int(fl) + int(fr)
                if fl:
                    state["L"].add_(g @ g.T)
                if fr:
                    state["R"].add_(g.T @ g)
                for side, on in (("L", fl), ("R", fr)):
                    key = side + "_inv_root"
                    if on and (refresh or state[key] is None):
                        root = inverse_root(state[side], p=2.0 * k, eps=eps)
                        if root is not None:           # keep the old root if eigh failed
                            state[key] = root
                if (fl and state["L_inv_root"] is None) or (fr and state["R_inv_root"] is None):
                    continue                           # nothing usable yet: skip this step
                upd = g.double()                       # roots are float64 (see _kron.inverse_root)
                upd = state["L_inv_root"] @ upd if fl else upd
                upd = upd @ state["R_inv_root"] if fr else upd
                p.data.add_(upd.to(p.dtype).reshape(p.shape), alpha=-lr)
        self._project()
        return loss
