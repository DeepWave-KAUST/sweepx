"""SOAP: Adam in the eigenbasis of Shampoo's preconditioner.

This follows the official implementation of Vyas et al. (2024)
(https://github.com/nikhilvyas/SOAP, commit a1e5535) step for step;
``tests/test_soap_parity.py`` checks it bit-for-bit against that code.

Every parameter keeps one Shampoo factor per tensor axis (``L`` and ``R`` for
a matrix), an exponential moving average of the gradient contracted over all
other axes::

    GG_k <- shampoo_beta * GG_k + (1 - shampoo_beta) * contract(G, G, all axes but k)

and Adam runs on the gradient rotated into the eigenbases ``Q_k`` of those
factors::

    G'   = G projected on Q_1, ..., Q_n
    m, v = Adam moments of G'
    W   <- W - lr * project_back(m_hat / (sqrt(v_hat) + eps))

Order of operations, as in the official code:

1. On the first step the factors are built from the gradient and
   eigendecomposed (columns in descending eigenvalue order); the parameter is
   not updated, so a gradient never enters its own projection.
2. On later steps Adam steps in the current basis, decoupled weight decay is
   applied, and only then are the factors updated with the gradient. Every
   ``update_freq`` steps the basis is refreshed by one power iteration + QR
   from the previous basis (Algorithm 4 of the paper), after sorting its
   columns by Rayleigh quotient and permuting ``v`` with them. ``m`` is moved
   into the new basis exactly (projected back, then re-projected); ``v`` is not
   rotated.

An axis longer than ``max_precond_dim`` gets no factor (identity), so a
``(1e5, 1)`` hash-grid table is updated by Adam. 1-D parameters are plain Adam
unless ``precondition_1d``. ``merge_dims`` merges neighbouring axes until
their product would exceed ``max_precond_dim``.

Deviations from the official code:

* ``weight_decay`` defaults to 0 (official: 0.01). FWI parameters are often
  physical quantities such as a velocity grid, which must not be decayed.
* The factors are kept in the parameter dtype, so float64 parameters work (the
  official code allocates them in the default dtype and fails on float64).
  Parameters must be float32 or float64.
* ``closure`` is evaluated with gradients enabled.
* ``update_freq`` is the official ``precondition_frequency``; the optional
  ``bounds`` clamp after the step is sweep-opt's; there is no ``data_format``
  option (channels-last 4-D convolution weights).

Reference:
    Vyas, N., Morwani, D., Zhao, R., Kwun, M., Shapira, I., Brandfonbrener,
    D., Janson, L., and Kakade, S. (2024). "SOAP: Improving and stabilizing
    Shampoo using Adam." arXiv:2409.11321. https://arxiv.org/abs/2409.11321.
"""
from __future__ import annotations

from itertools import chain
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from sweep_opt._utils import iter_params


class SOAP(Optimizer):
    """SOAP optimizer (Adam in Shampoo's eigenbasis), official algorithm.

    ``lr``, ``betas``, ``shampoo_beta``, ``eps``, ``weight_decay``,
    ``update_freq``, ``max_precond_dim``, ``merge_dims``, ``precondition_1d``,
    ``normalize_grads`` and ``correct_bias`` are per parameter group, so
    groups and ``torch.optim.lr_scheduler`` work as for any torch optimizer.
    ``shampoo_beta < 0`` uses ``betas[1]`` for the factors' moving average.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        *,
        betas: Tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_freq: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if shampoo_beta >= 1.0:
            raise ValueError(f"shampoo_beta must be < 1, got {shampoo_beta}")
        if eps < 0 or weight_decay < 0:
            raise ValueError("eps and weight_decay must be non-negative")
        if update_freq < 1:
            raise ValueError("update_freq must be >= 1")
        if max_precond_dim < 1:
            raise ValueError("max_precond_dim must be >= 1")
        super().__init__(params, dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
            weight_decay=weight_decay, update_freq=update_freq,
            max_precond_dim=max_precond_dim, merge_dims=merge_dims,
            precondition_1d=precondition_1d, normalize_grads=normalize_grads,
            correct_bias=correct_bias))
        self.bounds = bounds

    def _clamp_bounds(self) -> None:
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype not in (torch.float32, torch.float64):
                    raise TypeError(f"SOAP supports float32/float64 parameters, got {grad.dtype}")
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                if "Q" not in state:
                    _init_preconditioner(grad, state, group)
                    _update_preconditioner(grad, state, group)
                    continue                           # first step only builds the basis

                grad_projected = _project(grad, state, group)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                norm_grad = _project_back(exp_avg / denom, state, group)
                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)
                p.add_(norm_grad, alpha=-step_size)
                if group["weight_decay"] > 0.0:        # decoupled (AdamW) decay
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # factors and basis are updated after the step (current gradient
                # never enters its own projection)
                _update_preconditioner(grad, state, group)
        self._clamp_bounds()
        return loss


def _merge_dims(grad: Tensor, max_precond_dim: int) -> Tensor:
    """Merge neighbouring axes while their product stays <= ``max_precond_dim``."""
    new_shape = []
    curr_shape = 1
    for sh in grad.shape:
        temp_shape = curr_shape * sh
        if temp_shape > max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape
    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)
    return grad.reshape(new_shape)


def _init_preconditioner(grad: Tensor, state: dict, group: dict) -> None:
    """One factor per axis (``None`` = identity), basis not computed yet."""
    mpd = group["max_precond_dim"]
    state["GG"] = []
    if grad.dim() == 1:
        if not group["precondition_1d"] or grad.shape[0] > mpd:
            state["GG"].append(None)
        else:
            state["GG"].append(torch.zeros(grad.shape[0], grad.shape[0],
                                           dtype=grad.dtype, device=grad.device))
    else:
        if group["merge_dims"]:
            grad = _merge_dims(grad, mpd)
        for sh in grad.shape:
            state["GG"].append(None if sh > mpd else
                               torch.zeros(sh, sh, dtype=grad.dtype, device=grad.device))
    state["Q"] = None
    sb = group["shampoo_beta"]
    state["shampoo_beta"] = sb if sb >= 0 else group["betas"][1]


def _update_preconditioner(grad: Tensor, state: dict, group: dict) -> None:
    """Update the factors with ``grad``; (re)build the basis; keep ``m`` in it."""
    mpd = group["max_precond_dim"]
    if state["Q"] is not None:
        state["exp_avg"] = _project_back(state["exp_avg"], state, group)
    if grad.dim() == 1:
        if group["precondition_1d"] and grad.shape[0] <= mpd:
            state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"])
    else:
        g = _merge_dims(grad, mpd) if group["merge_dims"] else grad
        for idx, sh in enumerate(g.shape):
            if sh <= mpd:
                outer_product = torch.tensordot(
                    g, g, dims=[[*chain(range(idx), range(idx + 1, len(g.shape)))]] * 2)
                state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
    if state["Q"] is None:
        state["Q"] = _eigenbases(state["GG"])
    if state["step"] > 0 and state["step"] % group["update_freq"] == 0:
        state["Q"] = _refresh_bases(state, group)
    if state["step"] > 0:
        state["exp_avg"] = _project(state["exp_avg"], state, group)


def _project(grad: Tensor, state: dict, group: dict) -> Tensor:
    """Rotate ``grad`` into the eigenbases (``G x_k Q_k``, axis by axis)."""
    original_shape = grad.shape
    if group["merge_dims"]:
        grad = _merge_dims(grad, group["max_precond_dim"])
    for mat in state["Q"]:
        if mat is not None:
            grad = torch.tensordot(grad, mat, dims=[[0], [0]])
        else:
            grad = grad.permute(list(range(1, len(grad.shape))) + [0])
    if group["merge_dims"]:
        grad = grad.reshape(original_shape)
    return grad


def _project_back(grad: Tensor, state: dict, group: dict) -> Tensor:
    """Inverse of :func:`_project` (``G x_k Q_k^T``, axis by axis)."""
    original_shape = grad.shape
    if group["merge_dims"]:
        grad = _merge_dims(grad, group["max_precond_dim"])
    for mat in state["Q"]:
        if mat is not None:
            grad = torch.tensordot(grad, mat, dims=[[0], [1]])
        else:
            grad = grad.permute(list(range(1, len(grad.shape))) + [0])
    if group["merge_dims"]:
        grad = grad.reshape(original_shape)
    return grad


def _eigenbases(factors) -> list:
    """Eigenvectors of each factor, columns in descending eigenvalue order."""
    out = []
    for m in factors:
        if m is None:
            out.append(None)
            continue
        n = m.shape[0]
        try:
            _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(n, dtype=m.dtype, device=m.device))
        except RuntimeError:                           # retry in float64, as the official code
            _, Q = torch.linalg.eigh(m.to(torch.float64)
                                     + 1e-30 * torch.eye(n, dtype=torch.float64, device=m.device))
            Q = Q.to(m.dtype)
        out.append(torch.flip(Q, [1]))
    return out


def _refresh_bases(state: dict, group: dict) -> list:
    """One power iteration + QR per factor from the current basis (paper, Algorithm 4).

    Columns are first sorted by Rayleigh quotient (descending) and ``v`` is
    permuted with them, as in the official code.
    """
    exp_avg_sq = state["exp_avg_sq"]
    orig_shape = exp_avg_sq.shape
    if group["merge_dims"]:
        exp_avg_sq = _merge_dims(exp_avg_sq, group["max_precond_dim"])
    out = []
    for ind, (m, o) in enumerate(zip(state["GG"], state["Q"])):
        if m is None:
            out.append(None)
            continue
        est_eig = torch.diag(o.T @ m @ o)
        sort_idx = torch.argsort(est_eig, descending=True)
        exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
        o = o[:, sort_idx]
        power_iter = m @ o
        Q, _ = torch.linalg.qr(power_iter)
        out.append(Q)
    if group["merge_dims"]:
        exp_avg_sq = exp_avg_sq.reshape(orig_shape)
    state["exp_avg_sq"] = exp_avg_sq
    return out
