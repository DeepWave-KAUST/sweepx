"""Matrix-free Hessian-vector product via PyTorch autograd.

Given a scalar loss ``L(theta)`` and a vector ``v`` with the same shape as
the parameters, the Hessian-vector product is::

    H v = nabla_theta ( (nabla_theta L)^T v )

We get this in two backward passes by asking autograd to differentiate the
inner-product ``g . v`` with respect to ``theta``. The trick is to compute
the first gradient with ``create_graph=True`` so the second backward sees
the dependency on ``theta``.

Reference:
    Pearlmutter, B. A. (1994). "Fast exact multiplication by the Hessian."
    Neural Computation 6(1), 147-160.
    DOI:10.1162/neco.1994.6.1.147.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch
from torch import Tensor


def hessian_vector_product(
    loss_fn: Callable[[], Tensor],
    params: Sequence[Tensor],
    vec: Sequence[Tensor],
) -> List[Tensor]:
    """Compute ``H @ vec`` where ``H`` is the Hessian of ``loss_fn`` w.r.t. ``params``.

    Args:
        loss_fn: callable returning the scalar loss as a tensor that depends on
            ``params`` through the autograd graph. Will be called once.
        params: leaf tensors with ``requires_grad=True``.
        vec: tensors matching the shapes of ``params``.

    Returns:
        List of tensors with the same shapes as ``params``, representing
        ``H @ vec`` partitioned by parameter.

    Notes:
        Uses two backward passes (Pearlmutter trick). Memory is O(graph
        size), same as a plain backward; no Hessian is ever materialized.
    """
    if len(params) != len(vec):
        raise ValueError(f"len(params)={len(params)} != len(vec)={len(vec)}")
    for i, (p, v) in enumerate(zip(params, vec)):
        if p.shape != v.shape:
            raise ValueError(f"params[{i}].shape {tuple(p.shape)} != vec[{i}].shape {tuple(v.shape)}")
        if not p.requires_grad:
            raise ValueError(f"params[{i}] must require grad for HVP")

    loss = loss_fn()
    grads = torch.autograd.grad(
        loss, params, create_graph=True, allow_unused=False,
    )
    if not any(g.requires_grad for g in grads):
        raise RuntimeError(
            "hessian_vector_product needs a twice-differentiable loss, but its gradient "
            "came back without an autograd graph. This happens when the loss goes "
            "through an operation without double-backward support, e.g. sweep's native "
            "impl='c' propagator: build the solver with impl='eager' and "
            "eager_options=EagerOptions(use_compile=False)."
        )
    inner = sum((g * v).sum() for g, v in zip(grads, vec))
    hv = torch.autograd.grad(inner, params, retain_graph=False)
    return list(hv)


def gauss_newton_vector_product(
    residual_fn: Callable[[], Tensor],
    params: Sequence[Tensor],
    vec: Sequence[Tensor],
) -> List[Tensor]:
    """Compute ``J^T J @ vec`` for a non-linear least-squares ``L = 0.5 ||r||^2``.

    The Gauss-Newton matrix ``J^T J`` is the positive-semidefinite quadratic
    approximation of the Hessian that drops the second-derivative term
    involving the residual. We compute it matrix-free as a JVP through the
    residual followed by a VJP back to the parameters.

    Args:
        residual_fn: callable returning the residual tensor ``r(params)``.
        params: leaf tensors with ``requires_grad=True``.
        vec: tensors matching the shapes of ``params``.

    Returns:
        ``J^T J @ vec`` as a list of tensors matching the shapes of ``params``.
    """
    if len(params) != len(vec):
        raise ValueError(f"len(params)={len(params)} != len(vec)={len(vec)}")

    # JVP: dr = J @ vec, computed via forward-mode emulated by a double-backward.
    residual = residual_fn()
    if residual.requires_grad is False:
        raise RuntimeError("residual must depend on params with requires_grad=True")
    # Trick: build a dummy upstream gradient and compute J via two adjoints.
    # Simpler: use torch.autograd.functional.jvp / vjp.
    grad_outputs = torch.zeros_like(residual, requires_grad=True)
    grads = torch.autograd.grad(
        residual, params, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True,
    )
    # grads = J^T @ grad_outputs, depends on grad_outputs as a graph node.
    # Differentiating sum(grads_i * vec_i) wrt grad_outputs gives J @ vec.
    inner_v = sum((g * v).sum() for g, v in zip(grads, vec))
    jvp = torch.autograd.grad(inner_v, grad_outputs, retain_graph=True)[0]
    # jvp = J @ vec  (residual-shaped)
    # Now do an ordinary VJP: J^T @ jvp.
    jt_jv = torch.autograd.grad(residual, params, grad_outputs=jvp,
                                retain_graph=False)
    return list(jt_jv)
