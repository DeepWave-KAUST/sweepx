"""Diagonal pseudo-Hessian preconditioner (Plessix & Mulder, 2004).

The pseudo-Hessian approximates the diagonal of the FWI Hessian by the
zero-lag autocorrelation of the source-side wavefield times the receiver-side
wavefield, evaluated at every model grid point. Plessix & Mulder showed this
amounts to scaling the gradient by the inverse of the product of source and
receiver illumination::

    P_ii = 1 / (S_i * R_i + eps)

where ``S_i`` and ``R_i`` are the source and receiver illumination at grid
point ``i``. The construction approximately compensates for geometrical
spreading and uneven coverage; it is the de-facto industrial preconditioner
for FWI.

Reference:
    Plessix, R.-E., and Mulder, W. A. (2004). "Frequency-domain finite-difference
    amplitude-preserving migration." Geophysical Journal International 157(3),
    975-987. DOI:10.1111/j.1365-246X.2004.02289.x.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import torch
from torch import Tensor

from sweep_opt.precond.base import Preconditioner


_TensorLike = Union[Tensor, Dict[int, Tensor]]


def _as_dict(obj: Optional[_TensorLike], params_list) -> Dict[int, Tensor]:
    """Normalize a tensor / dict / None into a dict keyed by id(param)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, Tensor):
        raise TypeError(f"expected Tensor or dict, got {type(obj).__name__}")
    if len(params_list) != 1:
        raise ValueError(
            "Passing a single tensor for source_illum / receiver_illum is only "
            "valid when the optimizer holds a single parameter; got "
            f"{len(params_list)} parameters - use a dict {{id(param): tensor}}."
        )
    return {id(params_list[0]): obj}


class PseudoHessianPrecond(Preconditioner):
    """Diagonal pseudo-Hessian preconditioner.

    Args:
        eps: stabilization added to the source*receiver illumination before
            inversion. A small fraction of the maximum illumination is the
            standard choice; the default ``1e-3`` assumes the user has
            normalized the illumination tensors to unit max already.
        normalize: if True, divide the diagonal by its maximum so the largest
            preconditioned gradient component is unchanged in magnitude. This
            keeps the optimizer's ``lr`` interpretation comparable to the
            unpreconditioned run.

    Notes:
        Without illumination tensors (i.e. ``update`` is never called or is
        called with both ``source_illum`` and ``receiver_illum`` left as
        ``None``), the preconditioner falls back to the identity. Calling
        ``update`` overrides any prior diagonal.
    """

    def __init__(self, eps: float = 1e-3, normalize: bool = True) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        self.normalize = normalize
        self._diag: Dict[int, Tensor] = {}

    def update(
        self,
        params: Iterable[Tensor],
        *,
        source_illum: Optional[_TensorLike] = None,
        receiver_illum: Optional[_TensorLike] = None,
        **_kwargs,
    ) -> None:
        params_list = list(params)
        if source_illum is None and receiver_illum is None:
            self._diag = {}
            return

        s_dict = _as_dict(source_illum, params_list)
        r_dict = _as_dict(receiver_illum, params_list)

        new_diag: Dict[int, Tensor] = {}
        for p in params_list:
            s = s_dict.get(id(p))
            r = r_dict.get(id(p))
            if s is None and r is None:
                # Skip params with no illumination at all.
                continue
            ones = torch.ones_like(p)
            s_t = ones if s is None else s.detach().to(p.device, p.dtype)
            r_t = ones if r is None else r.detach().to(p.device, p.dtype)
            diag = 1.0 / (s_t * r_t + self.eps)
            if self.normalize:
                diag = diag / diag.max()
            new_diag[id(p)] = diag.reshape(p.shape)
        self._diag = new_diag

    def __call__(self, grad: Tensor, param: Tensor) -> Tensor:
        diag = self._diag.get(id(param))
        if diag is None:
            return grad
        return grad * diag

    def diagonal_for(self, param: Tensor) -> Optional[Tensor]:
        """Expose the cached diagonal for a parameter (mostly for tests / plots)."""
        return self._diag.get(id(param))
