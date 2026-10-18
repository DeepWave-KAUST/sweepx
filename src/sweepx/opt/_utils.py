"""Internal helpers shared across sweep-opt optimizers."""
from __future__ import annotations

from typing import Iterable, List

import torch
from torch import Tensor


def iter_params(param_groups) -> Iterable[Tensor]:
    """Yield every parameter across all param groups, in the order torch uses internally."""
    for group in param_groups:
        for p in group["params"]:
            yield p


def flat_params(param_groups) -> Tensor:
    """Concatenate all parameters into a single 1-D tensor (copy)."""
    return torch.cat([p.detach().reshape(-1) for p in iter_params(param_groups)])


def flat_grads(param_groups, fill_zero: bool = True) -> Tensor:
    """Concatenate every parameter's gradient into a single 1-D tensor.

    If a parameter has no .grad and ``fill_zero`` is True, treat it as zero.
    """
    chunks: List[Tensor] = []
    for p in iter_params(param_groups):
        if p.grad is None:
            if fill_zero:
                chunks.append(torch.zeros_like(p).reshape(-1))
            else:
                raise RuntimeError("Parameter has no gradient; call backward() first.")
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def write_flat_to_params(param_groups, flat: Tensor) -> None:
    """In-place copy a flat 1-D tensor back into the parameters."""
    offset = 0
    for p in iter_params(param_groups):
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n
    if offset != flat.numel():
        raise RuntimeError(
            f"flat tensor length {flat.numel()} does not match parameter count {offset}"
        )


def add_flat_to_params(param_groups, flat: Tensor, alpha: float = 1.0) -> None:
    """In-place add ``alpha * flat`` to the parameters."""
    offset = 0
    for p in iter_params(param_groups):
        n = p.numel()
        p.data.add_(flat[offset : offset + n].view_as(p), alpha=alpha)
        offset += n
