"""Tiny array-API shim: detect numpy vs torch input, dispatch accordingly.

Keeps the public functions in `filter`, `normalize`, etc., short and lets us
honor torch's autograd / device semantics when callers pass tensors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # optional
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]


def is_torch(x: Any) -> bool:
    return _torch is not None and isinstance(x, _torch.Tensor)


def to_numpy(x: Any) -> np.ndarray:
    if is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def like(x_ref: Any, arr: np.ndarray) -> Any:
    """Cast `arr` (numpy) back to whatever array library `x_ref` came from."""
    if is_torch(x_ref):
        return _torch.from_numpy(arr).to(x_ref.device).to(x_ref.dtype)
    return arr
