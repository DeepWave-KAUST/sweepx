"""Trace normalization.

Default mode is RMS — divide each trace by its root-mean-square amplitude.
"""

from __future__ import annotations

import numpy as np

from . import _xp


def trace_normalize(
    data,
    *,
    mode: str = "rms",
    time_axis: int = 0,
    eps: float = 1e-12,
):
    """Normalize per-trace amplitudes.

    Parameters
    ----------
    data
        Array containing one or more traces. The reduction is along
        ``time_axis``; everything else is treated as the trace index.
    mode
        ``"rms"`` (default), ``"max"`` (peak amplitude), or ``"l2"``.
    time_axis
        Axis to reduce over.
    eps
        Floor on the divisor to avoid divide-by-zero.
    """
    arr = _xp.to_numpy(data)
    if mode == "rms":
        scale = np.sqrt(np.mean(arr * arr, axis=time_axis, keepdims=True))
    elif mode == "max":
        scale = np.max(np.abs(arr), axis=time_axis, keepdims=True)
    elif mode == "l2":
        scale = np.sqrt(np.sum(arr * arr, axis=time_axis, keepdims=True))
    else:
        raise ValueError(f"mode must be 'rms', 'max', or 'l2'; got {mode!r}")
    out = arr / np.maximum(scale, eps)
    return _xp.like(data, out.astype(arr.dtype, copy=False))


__all__ = ["trace_normalize"]
