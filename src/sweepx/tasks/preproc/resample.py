"""Time-axis resampling via polyphase filtering.

Thin wrapper around ``scipy.signal.resample_poly`` so callers don't have to
compute up/down ratios by hand.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import _xp


def resample_time(data, dt_in: float, dt_out: float, *, axis: int = 0):
    """Resample along the time axis from ``dt_in`` to ``dt_out``.

    Uses polyphase filtering (``scipy.signal.resample_poly``), so the rate
    change must be expressible as a rational; the helper finds the
    smallest ``(up, down)`` automatically.
    """
    if dt_in <= 0 or dt_out <= 0:
        raise ValueError(f"dt must be positive; got dt_in={dt_in}, dt_out={dt_out}")
    arr = _xp.to_numpy(data)
    # resample_poly: out_rate = in_rate * (up / down). We want
    # out_rate / in_rate = dt_in / dt_out, so up/down = dt_in/dt_out.
    # Express that ratio as integers with microsecond resolution.
    up_raw = round(dt_in * 1e6)
    down_raw = round(dt_out * 1e6)
    g = gcd(up_raw, down_raw)
    up, down = up_raw // g, down_raw // g
    y = resample_poly(arr, up=up, down=down, axis=axis)
    return _xp.like(data, np.ascontiguousarray(y).astype(arr.dtype, copy=False))


__all__ = ["resample_time"]
