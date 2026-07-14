"""Time-axis resampling via polyphase filtering (``scipy.signal.resample_poly``).

Self-contained numpy-only copy of the resampler formerly in
``sweep_preproc.resample`` — that package was absorbed into
``sweep_tasks.preproc`` and retired. Kept here so sweep-io's optional
``DataPlan.dt_target_s`` feature has no cross-package dependency, and so
sweep-io never has to import sweep-tasks (which would be a dependency cycle:
sweep-tasks already imports sweep-io).

The numerical path is identical to the original (same ``(up, down)`` rational
at microsecond resolution, same ``resample_poly`` call, same dtype handling).
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_time(data: "np.ndarray", dt_in: float, dt_out: float, *, axis: int = 0) -> "np.ndarray":
    """Resample a numpy array along ``axis`` from ``dt_in`` to ``dt_out``.

    Uses polyphase filtering, so the rate change must be rational; the smallest
    ``(up, down)`` is found automatically with microsecond resolution.
    """
    if dt_in <= 0 or dt_out <= 0:
        raise ValueError(f"dt must be positive; got dt_in={dt_in}, dt_out={dt_out}")
    arr = np.asarray(data)
    up_raw = round(dt_in * 1e6)
    down_raw = round(dt_out * 1e6)
    g = gcd(up_raw, down_raw)
    up, down = up_raw // g, down_raw // g
    y = resample_poly(arr, up=up, down=down, axis=axis)
    return np.ascontiguousarray(y).astype(arr.dtype, copy=False)
