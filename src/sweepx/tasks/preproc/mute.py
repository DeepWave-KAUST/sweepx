"""Direct-wave / linear muting with optional cosine taper.

Time axis is expected to be the first axis by default (`(nt, nrec, ...)`).
Pass ``time_axis=`` to override.
"""

from __future__ import annotations

import numpy as np

from . import _xp


def linear_mute(
    data,
    *,
    dt: float,
    dh: float,
    offsets,
    vmute: float,
    t0: float = 0.0,
    taper: int = 20,
    keep: str = "below",
    time_axis: int = 0,
):
    """Mute samples above/below a linear time-offset line.

    Parameters
    ----------
    data
        Array of shape ``(..., nt, nrec, ...)`` — pass ``time_axis`` to pick.
    dt
        Time sample interval (s).
    dh
        Receiver spacing if `offsets` is given as integer trace indices,
        otherwise unused. (Kept for API symmetry; pass ``1.0`` for true offsets.)
    offsets
        Per-trace offsets (m). 1-D array of length ``nrec``.
    vmute
        Mute velocity (m/s). The mute line is ``t(off) = t0 + off / vmute``.
    t0
        Mute time intercept (s).
    taper
        Cosine taper length in samples; ``0`` for a hard cut.
    keep
        ``"below"`` → keep samples at ``t > t_mute`` (remove direct arrival);
        ``"above"`` → keep samples at ``t < t_mute`` (top mute).
    time_axis
        Position of the time axis.

    Returns
    -------
    Same type as `data`, same shape.
    """
    arr = _xp.to_numpy(data)
    offsets = np.asarray(offsets, dtype="float64")
    nt = arr.shape[time_axis]
    nrec = offsets.shape[0]

    t_mute = t0 + offsets / float(vmute)
    i_mute = np.round(t_mute / dt).astype("int64")
    i_mute = np.clip(i_mute, 0, nt)

    mask = np.ones((nt, nrec), dtype=arr.dtype)
    sample_idx = np.arange(nt)[:, None]
    if keep == "below":
        mask = (sample_idx >= i_mute[None, :]).astype(arr.dtype)
    elif keep == "above":
        mask = (sample_idx <= i_mute[None, :]).astype(arr.dtype)
    else:
        raise ValueError(f"keep must be 'below' or 'above'; got {keep!r}")

    if taper > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper + 2)[1:-1])).astype(arr.dtype)
        for j in range(nrec):
            c = int(i_mute[j])
            if keep == "below":
                lo = max(0, c)
                hi = min(nt, c + taper)
                mask[lo:hi, j] = ramp[: hi - lo]
            else:  # above
                lo = max(0, c - taper)
                hi = min(nt, c)
                mask[lo:hi, j] = ramp[::-1][: hi - lo]

    # broadcast mask onto arr along (time_axis, time_axis+1) under the
    # assumption "(..., nt, nrec, ...)". For arbitrary axes we fall back
    # to explicit reshape.
    arr_moved = np.moveaxis(arr, time_axis, 0)
    if arr_moved.shape[1] != nrec:
        raise ValueError(
            f"linear_mute expects receiver axis right after time axis "
            f"(saw shape {arr.shape}, time_axis={time_axis})."
        )
    out_moved = arr_moved * mask.reshape((nt, nrec) + (1,) * (arr_moved.ndim - 2))
    out = np.moveaxis(out_moved, 0, time_axis).astype(arr.dtype, copy=False)
    return _xp.like(data, out)


__all__ = ["linear_mute"]
