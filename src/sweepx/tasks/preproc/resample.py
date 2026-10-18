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


def _poly_ratio(dt_in: float, dt_out: float) -> tuple[int, int]:
    """Smallest ``(up, down)`` for ``dt_in -> dt_out`` at microsecond resolution."""
    if dt_in <= 0 or dt_out <= 0:
        raise ValueError(f"dt must be positive; got dt_in={dt_in}, dt_out={dt_out}")
    up_raw, down_raw = round(dt_in * 1e6), round(dt_out * 1e6)
    g = gcd(up_raw, down_raw)
    return up_raw // g, down_raw // g


def resample_poly_coeffs(up: int, down: int, *, window=("kaiser", 5.0)):
    """The FIR scipy's ``resample_poly`` uses, plus its trim bookkeeping.

    Returned so a GPU implementation can reproduce ``resample_poly`` exactly
    rather than approximate it — the anti-alias filter *is* the correctness
    of a decimation, so an implementation that differs here silently changes
    the data.
    """
    from scipy.signal import firwin

    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, 1.0 / max_rate, window=window) * up
    return h, half_len


def resample_time_torch(x, dt_in: float, dt_out: float, *, axis: int = -1):
    """GPU polyphase resample, numerically equal to :func:`resample_time`.

    Mirrors ``scipy.signal.resample_poly`` step for step (same Kaiser-windowed
    FIR, same zero-stuffing, same pre/post padding and trim), evaluated with
    ``conv1d`` so it runs on whatever device ``x`` is on.

    Why this and not interpolation: ``dt_in -> dt_out`` here is a *decimation*
    (2 ms -> 6.8 ms drops the Nyquist from 250 Hz to 73.5 Hz), so everything
    above the new Nyquist has to be filtered out or it folds back into the
    retained band. Linear interpolation has no such filter — measured on
    field OBN traces it leaves an error several times larger than the 1.5-2 Hz
    signal being inverted. An FFT resample does filter, but assumes the record
    is periodic and rings at the wrap-around.
    """
    import torch
    import torch.nn.functional as F

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expects a torch.Tensor; got {type(x).__name__}")
    up, down = _poly_ratio(dt_in, dt_out)
    if up == down:
        return x
    xm = x.movedim(axis, -1)
    lead, n_in = xm.shape[:-1], xm.shape[-1]
    xf = xm.reshape(-1, 1, n_in)

    h_np, half_len = resample_poly_coeffs(up, down)
    # scipy's padding so the output is phase-aligned and long enough
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)
    n_pre_pad = (down - half_len % down)
    n_pre_remove = (half_len + n_pre_pad) // down

    def _olen(len_h, n, u, dwn):
        return (((n - 1) * u + len_h) - 1) // dwn + 1

    n_post_pad = 0
    while _olen(len(h_np) + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h_np = np.concatenate([np.zeros(n_pre_pad), h_np, np.zeros(n_post_pad)])
    h = torch.as_tensor(h_np, device=xf.device, dtype=torch.float64)

    # upfirdn: zero-stuff by ``up``, convolve, keep every ``down``-th sample
    xu = xf.new_zeros(xf.shape[0], 1, (n_in - 1) * up + 1, dtype=torch.float64)
    xu[:, :, ::up] = xf.to(torch.float64)
    # conv1d is correlation with 'valid' padding; pad both sides by len(h)-1
    # so we get the FULL convolution upfirdn produces.
    xp = F.pad(xu, (h.numel() - 1, h.numel() - 1))
    y = F.conv1d(xp, h.flip(0).reshape(1, 1, -1))[:, 0, ::down]
    y = y[:, n_pre_remove:n_pre_remove + n_out]
    return y.reshape(*lead, n_out).to(dtype=x.dtype).movedim(-1, axis)


__all__ = ["resample_time", "resample_time_torch", "resample_poly_coeffs"]
