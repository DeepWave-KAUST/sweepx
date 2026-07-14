"""Source-wavelet estimation.

Currently provides the canonical match-filter / Wiener estimator:
solve in the frequency domain for the wavelet ``w`` that minimizes
``|| obs - w * syn ||_2`` per shot, then optionally average across shots.
"""

from __future__ import annotations

import numpy as np

from . import _xp


def estimate_match_filter(
    obs,
    syn,
    *,
    time_axis: int = 0,
    eps: float = 1e-8,
    average_shots: bool = True,
):
    """Wiener / match-filter source-wavelet estimate.

    Parameters
    ----------
    obs, syn
        Observed and synthetic data of the same shape. The wavelet is
        estimated along ``time_axis``; all other axes are treated as
        independent samples (receivers / shots).
    eps
        Regularization on the spectrum denominator (relative to its max).
    average_shots
        If ``True`` (default), reduce the estimate to a single wavelet by
        averaging across all non-time axes. If ``False`` return one
        wavelet per receiver/shot.

    Returns
    -------
    Same array type as `obs`. Shape:
    ``(nt,)`` if ``average_shots`` else `obs`-shape with time axis preserved.
    """
    obs_n = _xp.to_numpy(obs)
    syn_n = _xp.to_numpy(syn)
    if obs_n.shape != syn_n.shape:
        raise ValueError(f"obs/syn shape mismatch: {obs_n.shape} vs {syn_n.shape}")
    obs_t = np.moveaxis(obs_n, time_axis, 0)
    syn_t = np.moveaxis(syn_n, time_axis, 0)
    nt = obs_t.shape[0]

    O = np.fft.rfft(obs_t, axis=0)
    S = np.fft.rfft(syn_t, axis=0)
    denom = (S * np.conj(S)).real
    reg = eps * float(denom.max())
    W = (O * np.conj(S)) / (denom + reg)

    if average_shots:
        if W.ndim > 1:
            W = W.reshape(W.shape[0], -1).mean(axis=1)
        w = np.fft.irfft(W, n=nt)
        return _xp.like(obs, w.astype(obs_n.dtype, copy=False))

    w = np.fft.irfft(W, n=nt, axis=0)
    w = np.moveaxis(w, 0, time_axis)
    return _xp.like(obs, w.astype(obs_n.dtype, copy=False))


__all__ = ["estimate_match_filter"]
