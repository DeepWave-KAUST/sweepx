"""Estimate a source wavelet from the direct arrival via rank-1 alternation.

This module implements the algorithm used by the upstream OBN direct-wave
wavelet script: take a small set of
near-offset traces, align each one to its predicted direct-wave arrival
``tau_i = offset_i / water_velocity``, and fit the aligned window with the
rank-1 model ``aligned[i, t] ~= amplitudes[i] * wavelet[t]`` by alternating
weighted least-squares updates.

The algorithm needs only NumPy, so it runs without a wave-equation solver
and is suitable for producing the *initial* wavelet that a downstream SIREN
or sweep wavelet inversion can then refine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectWaveletResult:
    """Outputs of the rank-1 direct-wave wavelet estimator."""

    rel_t_s: np.ndarray
    wavelet: np.ndarray
    amplitudes: np.ndarray
    aligned: np.ndarray
    predicted_aligned: np.ndarray
    valid: np.ndarray
    iterations: int


def align_traces_by_travel_time(
    traces: np.ndarray,
    dt_s: float,
    travel_times_s: np.ndarray,
    rel_t_min_s: float,
    rel_t_max_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample each trace so that t = 0 corresponds to its direct-wave time.

    Parameters
    ----------
    traces:
        Observed traces with shape ``(ntrace, nt)``.
    dt_s:
        Time-axis sample interval in seconds.
    travel_times_s:
        Predicted direct-wave arrival per trace in seconds, shape ``(ntrace,)``.
    rel_t_min_s, rel_t_max_s:
        Inclusive relative-time window for the aligned wavelet support. Use
        a small negative value (for example ``-0.15``) to keep some data
        before the arrival for residual checking.

    Returns
    -------
    rel_t:
        ``(nrel,)`` array of relative times, ``rel_t[0] == rel_t_min_s``.
    aligned:
        ``(ntrace, nrel)`` aligned traces (linear interpolation, zero outside
        the original support).
    valid:
        ``(ntrace, nrel)`` boolean mask where the aligned sample falls inside
        the original trace.
    """

    traces_arr = np.asarray(traces, dtype=np.float64)
    if traces_arr.ndim != 2:
        raise ValueError("traces must be a 2D array of shape (ntrace, nt)")
    if traces_arr.shape[0] != travel_times_s.shape[0]:
        raise ValueError("travel_times_s must have one entry per trace")
    if rel_t_max_s <= rel_t_min_s:
        raise ValueError("rel_t_max_s must be greater than rel_t_min_s")

    rel_t = np.arange(
        float(rel_t_min_s),
        float(rel_t_max_s) + 0.5 * float(dt_s),
        float(dt_s),
        dtype=np.float64,
    )
    sample_t = np.arange(traces_arr.shape[1], dtype=np.float64) * float(dt_s)
    aligned = np.empty((traces_arr.shape[0], rel_t.size), dtype=np.float32)
    valid = np.empty_like(aligned, dtype=bool)
    for itrace, tau in enumerate(np.asarray(travel_times_s, dtype=np.float64)):
        query_t = float(tau) + rel_t
        valid[itrace] = (query_t >= sample_t[0]) & (query_t <= sample_t[-1])
        aligned[itrace] = np.interp(
            query_t, sample_t, traces_arr[itrace], left=0.0, right=0.0
        ).astype(np.float32)
    return rel_t.astype(np.float32), aligned, valid


def estimate_direct_wavelet_rank1(
    traces: np.ndarray,
    dt_s: float,
    travel_times_s: np.ndarray,
    rel_t_min_s: float = -0.15,
    rel_t_max_s: float = 0.65,
    iterations: int = 20,
    normalize_each_iteration: bool = True,
) -> DirectWaveletResult:
    """Fit a rank-1 wavelet * amplitudes model to direct-wave-aligned traces.

    The wavelet is updated in closed form, normalised to absolute-max one,
    and the amplitudes are then refit by least squares on the still-valid
    samples. The procedure converges quickly; 20 iterations is plenty for
    typical near-offset gathers and is the upstream default.

    Parameters
    ----------
    traces, dt_s, travel_times_s, rel_t_min_s, rel_t_max_s:
        Same as :func:`align_traces_by_travel_time`.
    iterations:
        Number of alternating updates. Must be at least one.
    normalize_each_iteration:
        Re-scale the wavelet to absmax one (and absorb the scale into
        ``amplitudes``) at every iteration. The upstream code does this; the
        flag is exposed here for testing only.

    Returns
    -------
    DirectWaveletResult.
    """

    rel_t, aligned, valid = align_traces_by_travel_time(
        traces, dt_s, travel_times_s, rel_t_min_s, rel_t_max_s
    )
    aligned64 = aligned.astype(np.float64)
    valid_f = valid.astype(np.float64)

    amplitudes = np.ones((aligned64.shape[0],), dtype=np.float64)
    wavelet = np.mean(aligned64, axis=0)
    iters = max(1, int(iterations))
    for _ in range(iters):
        # 1. wavelet update (weighted LS)
        numerator = np.sum((amplitudes[:, None] * aligned64) * valid_f, axis=0)
        denominator = np.sum((amplitudes[:, None] ** 2) * valid_f, axis=0)
        wavelet = numerator / np.maximum(denominator, 1.0e-12)
        # 2. normalise wavelet to absmax 1, absorbing scale into amplitudes
        if normalize_each_iteration:
            scale = float(np.max(np.abs(wavelet)))
            if scale > 0.0:
                wavelet = wavelet / scale
                amplitudes = amplitudes * scale
        ww = float(np.sum(wavelet * wavelet))
        if ww <= 0.0:
            break
        # 3. amplitude update (per-trace LS on valid samples)
        for itrace in range(aligned64.shape[0]):
            mask = valid[itrace]
            if not np.any(mask):
                amplitudes[itrace] = 0.0
                continue
            denom = float(np.sum(wavelet[mask] ** 2))
            amplitudes[itrace] = (
                float(np.sum(aligned64[itrace, mask] * wavelet[mask]))
                / max(denom, 1.0e-12)
            )

    predicted_aligned = (amplitudes[:, None] * wavelet[None, :]).astype(np.float32)
    return DirectWaveletResult(
        rel_t_s=rel_t.astype(np.float32),
        wavelet=wavelet.astype(np.float32),
        amplitudes=amplitudes.astype(np.float32),
        aligned=aligned.astype(np.float32),
        predicted_aligned=predicted_aligned,
        valid=valid,
        iterations=iters,
    )


def synthesize_direct_record(
    record_shape: tuple[int, int],
    dt_s: float,
    travel_times_s: np.ndarray,
    rel_t_s: np.ndarray,
    wavelet: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Reconstruct full-length traces by injecting ``a_i * wavelet`` at ``tau_i``.

    Useful for the obs / synthetic / residual QC panel.
    """

    if len(record_shape) != 2:
        raise ValueError("record_shape must be (ntrace, nt)")
    sample_t = np.arange(record_shape[1], dtype=np.float64) * float(dt_s)
    out = np.zeros(record_shape, dtype=np.float32)
    rel_t64 = np.asarray(rel_t_s, dtype=np.float64)
    wavelet64 = np.asarray(wavelet, dtype=np.float64)
    for itrace, tau in enumerate(np.asarray(travel_times_s, dtype=np.float64)):
        out[itrace] = (
            float(amplitudes[itrace])
            * np.interp(sample_t - float(tau), rel_t64, wavelet64, left=0.0, right=0.0)
        ).astype(np.float32)
    return out


def to_causal_wavelet(
    rel_t_s: np.ndarray,
    wavelet: np.ndarray,
    dt_s: float,
    prepad_s: float = 0.0,
    drop_lead_in: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a relative-time wavelet so the time axis starts at zero.

    The downstream wavelet consumers (SIREN wavelet inversion, FWI
    `fwi.wavelet_path`) treat the wavelet as a *source time function*, where
    ``time_s = 0`` is the source-emission start. The rank-1 estimator works
    in *relative* time, where ``rel_t = 0`` is the predicted direct arrival
    and ``rel_t < 0`` covers the noise / lead-in samples *before* the
    predicted arrival. Those negative-``rel_t`` samples have no physical
    meaning as part of the source signature, so by default this helper
    *drops* them before constructing the causal output, ensuring that
    ``causal_wavelet[i] = wavelet[rel_t = i*dt]`` for ``i >= 0``.

    Parameters
    ----------
    rel_t_s, wavelet:
        Relative-time wavelet from :func:`estimate_direct_wavelet_rank1`.
    dt_s:
        Sample interval. Must match the spacing of ``rel_t_s``.
    prepad_s:
        Extra zero pad added at the front (useful when a downstream solver
        expects a small lead-in before the first non-zero wavelet sample).
    drop_lead_in:
        When ``True`` (the default), drop the part of the wavelet with
        ``rel_t < 0`` before constructing the causal output. Set to
        ``False`` to keep the original behaviour (causal sample 0 maps to
        ``rel_t = rel_t_min``).

    Returns
    -------
    time_s, causal_wavelet
        Both have the same length. ``time_s[npad] = prepad_s`` always
        corresponds to the predicted direct-arrival time
        (``rel_t = 0``) when ``drop_lead_in=True``.
    """

    rel_t = np.asarray(rel_t_s, dtype=np.float64)
    wavelet_arr = np.asarray(wavelet, dtype=np.float32)
    if rel_t.shape[0] != wavelet_arr.shape[0]:
        raise ValueError("rel_t_s and wavelet must have the same length")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    if drop_lead_in:
        keep_mask = rel_t >= -0.5 * float(dt_s)
        if np.any(keep_mask):
            first_idx = int(np.argmax(keep_mask))
            wavelet_arr = wavelet_arr[first_idx:]
        else:
            wavelet_arr = wavelet_arr[:0]

    npad = max(0, int(round(float(prepad_s) / float(dt_s))))
    causal = np.concatenate(
        [np.zeros((npad,), dtype=np.float32), wavelet_arr]
    ).astype(np.float32)
    time_s = (np.arange(causal.size, dtype=np.float64) * float(dt_s)).astype(np.float32)
    return time_s, causal


def causal_event_time_s(
    rel_t_s: np.ndarray,
    prepad_s: float = 0.0,
    drop_lead_in: bool = True,
) -> float:
    """Return the time (in causal coordinates) of ``rel_t = 0``.

    With ``drop_lead_in=True`` (matches :func:`to_causal_wavelet` default),
    ``rel_t = 0`` lands exactly at ``time_s = prepad_s``. With
    ``drop_lead_in=False``, it lands at ``time_s = prepad_s - rel_t_min``.
    """

    rel_t = np.asarray(rel_t_s, dtype=np.float64)
    if rel_t.size == 0:
        raise ValueError("rel_t_s must be non-empty")
    if drop_lead_in:
        return float(prepad_s)
    return float(prepad_s) + float(-rel_t[0])


def _l2_normalize_rows(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """L2-normalize each row of a 2D array."""

    arr = np.asarray(values, dtype=np.float64)
    scale = np.sqrt(np.sum(arr * arr, axis=-1, keepdims=True))
    return arr / np.maximum(scale, eps)


def _max_normalize_rows(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Normalize each row of a 2D array by its absolute maximum."""

    arr = np.asarray(values, dtype=np.float64)
    scale = np.max(np.abs(arr), axis=-1, keepdims=True)
    return arr / np.maximum(scale, eps)


def polarity_align_wavelets(
    wavelets: np.ndarray,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip wavelets so they share polarity with a robust reference shape.

    The reference is the L2-normalized median of the L2-normalized inputs
    (or the supplied ``reference``). Each wavelet is multiplied by the sign
    of its dot product with the reference. Wavelets that already align with
    the reference get polarity ``+1``.

    Parameters
    ----------
    wavelets:
        ``(nw, nrel)`` array of estimated wavelets.
    reference:
        Optional reference shape ``(nrel,)``. When ``None``, the median of
        the L2-normalised wavelets is used (matches the upstream
        ``make_average_crg_wavelet`` behaviour).

    Returns
    -------
    aligned:
        Wavelets multiplied by their polarity (max-normalized rows).
    polarity:
        ``+1`` / ``-1`` per wavelet.
    reference:
        The reference shape used (L2-normalized).
    """

    arr = np.asarray(wavelets, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("wavelets must be a 2D array of shape (nw, nrel)")
    normalized = _max_normalize_rows(arr)
    l2_norm = _l2_normalize_rows(normalized)
    if reference is None:
        ref = np.median(l2_norm, axis=0)
    else:
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        if ref.shape[0] != arr.shape[1]:
            raise ValueError("reference length must match the wavelet length")
    ref = ref / max(float(np.linalg.norm(ref)), 1.0e-12)
    polarity = np.sign(l2_norm @ ref)
    polarity[polarity == 0.0] = 1.0
    aligned = normalized * polarity[:, None]
    return aligned.astype(np.float32), polarity.astype(np.float32), ref.astype(np.float32)


def robust_average_wavelets(
    wavelets: np.ndarray,
    initial_mask: np.ndarray | None = None,
    min_shape_corr_to_average: float = 0.9,
) -> dict[str, np.ndarray]:
    """Polarity-align, mean-normalize, and average a stack of wavelets.

    Mirrors the upstream ``make_average_crg_wavelet`` routine: build a
    polarity-aligned and max-normalized stack, take a preliminary mean of the
    quality-passing subset, then keep only wavelets whose shape correlation
    with the preliminary mean exceeds ``min_shape_corr_to_average``. Their
    arithmetic mean (re-normalized to absmax 1) is the "robust average".

    Parameters
    ----------
    wavelets:
        ``(nw, nrel)`` stack.
    initial_mask:
        Boolean ``(nw,)`` mask of candidates that pass an upstream quality
        filter (residual RMS, finite values, etc.). When ``None`` all
        wavelets are considered candidates.
    min_shape_corr_to_average:
        Minimum L2-normalized correlation between an aligned wavelet and the
        preliminary average for inclusion in the final mean.

    Returns
    -------
    Dictionary with keys ``aligned``, ``polarity``, ``reference``, ``mean``,
    ``std``, ``preliminary_mean``, ``shape_corr_to_average``, ``final_mask``,
    and ``average`` (the robust mean, absmax-normalized).
    """

    arr = np.asarray(wavelets, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("wavelets must be a 2D array of shape (nw, nrel)")
    if initial_mask is None:
        initial_mask = np.ones((arr.shape[0],), dtype=bool)
    initial_mask = np.asarray(initial_mask, dtype=bool)
    if initial_mask.shape[0] != arr.shape[0]:
        raise ValueError("initial_mask length must match the number of wavelets")

    aligned, polarity, reference = polarity_align_wavelets(arr)
    aligned64 = aligned.astype(np.float64)

    if not np.any(initial_mask):
        raise ValueError("initial_mask removes every wavelet; cannot average")

    preliminary = np.mean(aligned64[initial_mask], axis=0)
    preliminary_norm = preliminary / max(float(np.linalg.norm(preliminary)), 1.0e-12)
    shape_corr = _l2_normalize_rows(aligned64) @ preliminary_norm
    final_mask = initial_mask & (shape_corr >= float(min_shape_corr_to_average))
    if not np.any(final_mask):
        raise ValueError(
            "No wavelets passed the shape-correlation filter; lower min_shape_corr_to_average"
        )

    average = np.mean(aligned64[final_mask], axis=0)
    scale = max(float(np.max(np.abs(average))), 1.0e-12)
    average = (average / scale).astype(np.float32)
    mean = np.mean(aligned64, axis=0).astype(np.float32)
    std = np.std(aligned64, axis=0).astype(np.float32)
    return {
        "aligned": aligned,
        "polarity": polarity,
        "reference": reference,
        "preliminary_mean": preliminary.astype(np.float32),
        "shape_corr_to_average": shape_corr.astype(np.float32),
        "final_mask": final_mask,
        "mean": mean,
        "std": std,
        "average": average,
    }
