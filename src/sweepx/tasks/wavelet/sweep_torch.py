"""Wavelet inversion utilities built on the sweep PyTorch propagator.

Ported from ``fwi_workflow.wavelet.sweep_torch``. Two adapter-level
changes from the legacy code:

* Data input is a :class:`sweep_io.seismic_plan.SeismicPlan` npz path
  (``WaveletInversionConfig.plan_path``) rather than a legacy
  ``csg_index_v2`` npz + a separate raw SEG-Y path. The plan already
  knows where the SEG-Y bytes live (``row_file_id`` + ``row_trace_offset``)
  and what the time-axis is (``plan.dt_s``, ``plan.samples_per_trace``).
* ``SirenWavelet`` comes from :mod:`sweep_nn.wavelet` (bit-identical to the old
  in-package copy), imported lazily inside the siren-mode branch so the
  CPU-only analyze / convert-farfield paths stay torch-free.

Inner algorithm (rank-1 SIREN prefit + sweep wave-equation refinement)
is unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sweep_io.seismic_plan import PlanReader, SeismicPlan

from .estimation import normalize_wavelet, ricker


@dataclass(frozen=True)
class WaveletInversionConfig:
    """Parameters for constant-velocity wavelet inversion.

    ``plan_path`` is the canonical input: a ``seismic_plan_v1`` npz
    produced by ``sweep-tasks build-plan``. ``shot_start`` / ``shot_stop``
    are 1-based inclusive indices into ``plan.group_id`` (i.e. shot
    ordinals within the plan, not SEG-Y FFIDs).
    """

    plan_path: Path
    output_dir: Path
    shot_start: int = 1
    shot_stop: int = 20
    tmax_s: float = 1.0
    velocity_m_s: float = 1500.0
    dx_m: float = 12.5
    dz_m: float = 12.5
    dy_m: float | None = None  # 3D only; defaults to dx_m when equation='Acoustic3D'
    x_padding_m: float = 500.0
    z_padding_m: float = 100.0
    y_padding_m: float | None = None  # 3D only; defaults to x_padding_m
    model_depth_m: float = 1200.0
    equation: str = "Acoustic"  # 'Acoustic' (2D) or 'Acoustic3D'
    # Optional finer-dt forward: if set, the propagator runs at this dt while
    # obs stays at plan.dt_s. syn is stride-sampled back to plan.dt_s before
    # being compared to obs. Must satisfy plan.dt_s = k * simulation_dt_s for
    # an integer k>=1. None ⇒ propagator dt = plan.dt_s (no resample).
    simulation_dt_s: float | None = None
    # Right-shift obs by this many seconds before the inversion loop so the
    # real direct arrival enters the wavelet's zero-prepad frame (matches
    # legacy fwi_workflow-dev ``delay_traces``). The estimated wavelet's
    # peak then lands near ``observed_delay_s`` instead of chasing the raw
    # obs onset. None ⇒ derive from the initial wavelet's analyze metadata
    # via ``max(0, -rel_t.min()) + prepad_s``; 0.0 ⇒ no shift (sweep-tasks
    # pre-2026-05-22 behaviour, also FWI-runner-compatible if you set the
    # output's ``source_delay_s`` to the measured peak yourself).
    observed_delay_s: float | None = None
    # Optional direct-arrival window mask for the inversion loss. When both
    # min/max are set, the loss only counts samples in
    # ``[arrival + min, arrival + max]`` per (source, receiver) pair, where
    # arrival = |src-rec|/direct_water_velocity_m_s + observed_delay_s.
    # Mirrors legacy ``masked_cosine_loss`` (invert_3d_direct_wavelet_siren_sweep.py:333)
    # — without this mask, SIREN learns "fake source bursts" to fit reflections /
    # multiples that the constant-vp propagator can't reproduce. None on
    # either end ⇒ full-trace loss (sweep-tasks pre-2026-05-22 behaviour).
    direct_rel_t_min_s: float | None = None
    direct_rel_t_max_s: float | None = None
    # Velocity used to predict direct arrivals when building the mask.
    # None ⇒ fall back to ``velocity_m_s`` (the propagator's background v).
    direct_water_velocity_m_s: float | None = None
    backend: str = "cuda"
    geophyai_root: Path | None = None
    mode: str = "discrete"
    initial_frequency_hz: float = 8.0
    epochs: int = 100
    lr: float = 0.01
    loss_type: str = "mse"
    filter_lowcut_hz: float | None = None
    filter_highcut_hz: float | None = None
    filter_order: int = 4
    max_abs_offset_m: float | None = None
    nearest_receivers: int | None = None
    spatial_order: int = 4
    abcn: int = 40
    free_surface: bool = True
    inr_hidden_features: int = 64
    inr_hidden_layers: int = 3
    inr_omega0: float = 30.0
    inr_first_omega0: float | None = None
    inr_hidden_omega0: float = 1.0
    wavelet_taper_start_s: float | None = None
    wavelet_taper_end_s: float | None = None
    wavelet_zero_initial_samples: int = 0
    seed: int = 0
    save_every: int = 10
    initial_wavelet_path: Path | None = None
    prefit_steps: int = 0
    prefit_lr: float = 0.01
    initial_wavelet_prior_weight: float = 0.0
    command: str | None = None


@dataclass(frozen=True)
class PreparedWaveletData:
    """Observed data and gridded acquisition geometry for wavelet inversion.

    ``dt_s`` / ``nt`` describe ``observed`` (plan-side, the recording dt).
    ``simulation_dt_s`` / ``simulation_nt`` describe the propagator-side
    time axis used for the forward wavelet and the syn record before it is
    stride-sampled back to ``dt_s`` for loss against obs. When the user
    leaves ``WaveletInversionConfig.simulation_dt_s`` at None, the two pairs
    are equal and the resample is a no-op.
    """

    observed: np.ndarray
    sources: np.ndarray
    receivers: np.ndarray
    model: np.ndarray
    dt_s: float
    nt: int
    shot_ids: np.ndarray
    x_origin_m: float
    geometry_summary: dict[str, float | int]
    selected_offsets_m: np.ndarray
    simulation_dt_s: float = 0.0
    simulation_nt: int = 0
    simulation_to_plan_stride: int = 1
    # Right-shift already applied to ``observed`` (seconds). The inversion
    # loop relies on this to interpret the SIREN-learned wavelet inside the
    # legacy zero-prepad frame; ``save_wavelet_outputs`` writes the same
    # value as ``source_delay_s`` so the FWI runner re-applies the matching
    # obs shift downstream.
    observed_delay_s: float = 0.0
    # Optional direct-window mask (shape matches ``observed``, dtype
    # float32). 1.0 inside the direct-arrival window, 0.0 outside.
    # None ⇒ full-trace loss.
    direct_window_mask: np.ndarray | None = None

    def __post_init__(self):
        # Default to equality with dt_s/nt when not provided explicitly.
        if self.simulation_dt_s <= 0.0:
            object.__setattr__(self, "simulation_dt_s", float(self.dt_s))
        if self.simulation_nt <= 0:
            object.__setattr__(self, "simulation_nt", int(self.nt))


def _as_optional_float(value: float | str | None) -> float | None:
    """Convert optional numeric config values into floats."""

    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null", "off"):
        return None
    return float(value)


def apply_wavelet_postprocessing(
    wavelet: np.ndarray,
    dt_s: float,
    taper_start_s: float | None,
    taper_end_s: float | None,
    zero_initial_samples: int = 0,
) -> np.ndarray:
    """Apply final wavelet cosine tapering and early-sample zeroing."""

    values = np.asarray(wavelet, dtype=np.float32).copy()
    if taper_start_s is not None:
        start = int(np.floor(float(taper_start_s) / float(dt_s))) + 1
        start = max(0, min(start, values.size))
        if taper_end_s is None:
            end = values.size - 1
        else:
            end = int(np.floor(float(taper_end_s) / float(dt_s))) + 1
        end = max(start, min(end, values.size))
        if end > start:
            phase = np.linspace(0.0, np.pi, end - start, endpoint=False, dtype=np.float32)
            values[start:end] *= 0.5 * (1.0 + np.cos(phase))
        values[end:] = 0.0
    nzero = max(0, min(int(zero_initial_samples), values.size))
    if nzero:
        values[:nzero] = 0.0
    return values


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an npz file into memory and close the underlying zip handle."""

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _normalise_backend_record(record: Any, nshots: int, nreceivers: int, nt: int):
    """Return a sweep record tensor with shape ``(nshot, nreceiver, nt)``."""

    tensor = record
    original_shape = tuple(tensor.shape)
    while tensor.ndim > 3:
        squeeze_axis = None
        for axis, size in enumerate(tensor.shape):
            if int(size) == 1:
                squeeze_axis = axis
                break
        if squeeze_axis is None:
            break
        tensor = tensor.squeeze(squeeze_axis)
    if tensor.ndim == 2 and nshots == 1:
        if tuple(tensor.shape) == (nreceivers, nt):
            return tensor.reshape(1, nreceivers, nt)
        if tuple(tensor.shape) == (nt, nreceivers):
            return tensor.transpose(0, 1).reshape(1, nreceivers, nt)
    if tensor.ndim == 3:
        shape = tuple(int(v) for v in tensor.shape)
        axes = range(3)
        candidates = []
        for shot_axis in axes:
            if shape[shot_axis] != int(nshots):
                continue
            for receiver_axis in axes:
                if receiver_axis == shot_axis or shape[receiver_axis] != int(nreceivers):
                    continue
                for time_axis in axes:
                    if time_axis in (shot_axis, receiver_axis) or shape[time_axis] != int(nt):
                        continue
                    candidates.append((shot_axis, receiver_axis, time_axis))
        if candidates:
            shot_axis, receiver_axis, time_axis = candidates[0]
            return tensor.permute(shot_axis, receiver_axis, time_axis).contiguous()
        if shape[0] == int(nshots) and shape[-1] == int(nt):
            return tensor.reshape(nshots, nreceivers, nt)
        if shape[0] == int(nshots) and shape[1] == int(nt):
            return tensor.transpose(1, 2).contiguous().reshape(nshots, nreceivers, nt)
    raise ValueError(
        "Cannot map sweep record shape "
        f"{original_shape} after squeezing to {tuple(tensor.shape)} "
        f"into canonical (nshot, nreceiver, nt)=({nshots}, {nreceivers}, {nt}). "
        "Check the sweep backend output axis order."
    )


def _decide_filter_type(lowcut_hz: float | None, highcut_hz: float | None) -> str | None:
    """Return the scipy Butterworth filter type for optional low/high cuts."""

    if lowcut_hz is not None and highcut_hz is not None:
        return "bandpass"
    if lowcut_hz is not None:
        return "highpass"
    if highcut_hz is not None:
        return "lowpass"
    return None


def filter_dw(data, freqs, dt: float = 0.001, forder: int = 3, btype: str | None = None, **kwargs):
    """Apply a zero-phase Butterworth filter along the last axis with torch autograd support.

    Args:
        data: Tensor with time samples on the last axis, for example
            ``(nshots, nreceivers, nt)``.
        freqs: Single cutoff frequency or a two-value pass band in Hz.
        dt: Sampling interval in seconds.
        forder: Butterworth filter order.
        btype: Optional scipy Butterworth filter type. If omitted it is inferred
            from the number of cutoff frequencies.
        **kwargs: Reserved for API compatibility with existing filtering helpers.
    """

    import torch
    from scipy import signal

    del kwargs
    try:
        from torchaudio.functional import filtfilt
    except (ImportError, OSError) as exc:
        raise ImportError(
            "torchaudio is required for differentiable Butterworth filtering. "
            "Install torchaudio in the wavelet inversion environment or run with "
            "an environment that provides torchaudio.functional.filtfilt."
        ) from exc

    if isinstance(freqs, (int, float)):
        freqs = [float(freqs)]
    else:
        freqs = [float(freq) for freq in freqs]
    if not freqs:
        return data

    filter_type = btype
    if filter_type is None:
        if len(freqs) == 1:
            filter_type = "lowpass"
        elif len(freqs) == 2:
            filter_type = "bandpass"
        else:
            raise ValueError("freqs must contain one cutoff frequency or a two-frequency band.")

    wn = [2.0 * freq / (1.0 / float(dt)) for freq in freqs]
    wn_value = wn[0] if len(wn) == 1 else wn
    b, a = signal.butter(int(forder), Wn=wn_value, btype=filter_type)
    b_tensor = torch.from_numpy(np.asarray(b)).double().to(data.device)
    a_tensor = torch.from_numpy(np.asarray(a)).double().to(data.device)
    return filtfilt(data.double(), a_tensor, b_tensor, clamp=False).to(dtype=data.dtype)


def _filter_cutoffs(lowcut_hz: float | None, highcut_hz: float | None):
    """Return Butterworth cutoff frequencies in Hz for optional low/high cuts."""

    filter_type = _decide_filter_type(lowcut_hz, highcut_hz)
    if filter_type is None:
        return None, None
    if filter_type == "bandpass":
        return [float(lowcut_hz), float(highcut_hz)], filter_type
    if filter_type == "highpass":
        return [float(lowcut_hz)], filter_type
    return [float(highcut_hz)], filter_type


def _apply_torch_filter(data, freqs, dt_s: float, order: int, btype: str | None):
    """Apply the configured Butterworth filter along the last axis."""

    if freqs is None:
        return data
    return filter_dw(data, freqs, dt=dt_s, forder=order, btype=btype)


def trace_cosine_loss(syn, obs, mask=None, eps: float = 1.0e-8):
    """Return one minus the mean per-trace cosine similarity.

    The input tensors must have time on the last axis and trace dimensions in
    the leading axes. Each trace is demeaned (over the full time axis) before
    computing the dot product. If ``mask`` is provided, the demeaned signals
    are zeroed outside the mask before the inner product — this mirrors
    legacy fwi_workflow-dev's ``masked_cosine_loss`` and lets the user
    restrict the loss to a direct-arrival window.
    """

    syn_centered = syn - syn.mean(dim=-1, keepdim=True)
    obs_centered = obs - obs.mean(dim=-1, keepdim=True)
    if mask is not None:
        syn_centered = syn_centered * mask
        obs_centered = obs_centered * mask
    numerator = (syn_centered * obs_centered).sum(dim=-1)
    syn_norm = syn_centered.square().sum(dim=-1).sqrt()
    obs_norm = obs_centered.square().sum(dim=-1).sqrt()
    cosine = numerator / (syn_norm * obs_norm).clamp_min(eps)
    return 1.0 - cosine.mean()


def compute_matching_loss(syn, obs, loss_type: str, mask=None):
    """Compute the configured wavelet matching loss.

    When ``mask`` is supplied (same shape as ``syn``/``obs``, float 0/1), the
    loss only counts samples where ``mask == 1``. Used by the direct-arrival
    window option to keep reflections / multiples out of the wavelet fit.
    """

    if loss_type == "mse":
        diff_sq = (syn - obs).square()
        if mask is not None:
            mass = mask.sum().clamp_min(1.0)
            return (diff_sq * mask).sum() / mass
        return diff_sq.mean()
    if loss_type == "trace_cosine":
        return trace_cosine_loss(syn, obs, mask=mask)
    raise ValueError("loss_type must be 'mse' or 'trace_cosine'")


def configure_sweep_import(geophyai_root: str | Path | None) -> None:
    """Add a local geophyai checkout to ``sys.path`` when requested."""

    if geophyai_root is None:
        return
    root = Path(geophyai_root).expanduser().resolve()
    for candidate in (root / "src", root):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _deduplicate_receiver_grid_points(
    rows_by_shot: list[np.ndarray],
    metrics_by_shot: list[np.ndarray],
    receiver_xyz_m: np.ndarray,
    receiver_grid: np.ndarray,
    grid_origin_m: np.ndarray,
    grid_spacing_m: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict[str, int]]:
    """Keep the trace closest to each receiver grid point for every shot.

    Works for 2D (last dim = 2, x/z) and 3D (last dim = 3, x/y/z) grids;
    the grid key is the full integer tuple of the last axis.

    Ties are resolved by preserving the first trace in the existing receiver
    order. The output is truncated to the minimum deduplicated receiver count
    across shots so the selected batch remains rectangular.

    ``receiver_xyz_m`` shape: ``(nshots, nreceivers, ndim)`` (metres, physical).
    ``receiver_grid`` shape:  ``(nshots, nreceivers, ndim)`` (integer indices).
    ``grid_origin_m`` shape:  ``(ndim,)`` (physical metres at index 0).
    ``grid_spacing_m`` shape: ``(ndim,)`` (metres per index).
    """

    kept_indices_by_shot: list[np.ndarray] = []
    before_counts = []
    after_counts = []
    nshots, _nreceivers, _ndim = receiver_grid.shape
    for ishot in range(nshots):
        grid_to_best: dict[tuple[int, ...], tuple[int, float]] = {}
        for local_index in range(receiver_grid.shape[1]):
            grid_idx = receiver_grid[ishot, local_index]
            grid_key = tuple(int(v) for v in grid_idx)
            grid_xyz_m = grid_origin_m + grid_idx.astype(np.float64) * grid_spacing_m
            phys_xyz = receiver_xyz_m[ishot, local_index]
            distance = float(np.sum((phys_xyz - grid_xyz_m) ** 2))
            current = grid_to_best.get(grid_key)
            if current is None or distance < current[1]:
                grid_to_best[grid_key] = (local_index, distance)
        kept = np.asarray(sorted(item[0] for item in grid_to_best.values()), dtype=np.int64)
        kept_indices_by_shot.append(kept)
        before_counts.append(int(receiver_grid.shape[1]))
        after_counts.append(int(kept.size))

    common_count = min(after_counts)
    if common_count <= 0:
        raise ValueError("All receivers were removed by receiver grid-point deduplication")

    deduped_rows = [rows[kept[:common_count]] for rows, kept in zip(rows_by_shot, kept_indices_by_shot)]
    deduped_metrics = [metrics[kept[:common_count]] for metrics, kept in zip(metrics_by_shot, kept_indices_by_shot)]
    deduped_xyz = np.stack([values[kept[:common_count]] for values, kept in zip(receiver_xyz_m, kept_indices_by_shot)])
    deduped_grid = np.stack([values[kept[:common_count]] for values, kept in zip(receiver_grid, kept_indices_by_shot)])
    stats = {
        "before_min": min(before_counts),
        "after_min": common_count,
        "removed_total": int(sum(before_counts) - common_count * len(before_counts)),
    }
    return deduped_rows, deduped_metrics, deduped_xyz, deduped_grid.astype(np.int64), stats


def _deduplicate_source_grid_points(
    shot_ids: np.ndarray,
    rows_by_shot: list[np.ndarray],
    metrics_by_shot: list[np.ndarray],
    source_xyz_m: np.ndarray,
    receiver_xyz_m: np.ndarray,
    source_grid: np.ndarray,
    receiver_grid: np.ndarray,
    grid_origin_m: np.ndarray,
    grid_spacing_m: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Keep the shot closest to each source grid point.

    Works for 2D (ndim=2) and 3D (ndim=3) grids. Ties are resolved by
    preserving the first shot in the current shot order.

    ``source_xyz_m`` shape:   ``(nshots, ndim)`` (metres, physical).
    ``source_grid`` shape:    ``(nshots, ndim)`` (integer indices).
    ``receiver_xyz_m`` shape: ``(nshots, nreceivers, ndim)``.
    ``receiver_grid`` shape:  ``(nshots, nreceivers, ndim)``.
    """

    grid_to_best: dict[tuple[int, ...], tuple[int, float]] = {}
    for ishot in range(source_grid.shape[0]):
        grid_idx = source_grid[ishot]
        grid_key = tuple(int(v) for v in grid_idx)
        grid_xyz_m = grid_origin_m + grid_idx.astype(np.float64) * grid_spacing_m
        distance = float(np.sum((source_xyz_m[ishot] - grid_xyz_m) ** 2))
        current = grid_to_best.get(grid_key)
        if current is None or distance < current[1]:
            grid_to_best[grid_key] = (ishot, distance)

    kept_shots = np.asarray(sorted(item[0] for item in grid_to_best.values()), dtype=np.int64)
    stats = {
        "before": int(source_grid.shape[0]),
        "after": int(kept_shots.size),
        "removed": int(source_grid.shape[0] - kept_shots.size),
    }
    return (
        shot_ids[kept_shots],
        [rows_by_shot[index] for index in kept_shots],
        [metrics_by_shot[index] for index in kept_shots],
        source_xyz_m[kept_shots],
        receiver_xyz_m[kept_shots],
        source_grid[kept_shots].astype(np.int64),
        receiver_grid[kept_shots].astype(np.int64),
        stats,
    )


def prepare_wavelet_inversion_data(config: WaveletInversionConfig) -> PreparedWaveletData:
    """Read selected groups from a SeismicPlan, truncate time, map onto a 2D grid.

    Reads the plan, picks ``shot_start..shot_stop`` groups, optionally
    sub-selects the ``nearest_receivers`` closest-offset rows per group,
    deduplicates source/receiver grid points, then reads the selected
    trace bytes via :class:`PlanReader`. Returns observed data and
    grid-mapped source / receiver indices.

    Supports both CSG and CRG plans. Under CSG the propagator source is
    the physical source (one per group); under CRG it is the physical
    receiver (OBN node, one per group) and the row-level physical
    sources are treated as the propagator's receivers — i.e. wavelet
    inversion runs in the reciprocal frame. Downstream variable names
    (``source_x``, ``receiver_x``, ``sources``, ``receivers``) refer to
    the *propagator-frame* roles, not the physical acquisition roles.
    """

    plan = SeismicPlan.load(config.plan_path)
    if plan.grouping not in ("csg", "crg"):
        raise ValueError(
            f"WaveletInversionConfig.plan_path must point at a CSG or CRG plan; "
            f"got grouping={plan.grouping!r}"
        )
    shot_ids = np.asarray(plan.group_id, dtype=np.int64)
    if config.shot_start < 1 or config.shot_stop < config.shot_start:
        raise ValueError("shot_start and shot_stop are one-based inclusive shot ordinals")
    start_i = int(config.shot_start) - 1
    stop_i = int(config.shot_stop)
    if stop_i > shot_ids.size:
        raise IndexError(
            f"Requested shot_stop={config.shot_stop}, but the plan only "
            f"contains {shot_ids.size} {plan.grouping.upper()} groups"
        )

    selected_shot_ids = shot_ids[start_i:stop_i]
    starts = np.asarray(plan.group_offsets[:-1], dtype=np.int64)[start_i:stop_i]
    counts = np.diff(plan.group_offsets).astype(np.int64)[start_i:stop_i]

    # Per-row source / receiver positions live on the plan itself. xy-only
    # offset metric matches what the legacy CSG-index npz stored under
    # the "offset" key (signed for 2-D streamer lines, but |offset| is
    # what the wavelet selection actually uses). For CRG plans the
    # propagator-frame source is the physical receiver (one per group)
    # and the propagator-frame receivers are the physical sources
    # (varying within a group); |offset| = |rec - src| is reciprocity-
    # symmetric so the downstream selection/dedup logic is unchanged.
    if plan.grouping == "crg":
        src_xyz = np.asarray(plan.row_receiver_xyz, dtype=np.float64)
        rec_xyz = np.asarray(plan.row_source_xyz, dtype=np.float64)
    else:
        src_xyz = np.asarray(plan.row_source_xyz, dtype=np.float64)
        rec_xyz = np.asarray(plan.row_receiver_xyz, dtype=np.float64)
    header_offsets = np.hypot(rec_xyz[:, 0] - src_xyz[:, 0],
                              rec_xyz[:, 1] - src_xyz[:, 1])
    # Preserve 2-D streamer sign convention so negative offsets survive
    # the existing diagnostics; |offset| is what the selection uses.
    sign = np.sign(rec_xyz[:, 0] - src_xyz[:, 0])
    sign[sign == 0.0] = 1.0
    header_offsets = header_offsets * sign

    max_abs_offset = _as_optional_float(config.max_abs_offset_m)
    nearest_receivers = None if config.nearest_receivers is None else int(config.nearest_receivers)
    selected_rows_by_shot: list[np.ndarray] = []
    selected_metric_by_shot: list[np.ndarray] = []
    for start, count in zip(starts, counts):
        rows = np.arange(int(start), int(start + count), dtype=np.int64)
        metric = np.abs(header_offsets[rows])
        if max_abs_offset is not None:
            keep = metric <= max_abs_offset
            rows = rows[keep]
            metric = metric[keep]
        if rows.size == 0:
            raise ValueError("Offset selection removed all traces for at least one selected shot")
        order = np.argsort(metric, kind="mergesort")
        rows = rows[order]
        metric = metric[order]
        if nearest_receivers is not None:
            if rows.size < nearest_receivers:
                raise ValueError(
                    f"Requested nearest_receivers={nearest_receivers}, but one shot has only {rows.size} traces"
                )
            rows = rows[:nearest_receivers]
            metric = metric[:nearest_receivers]
        selected_rows_by_shot.append(rows)
        selected_metric_by_shot.append(metric)

    if nearest_receivers is None:
        common_count = min(rows.size for rows in selected_rows_by_shot)
        selected_rows_by_shot = [rows[:common_count] for rows in selected_rows_by_shot]
        selected_metric_by_shot = [metric[:common_count] for metric in selected_metric_by_shot]

    # Per-group source xyz (one per group). For CSG, plan rows in a
    # group share src_xyz so taking starts[i] picks the canonical value;
    # for CRG we already swapped roles above so src_xyz here is the
    # group-level OBN node coord.
    selected_source_xyz_m = np.stack(
        [src_xyz[start] for start in starts], axis=0
    ).astype(np.float64)
    # Row-level receiver xyz (multiple per group, per selected_rows).
    selected_receiver_xyz_m = np.stack(
        [rec_xyz[rows] for rows in selected_rows_by_shot], axis=0
    ).astype(np.float64)

    is_3d = config.equation == "Acoustic3D"
    dx_m = float(config.dx_m)
    dz_m = float(config.dz_m)
    dy_m = float(config.dy_m if config.dy_m is not None else config.dx_m)
    x_padding_m = float(config.x_padding_m)
    z_padding_m = float(config.z_padding_m)
    y_padding_m = float(
        config.y_padding_m if config.y_padding_m is not None else config.x_padding_m
    )

    # ---- grid extent ----
    # x: span of all sources + receivers ± padding
    src_x = selected_source_xyz_m[:, 0]
    rec_x = selected_receiver_xyz_m[..., 0]
    xmin = float(min(src_x.min(), rec_x.min()) - x_padding_m)
    xmax = float(max(src_x.max(), rec_x.max()) + x_padding_m)
    nx = int(math.ceil((xmax - xmin) / dx_m)) + 1

    # z: 0 .. max(model_depth, deepest receiver + padding); grid origin
    # in z is "-z_padding_m" so positive subsurface depths map to
    # nonnegative indices, matching the legacy 2D convention.
    rec_z = selected_receiver_xyz_m[..., 2]
    nz = int(math.ceil(max(config.model_depth_m, rec_z.max() + z_padding_m) / dz_m)) + 1
    z_origin_m = -z_padding_m

    if is_3d:
        src_y = selected_source_xyz_m[:, 1]
        rec_y = selected_receiver_xyz_m[..., 1]
        ymin = float(min(src_y.min(), rec_y.min()) - y_padding_m)
        ymax = float(max(src_y.max(), rec_y.max()) + y_padding_m)
        ny = int(math.ceil((ymax - ymin) / dy_m)) + 1
    else:
        ymin = 0.0
        ny = 1

    if is_3d:
        grid_origin_m = np.array([xmin, ymin, z_origin_m], dtype=np.float64)
        grid_spacing_m = np.array([dx_m, dy_m, dz_m], dtype=np.float64)
        # axis_lengths order matches (x, y, z) — same as sources/receivers.
        axis_lengths = np.array([nx, ny, nz], dtype=np.int64)
    else:
        grid_origin_m = np.array([xmin, z_origin_m], dtype=np.float64)
        grid_spacing_m = np.array([dx_m, dz_m], dtype=np.float64)
        axis_lengths = np.array([nx, nz], dtype=np.int64)

    # ---- grid mapping (physical → integer index) ----
    if is_3d:
        # Drop y when 2D; keep all three when 3D.
        src_select = selected_source_xyz_m[:, [0, 1, 2]]
        rec_select = selected_receiver_xyz_m[..., [0, 1, 2]]
    else:
        src_select = selected_source_xyz_m[:, [0, 2]]
        rec_select = selected_receiver_xyz_m[..., [0, 2]]

    sources = np.rint((src_select - grid_origin_m) / grid_spacing_m).astype(np.int64)
    receivers = np.rint((rec_select - grid_origin_m) / grid_spacing_m).astype(np.int64)
    for axis in range(sources.shape[-1]):
        sources[..., axis] = np.clip(sources[..., axis], 0, int(axis_lengths[axis] - 1))
        receivers[..., axis] = np.clip(receivers[..., axis], 0, int(axis_lengths[axis] - 1))

    # Pass only the spatial coords we actually mapped (drop y in 2D so
    # the dedup distance metric stays in the model frame).
    src_phys = src_select.astype(np.float64)
    rec_phys = rec_select.astype(np.float64)

    (
        selected_shot_ids,
        selected_rows_by_shot,
        selected_metric_by_shot,
        src_phys,
        rec_phys,
        sources,
        receivers,
        source_dedup_stats,
    ) = _deduplicate_source_grid_points(
        selected_shot_ids,
        selected_rows_by_shot,
        selected_metric_by_shot,
        src_phys,
        rec_phys,
        sources,
        receivers,
        grid_origin_m,
        grid_spacing_m,
    )

    (
        selected_rows_by_shot,
        selected_metric_by_shot,
        rec_phys,
        receivers,
        dedup_stats,
    ) = _deduplicate_receiver_grid_points(
        selected_rows_by_shot,
        selected_metric_by_shot,
        rec_phys,
        receivers,
        grid_origin_m,
        grid_spacing_m,
    )
    selected_count = int(receivers.shape[1])
    if selected_count <= 0:
        raise ValueError("No receivers remain after grid-point deduplication")

    selected_rows = np.concatenate(selected_rows_by_shot)
    selected_offsets_m = np.stack(selected_metric_by_shot).astype(np.float32)

    # Pull dt + samples_per_trace from the plan itself; the legacy code
    # parsed the SEG-Y binary header here, but a SeismicPlan caches both
    # at build time so this is a constant-time lookup.
    dt_s = float(plan.dt_s)
    nt = min(int(round(config.tmax_s / dt_s)) + 1, int(plan.samples_per_trace))

    # Optional sub-plan-dt forward: validate the user-supplied
    # ``simulation_dt_s`` divides ``plan.dt_s`` into an integer stride,
    # then size simulation_nt so ``syn[..., ::stride]`` exactly aligns
    # with the plan timeline (giving plan_nt obs-aligned samples).
    if config.simulation_dt_s is None:
        simulation_dt_s = dt_s
        simulation_nt = nt
        simulation_stride = 1
    else:
        sim_dt = float(config.simulation_dt_s)
        if sim_dt <= 0.0 or sim_dt > dt_s + 1.0e-12:
            raise ValueError(
                f"simulation_dt_s must be in (0, plan.dt_s={dt_s}]; got {sim_dt}"
            )
        ratio = dt_s / sim_dt
        simulation_stride = int(round(ratio))
        if not math.isclose(ratio, float(simulation_stride), rel_tol=1.0e-6, abs_tol=1.0e-9):
            raise ValueError(
                f"plan.dt_s={dt_s} must be an integer multiple of "
                f"simulation_dt_s={sim_dt}; got ratio {ratio}"
            )
        simulation_dt_s = dt_s / simulation_stride
        simulation_nt = (nt - 1) * simulation_stride + 1

    # Trace bytes are read lazily via PlanReader. Row order is preserved
    # so the final reshape below matches the (shots, receivers, nt) layout
    # the rest of the pipeline expects. cache_all=False keeps memory
    # bounded for large plans; row order is arbitrary so we don't pay
    # the LRU coalesce cost of read_rows here either.
    with PlanReader(plan, cache_all=False) as reader:
        traces = reader.read_rows(selected_rows)
    observed = traces[:, :nt].reshape(selected_shot_ids.size, selected_count, nt)
    observed = observed.astype(np.float32, copy=False)

    # Demean first so the legacy obs right-shift (below) zeros into a
    # zero-mean buffer; otherwise the prepad samples would be non-zero
    # after demean and break the wavelet-prepad-frame alignment.
    observed -= observed.mean(axis=-1, keepdims=True)

    # ---- legacy-aligned obs delay (right-shift into wavelet prepad frame).
    # Equivalent to ``delay_traces(...)`` at invert_3d_direct_wavelet_siren_sweep.py:223:
    # the SIREN learns a wavelet whose main bang sits at ``observed_delay_s``,
    # not chasing the raw obs onset. Resolved in priority order:
    #   1. explicit config.observed_delay_s
    #   2. analyze npz's legacy formula  max(0, -rel_t.min()) + prepad_s
    #   3. 0.0 (no shift; sweep-tasks pre-alignment behaviour)
    observed_delay_s = 0.0
    if config.observed_delay_s is not None:
        observed_delay_s = float(config.observed_delay_s)
    elif config.initial_wavelet_path is not None:
        try:
            with np.load(config.initial_wavelet_path) as _nz:
                rel_t_min_val = 0.0
                if "rel_t_min_s" in _nz.files:
                    rel_t_min_val = float(np.asarray(_nz["rel_t_min_s"]).item())
                elif "time_relative_s" in _nz.files:
                    rel_t_min_val = float(np.asarray(_nz["time_relative_s"]).min())
                prepad_legacy = 0.0
                if "prepad_s" in _nz.files:
                    prepad_legacy = float(np.asarray(_nz["prepad_s"]).item())
                observed_delay_s = max(0.0, -rel_t_min_val) + max(0.0, prepad_legacy)
        except Exception:
            observed_delay_s = 0.0

    if observed_delay_s > 0.0:
        observed_delay_samples = int(round(observed_delay_s / dt_s))
        if 0 < observed_delay_samples < nt:
            observed = np.roll(observed, shift=observed_delay_samples, axis=-1)
            observed[..., :observed_delay_samples] = 0.0
        else:
            # Caller asked for a shift that doesn't fit the obs window —
            # silently fall back to no shift (the loop still functions, the
            # wavelet just learns the raw frame).
            observed_delay_s = 0.0

    obs_scale = float(np.sqrt(np.mean(observed**2)))
    if obs_scale > 0.0:
        observed /= obs_scale

    # ---- direct-arrival window mask (legacy masked_cosine_loss parity) ----
    # Build only when both rel_t_min and rel_t_max are supplied. Predicted
    # arrival time per (source, receiver) pair = |src-rec| / v_water +
    # observed_delay_s (obs has already been right-shifted by that amount
    # above). Mask[s, r, k] = 1.0 iff t_k ∈ [arrival + min, arrival + max].
    direct_window_mask = None
    if (config.direct_rel_t_min_s is not None
            and config.direct_rel_t_max_s is not None):
        v_dir = float(config.direct_water_velocity_m_s
                       if config.direct_water_velocity_m_s is not None
                       else config.velocity_m_s)
        # Reconstruct propagator-frame physical coords from grid indices,
        # using the same origin/spacing the dedup step used. shape:
        # sources_m (nshots, ndim); receivers_m (nshots, nrec, ndim).
        sources_m = sources.astype(np.float64) * grid_spacing_m + grid_origin_m
        receivers_m = receivers.astype(np.float64) * grid_spacing_m + grid_origin_m
        dists = np.linalg.norm(
            receivers_m - sources_m[:, None, :], axis=-1
        )  # (nshots, nrec)
        arrival = dists / v_dir + float(observed_delay_s)
        t_grid = (np.arange(nt, dtype=np.float64) * dt_s)
        rel_lo = float(config.direct_rel_t_min_s)
        rel_hi = float(config.direct_rel_t_max_s)
        if rel_lo > rel_hi:
            raise ValueError(
                f"direct_rel_t_min_s ({rel_lo}) must be <= direct_rel_t_max_s ({rel_hi})"
            )
        # Broadcast: (nshots, nrec, 1) ± window vs (1, 1, nt).
        lo = arrival[:, :, None] + rel_lo
        hi = arrival[:, :, None] + rel_hi
        mask_bool = (t_grid[None, None, :] >= lo) & (t_grid[None, None, :] <= hi)
        direct_window_mask = mask_bool.astype(np.float32)
        if direct_window_mask.sum() == 0.0:
            raise ValueError(
                "direct-window mask is all zeros — arrival times fell outside "
                f"the trace [0, {nt*dt_s:g}] s window. Check observed_delay_s "
                "and direct_rel_t_min/max_s."
            )

    if is_3d:
        model = np.full((nz, ny, nx), float(config.velocity_m_s), dtype=np.float32)
    else:
        model = np.full((nz, nx), float(config.velocity_m_s), dtype=np.float32)
    summary = {
        "nshots": int(selected_shot_ids.size),
        "nshots_before_grid_dedup": int(source_dedup_stats["before"]),
        "nshots_after_grid_dedup": int(source_dedup_stats["after"]),
        "grid_duplicate_shots_removed": int(source_dedup_stats["removed"]),
        "nreceivers": int(selected_count),
        "nreceivers_before_grid_dedup_min": int(dedup_stats["before_min"]),
        "nreceivers_after_grid_dedup_min": int(dedup_stats["after_min"]),
        "grid_duplicate_traces_removed": int(dedup_stats["removed_total"]),
        "max_abs_offset_m": None if max_abs_offset is None else float(max_abs_offset),
        "nearest_receivers": None if nearest_receivers is None else int(nearest_receivers),
        "selected_offset_min_m": float(selected_offsets_m.min()),
        "selected_offset_max_m": float(selected_offsets_m.max()),
        "nt": int(nt),
        "dt_s": float(dt_s),
        "nx": int(nx),
        "nz": int(nz),
        "dx_m": dx_m,
        "dz_m": dz_m,
        "x_origin_m": float(xmin),
        "equation": str(config.equation),
        "obs_rms_before_normalization": float(obs_scale),
        "simulation_dt_s": float(simulation_dt_s),
        "simulation_nt": int(simulation_nt),
        "simulation_to_plan_stride": int(simulation_stride),
        "observed_delay_s": float(observed_delay_s),
        "direct_window_active": direct_window_mask is not None,
        "direct_window_rel_t_min_s": (
            float(config.direct_rel_t_min_s) if direct_window_mask is not None else None
        ),
        "direct_window_rel_t_max_s": (
            float(config.direct_rel_t_max_s) if direct_window_mask is not None else None
        ),
        "direct_window_active_fraction": (
            float(direct_window_mask.mean()) if direct_window_mask is not None else None
        ),
    }
    if is_3d:
        summary["ny"] = int(ny)
        summary["dy_m"] = dy_m
        summary["y_origin_m"] = float(ymin)
    return PreparedWaveletData(
        observed=observed,
        sources=sources,
        receivers=receivers,
        model=model,
        dt_s=dt_s,
        nt=nt,
        shot_ids=selected_shot_ids,
        x_origin_m=xmin,
        geometry_summary=summary,
        selected_offsets_m=selected_offsets_m,
        simulation_dt_s=float(simulation_dt_s),
        simulation_nt=int(simulation_nt),
        simulation_to_plan_stride=int(simulation_stride),
        observed_delay_s=float(observed_delay_s),
        direct_window_mask=direct_window_mask,
    )


def build_sweep_solver(config: WaveletInversionConfig, data: PreparedWaveletData, device):
    """Create a sweep acoustic PropTorch solver (2D or 3D per config.equation)."""

    from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    if config.equation == "Acoustic3D":
        from sweep.equations import Acoustic3D
        equation = Acoustic3D(spatial_order=config.spatial_order, device=device, backend="torch")
    elif config.equation == "Acoustic":
        from sweep.equations import Acoustic
        equation = Acoustic(spatial_order=config.spatial_order, device=device, backend="torch")
    else:
        raise ValueError(
            f"equation must be 'Acoustic' (2D) or 'Acoustic3D'; got {config.equation!r}"
        )
    common = dict(
        shape=data.model.shape,
        dev=device,
        dh=config.dx_m,
        # Propagator runs at simulation_dt_s (defaults to plan dt_s when the
        # user leaves config.simulation_dt_s at None). Syn output is then
        # stride-sampled back to plan dt_s before being compared to obs.
        dt=data.simulation_dt_s,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=config.abcn,
        free_surface=config.free_surface,
    )
    if config.backend == "cuda":
        cuda_options = CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(storage="gpu"),
            )
        )
        return PropTorch(
            equation, **common,
            backend="torch", impl="c",
            cuda_options=cuda_options,
        )
    if config.backend == "eager":
        return PropTorch(equation, **common, backend="torch", impl="eager")
    raise ValueError("backend must be 'cuda' or 'eager'")


def save_wavelet_outputs(
    output_dir: Path,
    wavelet: np.ndarray,
    initial_wavelet: np.ndarray | None,
    losses: list[float],
    data: PreparedWaveletData,
    config: WaveletInversionConfig,
    obs_filtered: np.ndarray | None = None,
    syn_filtered: np.ndarray | None = None,
    raw_estimated_wavelet: np.ndarray | None = None,
) -> None:
    """Save wavelet inversion arrays, metadata, loss CSV, and QC figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # The wavelet lives on the simulation timeline (which equals the plan
    # timeline when simulation_dt_s is left unset). The npz must report
    # this dt — both ``dt_s`` (for sweep_io's wavelet loader) and the
    # ``time_s`` array are at the wavelet's own sampling rate.
    dt_s = float(data.simulation_dt_s)
    time_s = np.arange(wavelet.size, dtype=np.float32) * dt_s

    # ----- FWI-runner compatibility metadata --------------------------------
    # The runner's ``kind: siren_pipeline_npz`` wavelet loader (via
    # :func:`sweep_io.wavelet.load_wavelet_npz`) needs ``dt_s`` and uses an
    # optional ``source_delay_s`` to right-shift obs so it lines up with
    # syn's main bang.
    #
    # When ``prepare_wavelet_inversion_data`` shifted obs by
    # ``data.observed_delay_s`` (legacy alignment), the SIREN learned a
    # wavelet inside that prepad frame: its main bang sits at — or near —
    # ``observed_delay_s``. Report that as the FWI runner's anchor so the
    # runner reapplies an identical obs shift downstream and syn/obs stay
    # aligned. If obs was NOT shifted (legacy-disabled path), fall back to
    # the wavelet's measured peak so source_delay_s remains self-consistent.
    if data.observed_delay_s > 0.0:
        src_delay_s = float(data.observed_delay_s)
    else:
        src_delay_s = float(np.argmax(np.abs(wavelet))) * dt_s
    src_prepad_s = 0.0
    if config.initial_wavelet_path is not None:
        try:
            with np.load(config.initial_wavelet_path) as _nz:
                if "prepad_s" in _nz.files:
                    src_prepad_s = max(0.0, float(np.asarray(_nz["prepad_s"]).item()))
        except Exception:
            # Best-effort metadata only; absence does not affect correctness.
            pass

    payload = {
        # Primary key. sweep-io's load_wavelet_npz tries "wavelet" first
        # then "optimized_siren_wavelet" — we write both so the npz is
        # compatible with code that hard-codes either.
        "wavelet": wavelet.astype(np.float32),
        "optimized_siren_wavelet": wavelet.astype(np.float32),
        "time_s": time_s,
        # FWI runner / sweep-io needs ``dt_s`` as a scalar (preferred over
        # inferring from ``time_s``).
        "dt_s": np.float64(dt_s),
        # FWI runner reads ``source_delay_s`` via
        # :func:``sweep_tasks._helpers.wavelet_build._get_wavelet_source_delay_s``
        # and right-
        # shifts obs by that many seconds at iter time.
        "source_delay_s": np.float32(src_delay_s),
        # Informational mirrors of the legacy SIREN-pipeline schema —
        # consumers that target the legacy schema find familiar keys.
        "source_prepad_s": np.float32(src_prepad_s),
        # Amount of right-shift actually applied to obs during training.
        # Matches source_delay_s when the legacy delay-traces alignment was
        # active; 0.0 when it was skipped.
        "observed_delay_s": np.float32(float(data.observed_delay_s)),
        # Training artefacts.
        "losses": np.asarray(losses, dtype=np.float64),
        "shot_ids": data.shot_ids,
        "selected_offsets_m": data.selected_offsets_m,
    }
    if raw_estimated_wavelet is not None:
        payload["raw_estimated_wavelet"] = raw_estimated_wavelet.astype(np.float32)
    if initial_wavelet is not None:
        key = "initial_siren_wavelet" if config.mode == "siren" else "initial_wavelet"
        payload[key] = initial_wavelet.astype(np.float32)
    np.savez_compressed(output_dir / "estimated_wavelet.npz", **payload)
    if initial_wavelet is not None:
        initial_payload = {
            "wavelet": np.asarray(initial_wavelet, dtype=np.float32),
            "time_s": time_s,
            "dt_s": np.float64(dt_s),
            "mode": np.asarray(str(config.mode)),
        }
        np.savez_compressed(output_dir / "initial_wavelet.npz", **initial_payload)
    if obs_filtered is not None and syn_filtered is not None:
        np.savez_compressed(
            output_dir / "final_filtered_records.npz",
            obs_filtered=obs_filtered.astype(np.float32),
            syn_filtered=syn_filtered.astype(np.float32),
            time_s=time_s,
            shot_ids=data.shot_ids,
            selected_offsets_m=data.selected_offsets_m,
        )
    with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch, loss in enumerate(losses, start=1):
            writer.writerow([epoch, f"{loss:.10e}"])
    metadata = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "geometry": data.geometry_summary,
        "shot_ids": [int(value) for value in data.shot_ids],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if config.command:
        with (output_dir / "command.txt").open("w", encoding="utf-8") as f:
            f.write(config.command.rstrip() + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        if initial_wavelet is not None:
            label = "initial_siren" if config.mode == "siren" else "initial"
            axes[0].plot(time_s, initial_wavelet, color="0.55", linewidth=1.2, label=label)
        axes[0].plot(time_s, wavelet, color="black", linewidth=1.4, label="estimated")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.25)
        axes[1].plot(np.arange(1, len(losses) + 1), losses, color="tab:red", linewidth=1.4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(config.loss_type)
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "estimated_wavelet.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
        ax.plot(np.arange(1, len(losses) + 1), losses, color="tab:red", linewidth=1.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(config.loss_type)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        if initial_wavelet is not None:
            fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.0))
            label = "initial_siren" if config.mode == "siren" else "initial"
            ax.plot(time_s, initial_wavelet, color="0.35", linewidth=1.4, label=label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(frameon=False)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / "initial_wavelet.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
        if obs_filtered is not None and syn_filtered is not None:
            save_wiggle_comparison(
                output_dir / "wiggle_obs_syn_comparison.png",
                obs_filtered,
                syn_filtered,
                time_s,
                data.shot_ids,
                data.selected_offsets_m,
            )
            save_record_spectrum_comparison(
                output_dir / "spectrum_obs_syn_comparison.png",
                obs_filtered,
                syn_filtered,
                data.dt_s,
            )
    except Exception as exc:  # pragma: no cover - plotting is best-effort on headless clusters.
        print(f"warning: failed to save wavelet QC figure: {exc}")


def save_record_spectrum_comparison(
    output_path: Path,
    obs: np.ndarray,
    syn: np.ndarray,
    dt_s: float,
) -> None:
    """Save mean observed/synthetic amplitude spectra for final matched records."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs_values = np.asarray(obs, dtype=np.float32)
    syn_values = np.asarray(syn, dtype=np.float32)
    nt = obs_values.shape[-1]
    freqs = np.fft.rfftfreq(nt, d=float(dt_s))
    obs_amp = np.abs(np.fft.rfft(obs_values, axis=-1)).mean(axis=(0, 1))
    syn_amp = np.abs(np.fft.rfft(syn_values, axis=-1)).mean(axis=(0, 1))
    obs_amp = obs_amp / max(float(obs_amp.max()), 1.0e-12)
    syn_amp = syn_amp / max(float(syn_amp.max()), 1.0e-12)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
    ax.plot(freqs, obs_amp, color="black", linewidth=1.4, label="obs filtered")
    ax.plot(freqs, syn_amp, color="tab:red", linewidth=1.4, label="syn filtered")
    ax.set_xlim(0.0, min(100.0, float(freqs[-1])))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized mean amplitude")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_wiggle_comparison(
    output_path: Path,
    obs: np.ndarray,
    syn: np.ndarray,
    time_s: np.ndarray,
    shot_ids: np.ndarray,
    offsets_m: np.ndarray,
    max_traces: int = 40,
) -> None:
    """Save normalized obs vs final-wavelet syn wiggle overlays for representative shots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nshots, nreceivers, _ = obs.shape
    shot_indices = np.unique(np.linspace(0, nshots - 1, min(3, nshots), dtype=np.int64))
    fig, axes = plt.subplots(1, shot_indices.size, figsize=(6.0 * shot_indices.size, 5.0), squeeze=False)
    for ax, ishot in zip(axes[0], shot_indices):
        stride = max(1, int(math.ceil(nreceivers / max_traces)))
        rec_indices = np.arange(0, nreceivers, stride, dtype=np.int64)
        obs_panel = _normalize_wiggle_traces(obs[ishot, rec_indices])
        syn_panel = _normalize_wiggle_traces(syn[ishot, rec_indices])
        scale = np.percentile(np.abs(np.concatenate([obs_panel.ravel(), syn_panel.ravel()])), 99.0)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        trace_spacing = 1.0
        amp_scale = 0.45 * trace_spacing / scale
        for itrace, receiver_index in enumerate(rec_indices):
            x0 = float(itrace)
            ax.plot(x0 + obs_panel[itrace] * amp_scale, time_s, color="black", linewidth=0.7)
            ax.plot(x0 + syn_panel[itrace] * amp_scale, time_s, color="tab:red", linewidth=0.7, alpha=0.85)
        tick_positions = np.linspace(0, rec_indices.size - 1, min(6, rec_indices.size), dtype=np.int64)
        tick_offsets = offsets_m[ishot, rec_indices[tick_positions]]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{value:.0f}" for value in tick_offsets], rotation=30, ha="right")
        ax.set_title(f"FFID {int(shot_ids[ishot])}")
        ax.set_xlabel("Offset (m)")
        ax.set_ylabel("Time (s)")
        ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].plot([], [], color="black", linewidth=1.0, label="obs filtered")
    axes[0, 0].plot([], [], color="tab:red", linewidth=1.0, label="final-wavelet syn (sweep solver)")
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _normalize_wiggle_traces(traces: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Normalize each wiggle trace by its own maximum absolute amplitude."""

    values = np.asarray(traces, dtype=np.float32)
    scale = np.max(np.abs(values), axis=-1, keepdims=True)
    return values / np.maximum(scale, eps)


def _load_initial_wavelet_to_grid(
    path: str | Path,
    dt_s: float,
    nt: int,
) -> np.ndarray:
    """Load a wavelet npz / npy and resample-pad it onto a (dt_s, nt) grid.

    The output is a length-``nt`` causal source time function suitable for use
    as an initial wavelet target for the SIREN/discrete inversion. If the
    source file stores ``time_s``, the wavelet is resampled to ``dt_s`` via
    linear interpolation; otherwise it is assumed to share ``dt_s`` already.
    Excess samples are truncated; a shorter source wavelet is zero-padded to
    fill the trailing samples.
    """

    src_path = Path(path)
    if src_path.suffix == ".npz":
        with np.load(src_path) as data:
            if "wavelet" in data.files:
                wavelet = np.asarray(data["wavelet"], dtype=np.float64).reshape(-1)
            else:
                wavelet = np.asarray(data[data.files[0]], dtype=np.float64).reshape(-1)
            src_dt = float(dt_s)
            if "time_s" in data.files:
                time_s = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
                if time_s.size > 1:
                    src_dt = float(np.median(np.diff(time_s)))
    else:
        wavelet = np.asarray(np.load(src_path), dtype=np.float64).reshape(-1)
        src_dt = float(dt_s)

    if not math.isclose(src_dt, float(dt_s), rel_tol=1.0e-9, abs_tol=1.0e-12):
        old_t = np.arange(wavelet.size, dtype=np.float64) * src_dt
        new_t = np.arange(int(nt), dtype=np.float64) * float(dt_s)
        wavelet = np.interp(new_t, old_t, wavelet, left=0.0, right=0.0)

    out = np.zeros(int(nt), dtype=np.float32)
    n_copy = min(wavelet.size, int(nt))
    out[:n_copy] = wavelet[:n_copy].astype(np.float32)
    return out


def run_wavelet_inversion(config: WaveletInversionConfig) -> Path:
    """Invert a source wavelet from selected real-data CSGs."""

    import torch

    configure_sweep_import(config.geophyai_root)
    if config.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested, but torch.cuda.is_available() is false")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = prepare_wavelet_inversion_data(config)
    device = torch.device("cuda" if config.backend == "cuda" else "cpu")
    solver = build_sweep_solver(config, data, device)

    model = torch.as_tensor(data.model, dtype=torch.float32, device=device)
    sources = data.sources
    receivers = data.receivers
    obs = torch.as_tensor(data.observed, dtype=torch.float32, device=device)
    filter_freqs, filter_type = _filter_cutoffs(config.filter_lowcut_hz, config.filter_highcut_hz)
    obs_filt = _apply_torch_filter(obs, filter_freqs, data.dt_s, config.filter_order, filter_type)
    # Direct-window mask (legacy parity). Same shape as obs/syn at plan dt;
    # forward loop applies it inside compute_matching_loss.
    if data.direct_window_mask is not None:
        loss_mask = torch.as_tensor(data.direct_window_mask, dtype=torch.float32, device=device)
        print(
            f"[estimate-wavelet] direct-window mask active "
            f"({float(loss_mask.mean()):.2%} of samples kept)",
            flush=True,
        )
    else:
        loss_mask = None

    # Wavelet lives on the simulation timeline (so the propagator can use a
    # finer dt than the plan); obs lives on the plan timeline. They're kept
    # apart by simulation_dt_s vs data.dt_s and reconciled via stride
    # sampling of syn before the loss.
    initial_wavelet_np: np.ndarray | None = None
    if config.mode == "discrete":
        initial_wavelet_np = normalize_wavelet(
            ricker(config.initial_frequency_hz, data.simulation_dt_s, data.simulation_nt,
                   peak_time_s=1.0 / config.initial_frequency_hz)
        ).astype(np.float32)
        wavelet_param = torch.nn.Parameter(torch.as_tensor(initial_wavelet_np, dtype=torch.float32, device=device))
        parameters = [wavelet_param]

        def current_wavelet():
            return wavelet_param

    elif config.mode == "siren":
        # Local import: sweep_nn.wavelet pulls torch at module load, so defer it
        # to this siren-inversion branch (which needs torch anyway) rather than
        # the module top. Bit-identical to the retired in-package
        # _siren.SirenWavelet (verified: same init + forward under one seed).
        from sweep_nn.wavelet import SirenWavelet

        siren = SirenWavelet(
            data.simulation_nt,
            hidden_features=config.inr_hidden_features,
            hidden_layers=config.inr_hidden_layers,
            first_omega0=(config.inr_first_omega0
                          if config.inr_first_omega0 is not None
                          else config.inr_omega0),
            hidden_omega0=config.inr_hidden_omega0,
            bias=True,
        ).to(device)
        parameters = list(siren.parameters())

        def current_wavelet():
            return siren()

    else:
        raise ValueError("mode must be 'discrete' or 'siren'")

    if config.mode == "siren":
        initial_wavelet_np = current_wavelet().detach().cpu().numpy().astype(np.float32)

    target_initial_wavelet_np: np.ndarray | None = None
    prefit_wavelet_np: np.ndarray | None = None
    prefit_losses: list[float] = []
    if config.initial_wavelet_path is not None:
        target_initial_wavelet_np = _load_initial_wavelet_to_grid(
            config.initial_wavelet_path, data.simulation_dt_s, data.simulation_nt,
        )
        target_tensor = torch.as_tensor(
            target_initial_wavelet_np, dtype=torch.float32, device=device
        )
        if config.mode == "discrete":
            with torch.no_grad():
                wavelet_param.copy_(target_tensor)
            prefit_wavelet_np = target_initial_wavelet_np.copy()
            print(
                f"prefit (discrete): wavelet initialised from {config.initial_wavelet_path}",
                flush=True,
            )
        else:
            prefit_steps = max(0, int(config.prefit_steps))
            if prefit_steps > 0:
                # LBFGS converges the MSE prefit in O(5) outer iterations and is
                # far more stable than Adam for this fixed-target fitting task.
                # config.prefit_steps acts as the number of LBFGS outer steps.
                prefit_optimizer = torch.optim.LBFGS(
                    parameters,
                    lr=float(config.prefit_lr) if float(config.prefit_lr) > 0.0 else 1.0,
                    max_iter=20,
                    line_search_fn="strong_wolfe",
                )

                def _prefit_closure():
                    prefit_optimizer.zero_grad(set_to_none=True)
                    current = current_wavelet()
                    loss = torch.mean((current - target_tensor) ** 2)
                    loss.backward()
                    return loss

                target_norm = float(torch.sum(target_tensor * target_tensor).clamp(min=1.0e-12))
                for step in range(1, prefit_steps + 1):
                    loss = prefit_optimizer.step(_prefit_closure)
                    loss_value = float(loss.detach().cpu()) if loss is not None else 0.0
                    prefit_losses.append(loss_value)
                    if (
                        step == 1
                        or step % max(config.save_every, 1) == 0
                        or step == prefit_steps
                    ):
                        rel_err = loss_value * float(data.simulation_nt) / target_norm
                        print(
                            f"prefit {step:05d} mse {loss_value:.6e} rel {rel_err:.4e}",
                            flush=True,
                        )
            prefit_wavelet_np = current_wavelet().detach().cpu().numpy().astype(np.float32)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        prefit_time_s = (
            np.arange(int(data.simulation_nt), dtype=np.float64) * float(data.simulation_dt_s)
        ).astype(np.float32)
        np.savez_compressed(
            config.output_dir / "initial_wavelet_target.npz",
            wavelet=target_initial_wavelet_np.astype(np.float32),
            time_s=prefit_time_s,
            source_path=np.asarray(str(config.initial_wavelet_path)),
        )
        prefit_payload = {
            "wavelet": prefit_wavelet_np.astype(np.float32),
            "target": target_initial_wavelet_np.astype(np.float32),
            "time_s": prefit_time_s,
            "mode": np.asarray(str(config.mode)),
        }
        if prefit_losses:
            prefit_payload["mse_losses"] = np.asarray(prefit_losses, dtype=np.float64)
        np.savez_compressed(
            config.output_dir / "prefit_siren_wavelet.npz", **prefit_payload
        )

    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    losses: list[float] = []
    prior_weight = float(config.initial_wavelet_prior_weight)
    prior_target = (
        target_tensor
        if (prior_weight > 0.0 and target_initial_wavelet_np is not None)
        else None
    )
    stride = int(data.simulation_to_plan_stride)
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        wavelet = current_wavelet()
        syn = solver(wavelet, sources, receivers, models=[model])
        # syn is on the simulation timeline (simulation_nt samples at
        # simulation_dt_s). Stride-sample it back to the plan timeline
        # (data.nt samples at data.dt_s) so the loss compares syn and obs
        # in the same frame as the SEG-Y recording.
        syn = _normalise_backend_record(syn, data.observed.shape[0], data.observed.shape[1], data.simulation_nt)
        if stride > 1:
            syn = syn[..., ::stride].contiguous()
        syn_filt = _apply_torch_filter(syn, filter_freqs, data.dt_s, config.filter_order, filter_type)
        match_loss = compute_matching_loss(syn_filt, obs_filt, config.loss_type, mask=loss_mask)
        if prior_target is not None:
            prior_loss = torch.mean((wavelet - prior_target) ** 2)
            loss = match_loss + prior_weight * prior_loss
        else:
            prior_loss = None
            loss = match_loss
        loss.backward()
        optimizer.step()
        loss_value = float(match_loss.detach().cpu())
        losses.append(loss_value)
        if epoch == 1 or epoch % max(config.save_every, 1) == 0 or epoch == config.epochs:
            if prior_target is not None:
                prior_value = float(prior_loss.detach().cpu())
                total_value = float(loss.detach().cpu())
                print(
                    f"epoch {epoch:05d} match {loss_value:.6e} prior {prior_value:.6e} total {total_value:.6e}",
                    flush=True,
                )
            else:
                print(f"epoch {epoch:05d} loss {loss_value:.6e}", flush=True)

    raw_wavelet_np = current_wavelet().detach().cpu().numpy().astype(np.float32)
    wavelet_np = apply_wavelet_postprocessing(
        raw_wavelet_np,
        # Wavelet is on the simulation timeline; taper-time params (taper
        # start/end, zero-initial-samples) are interpreted at that dt so
        # the integer sample counts come out right.
        data.simulation_dt_s,
        config.wavelet_taper_start_s,
        config.wavelet_taper_end_s,
        config.wavelet_zero_initial_samples,
    )
    with torch.no_grad():
        final_wavelet = torch.as_tensor(wavelet_np, dtype=torch.float32, device=device)
        final_syn = solver(final_wavelet, sources, receivers, models=[model])
        final_syn = _normalise_backend_record(
            final_syn, data.observed.shape[0], data.observed.shape[1], data.simulation_nt
        )
        if stride > 1:
            final_syn = final_syn[..., ::stride].contiguous()
        final_syn_filt = _apply_torch_filter(
            final_syn,
            filter_freqs,
            data.dt_s,
            config.filter_order,
            filter_type,
        ).detach().cpu().numpy()
        obs_filt_np = obs_filt.detach().cpu().numpy()
    save_wavelet_outputs(
        config.output_dir,
        wavelet_np,
        initial_wavelet_np,
        losses,
        data,
        config,
        obs_filtered=obs_filt_np,
        syn_filtered=final_syn_filt,
        raw_estimated_wavelet=raw_wavelet_np,
    )
    return config.output_dir / "estimated_wavelet.npz"
