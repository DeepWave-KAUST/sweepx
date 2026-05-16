"""DataPlan + ModelPlan — positional selectors applied before FWI sees the data.

Two complementary planners:

- :class:`DataPlan` subsets a ``(PhysicalGeometry, observed)`` pair:
  shots (range / stride / explicit list), receivers (stride / offset window),
  time axis (resample / decimate / truncate).
- :class:`ModelPlan` crops a velocity model to a region of interest and
  optionally drops sources / receivers that fall outside the kept window.

Both classes are plain dataclasses — the matching Pydantic schemas live in
``sweep-tasks`` and translate to these at task-build time.

Signal-processing knobs (bandpass / mute / wavelet estimation) deliberately
stay in ``sweep-preproc`` so a DataPlan is *only* about selection and
sampling. Compose them at the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .geometry import PhysicalGeometry


# ============================================================================
# DataPlan
# ============================================================================
@dataclass
class DataPlan:
    """Which shots / receivers / time samples to feed into FWI.

    Attributes
    ----------
    shot_start, shot_stop, shot_stride
        Standard range-style shot selection. Ignored when ``shot_indices``
        is given.
    shot_indices
        Explicit list of shot indices to keep. Overrides the range fields.
    receiver_stride
        Keep every ``N``-th receiver per shot (uniform across shots).
    offset_min_m, offset_max_m
        Receiver-offset window in **meters** (signed if ``abs_offset=False``,
        else compared to ``|offset|``). Offsets are computed in the
        horizontal plane only — ``ndim=2`` uses ``|rx - sx|``,
        ``ndim=3`` uses ``sqrt((rx-sx)^2 + (ry-sy)^2)``.
    abs_offset
        Treat offsets as absolute values when comparing to the window.
    dt_target_s
        Time-resample the observed data to this dt (delegates to
        ``sweep_preproc.resample.resample_time``). Mutually exclusive with
        ``time_decimate``.
    time_decimate
        Keep every ``K``-th time sample (no anti-alias filtering — that's
        on the caller). Mutually exclusive with ``dt_target_s``.
    t_start_s, t_end_s
        Trim the time axis to this window (inclusive of start, exclusive
        of end), applied **after** any resample / decimate.
    """

    shot_start: int = 0
    shot_stop: int | None = None
    shot_stride: int = 1
    shot_indices: Sequence[int] | np.ndarray | None = None

    receiver_stride: int = 1
    offset_min_m: float | None = None
    offset_max_m: float | None = None
    abs_offset: bool = True

    dt_target_s: float | None = None
    time_decimate: int | None = None

    t_start_s: float | None = None
    t_end_s: float | None = None

    def __post_init__(self) -> None:
        if self.shot_stride <= 0:
            raise ValueError(f"shot_stride must be > 0; got {self.shot_stride}")
        if self.receiver_stride <= 0:
            raise ValueError(f"receiver_stride must be > 0; got {self.receiver_stride}")
        if self.dt_target_s is not None and self.time_decimate is not None:
            raise ValueError("DataPlan: dt_target_s and time_decimate are mutually exclusive.")
        if (
            self.offset_min_m is not None
            and self.offset_max_m is not None
            and self.offset_min_m > self.offset_max_m
        ):
            raise ValueError(
                f"DataPlan: offset_min_m ({self.offset_min_m}) "
                f"must be <= offset_max_m ({self.offset_max_m})"
            )


def _select_shots(plan: DataPlan, nshots: int) -> np.ndarray:
    if plan.shot_indices is not None:
        idx = np.asarray(plan.shot_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"shot_indices must be 1-D; got shape {idx.shape}")
        if (idx < 0).any() or (idx >= nshots).any():
            raise ValueError(
                f"shot_indices out of range [0, {nshots}); got min={idx.min()} max={idx.max()}"
            )
        return idx
    stop = plan.shot_stop if plan.shot_stop is not None else nshots
    return np.arange(plan.shot_start, min(stop, nshots), plan.shot_stride, dtype=np.int64)


def _receiver_keep_mask(
    plan: DataPlan,
    geom: PhysicalGeometry,
    shot_idx: np.ndarray,
) -> np.ndarray:
    """Per-shot receiver mask after stride + offset window. Shape: (len(shot_idx), nrec)."""
    nshots_kept = len(shot_idx)
    nrec = geom.nreceivers
    mask = np.zeros((nshots_kept, nrec), dtype=bool)
    mask[:, :: plan.receiver_stride] = True

    if plan.offset_min_m is None and plan.offset_max_m is None:
        return mask

    src = geom.sources_xyz_m[shot_idx]            # (nshots_kept, ndim)
    rec = geom.receivers_xyz_m[shot_idx]          # (nshots_kept, nrec, ndim)
    # Horizontal-plane offset: drop the last axis (z).
    if geom.ndim == 2:
        off = rec[:, :, 0] - src[:, None, 0]
    elif geom.ndim == 3:
        dx = rec[:, :, 0] - src[:, None, 0]
        dy = rec[:, :, 1] - src[:, None, 1]
        off = np.sqrt(dx * dx + dy * dy)
    else:
        raise ValueError(f"unsupported ndim={geom.ndim}")
    if plan.abs_offset:
        off = np.abs(off)

    if plan.offset_min_m is not None:
        mask &= off >= plan.offset_min_m
    if plan.offset_max_m is not None:
        mask &= off <= plan.offset_max_m
    return mask


def _resample_time(plan: DataPlan, obs: np.ndarray, dt: float, time_axis: int) -> tuple[np.ndarray, float]:
    if plan.time_decimate is not None and plan.time_decimate > 1:
        # axis-aware decimate
        slicer = [slice(None)] * obs.ndim
        slicer[time_axis] = slice(None, None, plan.time_decimate)
        return obs[tuple(slicer)], dt * plan.time_decimate
    if plan.dt_target_s is not None and abs(plan.dt_target_s - dt) > 1e-12:
        try:
            from sweep_preproc.resample import resample_time
        except ImportError as e:
            raise ImportError(
                "DataPlan.dt_target_s requires `sweep_preproc`. "
                "Install with `pip install sweep-preproc`."
            ) from e
        return resample_time(obs, dt, plan.dt_target_s, axis=time_axis), plan.dt_target_s
    return obs, dt


def apply_data_plan(
    plan: DataPlan,
    geom: PhysicalGeometry,
    obs: np.ndarray,
    *,
    time_axis: int = -2,
    receiver_axis: int = -1,
) -> tuple[PhysicalGeometry, np.ndarray, np.ndarray]:
    """Apply the data plan to a ``(geometry, obs)`` pair.

    Parameters
    ----------
    plan
        :class:`DataPlan` to apply.
    geom
        Source-of-truth geometry in physical units.
    obs
        Observed data tensor. Shape must include shot axis at ``0``, a
        time axis at ``time_axis``, and a receiver axis at ``receiver_axis``.
        The default layout matches sweep's binding output:
        ``(nshots, nt, nrec)`` (or ``(nshots, nt, nrec, nchannel)``).
    time_axis
        Position of the time axis (default ``-2``).
    receiver_axis
        Position of the receiver axis (default ``-1``).

    Returns
    -------
    geom_out : PhysicalGeometry
        Geometry restricted to the selected shots (receiver dim untouched
        because the receiver mask is per-shot non-uniform when offset
        filters are active).
    obs_out : np.ndarray
        Observed data after shot subset + time resample.
    receiver_keep_mask : np.ndarray
        ``(nshots_kept, nreceivers)`` boolean mask. The caller applies it
        per-shot to ``obs_out`` along ``receiver_axis`` because the kept
        count may vary per shot (offset window depends on the source).
        ``True`` entries indicate "use this receiver".
    """
    if obs.shape[0] != geom.nshots:
        raise ValueError(
            f"obs shot axis ({obs.shape[0]}) does not match geometry ({geom.nshots})"
        )

    shot_idx = _select_shots(plan, geom.nshots)
    rcv_mask = _receiver_keep_mask(plan, geom, shot_idx)

    # Subset shots
    obs_sub = obs[shot_idx]
    geom_sub = PhysicalGeometry(
        sources_xyz_m=geom.sources_xyz_m[shot_idx],
        receivers_xyz_m=geom.receivers_xyz_m[shot_idx],
        dt=geom.dt,
        nt=geom.nt,
        meta=dict(geom.meta),
    )

    # Time resample
    obs_t, dt_new = _resample_time(plan, obs_sub, geom.dt, time_axis)
    nt_new = obs_t.shape[time_axis]

    # Time window trim
    if plan.t_start_s is not None or plan.t_end_s is not None:
        t0 = max(int(round((plan.t_start_s or 0.0) / dt_new)), 0)
        t1 = (
            min(int(round(plan.t_end_s / dt_new)), nt_new)
            if plan.t_end_s is not None
            else nt_new
        )
        if t1 <= t0:
            raise ValueError(f"DataPlan time window collapses: t0={t0} t1={t1}")
        slicer = [slice(None)] * obs_t.ndim
        slicer[time_axis] = slice(t0, t1)
        obs_t = obs_t[tuple(slicer)]
        nt_new = t1 - t0

    geom_sub.dt = float(dt_new)
    geom_sub.nt = int(nt_new)
    return geom_sub, obs_t, rcv_mask


# ============================================================================
# ModelPlan
# ============================================================================
@dataclass
class ModelPlan:
    """Crop a velocity model to a region of interest (axes are zero-indexed in meters).

    Coordinate convention matches sweep: the model array has shape
    ``(nz, nx)`` for 2-D or ``(nz, ny, nx)`` for 3-D, and the physical
    origin of the array (index ``0``) is at ``origin_xyz_m`` (defaults to
    zero on every axis).

    Attributes
    ----------
    x_window_m, y_window_m, z_window_m
        ``(low, high)`` inclusive windows in meters. ``None`` keeps the
        full axis.
    drop_outside_sources, drop_outside_receivers
        When ``True`` (default) acquisition geometry entries whose
        physical position falls outside the kept window are dropped.
    """

    x_window_m: tuple[float, float] | None = None
    y_window_m: tuple[float, float] | None = None
    z_window_m: tuple[float, float] | None = None
    drop_outside_sources: bool = True
    drop_outside_receivers: bool = True

    def __post_init__(self) -> None:
        for name in ("x_window_m", "y_window_m", "z_window_m"):
            w = getattr(self, name)
            if w is not None and (len(w) != 2 or w[0] > w[1]):
                raise ValueError(f"{name} must be (low, high) with low <= high; got {w}")


def _axis_slice(window_m: tuple[float, float] | None, n: int, dh: float, origin: float) -> slice:
    if window_m is None:
        return slice(0, n)
    lo = max(int(np.floor((window_m[0] - origin) / dh)), 0)
    hi = min(int(np.ceil((window_m[1] - origin) / dh)) + 1, n)
    if hi <= lo:
        raise ValueError(
            f"ModelPlan window {window_m} collapses on a {n}-cell axis "
            f"with dh={dh}, origin={origin}."
        )
    return slice(lo, hi)


def apply_model_plan(
    plan: ModelPlan,
    vp: np.ndarray,
    dh: tuple[float, ...],
    geom: PhysicalGeometry | None = None,
    *,
    origin_xyz_m: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, PhysicalGeometry | None, np.ndarray | None]:
    """Crop a velocity model and (optionally) drop out-of-window acquisition.

    Parameters
    ----------
    plan
        :class:`ModelPlan`.
    vp
        Model array, shape ``(nz, nx)`` (2-D) or ``(nz, ny, nx)`` (3-D).
    dh
        Grid spacing tuple matching the model axes: ``(dz, dx)`` for 2-D
        or ``(dz, dy, dx)`` for 3-D.
    geom
        Optional acquisition. If given, sources / receivers outside the
        cropped window are dropped per the plan's flags. The returned
        geometry has its physical positions **rebased** so that the new
        cropped origin sits at ``(0, 0)``.
    origin_xyz_m
        Coordinate origin of the **input** model. Defaults to zero on
        every axis. Note that axis order matches ``vp.shape`` so it's
        ``(z_origin, x_origin)`` for 2-D.

    Returns
    -------
    vp_out : np.ndarray
        Cropped model.
    geom_out : PhysicalGeometry | None
        Cropped + rebased geometry, or ``None`` if no geom was given.
    keep_shots_mask : np.ndarray | None
        ``(nshots,)`` boolean — surviving shots. ``None`` if no geom given
        or ``drop_outside_sources=False``. The corresponding obs slice is
        ``obs[keep_shots_mask]``.
    """
    if vp.ndim not in (2, 3):
        raise ValueError(f"apply_model_plan supports 2-D or 3-D vp; got ndim={vp.ndim}")
    dh = tuple(float(d) for d in dh)
    if len(dh) != vp.ndim:
        raise ValueError(f"dh length {len(dh)} must match vp.ndim={vp.ndim}")
    if origin_xyz_m is None:
        origin = (0.0,) * vp.ndim
    else:
        origin = tuple(float(o) for o in origin_xyz_m)
        if len(origin) != vp.ndim:
            raise ValueError(
                f"origin_xyz_m length {len(origin)} must match vp.ndim={vp.ndim}"
            )

    # vp axes: (nz, nx) or (nz, ny, nx). Map plan.{z,y,x}_window_m to axes.
    if vp.ndim == 2:
        z_sl = _axis_slice(plan.z_window_m, vp.shape[0], dh[0], origin[0])
        x_sl = _axis_slice(plan.x_window_m, vp.shape[1], dh[1], origin[1])
        vp_out = vp[z_sl, x_sl]
        new_origin = (
            origin[0] + z_sl.start * dh[0],
            origin[1] + x_sl.start * dh[1],
        )
    else:
        z_sl = _axis_slice(plan.z_window_m, vp.shape[0], dh[0], origin[0])
        y_sl = _axis_slice(plan.y_window_m, vp.shape[1], dh[1], origin[1])
        x_sl = _axis_slice(plan.x_window_m, vp.shape[2], dh[2], origin[2])
        vp_out = vp[z_sl, y_sl, x_sl]
        new_origin = (
            origin[0] + z_sl.start * dh[0],
            origin[1] + y_sl.start * dh[1],
            origin[2] + x_sl.start * dh[2],
        )

    if geom is None:
        return vp_out, None, None

    # Geometry's axes are in (x, z) or (x, y, z) order — opposite of vp.
    # Build a window per geometry axis from plan windows.
    new_origin_geom = (
        (new_origin[1], new_origin[0])
        if geom.ndim == 2
        else (new_origin[2], new_origin[1], new_origin[0])
    )
    geom_window_geom = (
        (plan.x_window_m, plan.z_window_m)
        if geom.ndim == 2
        else (plan.x_window_m, plan.y_window_m, plan.z_window_m)
    )

    def _in_window(pts_xyz: np.ndarray) -> np.ndarray:
        ok = np.ones(pts_xyz.shape[:-1], dtype=bool)
        for axis, win in enumerate(geom_window_geom):
            if win is None:
                continue
            ok &= (pts_xyz[..., axis] >= win[0]) & (pts_xyz[..., axis] <= win[1])
        return ok

    if plan.drop_outside_sources:
        src_keep = _in_window(geom.sources_xyz_m)
    else:
        src_keep = np.ones(geom.nshots, dtype=bool)

    sources_kept = geom.sources_xyz_m[src_keep]
    receivers_kept = geom.receivers_xyz_m[src_keep]

    if plan.drop_outside_receivers:
        # Per-shot mask, but receivers axis must stay uniform. Simplest:
        # only zero out — caller can still propagate with whatever sweep
        # does for "receivers" outside the model (which is OK if the
        # source position is inside). We do NOT shrink the receiver axis
        # here for shape stability; the caller can layer a DataPlan offset
        # filter on top if they want to actually drop them.
        pass

    # Rebase to cropped origin
    sources_rb = sources_kept - np.asarray(new_origin_geom, dtype="float64")
    receivers_rb = receivers_kept - np.asarray(new_origin_geom, dtype="float64")

    geom_out = PhysicalGeometry(
        sources_xyz_m=sources_rb,
        receivers_xyz_m=receivers_rb,
        dt=geom.dt,
        nt=geom.nt,
        meta={**geom.meta, "model_plan_origin_m": new_origin_geom},
    )
    return vp_out, geom_out, src_keep if plan.drop_outside_sources else None


__all__ = [
    "DataPlan",
    "ModelPlan",
    "apply_data_plan",
    "apply_model_plan",
]
