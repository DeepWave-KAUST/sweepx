"""Synchronous TaskRunner: turns a validated TaskSpec into actual work.

Each `_run_*` method mirrors the matching script in `examples/`:

  forward    -> examples/wavefields/free_surface_forward/acoustic_free_surface.py
  wavefield  -> same, with snapshot times + plots
  fwi        -> examples/FWI/2d/acoustic/torch/_fwi_marmousi_common.py
  lsrtm      -> examples/LSRTM/2d/acoustic/torch/lsrtm.py

The runner stays close to those scripts on purpose — they have been
validated end-to-end on Marmousi. Any new physics knob should land in the
schema, not as a runtime kwarg.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sweep
import sweep.equations as eq_mod
from sweep.signal import ricker

# Canonical filter primitives — used across every FWI / RTM bandpass site
# in this module. ``bandpass`` is scipy ``sosfiltfilt`` on numpy (CPU);
# ``bandpass_torch`` is the FFT |H(z)|² Butterworth on torch (GPU/CPU,
# autograd-friendly). Importing once at module load avoids the ~50 µs
# import-on-call cost previously paid by half a dozen inline imports.
from sweep_preproc.filter import bandpass as _bandpass_cpu
from sweep_preproc.filter import bandpass_torch as _bandpass_torch_fft
from sweep_tasks.schemas import (
    BaseTaskSpec,
    ForwardSpec,
    FWISpec,
    IntrospectSpec,
    LSRTMSpec,
    LineSet,
    ModelRef,
    PhysicsSpec,
    RTMSpec,
    TaskSpec,
    WavefieldSpec,
)


# ---------- result + status containers -----------------------------------

@dataclass
class TaskStatus:
    task_id: str
    task_type: str
    state: str = "pending"  # pending | running | success | failed
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "artifacts": self.artifacts,
            "summary": self.summary,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


@dataclass
class TaskResult:
    status: TaskStatus
    task_dir: Path


# ---------- helpers shared by the _run_* methods --------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_task_id(task_type: str, override: str | None) -> str:
    if override:
        return override
    return f"{task_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _resolve_device(device: str) -> "torch.device":
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _apply_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)


def _get_equation_class(name: str) -> type:
    classes = eq_mod._equation_classes()
    if name not in classes:
        raise ValueError(
            f"Unknown equation '{name}'. Available: {sorted(classes)}"
        )
    return classes[name]


def _model_names_for_equation(equation_cls: type) -> list[str]:
    specs = getattr(equation_cls, "MODEL_SPECS", None)
    if specs:
        return [s.name for s in specs]
    return _read_class_property(equation_cls, "models") or []


def _wavefield_names_for_equation(equation_cls: type) -> list[str]:
    specs = getattr(equation_cls, "FIELD_SPECS", None)
    if specs:
        return [s.name for s in specs]
    return _read_class_property(equation_cls, "wavefields") or []


def _read_class_property(cls: type, name: str):
    import inspect as _inspect

    descriptor = _inspect.getattr_static(cls, name, None)
    if isinstance(descriptor, property):
        try:
            return list(descriptor.fget(cls))
        except Exception:
            return None
    if descriptor is not None and not callable(descriptor):
        try:
            return list(descriptor)
        except Exception:
            return None
    return None


def _load_model_tensor(ref: ModelRef, base_dir: Path | None = None) -> "torch.Tensor":
    import torch

    if ref.constant is not None:
        return torch.full(tuple(ref.shape), float(ref.constant), dtype=torch.float32)
    path = ref.path
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr)


def _infer_shape(models: list[ModelRef], grid_shape: tuple | None) -> tuple[int, ...]:
    if grid_shape is not None:
        return tuple(int(v) for v in grid_shape)
    first = models[0]
    if first.constant is not None:
        return tuple(int(v) for v in first.shape)
    arr = np.load(first.path, mmap_mode="r")
    return tuple(arr.shape)


def _line_array(line: LineSet, fallback_stop: int) -> "np.ndarray":
    stop = line.stop if line.stop is not None else fallback_stop
    if stop <= line.start:
        raise ValueError(
            f"LineSet stop ({stop}) must be greater than start ({line.start})."
        )
    xs = np.arange(line.start, stop, line.step, dtype=np.int64).reshape(-1, 1)
    zs = np.full_like(xs, line.depth)
    return np.concatenate([xs, zs], axis=1)


def _build_geometry(
    geometry,
    shape: tuple[int, ...],
    *,
    dh: float | None = None,
    segy_cache: dict | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Dispatch on geometry.kind and return (sources, receivers) numpy arrays.

    Output shapes are always sources=(nshots, ndim) and receivers=(nshots, nrec, ndim).
    ``ndim`` is 2 when ``shape`` is ``(nz, nx)`` and 3 when ``shape`` is
    ``(nz, ny, nx)``. ``explicit`` / ``from_file`` accept either; ``line``
    is 2-D-only; the SEG-Y kinds are 2-D-only because their header-derived
    geometry is built around ``(x, z)`` only.

    ``dh`` and ``segy_cache`` are only used by the SEG-Y-backed kinds
    (``from_segy_headers`` / ``from_segy_index``). The cache lets the runner
    share a single SEG-Y scan between the geometry resolver and the obs
    loader when both point at the same file / same index.
    """

    kind = getattr(geometry, "kind", None)
    grid_ndim = len(shape)
    if kind == "line":
        if grid_ndim != 2:
            raise ValueError(
                f"LineGeometry only supports 2-D grids (shape ndim=2); "
                f"got shape={shape}. Use ExplicitGeometry / FromFileGeometry "
                f"for 3-D."
            )
        nx = int(shape[-1])
        sources = _line_array(geometry.sources, nx)
        rec = _line_array(geometry.receivers, nx)
        receivers = rec[None, ...].repeat(sources.shape[0], axis=0)
        return sources, receivers
    if kind == "explicit":
        sources, receivers = _explicit_geometry_arrays(geometry)
        if sources.shape[1] != grid_ndim:
            raise ValueError(
                f"ExplicitGeometry ndim={sources.shape[1]} does not match grid "
                f"ndim={grid_ndim} (shape={shape})."
            )
        return sources, receivers
    if kind == "from_file":
        sources, receivers = _from_file_geometry_arrays(geometry)
        if sources.shape[1] != grid_ndim:
            raise ValueError(
                f"FromFileGeometry ndim={sources.shape[1]} does not match grid "
                f"ndim={grid_ndim} (shape={shape})."
            )
        return sources, receivers
    if kind == "from_segy_headers":
        if grid_ndim != 2:
            raise ValueError(
                f"from_segy_headers geometry is 2-D-only; got grid shape={shape}. "
                f"For 3-D OBN data use the CRG-plan geometry adapter instead."
            )
        if dh is None:
            raise ValueError("from_segy_headers requires `dh` (spec.grid.dh).")
        payload = _load_segy_single_file_payload(
            geometry.path,
            byte_map=geometry.byte_map,
            source_depth_m_override=geometry.source_depth_m_override,
            receiver_depth_m_override=geometry.receiver_depth_m_override,
            coord_scalar_override=geometry.coord_scalar_override,
            cache=segy_cache,
        )
        # Initial parse always keeps ALL receivers (dedupe=False internally).
        # Per-stage dedupe (when ``geometry.dedupe=True``) happens inside
        # ``_prepare_stage`` against the pristine 120-receiver geometry, so
        # finer stages can recover the full receiver count even if a coarser
        # stage drops some.
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=False, dedup_method=geometry.dedup_method,
        )
        return sources_idx, receivers_idx
    if kind == "from_segy_index":
        if grid_ndim != 2:
            raise ValueError(
                f"from_segy_index geometry is 2-D-only; got grid shape={shape}."
            )
        if dh is None:
            raise ValueError("from_segy_index requires `dh` (spec.grid.dh).")
        payload = _load_segy_index_payload(
            geometry.index_path, shot_ids=geometry.shot_ids,
            cache=segy_cache,
        )
        # Initial parse always keeps ALL receivers (dedupe=False internally).
        # Per-stage dedupe (when ``geometry.dedupe=True``) happens inside
        # ``_prepare_stage`` against the pristine 120-receiver geometry, so
        # finer stages can recover the full receiver count even if a coarser
        # stage drops some.
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=False, dedup_method=geometry.dedup_method,
        )
        return sources_idx, receivers_idx
    if kind == "from_plan":
        if dh is None:
            raise ValueError("from_plan requires `dh` (spec.grid.dh).")
        if grid_ndim not in (2, 3):
            raise ValueError(
                f"from_plan: grid ndim must be 2 or 3 (shape={shape})."
            )
        payload = _load_seismic_plan_payload(
            geometry.plan_path,
            cache_all=False,
            grid_ndim=grid_ndim,
            cache=segy_cache,
            geometry_only=True,        # don't read traces just to compute geometry
        )
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=False, dedup_method=geometry.dedup_method,
        )
        return sources_idx, receivers_idx
    raise ValueError(f"Unknown geometry.kind '{kind}'.")


# Back-compat alias: callers that imported the old 2-D-only name keep working.
# The new dispatcher handles both 2-D and 3-D based on the ``shape`` arg.
_build_geometry_2d = _build_geometry


def _explicit_geometry_arrays(geometry) -> tuple["np.ndarray", "np.ndarray"]:
    sources = np.asarray(geometry.sources, dtype=np.int64)
    rec_raw = geometry.receivers
    first = rec_raw[0]
    is_per_shot = bool(first) and isinstance(first[0], list)
    if is_per_shot:
        receivers = np.asarray(rec_raw, dtype=np.int64)
        if receivers.shape[0] != sources.shape[0]:
            raise ValueError(
                f"Explicit per-shot receivers shape {receivers.shape} first axis must equal "
                f"nshots {sources.shape[0]}."
            )
    else:
        receivers_2d = np.asarray(rec_raw, dtype=np.int64)
        receivers = receivers_2d[None, ...].repeat(sources.shape[0], axis=0)
    return sources, receivers


def _load_segy_single_file_payload(
    path: Path,
    *,
    byte_map: dict | None,
    source_depth_m_override: float | None,
    receiver_depth_m_override: float | None,
    coord_scalar_override: float | None,
    shot_ids: list[int] | None = None,
    cache: dict | None = None,
):
    """Open + scan + read a single SEG-Y; cache the result keyed by ``path``.

    Returns a dict with keys ``physical_geometry``, ``obs`` (``(nshots, nrec,
    nt)`` float32), ``shot_ids`` (1-D int64). Both ``ObsSegyConfig`` and
    ``FromSegyGeometry`` flow through this helper, so when both reference the
    same file the runner reads it once.
    """
    from sweep_io.segy_index import (
        SEGYIndex,
        build_segy_index,
        IndexedShotGatherDataset,
    )

    key = str(Path(path).resolve())
    if cache is not None and key in cache:
        return cache[key]

    idx = build_segy_index(
        [path],
        byte_map=byte_map,
        source_depth_m_override=source_depth_m_override,
        receiver_depth_m_override=receiver_depth_m_override,
        coord_scalar_override=coord_scalar_override,
        num_workers=1,
    )
    sids = (
        np.asarray(shot_ids, dtype=np.int64)
        if shot_ids is not None else np.asarray(idx.shot_ids, dtype=np.int64)
    )
    # Materialise obs eagerly (single-file path -> simple).
    ds = IndexedShotGatherDataset(idx, shot_ids=sids)
    obs = np.stack([ds[i]["obs"] for i in range(len(ds))], axis=0)   # (nshots, nrec, nt)
    ds.close()

    pg = idx.to_physical_geometry(shot_ids=sids)
    result = {
        "physical_geometry": pg,
        "obs": obs,
        "shot_ids": sids,
        "n_samples": idx.n_samples,
        "dt_s": idx.dt_s,
    }
    if cache is not None:
        cache[key] = result
    return result


def _load_seismic_plan_payload(
    plan_path: Path,
    *,
    cache_all: bool = False,
    grid_ndim: int | None = None,
    cache: dict | None = None,
    geometry_only: bool = False,
):
    """Load a ``seismic_plan_v1`` cache as a payload dict (mirrors
    :func:`_load_segy_single_file_payload`).

    This helper serves the static-obs CSG path (``grouping="csg"`` only,
    every group's row count must be uniform — typical for rigid
    streamers). CRG-grouped plans go through the per-iter multisource
    sampler in :meth:`TaskRunner._run_fwi_multisource` (dispatched when
    ``obs.plan.sampling`` is set), which calls
    :func:`sweep_io.seismic_plan.sample_shared_shots_from_plan` +
    :meth:`PlanReader.read_rows` directly instead of going through here.

    ``grid_ndim``: 2 → keep (x, z) only (drop y); 3 → keep (x, y, z).
    ``geometry_only``: when True, skip the eager ``PlanReader.read_all`` so
    the geometry resolver doesn't pull every trace into RAM (RTM's per-batch
    PlanReader loop will do its own reads later). The cache entry under
    ``geometry_only=True`` and ``geometry_only=False`` are distinct so a
    subsequent eager-obs lookup still triggers a full load.
    """
    from sweep_io.seismic_plan import SeismicPlan, PlanReader
    from sweep_io.geometry import PhysicalGeometry

    plan_path = Path(plan_path)
    key = ("seismic_plan", str(plan_path.resolve()), bool(cache_all),
           int(grid_ndim) if grid_ndim else None, bool(geometry_only))
    if cache is not None and key in cache:
        return cache[key]

    plan = SeismicPlan.load(plan_path)
    if plan.grouping != "csg":
        raise ValueError(
            f"_load_seismic_plan_payload (CSG static-obs path) requires "
            f"grouping='csg' (got {plan.grouping!r}). For CRG-grouped "
            "plans the runner uses the multisource pipeline instead — "
            "set obs.plan.sampling (PlanSamplingConfig) on the YAML so "
            "the runner routes to _run_fwi_multisource."
        )
    counts = plan.per_group_row_counts()
    if counts.size == 0:
        raise ValueError("from_plan: SeismicPlan has zero groups")
    if not (counts == counts[0]).all():
        raise ValueError(
            f"from_plan: non-uniform receiver count per shot "
            f"({int(counts.min())}..{int(counts.max())}). The runner's "
            "per-stage dedupe path needs a uniform pristine layout; rebuild "
            "the plan with a filter that yields constant nrec or use "
            "from_segy_headers (which has its own per-shot dedupe)."
        )
    nshots = int(plan.n_groups)
    nrec = int(counts[0])
    nt = int(plan.samples_per_trace)

    # row arrays are sorted by group already → straight reshape works.
    src_xyz = np.asarray(plan.group_xyz, dtype=np.float64)               # (nshots, 3)
    rec_xyz = np.asarray(plan.row_receiver_xyz, dtype=np.float64).reshape(nshots, nrec, 3)

    if grid_ndim == 2:
        # 2-D streamer: keep (x, z) only.
        src_xyz = src_xyz[:, [0, 2]]
        rec_xyz = rec_xyz[:, :, [0, 2]]
    elif grid_ndim is not None and grid_ndim != 3:
        raise ValueError(
            f"from_plan: grid_ndim must be 2 or 3, got {grid_ndim}"
        )

    pg = PhysicalGeometry(
        sources_xyz_m=src_xyz,
        receivers_xyz_m=rec_xyz,
        dt=float(plan.dt_s), nt=nt,
        meta={"source": "SeismicPlan", "plan_path": str(plan_path),
              "plan_grouping": plan.grouping,
              "plan_build_meta": dict(plan.build_meta)},
    )

    if geometry_only:
        # `_segy_geometry_to_grid_indices` inspects obs.shape[1] to decide
        # whether a receiver mask is needed; supply a zero-cost placeholder
        # with the right receiver-axis size so the geometry path runs without
        # materialising actual samples. ``_build_geometry_2d`` drops the
        # returned obs immediately.
        obs = np.zeros((nshots, nrec, 1), dtype=np.float32)
    else:
        # Read obs into a single (nshots, nrec, nt) f32 tensor via PlanReader.
        reader = PlanReader(plan, cache_all=cache_all)
        try:
            obs_flat = reader.read_all().astype(np.float32, copy=False)  # (n_rows, nt)
        finally:
            reader.close()
        obs = obs_flat.reshape(nshots, nrec, nt)

    result = {
        "physical_geometry": pg,
        "obs": obs,
        "shot_ids": np.asarray(plan.group_id, dtype=np.int64),
        "n_samples": nt,
        "dt_s": float(plan.dt_s),
    }
    if cache is not None:
        cache[key] = result
    return result


def _load_segy_index_payload(
    index_path: Path,
    *,
    shot_ids: list[int] | None = None,
    cache: dict | None = None,
    lazy: bool = False,
    coalesce_gap: int = 0,
):
    """Load a pre-built SEGYIndex and materialise per-shot obs.

    ``lazy=False`` (default): read every shot eagerly, in input order, into
    a single ``(nshots, nrec, nt)`` tensor.

    ``lazy=True``: same final layout, but each shot is read by a background
    :class:`sweep_io.prefetch.Prefetcher` worker — useful when SEG-Y access
    is slow (network mount, cold cache) so the GIL-releasing I/O overlaps
    with whatever the consumer is doing at task-start. True per-step
    streaming during training is a future task: the dataset object is built
    but the materialised tensor is still what the train loop indexes.
    """
    from sweep_io.segy_index import SEGYIndex, IndexedShotGatherDataset
    from sweep_io.prefetch import Prefetcher

    key = (str(Path(index_path).resolve()), tuple(shot_ids) if shot_ids else None,
           bool(lazy), int(coalesce_gap))
    if cache is not None and key in cache:
        return cache[key]

    idx = SEGYIndex.load(index_path)
    sids = (
        np.asarray(shot_ids, dtype=np.int64)
        if shot_ids is not None else np.asarray(idx.shot_ids, dtype=np.int64)
    )

    ds = IndexedShotGatherDataset(idx, shot_ids=sids, coalesce_gap=coalesce_gap)
    if lazy:
        # Background-thread reader; results stay in input order.
        with Prefetcher((ds[i]["obs"] for i in range(len(ds))), queue_depth=2) as pf:
            obs = np.stack(list(pf), axis=0)
    else:
        obs = np.stack([ds[i]["obs"] for i in range(len(ds))], axis=0)
    ds.close()

    pg = idx.to_physical_geometry(shot_ids=sids)
    result = {
        "physical_geometry": pg,
        "obs": obs,
        "shot_ids": sids,
        "n_samples": idx.n_samples,
        "dt_s": idx.dt_s,
    }
    if cache is not None:
        cache[key] = result
    return result


def _segy_geometry_to_grid_indices(payload, dh: float, *, dedupe: bool, dedup_method: str):
    """Snap a SEG-Y-derived PhysicalGeometry to integer grid indices.

    Returns ``(sources_idx, receivers_idx, obs_aligned)`` — all rectangular
    ``(nshots, ...)`` tensors. Two modes:

    * **Uniform-mask path** (``dedupe=False`` or the dedup mask is identical
      across all shots): obs is sliced once with the shared keep_idx and the
      receivers_idx is returned post-slice. This is the fast path used by
      symmetric layouts (a stationary land array, streamer where source
      offsets are integer multiples of dh, etc.).

    * **Non-uniform-mask path** (``dedupe=True`` and shots disagree on which
      receivers survive — typical for streamer acquisition where the source
      shift per shot is not a multiple of dh): build per-shot keep_idx
      arrays, truncate to the minimum keep-count, and fancy-index obs
      per shot to a rectangular ``(nshots, n_common, nt)`` tensor.
    """
    pg = payload["physical_geometry"]
    obs = payload["obs"]
    gg, mask = pg.to_grid(dh=(dh, dh), dedupe=dedupe, dedup_method=dedup_method)
    uniform = bool(np.all(mask == mask[0:1]))
    if not dedupe or uniform:
        keep = np.flatnonzero(mask[0])
        if keep.size != obs.shape[1]:
            obs = np.ascontiguousarray(obs[:, keep, :])
            receivers_idx = gg.receivers[:, keep, :]
        else:
            receivers_idx = gg.receivers
        return gg.sources.astype(np.int64), receivers_idx.astype(np.int64), obs

    # Non-uniform path: per-shot keep_idx, truncate to common min.
    per_shot_keep = [np.flatnonzero(mask[s]) for s in range(mask.shape[0])]
    counts = np.asarray([k.size for k in per_shot_keep])
    n_common = int(counts.min())
    if n_common <= 0:
        raise ValueError("dedupe removed all receivers in at least one shot")
    truncated = np.stack(
        [k[:n_common].astype(np.int64) for k in per_shot_keep], axis=0
    )
    nshots = mask.shape[0]
    rec_out = np.empty((nshots, n_common, gg.receivers.shape[-1]), dtype=np.int64)
    obs_out = np.empty((nshots, n_common, obs.shape[-1]), dtype=obs.dtype)
    for s in range(nshots):
        rec_out[s] = gg.receivers[s, truncated[s], :].astype(np.int64)
        obs_out[s] = obs[s, truncated[s], :]
    return gg.sources.astype(np.int64), rec_out, np.ascontiguousarray(obs_out)


def _from_file_geometry_arrays(geometry) -> tuple["np.ndarray", "np.ndarray"]:
    sources = np.load(geometry.sources_file).astype(np.int64)
    receivers = np.load(geometry.receivers_file).astype(np.int64)
    if sources.ndim != 2:
        raise ValueError(
            f"sources_file array must be 2D (nshots, ndim), got shape {sources.shape}."
        )
    if receivers.ndim == 2:
        receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)
    elif receivers.ndim == 3:
        if receivers.shape[0] != sources.shape[0]:
            raise ValueError(
                f"receivers_file first axis {receivers.shape[0]} must equal nshots "
                f"{sources.shape[0]}."
            )
    else:
        raise ValueError(
            f"receivers_file array must be 2D or 3D, got shape {receivers.shape}."
        )
    return sources, receivers


def _build_wavelet(wavelet_spec, time_spec, *, override_dt: float | None = None,
                   override_nt: int | None = None) -> "np.ndarray":
    """Dispatch on wavelet.kind. Returned array has length nt.

    ``override_dt`` / ``override_nt`` let the runner pass in effective
    values when DataPlan / per-stage dt-sync changes the solver grid.
    """

    nt = int(override_nt) if override_nt is not None else int(time_spec.nt)
    dt = float(override_dt) if override_dt is not None else float(time_spec.dt)
    kind = getattr(wavelet_spec, "kind", None)
    if kind == "ricker":
        t = np.arange(nt, dtype=np.float32) * dt
        wave = ricker(t - float(wavelet_spec.delay), f=float(wavelet_spec.fm)).astype(np.float32)
        return float(wavelet_spec.scale) * wave
    if kind == "from_npy":
        arr = np.load(wavelet_spec.path).astype(np.float32)
        if arr.ndim != 1:
            raise ValueError(
                f"FromNpyWavelet expects a 1D array, got shape {arr.shape}."
            )
        if arr.shape[0] != nt:
            raise ValueError(
                f"FromNpyWavelet length {arr.shape[0]} != time.nt {nt}."
            )
        return float(wavelet_spec.scale) * arr
    if kind == "siren_pipeline_npz":
        from sweep_io.wavelet import load_wavelet_npz
        loaded = load_wavelet_npz(
            wavelet_spec.path, explicit_key=wavelet_spec.explicit_key,
        )
        samples = loaded.samples
        # Note: the SIREN-pipeline npz also reports ``source_delay_s``
        # (the zero-prepad applied during SIREN training); consumers
        # that need to align obs to the wavelet's frame should call
        # :func:`_get_wavelet_source_delay_s` and left-shift obs by
        # the same amount.
        # Resample to solver dt when the wavelet was sampled at a different
        # rate (typical for SIREN wavelets fit at SEG-Y dt while the solver
        # runs at a finer step). Uses sweep_preproc.resample_time.
        if abs(loaded.dt_s - dt) > 1.0e-12:
            from sweep_preproc.resample import resample_time
            samples = resample_time(
                samples.astype(np.float32), loaded.dt_s, dt, axis=0,
            ).astype(np.float32)
        # Length-align: truncate excess samples, zero-pad short tails.
        if samples.shape[0] > nt:
            samples = samples[:nt]
        elif samples.shape[0] < nt:
            samples = np.concatenate(
                [samples, np.zeros(nt - samples.shape[0], dtype=np.float32)]
            )
        return float(wavelet_spec.scale) * samples.astype(np.float32, copy=False)
    raise ValueError(f"Unknown wavelet.kind '{kind}'.")


def _get_wavelet_source_delay_s(wavelet_spec) -> float:
    """Return the wavelet's zero-prepad delay in seconds, or 0.0.

    Only the SIREN-pipeline npz format carries this metadata
    (``source_delay_s`` scalar). Consumers should left-shift observed
    traces by ``int(round(source_delay_s / dt_obs))`` samples so the
    main wavelet bang lines up with the actual event in obs — without
    this, syn and obs are off by the prepad (a systematic
    cycle-skip).

    For any other wavelet kind returns ``0.0``.
    """
    kind = getattr(wavelet_spec, "kind", None)
    if kind != "siren_pipeline_npz":
        return 0.0
    from sweep_io.wavelet import load_wavelet_npz
    loaded = load_wavelet_npz(
        wavelet_spec.path,
        explicit_key=getattr(wavelet_spec, "explicit_key", None),
    )
    return float(loaded.source_delay_s or 0.0)


def _resolve_modeling_inputs(spec, base_wavelet, base_sources, base_receivers, shape):
    """If spec.modeling_override is set, build wavelet/geometry overrides used only
    for the obs-synthesis forward pass. Validates shot/receiver count compatibility.
    """

    mod_override = getattr(spec, "modeling_override", None)
    if mod_override is None:
        return base_wavelet, base_sources, base_receivers, False

    wavelet = base_wavelet
    sources = base_sources
    receivers = base_receivers
    if mod_override.wavelet is not None:
        wavelet = _build_wavelet(mod_override.wavelet, spec.time)
    if mod_override.geometry is not None:
        sources, receivers = _build_geometry_2d(mod_override.geometry, shape)
        if sources.shape[0] != base_sources.shape[0]:
            raise ValueError(
                f"modeling_override.geometry produced {sources.shape[0]} shots but "
                f"inversion expects {base_sources.shape[0]}. Shot count must match "
                "so obs[shot_idx] aligns with the inversion shot indexing."
            )
        if receivers.shape[1] != base_receivers.shape[1]:
            raise ValueError(
                f"modeling_override.geometry produced {receivers.shape[1]} receivers per "
                f"shot but inversion expects {base_receivers.shape[1]}. Receiver count "
                "must match so the (syn - obs) residual is well-defined."
            )
    return wavelet, sources, receivers, True


def _cfl_check(vmax_m_s: float, dh_m: float, dt_s: float, *, threshold: float = 0.85) -> None:
    """Warn / raise if the FD time step violates the Courant condition.

    For 8th-order acoustic FD in 2D the practical safe range is roughly
    ``vmax * dt / dh ≤ 0.5–0.6``. We warn at 0.85 and raise above 1.0
    because anything above 1.0 will produce NaNs on the first step.
    """
    cfl = float(vmax_m_s) * float(dt_s) / float(dh_m)
    if cfl > 1.0:
        raise ValueError(
            f"CFL violation: vmax({vmax_m_s:.0f}) * dt({dt_s}) / dh({dh_m}) = "
            f"{cfl:.3f} > 1.0 — the forward propagator will produce NaNs. "
            f"Lower `time.dt` (try {0.6 * dh_m / vmax_m_s:.4f}s) or coarsen "
            f"`grid.dh`."
        )
    if cfl > threshold:
        print(
            f"[cfl] WARNING: vmax * dt / dh = {cfl:.3f} > {threshold:.2f}. "
            f"Stable forward modeling is not guaranteed; consider dt <= "
            f"{0.6 * dh_m / vmax_m_s:.4f}s for vmax={vmax_m_s:.0f} m/s."
        )


def _build_solver(physics: PhysicsSpec, backend, shape: tuple[int, ...], dh: float, dt: float,
                  nt: int, dev: "torch.device") -> "Any":
    from sweep.propagator.torch import PropTorch

    equation_cls = _get_equation_class(physics.equation)
    equation = equation_cls(
        spatial_order=physics.spatial_order,
        device=dev,
        backend="torch",
    )

    prop_kwargs: dict[str, Any] = dict(
        shape=tuple(shape),
        dev=dev,
        dh=dh,
        dt=dt,
        nt=nt,
        abcn=physics.abcn,
        free_surface=physics.free_surface,
        pml_type=physics.pml_type,
        source_type=list(physics.source_type),
        receiver_type=list(physics.receiver_type),
        use_ckpt=backend.use_ckpt,
    )
    if backend.use_ckpt:
        prop_kwargs["ckpt_chunks"] = backend.ckpt_chunks

    # Irregular free-surface topography → boundary-fitted curvilinear grid.
    # PropTorch builds the metric tensors from the 1-D surface-row array.
    if getattr(physics, "topography", None) is not None:
        prop_kwargs["topography"] = np.load(physics.topography)

    if backend.impl == "eager":
        eager = (backend.eager_options.to_dataclass()
                 if backend.eager_options is not None
                 else None)
        return PropTorch(
            equation,
            **prop_kwargs,
            backend="torch",
            impl="eager",
            eager_options=eager,
        )

    cuda = (backend.cuda_options.to_dataclass()
            if backend.cuda_options is not None
            else None)
    return PropTorch(
        equation,
        **prop_kwargs,
        backend="torch",
        impl="c",
        cuda_options=cuda,
    )


def _solver_models_in_order(equation_cls: type, refs: list[ModelRef], base_dir: Path | None,
                            dev: "torch.device") -> list["torch.Tensor"]:
    """Return tensors in the order the equation declares its MODEL_SPECS."""

    required = _model_names_for_equation(equation_cls)
    by_name = {r.name: r for r in refs}
    missing = set(required) - set(by_name)
    if missing:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' requires models {required}; "
            f"missing: {sorted(missing)}"
        )
    extra = set(by_name) - set(required)
    if extra:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' does not use model(s) {sorted(extra)}."
        )
    return [_load_model_tensor(by_name[name], base_dir).to(dev) for name in required]


def _validate_single_model(equation_cls: type, ref: ModelRef) -> None:
    required = _model_names_for_equation(equation_cls)
    if len(required) != 1:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' expects models {required}, "
            "but this task type only supports single-model equations (e.g. Acoustic)."
        )
    if ref.name != required[0]:
        raise ValueError(
            f"ModelRef name '{ref.name}' does not match equation's required model '{required[0]}'."
        )


# ---------- FWI / LSRTM training helpers ---------------------------------

def _apply_model_plan_to_fwi(
    spec, init_models_loaded: list[tuple[str, "np.ndarray"]],
    sources_idx, receivers_idx, dh: float,
):
    """Apply ``spec.model_plan`` (Gap 2): crop each loaded vp array to the
    window + shift / drop sources / receivers so their grid indices line up
    with the cropped model.

    Returns:
        cropped_vp_by_name: dict[name -> cropped ndarray]
        sources_idx_new, receivers_idx_new (or unchanged if no plan / no drop)
        keep_shots_mask (None when drop_outside_sources is False)
    """
    if spec.model_plan is None:
        return ({name: vp for name, vp in init_models_loaded},
                sources_idx, receivers_idx, None)

    plan = spec.model_plan
    from sweep_io.plan import ModelPlan, apply_model_plan
    from sweep_io.geometry import PhysicalGeometry

    # Lift current grid-index geometry into meters using `dh`.
    pg = PhysicalGeometry(
        sources_xyz_m=sources_idx.astype("float64") * float(dh),
        receivers_xyz_m=receivers_idx.astype("float64") * float(dh),
        dt=float(spec.time.dt), nt=int(spec.time.nt),
    )

    dp = ModelPlan(
        x_window_m=plan.x_window_m,
        y_window_m=plan.y_window_m,
        z_window_m=plan.z_window_m,
        drop_outside_sources=plan.drop_outside_sources,
        drop_outside_receivers=plan.drop_outside_receivers,
    )

    cropped_by_name: dict = {}
    new_pg = pg
    keep_mask = None
    # Apply the same crop to every model tensor. Geometry is rebased only on
    # the FIRST pass (subsequent calls would double-shift).
    first = True
    for name, vp in init_models_loaded:
        if vp.ndim == 2:
            vp_dh = (float(dh), float(dh))
        elif vp.ndim == 3:
            vp_dh = (float(dh), float(dh), float(dh))
        else:
            raise ValueError(f"ModelPlan: unsupported vp ndim {vp.ndim} for {name!r}")
        vp_out, geom_out, src_keep = apply_model_plan(
            dp, vp.astype("float32", copy=False), dh=vp_dh,
            geom=new_pg if first else None,
        )
        cropped_by_name[name] = vp_out
        if first and geom_out is not None:
            new_pg = geom_out
            keep_mask = src_keep
        first = False

    # Re-snap to grid indices at the SAME dh, no dedupe (positions are exact).
    gg, _ = new_pg.to_grid(
        dh=(float(dh),) * pg.ndim, dedupe=False,
    )
    return cropped_by_name, gg.sources.astype(np.int64), gg.receivers.astype(np.int64), keep_mask


def _apply_data_plan_to_fwi(
    spec, sources_idx, receivers_idx, obs, dh: float, dt: float, nt: int, dev
):
    """Apply ``spec.data_plan``.

    Subsets shots / receivers / time *after* obs has been generated. Lifts
    the grid-index geometry into ``sweep_io.PhysicalGeometry`` so the
    offset filter works in real meters, then drops back to grid indices.

    Receiver-axis subsetting requires a uniform mask across shots (typical
    for streamers). A non-uniform mask (offset filter where each shot picks
    a different receiver set) raises a clear error.
    """
    if spec.data_plan is None:
        return sources_idx, receivers_idx, obs

    import torch
    from sweep_io.geometry import PhysicalGeometry
    from sweep_io.plan import DataPlan, apply_data_plan

    dh_f = float(dh)
    sources_xyz = sources_idx.astype("float64") * dh_f
    receivers_xyz = receivers_idx.astype("float64") * dh_f

    pg = PhysicalGeometry(
        sources_xyz_m=sources_xyz,
        receivers_xyz_m=receivers_xyz,
        dt=dt,
        nt=nt,
    )

    plan_kwargs = spec.data_plan.model_dump(exclude_none=True)
    plan = DataPlan(**plan_kwargs)

    # Both backends now share the canonical ``(nshots, nt, nrec[, nfield])``
    # layout (the c-backend was aligned in geophyai commit 21041c5; the
    # eager backend was always canonical). Time is axis -3 / -2 depending
    # on whether the trailing channel axis is present.
    obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
    time_axis = -3 if obs_np.ndim >= 4 else -2
    receiver_axis = -2 if obs_np.ndim >= 4 else -1

    pg_planned, obs_planned, rcv_mask = apply_data_plan(
        plan, pg, obs_np, time_axis=time_axis, receiver_axis=receiver_axis,
    )

    # Receiver-axis subsetting: only handle uniform-across-shots masks.
    if not np.all(rcv_mask == rcv_mask[0:1]):
        raise NotImplementedError(
            "data_plan: non-uniform per-shot receiver mask (typical when an "
            "offset filter coexists with non-streamer geometry) is not yet "
            "supported. Drop the offset filter, or apply it upstream."
        )
    uniform_keep = rcv_mask[0]
    if not uniform_keep.all():
        slicer = [slice(None)] * obs_planned.ndim
        slicer[receiver_axis] = np.flatnonzero(uniform_keep)
        obs_planned = obs_planned[tuple(slicer)]
        # Also drop the same receivers from geometry.
        pg_planned = PhysicalGeometry(
            sources_xyz_m=pg_planned.sources_xyz_m,
            receivers_xyz_m=pg_planned.receivers_xyz_m[:, uniform_keep, :],
            dt=pg_planned.dt,
            nt=pg_planned.nt,
            meta=pg_planned.meta,
        )

    # Back to grid indices.
    gg, _ = pg_planned.to_grid(dh=(dh_f,) * pg.ndim, dedupe=False)
    sources_new = gg.sources
    receivers_new = gg.receivers

    # If time was resampled, push that into spec.time.nt and the obs.
    obs_t = torch.as_tensor(obs_planned, device=dev) if isinstance(obs, torch.Tensor) else obs_planned
    return sources_new, receivers_new, obs_t


def _normalize_fwi_init_models(spec) -> list:
    """Return the list of ModelRef objects for FWI, accepting either init_model or init_models."""

    if spec.init_models is not None:
        return list(spec.init_models)
    return [spec.init_model]


def _build_inv_tensors(init_models: list, dev, equation_cls,
                       *, overrides: dict[str, "np.ndarray"] | None = None):
    """Load each ModelRef as a requires_grad=True tensor and return both
    ordered list (per equation MODEL_SPECS) and name-keyed dict.

    ``overrides`` (used by Gap 2 model_plan) maps model name -> already-loaded
    numpy array; when present, the ModelRef's path / constant fields are
    ignored for that name.
    """

    required = _model_names_for_equation(equation_cls)
    by_name = {m.name: m for m in init_models}
    missing = set(required) - set(by_name)
    if missing:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' requires models {required}; "
            f"init_model(s) missing {sorted(missing)}."
        )
    extra = set(by_name) - set(required)
    if extra:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' does not use model(s) {sorted(extra)}."
        )
    in_order = []
    by_tensor: dict[str, "torch.Tensor"] = {}
    for name in required:
        if overrides is not None and name in overrides:
            import torch
            arr = np.ascontiguousarray(overrides[name].astype(np.float32, copy=False))
            t = torch.from_numpy(arr).to(dev).requires_grad_(True)
        else:
            t = _load_model_tensor(by_name[name]).to(dev).requires_grad_(True)
        in_order.append(t)
        by_tensor[name] = t
    return in_order, by_tensor, required


def _compute_loss(syn, obs, loss_spec):
    """Elementwise misfit via `sweep_loss` functional API.

    Returns the pointwise loss tensor; the caller is responsible for the
    final reduction (so multi-rank averaging stays in the runner's hands).

    For ``trace_cosine`` (a per-trace amplitude-normalised correlation misfit
    rather than a pointwise function), we still return a pointwise tensor
    by broadcasting the per-trace value back across the time axis. This
    keeps the caller's ``.sum() / global_norm`` recipe yielding the correct
    per-trace mean, regardless of the trace count or sample count.
    """
    from sweep_loss import huber_loss, l1_loss, l2_loss

    kind = loss_spec.kind
    if kind == "mse":
        # half=False matches the legacy `(syn-obs)**2` (not `0.5 * ...`).
        return l2_loss(syn, obs, reduction="none", half=False)
    if kind == "l1":
        return l1_loss(syn, obs, reduction="none")
    if kind == "huber":
        return huber_loss(syn, obs, delta=float(loss_spec.huber_delta), reduction="none")
    if kind == "trace_cosine":
        # sweep-loss expects canonical (ns, nt, nrec, nchan). Both backends
        # (eager + c, after geophyai 21041c5) deliver this layout natively
        # so no permutation is needed. The historical 3-D fallback
        # ``(nt, nrec, 1)`` is dropped: sweep no longer produces 3-D, and
        # the axis-0=time assumption disagreed with ``sweep_loss``'s 3-D
        # convention ``(nshots, nt, nrec)`` (time on axis 1) — a 3-D
        # input would have been silently misinterpreted.
        from sweep_loss import global_correlation_loss
        eps = float(getattr(loss_spec, "trace_cosine_eps", 1.0e-8))
        demean = bool(getattr(loss_spec, "trace_cosine_demean", True))

        if syn.ndim != 4:
            raise ValueError(
                f"trace_cosine: expected canonical 4-D syn "
                f"(ns, nt, nrec, nchan); got {syn.ndim}-D shape "
                f"{tuple(syn.shape)}. sweep backends always return 4-D."
            )

        per_trace = global_correlation_loss(
            syn, obs,
            offset_one=True, demean=demean, eps=eps, reduction="none",
        )  # (N,) where N = ns * nrec * nchan after sweep-loss flatten

        # Reshape back to per-trace canonical and broadcast across time so
        # the caller's .sum() / global_norm yields mean(1-cos) over traces.
        ns_c, nt_c, nr_c, nc_c = syn.shape
        return per_trace.view(ns_c, 1, nr_c, nc_c).expand(ns_c, nt_c, nr_c, nc_c)
    raise ValueError(f"Unknown loss kind '{kind}'.")


def _loss_sum(syn, obs_chunk, loss_spec, mask_chunk=None):
    """Pointwise misfit summed, optionally weighted by a per-sample data mask.
    ``mask_chunk=None`` is bit-identical to the legacy ``_compute_loss(...).sum()``."""
    pw = _compute_loss(syn, obs_chunk, loss_spec)
    if mask_chunk is not None:
        pw = pw * mask_chunk
    return pw.sum()


def _mask_chunk(data_mask, chunk, dev):
    """Slice the per-shot data mask for a chunk (or broadcast it when its
    leading dim is 1). Returns None when no mask is configured."""
    if data_mask is None:
        return None
    m = data_mask if data_mask.shape[0] == 1 else data_mask[chunk]
    return m.to(dev)


def _build_optimizer(opt_spec, inv_tensors_by_name, required_names):
    """Construct a torch optimizer; supports per-model lr via dict."""

    import torch

    params_in_order = [inv_tensors_by_name[name] for name in required_names]

    def _build_param_groups(lr_value):
        if isinstance(lr_value, dict):
            groups = []
            for name in required_names:
                if name not in lr_value:
                    raise ValueError(
                        f"optimizer.lr dict missing entry for inverted model '{name}'. "
                        f"Provided: {list(lr_value)}."
                    )
                groups.append({"params": [inv_tensors_by_name[name]], "lr": float(lr_value[name])})
            return groups
        return [{"params": params_in_order, "lr": float(lr_value)}]

    kind = opt_spec.kind
    if kind == "adam":
        return torch.optim.Adam(
            _build_param_groups(opt_spec.lr),
            eps=opt_spec.eps,
            betas=tuple(opt_spec.betas),
        )
    if kind == "sgd":
        return torch.optim.SGD(
            _build_param_groups(opt_spec.lr),
            momentum=opt_spec.momentum,
            nesterov=opt_spec.nesterov,
            weight_decay=opt_spec.weight_decay,
        )
    if kind == "lbfgs":
        return torch.optim.LBFGS(
            params_in_order,
            lr=float(opt_spec.lr),
            max_iter=opt_spec.max_iter,
            history_size=opt_spec.history_size,
            line_search_fn=opt_spec.line_search_fn,
        )
    raise ValueError(f"Unknown optimizer kind '{kind}'.")


def _build_reparam_optimizer(opt_spec, net_params, lr: float):
    """Build a torch optimizer over a reparam network's parameters.

    Mirrors :func:`_build_optimizer` for the network-as-vp case. The
    ``lr`` comes from ``spec.reparam.lr`` (≈ 1e-4); the optimizer kind
    and other hyperparameters come from ``spec.optimizer``.
    """
    import torch

    params = list(net_params)
    if not params:
        raise ValueError("reparam network has no trainable parameters")
    kind = opt_spec.kind
    if kind == "adam":
        return torch.optim.Adam(
            params, lr=float(lr), eps=opt_spec.eps, betas=tuple(opt_spec.betas),
        )
    if kind == "sgd":
        return torch.optim.SGD(
            params, lr=float(lr), momentum=opt_spec.momentum,
            nesterov=opt_spec.nesterov, weight_decay=opt_spec.weight_decay,
        )
    if kind == "lbfgs":
        return torch.optim.LBFGS(
            params, lr=float(lr), max_iter=opt_spec.max_iter,
            history_size=opt_spec.history_size,
            line_search_fn=opt_spec.line_search_fn,
        )
    raise ValueError(f"Unknown optimizer kind '{kind}' for reparam.")


def _build_scheduler(sched_spec, optimizer, total_epochs):
    """Dispatch the LR scheduler via `sweep_runner.scheduler.build`.

    The runner's `build()` accepts any object with the right `kind` +
    field attributes, so our Pydantic `Scheduler*` discriminated union
    plugs in directly without conversion.
    """
    from sweep_runner.scheduler import build as build_scheduler
    return build_scheduler(sched_spec, optimizer, total_epochs)


def _remember_initial_lrs(optimizer) -> list[float]:
    return [float(g["lr"]) for g in optimizer.param_groups]


def _apply_stage_lr_scale(optimizer, initial_lrs: list[float], scale: float) -> None:
    for group, base in zip(optimizer.param_groups, initial_lrs):
        group["lr"] = base * float(scale)


def _accumulate_illumination(solver, sill_sum, rill_sum):
    """Snapshot a chunk's solver illumination tensors into running buffers.

    Returns the updated ``(sill_sum, rill_sum)`` pair. The propagator
    backend zeros its ``source_illumination`` / ``receiver_illumination``
    attrs at the start of every forward call, so each chunk's backward
    must be snapshot here BEFORE the next chunk's forward kicks in.

    Either buffer may start as ``None``; the helper detach-clones the
    first snapshot and adds in-place thereafter.
    """
    import torch

    sill = getattr(solver, "source_illumination", None)
    rill = getattr(solver, "receiver_illumination", None)
    if sill is None or rill is None:
        return sill_sum, rill_sum
    sill_d = sill.detach()
    rill_d = rill.detach()
    if sill_sum is None:
        sill_sum = sill_d.clone()
    else:
        sill_sum.add_(sill_d)
    if rill_sum is None:
        rill_sum = rill_d.clone()
    else:
        rill_sum.add_(rill_d)
    return sill_sum, rill_sum


def _apply_illumination_precond(
    grad_tensor, sill_sum, rill_sum, *, eps: float, exponent: float,
) -> None:
    """Divide ``grad_tensor`` in place by ``(S·R + eps)**exponent``.

    Shapes must match (the illumination buffers are allocated by the
    backend to match the unpadded velocity tensor). No-op when either
    buffer is ``None`` (no chunk produced illumination, e.g. all chunks
    were empty on this rank).
    """
    import torch

    if sill_sum is None or rill_sum is None or grad_tensor is None:
        return
    sr = sill_sum * rill_sum + float(eps)
    if abs(exponent - 1.0) < 1.0e-12:
        scale = 1.0 / sr
    elif abs(exponent - 0.5) < 1.0e-12:
        scale = 1.0 / torch.sqrt(sr)
    else:
        scale = 1.0 / torch.pow(sr, float(exponent))
    grad_tensor.mul_(scale)


def _effective_bound(bounds_by_name, name):
    """Return the ModelBounds entry for ``name`` if active, else None.

    A bound is considered inactive when its ``enabled`` flag is False —
    callers that read ``min``/``max`` (clamp, reparam network, QC) all
    funnel through this helper so the YAML's ``enabled: false`` toggle
    is honoured uniformly. Backwards compatible: bounds without the
    ``enabled`` attribute (older schemas) are always active.
    """
    if not bounds_by_name:
        return None
    bound = bounds_by_name.get(name) if hasattr(bounds_by_name, "get") else None
    if bound is None:
        return None
    if not getattr(bound, "enabled", True):
        return None
    return bound


def _apply_bounds(inv_tensors_by_name, bounds_by_name, *, skip_names=()) -> None:
    """Clamp each named tensor to its model_bounds entry in place.

    ``skip_names`` (e.g. names handled by a reparam network) are passed
    through untouched — the network already enforces its own bounds via
    its render-time clamp.
    """
    if not bounds_by_name:
        return
    skip = set(skip_names)
    for name, t in inv_tensors_by_name.items():
        if name in skip:
            continue
        bound = _effective_bound(bounds_by_name, name)
        if bound is None:
            continue
        t.data.clamp_(min=bound.min, max=bound.max)


def _compute_local_window(sources_chunk, receivers_chunk, full_shape, dh, win_spec):
    """Bounding-box crop enclosing the batch's sources + receivers.

    2-D returns ``(z0, z1, x0, x1)`` on a ``(nz, nx)`` grid with the last
    coord axis ``[x, z]``.

    3-D returns ``(z0, z1, y0, y1, x0, x1)`` on a ``(nz, ny, nx)`` grid
    with the last coord axis ``[x, y, z]`` (matching sweep's 3-D
    propagator convention). The y dimension is padded by
    ``win_spec.padding_y_m`` if set, else falls back to
    ``win_spec.padding_x_m``. ``min_width_m`` enforces a floor on the x
    extent in both 2-D and 3-D; it is not applied to y.

    Both branches clamp to the full-model shape. Callers detect the
    grid dimensionality via ``len(full_shape)`` or by the length of the
    returned tuple (4 vs. 6).
    """
    import numpy as _np
    ndim = len(full_shape)
    pad_x = int(float(win_spec.padding_x_m) / float(dh))
    pad_z = int(float(win_spec.padding_z_m) / float(dh))
    min_w = int(float(win_spec.min_width_m) / float(dh))

    if ndim == 2:
        nz, nx = int(full_shape[0]), int(full_shape[1])
        src_x = sources_chunk[:, 0]; src_z = sources_chunk[:, 1]
        rec_x = receivers_chunk[..., 0].reshape(-1); rec_z = receivers_chunk[..., 1].reshape(-1)
        all_x = _np.concatenate([src_x, rec_x])
        all_z = _np.concatenate([src_z, rec_z])
        x0 = max(0, int(all_x.min()) - pad_x)
        x1 = min(nx, int(all_x.max()) + pad_x + 1)
        if win_spec.full_depth:
            z0, z1 = 0, nz
        else:
            z0 = max(0, int(all_z.min()) - pad_z)
            z1 = min(nz, int(all_z.max()) + pad_z + 1)
        if min_w > 0 and (x1 - x0) < min_w:
            extra = min_w - (x1 - x0)
            x0 = max(0, x0 - extra // 2)
            x1 = min(nx, x0 + min_w)
            if (x1 - x0) < min_w:
                x0 = max(0, x1 - min_w)
        return int(z0), int(z1), int(x0), int(x1)

    if ndim == 3:
        nz, ny, nx = (int(v) for v in full_shape)
        pad_y_m = win_spec.padding_y_m if win_spec.padding_y_m is not None else win_spec.padding_x_m
        pad_y = int(float(pad_y_m) / float(dh))
        src_x = sources_chunk[:, 0]; src_y = sources_chunk[:, 1]; src_z = sources_chunk[:, 2]
        rec_x = receivers_chunk[..., 0].reshape(-1)
        rec_y = receivers_chunk[..., 1].reshape(-1)
        rec_z = receivers_chunk[..., 2].reshape(-1)
        all_x = _np.concatenate([src_x, rec_x])
        all_y = _np.concatenate([src_y, rec_y])
        all_z = _np.concatenate([src_z, rec_z])
        x0 = max(0, int(all_x.min()) - pad_x)
        x1 = min(nx, int(all_x.max()) + pad_x + 1)
        y0 = max(0, int(all_y.min()) - pad_y)
        y1 = min(ny, int(all_y.max()) + pad_y + 1)
        if win_spec.full_depth:
            z0, z1 = 0, nz
        else:
            z0 = max(0, int(all_z.min()) - pad_z)
            z1 = min(nz, int(all_z.max()) + pad_z + 1)
        if min_w > 0 and (x1 - x0) < min_w:
            extra = min_w - (x1 - x0)
            x0 = max(0, x0 - extra // 2)
            x1 = min(nx, x0 + min_w)
            if (x1 - x0) < min_w:
                x0 = max(0, x1 - min_w)
        return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)

    raise NotImplementedError(
        f"_compute_local_window: only 2-D and 3-D grids supported; got shape {full_shape}."
    )


def _rebase_geometry_to_window(sources_chunk, receivers_chunk, z0, x0, *, y0=None):
    """Shift grid-index source / receiver coords to window-local origin.

    Args:
        sources_chunk    : ``(B, 2)`` or ``(B, 3)`` grid indices.
        receivers_chunk  : ``(B, nrec, 2)`` or ``(B, nrec, 3)`` grid indices.
        z0, x0           : window origin in z and x (always required).
        y0               : window origin in y. Pass for 3-D inputs; leave
                           ``None`` for 2-D. The function picks the axis
                           layout from ``y0`` rather than from the input
                           shape so 2-D callers can keep their existing
                           kwargs (``z0=..., x0=...``) unchanged.

    Returns fresh int64 arrays (not views); axis convention matches the
    rest of the runner: ``[x, z]`` (2-D) / ``[x, y, z]`` (3-D) on the
    last coord axis.
    """
    import numpy as _np
    s = _np.asarray(sources_chunk, dtype=_np.int64).copy()
    r = _np.asarray(receivers_chunk, dtype=_np.int64).copy()
    if y0 is None:
        s[:, 0] -= int(x0); s[:, 1] -= int(z0)
        r[..., 0] -= int(x0); r[..., 1] -= int(z0)
    else:
        s[:, 0] -= int(x0); s[:, 1] -= int(y0); s[:, 2] -= int(z0)
        r[..., 0] -= int(x0); r[..., 1] -= int(y0); r[..., 2] -= int(z0)
    return s, r


def _dump_run_metadata(
    spec,
    task_dir,
    *,
    extras: dict | None = None,
) -> None:
    """Write the resolved YAML + a runtime metadata json to ``task_dir``.

    Captures everything a reader of the run dir 6 months from now needs
    to reproduce / interpret the result:

    * ``config_resolved.yaml`` — the post-pydantic-validation spec with
      ALL fields explicit (defaults expanded). Diffing this against any
      hand-written YAML shows exactly what fields the pipeline saw.
    * ``run_meta.json`` — host, time, CUDA device, conda env path, key
      package versions (sweep-tasks, sweep-nn, sweep-io, sweep,
      sweep-loss, sweep-preproc, torch, numpy), git SHA / dirty-flag
      for each editable package (best-effort), plus any caller-supplied
      ``extras`` (typically the runner's derived setup quantities like
      ``shape``, ``origin``, ``n_groups``, ``net_params``, ...).
    """
    import json
    import os
    import socket
    import subprocess
    import sys
    from datetime import datetime, timezone

    import yaml

    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Resolved config (full pydantic dump, JSON-mode for paths
    # and enums → str). ``mode='json'`` makes the dict yaml.safe_dump-able.
    try:
        cfg = spec.model_dump(mode="json")
    except Exception:  # pydantic v1 fallback
        cfg = spec.dict()
    try:
        (task_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )
    except Exception as err:  # noqa: BLE001
        # Yaml dump can hit unrepresentable objects; fall back to json.
        (task_dir / "config_resolved.json").write_text(
            json.dumps(cfg, default=str, indent=2)
        )
        print(f"[run-meta] yaml dump failed ({err}); wrote config_resolved.json")

    # ---- 2) Package versions + git SHAs (best-effort).
    def _pkg_version(name: str) -> str | None:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            return None

    def _git_meta(pkg_name: str) -> dict | None:
        try:
            mod = __import__(pkg_name)
        except Exception:  # noqa: BLE001
            return None
        path = Path(getattr(mod, "__file__", "") or "").resolve().parent
        for _ in range(6):  # walk up looking for .git
            if (path / ".git").exists():
                break
            if path.parent == path:
                return None
            path = path.parent
        else:
            return None
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            dirty = bool(subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip())
            return {"path": str(path), "sha": sha, "dirty": dirty}
        except Exception:  # noqa: BLE001
            return None

    pkgs = [
        "sweep_tasks", "sweep_nn", "sweep_io", "sweep",
        "sweep_loss", "sweep_preproc", "sweep_viz", "sweep_runner",
        "torch", "numpy",
    ]
    versions = {n: _pkg_version(n) for n in pkgs}
    git = {n: _git_meta(n) for n in pkgs if _git_meta(n) is not None}

    # ---- 3) Host + CUDA snapshot.
    cuda_info: dict = {"available": False}
    try:
        import torch
        cuda_info["available"] = bool(torch.cuda.is_available())
        if cuda_info["available"]:
            cuda_info["device_count"] = int(torch.cuda.device_count())
            cuda_info["devices"] = [
                {
                    "index": i, "name": torch.cuda.get_device_name(i),
                    "total_mem_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 2,
                    ),
                }
                for i in range(int(torch.cuda.device_count()))
            ]
            cuda_info["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass

    meta = {
        "task_id": getattr(spec, "task_id", None),
        "task_type": getattr(spec, "task_type", None),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "argv": list(sys.argv),
        "cuda": cuda_info,
        "package_versions": versions,
        "git_repos": git,
        "env_vars_of_interest": {
            k: os.environ.get(k) for k in (
                "FWI_SEGY_ROOT", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )
        },
    }
    if extras:
        meta["runtime"] = extras
    (task_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _build_reparam_net(spec, base_vp, bounds, water_mask_override=None):
    """Construct a :class:`sweep_nn.VelocityINR` from a ReparamSpec.

    ``base_vp`` is the initial vp tensor at the first stage's grid. The net
    keeps it as a buffer (its forward returns ``base + delta``). ``bounds``
    is the optional :class:`ModelBounds` for vp — if provided, the network
    clamps its render output to those limits.
    """
    import torch  # local import — runner.py keeps torch imports per-function
    from sweep_nn import VelocityINR

    bounds_tuple = None
    if bounds is not None and bounds.min is not None and bounds.max is not None:
        bounds_tuple = (float(bounds.min), float(bounds.max))
    # Water-layer pin: build a boolean mask whose True voxels are
    # rendered as a fixed water velocity instead of the SIREN output.
    # When ``water_mask_override`` is supplied the caller has already
    # built the right mask (typically from a 2-D seabed_depth.npz,
    # cropped to the post-model_plan window) — use it verbatim. Else
    # fall back to ``init_vp == water_vp_m_s`` exact equality.
    water_mask = None
    if water_mask_override is not None:
        water_mask = water_mask_override.to(
            device=base_vp.device, dtype=torch.bool,
        )
        if tuple(water_mask.shape) != tuple(base_vp.shape):
            raise ValueError(
                f"water_mask_override shape {tuple(water_mask.shape)} != "
                f"base_vp shape {tuple(base_vp.shape)}"
            )
    elif bool(getattr(spec, "mask_water_layer", False)):
        water_vp_val = float(getattr(spec, "water_vp_m_s", 1500.0))
        water_mask = (base_vp.detach() == water_vp_val)
        if not bool(water_mask.any()):
            import warnings
            warnings.warn(
                f"reparam.mask_water_layer=True but no voxels in init_vp "
                f"equal water_vp_m_s={water_vp_val}; mask will be empty.",
                stacklevel=2,
            )
    return VelocityINR(
        base_vp.detach(),
        vp_mean=float(spec.vp_mean),
        vp_std=float(spec.vp_std),
        hidden_features=int(spec.hidden_features),
        hidden_layers=int(spec.hidden_layers),
        first_omega0=float(spec.first_omega0),
        hidden_omega0=float(spec.hidden_omega0),
        use_bias=bool(spec.use_bias),
        use_hash_encoding=bool(spec.hash.enabled),
        hash_levels=int(spec.hash.levels),
        hash_features_per_level=int(spec.hash.features_per_level),
        hash_log2_size=int(spec.hash.log2_size),
        hash_base_resolution=int(spec.hash.base_resolution),
        hash_finest_resolution=int(spec.hash.finest_resolution),
        direct_velocity=bool(spec.direct_velocity),
        coord_min=float(spec.coord_min),
        coord_max=float(spec.coord_max),
        bounds=bounds_tuple,
        water_mask=water_mask,
        water_vp=float(getattr(spec, "water_vp_m_s", 1500.0)),
        lateral_downsample=getattr(spec, "lateral_downsample", 1),
        compile_render=bool(getattr(spec, "compile_render", False)),
    ).to(base_vp.device)


def _zero_top_rows(inv_tensors_in_order, n_rows: int) -> None:
    if n_rows <= 0:
        return
    for t in inv_tensors_in_order:
        if t.grad is None:
            continue
        if t.grad.dim() >= 2:
            t.grad[:n_rows].zero_()
        else:
            t.grad[:n_rows] = 0


def _save_checkpoint(task_dir: Path, payload: dict) -> Path:
    """Atomic checkpoint write via `sweep_runner.checkpoint.save_payload`."""
    from sweep_runner.checkpoint import save_payload
    return save_payload(task_dir / "checkpoint.pt", payload)


def _load_checkpoint(prev_task_dir: Path) -> dict:
    """Counterpart to :func:`_save_checkpoint`. Raises FileNotFoundError if missing."""
    from sweep_runner.checkpoint import load_payload
    return load_payload(prev_task_dir / "checkpoint.pt", weights_only=False)


class _GracefulStopper:
    """Defer SIGINT / SIGTERM to the next training-iteration boundary.

    The training loops save a checkpoint at every epoch (see
    ``_save_checkpoint`` callsites in ``_run_fwi`` / ``_run_lsrtm``), so by
    the time the loop next checks :meth:`should_stop` there is already a
    fresh ``checkpoint.pt`` on disk capturing the just-completed epoch.
    The loop then breaks out cleanly, runs the usual final-outputs path,
    and exits with status="success" — but with ``interrupted=True`` /
    ``interrupted_at_epoch=N`` in the summary so callers can tell the
    difference.

    Distributed: every rank installs the handler (torchrun delivers
    SIGINT/SIGTERM to every worker process when the user hits ctrl-c on
    the launcher), but the canonical "should we stop" flag is rank 0's
    value, broadcast at each iter boundary. That keeps the ranks in
    lockstep even if individual workers race on signal delivery.
    """

    def __init__(self) -> None:
        self.requested = False
        self._signum: int | None = None
        self._installed = False
        self._prev: dict[int, Any] = {}

    def install(self, *, label: str = "runner") -> None:
        import signal

        if self._installed:
            return

        def _handler(signum, frame):  # noqa: ARG001
            # First signal: arm the stop flag, let the iter complete.
            # Second signal: restore the previous handler so a follow-up
            # ctrl-c terminates the process the usual way (escape hatch
            # when an iter is genuinely stuck).
            if self.requested:
                self._restore_handlers()
                print(f"\n[{label}] second signal {signum} — restoring default "
                      f"handler; next signal will abort.", flush=True)
                return
            self.requested = True
            self._signum = signum
            print(f"\n[{label}] signal {signum} received; will checkpoint and stop "
                  f"after the current iteration. Hit ctrl-c again to abort "
                  f"immediately.", flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev[sig] = signal.signal(sig, _handler)
            except (ValueError, OSError):
                # ValueError: not on the main thread. OSError: platform
                # doesn't support this signal. Either way, just don't
                # catch this one — the runner will still work, ctrl-c
                # just won't be graceful.
                pass
        self._installed = True

    def _restore_handlers(self) -> None:
        import signal

        for sig, prev in list(self._prev.items()):
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev.clear()

    def uninstall(self) -> None:
        self._restore_handlers()
        self._installed = False

    def should_stop(self, dist_info=None) -> bool:
        if dist_info is None or not getattr(dist_info, "is_distributed", False):
            return self.requested
        from sweep_runner import distributed as _dist

        flag = self.requested if dist_info.is_root else None
        flag = _dist.broadcast_object(flag, dist_info, src=0)
        return bool(flag)


def _save_illumination(solver, snapshots_dir: Path, epoch: int) -> list[Path]:
    artifacts: list[Path] = []
    for attr_name, label in (("source_illumination", "src"),
                             ("receiver_illumination", "rec")):
        tensor = getattr(solver, attr_name, None)
        if tensor is None:
            continue
        try:
            arr = tensor.detach().cpu().numpy()
        except AttributeError:
            continue
        path = snapshots_dir / f"{label}_illumination_epoch_{epoch:04d}.npy"
        np.save(path, arr)
        artifacts.append(path)
    return artifacts


def _normalise_stage_list(spec) -> list:
    """Return the effective stage list (single-stage fallback when spec.stages is None)."""

    if spec.stages:
        return list(spec.stages)
    from sweep_tasks.schemas import StageSpec
    return [StageSpec(epochs=spec.epochs, wavelet=None, lr_scale=1.0)]


# ---------------------------------------------------------------------------
# Per-stage rebuilds (Gaps 4 + 5)
# ---------------------------------------------------------------------------

def _resample_vp_tensor(vp: "torch.Tensor", new_shape: tuple[int, ...]) -> "torch.Tensor":
    """Resample a vp tensor between grid resolutions (bilinear for 2-D,
    trilinear for 3-D).

    Returns a fresh leaf tensor (requires_grad=True) — caller is responsible
    for re-initialising the optimizer because Adam state is shape-bound.
    """
    import torch
    import torch.nn.functional as F

    if tuple(vp.shape) == tuple(new_shape):
        return vp.detach().clone().requires_grad_(True)
    if vp.ndim == 2:
        if len(new_shape) != 2:
            raise ValueError(
                f"_resample_vp_tensor: vp is 2-D but new_shape={new_shape} is not."
            )
        mode = "bilinear"
    elif vp.ndim == 3:
        if len(new_shape) != 3:
            raise ValueError(
                f"_resample_vp_tensor: vp is 3-D but new_shape={new_shape} is not."
            )
        mode = "trilinear"
    else:
        raise ValueError(
            f"_resample_vp_tensor: unsupported vp.ndim {vp.ndim} (expected 2 or 3)."
        )
    src = vp.detach().unsqueeze(0).unsqueeze(0)
    dst = F.interpolate(src, size=tuple(new_shape), mode=mode, align_corners=True)
    return dst.squeeze(0).squeeze(0).contiguous().clone().requires_grad_(True)


def _resample_obs_time(obs_np: "np.ndarray", dt_old: float, dt_new: float,
                       *, time_axis: int = -1) -> "np.ndarray":
    """Resample obs along the time axis via sweep_preproc.resample.resample_time."""
    if abs(dt_old - dt_new) < 1e-12:
        return obs_np
    from sweep_preproc.resample import resample_time
    return resample_time(obs_np, dt_old, dt_new, axis=time_axis)


def _resample_obs_to_solver_dt(obs_t, dt_segy: float, dt_solver: float,
                               nt_solver: int):
    """Resample a torch obs tensor along the last axis to solver dt + length.

    Mirrors the legacy 3-D CRG FWI runner's per-iter obs prep: drop to
    numpy, run scipy.signal.resample_poly via sweep_preproc, truncate /
    zero-pad to nt_solver, and ship back to the original device.
    """
    if abs(dt_segy - dt_solver) < 1.0e-12 and obs_t.shape[-1] == nt_solver:
        return obs_t
    import torch
    from sweep_preproc.resample import resample_time

    arr = obs_t.detach().cpu().numpy()
    arr = resample_time(arr, dt_segy, dt_solver, axis=-1)
    cur_nt = arr.shape[-1]
    if cur_nt > nt_solver:
        arr = arr[..., :nt_solver]
    elif cur_nt < nt_solver:
        pad = [(0, 0)] * arr.ndim
        pad[-1] = (0, nt_solver - cur_nt)
        arr = np.pad(arr, pad)
    return torch.as_tensor(arr, dtype=obs_t.dtype, device=obs_t.device)


def _is_cuda_dev(dev) -> bool:
    """``True`` iff CUDA is available **and** ``dev`` names a CUDA device.

    Accepts ``torch.device`` or string ("cuda" / "cuda:N" / "cpu"). Used
    by every site that branches between GPU and CPU code paths (obs
    pipeline, local-window solver cache, illumination, etc.) — keeping
    the check in one place avoids the historical ~5 different inline
    repetitions that drifted in subtle ways (some only checked
    ``torch.cuda.is_available()``, others only the device type).
    """
    import torch  # lazy: runner is callable from torch-less environments
    if not torch.cuda.is_available():
        return False
    if isinstance(dev, torch.device):
        return dev.type == "cuda"
    if isinstance(dev, str):
        return dev.startswith("cuda")
    return False


def _bandpass_syn_torch(syn: "torch.Tensor", lo: float, hi: float, dt: float,
                        *, order: int) -> "torch.Tensor":
    """Differentiable bandpass on a synthetic torch tensor.

    Thin wrapper around :func:`sweep_preproc.filter.bandpass_torch` —
    the same canonical zero-phase GPU FFT Butterworth used by the obs
    stage-entry filter and by the multisource-encoded FWI path. Going
    through one impl across syn, obs, and wavelet ensures the filter
    response is identical and avoids the latent NaN risk of
    ``torchaudio.functional.filtfilt`` without padding on narrow bands.

    Both sweep backends (eager + c, after geophyai 21041c5) emit syn in
    canonical 4-D ``(n, nt, nrec, nchan)`` with time at axis 1, so this
    helper only accepts 4-D and raises on anything else. The historical
    3-D fallback ``(nt, nrec, 1)`` (time at axis 0) is removed because
    (a) sweep no longer produces it, and (b) the layout it assumed
    disagreed with ``sweep_loss.to_canonical``'s 3-D convention
    ``(nshots, nt, nrec)`` (time at axis 1), so any 3-D input would be
    silently filtered on the wrong axis. Manual 3-D inputs now fail
    loud here instead.
    """
    if syn.ndim != 4:
        raise ValueError(
            f"_bandpass_syn_torch: expected canonical 4-D syn "
            f"(n, nt, nrec, nchan); got {syn.ndim}-D shape {tuple(syn.shape)}. "
            "sweep backends always return 4-D — wrap or unsqueeze upstream."
        )
    return _bandpass_torch_fft(syn, lo=float(lo), hi=float(hi),
                               dt=float(dt), order=int(order), axis=1)


def _trim_or_pad_time(obs_np: "np.ndarray", target_nt: int, time_axis: int = -1) -> "np.ndarray":
    n = obs_np.shape[time_axis]
    if n == target_nt:
        return obs_np
    if n > target_nt:
        slicer = [slice(None)] * obs_np.ndim
        slicer[time_axis] = slice(0, target_nt)
        return obs_np[tuple(slicer)]
    pad_width = [(0, 0)] * obs_np.ndim
    pad_width[time_axis] = (0, target_nt - n)
    return np.pad(obs_np, pad_width)


def _shape_for_dh(orig_shape: tuple[int, ...], orig_dh: float, new_dh: float) -> tuple[int, ...]:
    """Pick a new grid shape that preserves physical extent (within rounding)."""
    if abs(orig_dh - new_dh) < 1e-12:
        return tuple(orig_shape)
    ratio = orig_dh / new_dh
    return tuple(max(1, int(round(s * ratio))) for s in orig_shape)


def _prepare_stage(
    *, spec, stage, state: dict, equation_cls, required_names: list[str],
    dev, dist_info, stage_idx: int,
) -> None:
    """Mutate ``state`` so the runner can run this stage end-to-end.

    Detects which of ``(dh, dt, nt, bandpass, wavelet, batch_size, lr_scale)``
    changed against the previous stage state. Triggers any combination of:

    - vp resample (bilinear) + fresh leaf tensors
    - solver rebuild at the new ``(shape, dh, dt, nt)``
    - geometry re-snap from the cached pristine PhysicalGeometry
    - obs rebuild from the cached pristine obs (time-resample + bandpass)
    - wavelet rebuild
    - optimizer re-init (Adam state is shape-bound) + initial_lrs cache
    """
    import torch

    new_dh = float(stage.dh_m) if stage.dh_m is not None else state["dh"]
    new_dt = float(stage.dt_s) if stage.dt_s is not None else state["dt"]
    if stage.nt is not None:
        new_nt = int(stage.nt)
    elif abs(new_dt - state["dt"]) > 1e-12:
        new_nt = int(round(state["nt"] * state["dt"] / new_dt))
    else:
        new_nt = state["nt"]

    grid_changed = abs(new_dh - state["dh"]) > 1e-12
    time_changed = (abs(new_dt - state["dt"]) > 1e-12) or (new_nt != state["nt"])
    bandpass_changed = stage.bandpass is not None or state.get("_active_bandpass") is not None
    wavelet_changed = stage.wavelet is not None or state.get("_stage_wavelet_idx", -1) != stage_idx

    if grid_changed and dist_info.is_root:
        print(f"[stage {stage_idx}] dh: {state['dh']} -> {new_dh}")
    if time_changed and dist_info.is_root:
        print(f"[stage {stage_idx}] dt: {state['dt']} -> {new_dt}, nt: {state['nt']} -> {new_nt}")

    # ---- Geometry re-snap from pristine physical positions -----------------
    # We re-snap whenever:
    #   - the stage changes dh (grid_changed)
    #   - state hasn't been initialised yet
    #   - dedupe is enabled and we haven't applied it at this dh yet (the
    #     initial geometry parse always keeps all receivers; per-stage
    #     dedupe is the first time it gets applied at the stage's dh)
    need_geom_resnap = (
        grid_changed
        or "sources" not in state
        or (state.get("dedupe_grid_snap", False)
            and state.get("_geom_applied_at_dh") != new_dh)
    )
    if need_geom_resnap:
        from sweep_io.geometry import PhysicalGeometry
        pg: PhysicalGeometry = state["pristine_physical_geom"]
        dedupe_flag = state.get("dedupe_grid_snap", True)
        dedup_method = state.get("dedup_method", "nearest")
        gg, mask = pg.to_grid(
            dh=(float(new_dh),) * pg.ndim,
            dedupe=dedupe_flag,
            dedup_method=dedup_method,
        )
        uniform = bool(np.all(mask == mask[0:1]))
        if not dedupe_flag or uniform:
            # Fast path: keep all (dedupe=false) or uniform mask across shots.
            keep_idx = np.flatnonzero(mask[0])
            state["sources"] = gg.sources.astype(np.int64)
            state["receivers"] = gg.receivers[:, keep_idx, :].astype(np.int64)
            state["receiver_keep_idx"] = keep_idx
            state["per_shot_keep_idx"] = None
            state["nshots"] = int(state["sources"].shape[0])
        else:
            # Non-uniform mask path: streamer-style acquisition where the
            # source position shifts per shot, so each shot's dedupe pattern
            # differs. Build per-shot ``keep_idx`` arrays, then TRUNCATE to
            # the minimum keep count across shots so the resulting tensors
            # stay rectangular (no ragged). For each shot, ``keep_idx`` is
            # already in receiver-index order from ``flatnonzero``; truncate
            # via ``[:n_common]`` preserves spatial order.
            per_shot_keep: list[np.ndarray] = [np.flatnonzero(mask[s]) for s in range(mask.shape[0])]
            counts = np.asarray([k.size for k in per_shot_keep])
            n_common = int(counts.min())
            if n_common <= 0:
                raise ValueError(
                    f"[stage {stage_idx}] dedupe removed all receivers in at least one shot"
                )
            truncated = np.stack(
                [k[:n_common].astype(np.int64) for k in per_shot_keep], axis=0
            )  # (nshots, n_common)
            # Build deduped receivers: gg.receivers is already the deduped
            # output (with dropped traces zeroed in the dropped slots but
            # mask tells us which to keep). However ``flatnonzero(mask[s])``
            # returns indices into the ORIGINAL receiver axis — we use those
            # against ``gg.receivers`` (which has the same axis length, with
            # mask[s][i] = True iff receiver i survived for shot s).
            # Truncated keep gives a rectangular (nshots, n_common) index.
            rec_out = np.zeros((mask.shape[0], n_common, gg.receivers.shape[-1]), dtype=np.int64)
            for s in range(mask.shape[0]):
                rec_out[s] = gg.receivers[s, truncated[s], :].astype(np.int64)
            state["sources"] = gg.sources.astype(np.int64)
            state["receivers"] = rec_out
            state["per_shot_keep_idx"] = truncated.astype(np.int64)
            # Maintain receiver_keep_idx for backward-compat (used to slice
            # pristine obs when uniform); keep first-shot keep as a
            # representative (obs masking path below uses per_shot_keep_idx
            # when available).
            state["receiver_keep_idx"] = truncated[0].copy()
            state["nshots"] = int(state["sources"].shape[0])
            if dist_info.is_root:
                drop = int(mask.shape[1] - n_common)
                print(f"[stage {stage_idx}] dedupe={dedup_method!r}: "
                      f"per-shot rec count {counts.min()}–{counts.max()}, "
                      f"truncated to common {n_common} (dropped {drop} per shot)")
        # Record the dh at which the current geometry was snapped so the
        # check above doesn't re-run on every stage entry.
        state["_geom_applied_at_dh"] = float(new_dh)

    # ---- Pick a new model shape (preserves physical extent) ----------------
    if grid_changed or "shape" not in state:
        new_shape = _shape_for_dh(state.get("pristine_shape", state["shape"]),
                                  state["pristine_dh"], new_dh)
        if dist_info.is_root and grid_changed:
            print(f"[stage {stage_idx}] vp shape: {state['shape']} -> {new_shape}")
        state["shape"] = new_shape

    # ---- vp resample ----------------------------------------------------
    # Two paths:
    #  - Reparam off: bilinear-resample each leaf tensor to the new shape
    #    and make a fresh `requires_grad=True` leaf (Adam state is shape-
    #    bound, so we rebuild the optimizer below).
    #  - Reparam on: keep the same VelocityINR. Update its `base_velocity`
    #    buffer from the pristine init, resampled to the new shape — and
    #    DO NOT rebuild the optimizer (its state on the network's params
    #    is independent of grid shape and is the entire point of the
    #    network-as-vp parameterization).
    if grid_changed:
        reparam_net = state.get("reparam_net")
        if reparam_net is not None:
            pristine_base = state["pristine_base_vp"]
            new_base = _resample_vp_tensor(pristine_base, state["shape"]).detach()
            reparam_net.update_base_velocity(new_base)
            with torch.no_grad():
                rendered = reparam_net().detach()
            state["inv_by_name"]["vp"] = rendered
            state["inv_in_order"] = [
                rendered if n == "vp" else state["inv_by_name"][n]
                for n in required_names
            ]
        else:
            new_inv: list = []
            new_by_name: dict = {}
            for name in required_names:
                old_t = state["inv_by_name"][name]
                new_t = _resample_vp_tensor(old_t, state["shape"])
                new_inv.append(new_t)
                new_by_name[name] = new_t
            state["inv_in_order"] = new_inv
            state["inv_by_name"] = new_by_name

    # ---- Rebuild solver ----------------------------------------------------
    if grid_changed or time_changed or "solver" not in state:
        state["solver"] = _build_solver(
            spec.physics, spec.backend, state["shape"], new_dh, new_dt, new_nt, dev,
        )
        # Drop cached local-window solvers — they were built for the
        # previous stage's (dh, dt, nt) and are no longer valid.
        if "local_solver_cache" in state:
            state["local_solver_cache"].clear()

    # ---- Rebuild wavelet ---------------------------------------------------
    # Rebuild from spec when:
    #   * time grid changed, or
    #   * stage explicitly overrides the wavelet, or
    #   * first stage (no cached wavelet yet), or
    #   * the stage is about to bandpass the wavelet (``target='wavelet'``):
    #     we MUST start from the pristine broadband wavelet rather than
    #     re-filtering whatever the previous stage's bandpass left behind,
    #     otherwise the per-stage filters compose (e.g. 1-4 Hz then 1-7 Hz
    #     stays effectively 1-4 Hz). The schema docs guarantee
    #     ``stages don't compose their filters``; this branch enforces it.
    rebuilt_wavelet = False
    bp_will_filter_wavelet = (
        stage.bandpass is not None
        and getattr(stage.bandpass, "target", "syn") == "wavelet"
    )
    if (
        time_changed
        or stage.wavelet is not None
        or "wavelet" not in state
        or bp_will_filter_wavelet
    ):
        wav_spec = stage.wavelet if stage.wavelet is not None else spec.wavelet
        state["wavelet"] = _build_wavelet(
            wav_spec, spec.time, override_dt=new_dt, override_nt=new_nt,
        )
        rebuilt_wavelet = True

    # ---- Optional wavelet bandpass (target='wavelet') ----------------------
    # When ``stage.bandpass.target == 'wavelet'``, we band-pass the source
    # wavelet ONCE at stage entry. Syn forward is then naturally band-limited
    # without per-iteration filtering in the autograd path. fwi_workflow-dev
    # filters syn (target='syn'); both yield equivalent gradients for a
    # linear wave equation.
    bp_spec = stage.bandpass
    if bp_spec is not None and getattr(bp_spec, "target", "syn") == "wavelet" \
            and (rebuilt_wavelet or state.get("_wavelet_bandpass_at") != (
                float(new_dt), int(new_nt), float(bp_spec.lo_hz), float(bp_spec.hi_hz),
                int(bp_spec.order), bp_spec.padtype)):
        # Bandpass the wavelet via scipy (non-autograd: wavelet is constant
        # input to the solver). One-shot per stage, so cheap. Handle both
        # numpy and torch wavelet objects (build_wavelet returns numpy for
        # ricker, torch for from_npy).
        wav_t = state["wavelet"]
        is_torch = hasattr(wav_t, "detach")
        if is_torch:
            wav_np = wav_t.detach().cpu().numpy()
            wav_dtype = wav_np.dtype
            wav_device = wav_t.device
        else:
            wav_np = np.asarray(wav_t)
            wav_dtype = wav_np.dtype
            wav_device = None
        wav_filt = _bandpass_cpu(
            wav_np, lo=bp_spec.lo_hz, hi=bp_spec.hi_hz,
            dt=new_dt, order=bp_spec.order,
            axis=-1, padtype=bp_spec.padtype,
        ).astype(wav_dtype, copy=False)
        wav_filt = np.ascontiguousarray(wav_filt)
        if is_torch:
            state["wavelet"] = torch.from_numpy(wav_filt).to(wav_device)
        else:
            state["wavelet"] = wav_filt
        state["_wavelet_bandpass_at"] = (
            float(new_dt), int(new_nt), float(bp_spec.lo_hz), float(bp_spec.hi_hz),
            int(bp_spec.order), bp_spec.padtype,
        )
        if dist_info.is_root:
            peak = float(np.abs(wav_filt).max())
            print(f"[stage {stage_idx}] wavelet bandpass "
                  f"{bp_spec.lo_hz}-{bp_spec.hi_hz} Hz applied (peak now {peak:.3f}) "
                  f"-> syn will be naturally band-limited; syn filter SKIPPED")

    # ---- Obs rebuild from pristine -----------------------------------------
    # Apply: receiver-mask -> time resample -> trim/pad -> optional bandpass.
    # Pristine obs is the snapshot at the END of step 5/5b (post-data_plan),
    # before any stage modification.
    pristine_obs = state["pristine_obs_np"]
    pristine_dt = state["pristine_dt"]
    receiver_axis_pristine = state["pristine_obs_receiver_axis"]
    time_axis_pristine = state["pristine_obs_time_axis"]

    # 1) receiver mask (apply along receiver axis)
    obs_np = pristine_obs
    per_shot_keep_idx = state.get("per_shot_keep_idx")
    if per_shot_keep_idx is not None:
        # Per-shot dedupe path: each shot keeps a different subset of
        # receivers. Apply a per-shot fancy-indexing reduction to obs along
        # the receiver axis. We do this in float32 numpy (CPU-friendly)
        # before time-resampling so downstream operations work on the
        # already-rectangular ``(nshots, ..., n_common, ...)`` tensor.
        # Canonical layout (post-geophyai 21041c5) is always 4-D for
        # solver-produced or SEG-Y-adapted obs.
        if obs_np.ndim == 4:
            # canonical: (nshots, nt, nrec, nchan). axis -2 = receiver.
            nshots, nt_pristine, nrec_pristine, nchan = obs_np.shape
            new = np.empty(
                (nshots, nt_pristine, per_shot_keep_idx.shape[1], nchan),
                dtype=obs_np.dtype,
            )
            # ``np.take`` keeps the receiver axis in place; plain fancy
            # indexing ``obs_np[s, :, idx, :]`` would move it to axis 0 and
            # give ``(n_keep, nt, nchan)`` instead of ``(nt, n_keep, nchan)``.
            for s in range(nshots):
                new[s] = np.take(obs_np[s], per_shot_keep_idx[s], axis=1)
            obs_np = new
        else:
            raise NotImplementedError(
                f"per-shot keep on obs.ndim={obs_np.ndim} not implemented "
                "(expected canonical 4-D (nshots, nt, nrec, nchan))"
            )
    else:
        # Uniform-mask path: existing behavior.
        keep_idx = state.get("receiver_keep_idx")
        if keep_idx is not None and keep_idx.size != obs_np.shape[receiver_axis_pristine]:
            slicer = [slice(None)] * obs_np.ndim
            slicer[receiver_axis_pristine] = keep_idx
            obs_np = obs_np[tuple(slicer)]

    # 2) time resample
    obs_np = _resample_obs_time(obs_np, pristine_dt, new_dt, time_axis=time_axis_pristine)

    # 3) trim/pad to new_nt
    obs_np = _trim_or_pad_time(obs_np, new_nt, time_axis=time_axis_pristine)

    # Upload obs to the solver device. The per-iter loop does
    # ``obs[chunk].to(dev)`` which becomes a free no-op once obs is on
    # dev. Full obs is ~1 GB float32 on Marmousi-scale; trivial on a
    # multi-GB GPU and saves ~9 s of repeated H2D per 100 iters.
    obs_np = np.ascontiguousarray(obs_np.astype(np.float32, copy=False))
    obs_t = torch.from_numpy(obs_np)
    on_cuda = _is_cuda_dev(dev)
    if on_cuda:
        obs_t = obs_t.to(dev, non_blocking=False)

    # 4) bandpass (per-stage; uses the new dt, so it's correctly normalised)
    #
    # GPU path (default when obs is on CUDA): reuse the canonical FFT
    # zero-phase Butterworth from sweep_preproc — same implementation
    # used by ``_run_fwi_multisource``'s per-iter encoded supershot path.
    # ~30× faster than scipy ``sosfiltfilt`` on Marmousi-scale obs
    # (1 GB float32), drops stage-entry time from ~10 s to <0.5 s.
    # CPU fallback: scipy ``sosfiltfilt`` with the configured padtype.
    if stage.bandpass is not None:
        if on_cuda:
            with torch.no_grad():
                # Chunk the bandpass over the shot axis (dim 0) so each cuFFT
                # plan stays under the 2^31-element limit; a full dense-OBN obs
                # (nshots*nt*nrec) can exceed INT_MAX and trip
                # CUFFT_INVALID_SIZE. Per-shot spectra are independent, so
                # filtering shot-chunks in place is exact.
                _bp_chunk = 64
                if obs_t.shape[0] > _bp_chunk and time_axis_pristine != 0:
                    for _b0 in range(0, obs_t.shape[0], _bp_chunk):
                        _b1 = min(_b0 + _bp_chunk, obs_t.shape[0])
                        obs_t[_b0:_b1] = _bandpass_torch_fft(
                            obs_t[_b0:_b1].contiguous(),
                            lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
                            dt=new_dt, order=stage.bandpass.order,
                            axis=time_axis_pristine,
                        )
                else:
                    obs_t = _bandpass_torch_fft(
                        obs_t, lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
                        dt=new_dt, order=stage.bandpass.order,
                        axis=time_axis_pristine,
                    )
            _flavor = "GPU FFT (sweep_preproc.bandpass_torch)"
        else:
            obs_np2 = _bandpass_cpu(
                obs_t.cpu().numpy(),
                lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
                dt=new_dt, order=stage.bandpass.order,
                axis=time_axis_pristine,
                padtype=stage.bandpass.padtype,
            )
            obs_t = torch.from_numpy(np.ascontiguousarray(
                obs_np2.astype(np.float32, copy=False)
            ))
            _flavor = f"scipy sosfiltfilt (padtype={stage.bandpass.padtype or 'none'})"
        if dist_info.is_root:
            print(f"[stage {stage_idx}] bandpass {stage.bandpass.lo_hz}-{stage.bandpass.hi_hz} Hz "
                  f"order={stage.bandpass.order} ({_flavor})")
    state["_active_bandpass"] = stage.bandpass
    state["obs"] = obs_t
    state["_stage_wavelet_idx"] = stage_idx

    # ---- Source dedupe + obs stacking --------------------------------------
    # At coarse dh, the source spacing (typically 25 m for Viking-class
    # streamer surveys) is smaller than dh, so multiple shots round to the
    # same source grid cell. Group shots by source cell, stack (mean) their
    # obs traces. This:
    #   - reduces redundant forward modeling (1 syn per cell vs many)
    #   - cleans the gradient (1 stacked obs vs 1 syn per cell — no more
    #     "3 obs vs 1 syn" redundancy on the source side)
    #   - matches what a properly-deduped acquisition would naturally do
    # Per-stage (cheap: rerun from pristine each stage), so finer stages
    # automatically recover all shots if their source spacing >= dh.
    if state.get("dedupe_grid_snap", False) and state.get("_source_dedup_at_dh") != new_dh:
        src_arr = state["sources"]
        keys = [(int(src_arr[s, 0]), int(src_arr[s, 1])) for s in range(src_arr.shape[0])]
        groups: dict[tuple[int, int], list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        if len(groups) < src_arr.shape[0]:
            sorted_keys = sorted(groups.keys())
            n_unique_src = len(sorted_keys)
            new_sources = np.zeros((n_unique_src, 2), dtype=np.int64)
            rec_curr = state["receivers"]
            new_receivers = np.zeros((n_unique_src, rec_curr.shape[1], 2), dtype=np.int64)
            obs_curr = state["obs"]  # tensor (n_pristine, n_common_rec, nt)
            new_obs_shape = (n_unique_src, *obs_curr.shape[1:])
            new_obs = torch.zeros(new_obs_shape, dtype=obs_curr.dtype)
            group_sizes = []
            for g, k in enumerate(sorted_keys):
                indices = np.asarray(groups[k], dtype=np.int64)
                group_sizes.append(int(indices.size))
                # Representative: first shot in the group (deterministic)
                new_sources[g] = src_arr[indices[0]]
                new_receivers[g] = rec_curr[indices[0]]
                # Stack obs: mean across the group's shots. obs_curr is
                # already receiver-deduped, so all shots in the group share
                # the same receiver layout (streamer geometry → constant
                # relative offsets) and can be averaged element-wise.
                if indices.size == 1:
                    new_obs[g] = obs_curr[int(indices[0])]
                else:
                    new_obs[g] = obs_curr[indices.tolist()].mean(dim=0)
            # Also dedupe per_shot_keep_idx if it's per-pristine-shot.
            per_shot_keep = state.get("per_shot_keep_idx")
            if per_shot_keep is not None:
                state["per_shot_keep_idx"] = np.stack(
                    [per_shot_keep[groups[k][0]] for k in sorted_keys]
                )
            state["sources"] = new_sources
            state["receivers"] = new_receivers
            state["obs"] = new_obs
            state["nshots"] = int(new_sources.shape[0])
            if dist_info.is_root:
                avg_grp = src_arr.shape[0] / n_unique_src
                print(f"[stage {stage_idx}] source dedupe: "
                      f"{src_arr.shape[0]} shots -> {n_unique_src} unique src cells "
                      f"(avg group size {avg_grp:.1f}, max {max(group_sizes)})")
        state["_source_dedup_at_dh"] = float(new_dh)

    # ---- Re-init optimizer (Adam state is shape-bound) ---------------------
    # Reparam mode: optimizer state is on network params (shape-invariant),
    # so DO NOT rebuild on grid change — preserving Adam moments across
    # stages is the main multi-scale benefit of network reparameterization.
    # However, an explicit ``stage.optimizer_reset=True`` forces a rebuild
    # even in reparam mode (e.g. when switching regimes drastically and
    # the old momentum has gone stale).
    force_reset = bool(getattr(stage, "optimizer_reset", False))
    if "optimizer" not in state:
        # Cold-init (e.g. resume-from path). Build appropriate optimizer.
        if state.get("reparam_net") is not None:
            state["optimizer"] = _build_reparam_optimizer(
                spec.optimizer, state["reparam_net"].parameters(),
                float(spec.reparam.lr),
            )
        else:
            state["optimizer"] = _build_optimizer(
                spec.optimizer, state["inv_by_name"], required_names,
            )
        state["initial_lrs"] = _remember_initial_lrs(state["optimizer"])
    elif force_reset:
        if state.get("reparam_net") is not None:
            state["optimizer"] = _build_reparam_optimizer(
                spec.optimizer, state["reparam_net"].parameters(),
                float(spec.reparam.lr),
            )
        else:
            state["optimizer"] = _build_optimizer(
                spec.optimizer, state["inv_by_name"], required_names,
            )
        state["initial_lrs"] = _remember_initial_lrs(state["optimizer"])
        if dist_info.is_root:
            print(f"[stage {stage_idx}] optimizer_reset=True — Adam moments dropped")
    elif grid_changed and state.get("reparam_net") is None:
        state["optimizer"] = _build_optimizer(
            spec.optimizer, state["inv_by_name"], required_names,
        )
        state["initial_lrs"] = _remember_initial_lrs(state["optimizer"])

    # ``lr_scale`` scales the optimizer's per-group initial lr; for the
    # reparam path the optimizer holds network params at ``spec.reparam.lr``,
    # so ``inr_lr_scale`` multiplies on top of that. For the grid path
    # ``inr_lr_scale`` is ignored (no INR).
    if state.get("reparam_net") is not None:
        effective_scale = float(stage.lr_scale) * float(
            getattr(stage, "inr_lr_scale", 1.0)
        )
    else:
        effective_scale = float(stage.lr_scale)
    _apply_stage_lr_scale(state["optimizer"], state["initial_lrs"], effective_scale)

    # ---- Batch size override ----------------------------------------------
    state["batchsize"] = (
        int(stage.batch_size) if stage.batch_size is not None
        else int(spec.batchsize)
    )

    # ---- Commit state ------------------------------------------------------
    state["dh"] = new_dh
    state["dt"] = new_dt
    state["nt"] = new_nt
    if dist_info.is_root:
        print(f"[stage {stage_idx}] obs shape: {tuple(state['obs'].shape)}, "
              f"nshots={state['nshots']}, batchsize={state['batchsize']}")


# ---------- the TaskRunner ------------------------------------------------

class TaskRunner:
    """Synchronous local task runner. One-shot: instantiate and call .run(spec)."""

    def run(self, spec: TaskSpec) -> TaskResult:
        from sweep_runner import distributed as _dist

        dist_info = _dist.init_distributed_if_needed()
        self._dist = dist_info

        # Rank 0 picks the task_id (timestamp-based) and broadcasts so every rank
        # writes to the same task directory under output_dir.
        if dist_info.is_root:
            task_id = _make_task_id(spec.task_type, spec.task_id)
        else:
            task_id = None
        task_id = _dist.broadcast_object(task_id, dist_info, src=0)

        output_dir = Path(spec.output_dir).expanduser()
        task_dir = (output_dir / task_id).resolve()
        if dist_info.is_root:
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "output").mkdir(exist_ok=True)
            (task_dir / "logs").mkdir(exist_ok=True)
        _dist.barrier(dist_info)

        status = TaskStatus(
            task_id=task_id,
            task_type=spec.task_type,
            state="running",
            started_at=_now_iso(),
        )
        if dist_info.is_root:
            status.write(task_dir / "status.json")

        dispatcher = {
            "introspect": self._run_introspect,
            "forward": self._run_forward,
            "wavefield": self._run_wavefield,
            "fwi": self._run_fwi,
            "lsrtm": self._run_lsrtm,
            "rtm": self._run_rtm,
        }
        runner_fn = dispatcher.get(spec.task_type)
        if runner_fn is None:
            status.state = "failed"
            status.error = f"Unknown task_type '{spec.task_type}'."
            status.finished_at = _now_iso()
            if dist_info.is_root:
                status.write(task_dir / "status.json")
            _dist.cleanup_distributed(dist_info)
            return TaskResult(status=status, task_dir=task_dir)

        try:
            artifacts, summary = runner_fn(spec, task_dir)
            status.state = "success"
            status.artifacts = [str(p) for p in artifacts]
            status.summary = summary
        except Exception as e:  # noqa: BLE001  (we want to surface any failure)
            import traceback as _tb

            status.state = "failed"
            status.error = f"{type(e).__name__}: {e}"
            # Print the full traceback BEFORE the finally-block's status
            # write, in case the write itself errors (e.g. the runner
            # deleted the task dir as part of its own cleanup) and ends
            # up masking the original exception in the stderr.
            print(f"[runner] task failed: {status.error}")
            _tb.print_exc()
        finally:
            status.finished_at = _now_iso()
            if dist_info.is_root:
                try:
                    task_dir.mkdir(parents=True, exist_ok=True)
                    status.write(task_dir / "status.json")
                except Exception as werr:  # noqa: BLE001
                    print(f"[runner] failed to write status.json: {werr!r}")
            _dist.barrier(dist_info)
            _dist.cleanup_distributed(dist_info)

        return TaskResult(status=status, task_dir=task_dir)

    # -- introspect --------------------------------------------------------

    def _run_introspect(self, spec: IntrospectSpec, task_dir: Path):
        action = spec.action
        result: Any
        torch_binding_available = sweep.is_torch_binding_available()

        if action == "list_equations":
            rows = []
            for name, cls in sorted(eq_mod._equation_classes().items()):
                supported = eq_mod.supports_torch_binding(cls)
                rows.append({
                    "name": name,
                    "models": _model_names_for_equation(cls),
                    "torch_binding_support": "yes" if supported else "no",
                    "torch_binding_available": "yes" if supported and torch_binding_available else "no",
                })
            result = rows

        elif action == "list_supported_bindings":
            result = eq_mod.torch_binding_supported_equations()

        elif action == "describe_equation":
            if not spec.target:
                raise ValueError("describe_equation requires `target` (equation name).")
            cls = _get_equation_class(spec.target)
            supported = eq_mod.supports_torch_binding(cls)
            result = {
                "name": spec.target,
                "models": _model_names_for_equation(cls),
                "wavefields": _wavefield_names_for_equation(cls),
                "torch_binding_support": "yes" if supported else "no",
                "torch_binding_available": "yes" if supported and torch_binding_available else "no",
            }

        elif action == "describe_field":
            if not spec.target or not spec.field_or_model:
                raise ValueError("describe_field requires `target` and `field_or_model`.")
            cls = _get_equation_class(spec.target)
            result = cls.describe_field(spec.field_or_model)

        elif action == "describe_model":
            if not spec.target or not spec.field_or_model:
                raise ValueError("describe_model requires `target` and `field_or_model`.")
            cls = _get_equation_class(spec.target)
            result = cls.describe_model(spec.field_or_model)

        else:
            raise ValueError(f"Unknown introspect action '{action}'.")

        payload = {"action": action, "target": spec.target,
                   "field_or_model": spec.field_or_model, "result": result}
        out_path = task_dir / "output" / "introspect.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str))

        # also print a human-friendly view to stdout for `sweep run` users
        print(json.dumps(payload, indent=2, default=str))
        summary = {"action": action, "row_count": len(result) if isinstance(result, list) else None}
        return [out_path], summary

    # -- forward -----------------------------------------------------------

    def _run_forward(self, spec: ForwardSpec, task_dir: Path):
        import torch

        _apply_seed(spec.seed)
        dev = _resolve_device(spec.device)
        equation_cls = _get_equation_class(spec.physics.equation)
        shape = _infer_shape(spec.models, spec.grid.shape)

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )
        wavelet = _build_wavelet(spec.wavelet, spec.time)
        sources, receivers = _build_geometry_2d(spec.geometry, shape)
        models = _solver_models_in_order(equation_cls, spec.models, base_dir=None, dev=dev)

        print(f"[forward] equation={spec.physics.equation} shape={shape} "
              f"nshots={sources.shape[0]} nreceivers={receivers.shape[1]} "
              f"nt={spec.time.nt} backend={spec.backend.impl}")

        t0 = time.perf_counter()
        with torch.no_grad():
            record = solver(wavelet, sources, receivers, models=models)
        elapsed = (time.perf_counter() - t0) * 1000.0
        record_np = record.detach().cpu().numpy()

        out_dir = task_dir / "output"
        record_path = out_dir / "record.npy"
        sources_path = out_dir / "sources.npy"
        receivers_path = out_dir / "receivers.npy"
        np.save(record_path, record_np)
        np.save(sources_path, sources)
        np.save(receivers_path, receivers)

        summary = {
            "elapsed_ms": elapsed,
            "record_shape": list(record_np.shape),
            "nshots": int(sources.shape[0]),
            "nreceivers": int(receivers.shape[1]),
            "nt": int(spec.time.nt),
        }
        print(f"[forward] done in {elapsed:.1f} ms, record shape {record_np.shape}")
        return [record_path, sources_path, receivers_path], summary

    # -- wavefield ---------------------------------------------------------

    def _run_wavefield(self, spec: WavefieldSpec, task_dir: Path):
        import torch

        _apply_seed(spec.seed)
        dev = _resolve_device(spec.device)
        equation_cls = _get_equation_class(spec.physics.equation)
        shape = _infer_shape(spec.models, spec.grid.shape)

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )
        wavelet = _build_wavelet(spec.wavelet, spec.time)
        sources, receivers = _build_geometry_2d(spec.geometry, shape)
        models = _solver_models_in_order(equation_cls, spec.models, base_dir=None, dev=dev)

        print(f"[wavefield] equation={spec.physics.equation} shape={shape} "
              f"snapshots={spec.snapshot_times}")

        t0 = time.perf_counter()
        record, snapshots = solver(
            wavelet, sources, receivers, models=models,
            return_wavefield=True, snapshot_times=list(spec.snapshot_times),
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        record_np = record.detach().cpu().numpy()
        snapshots_np = snapshots.detach().cpu().numpy()

        out_dir = task_dir / "output"
        record_path = out_dir / "record.npy"
        snapshots_path = out_dir / "snapshots.npy"
        np.save(record_path, record_np)
        np.save(snapshots_path, snapshots_np)

        artifacts: list[Path] = [record_path, snapshots_path]
        if spec.plot:
            try:
                fig_path = _plot_wavefield_snapshots(
                    snapshots_np, list(spec.snapshot_times), spec.physics.abcn,
                    shape, out_dir / "snapshots.png", spec.physics.free_surface,
                )
                artifacts.append(fig_path)
            except Exception as plot_err:  # noqa: BLE001
                print(f"[wavefield] plotting skipped: {plot_err}")

        summary = {
            "elapsed_ms": elapsed,
            "record_shape": list(record_np.shape),
            "snapshots_shape": list(snapshots_np.shape),
        }
        print(f"[wavefield] done in {elapsed:.1f} ms, snapshots {snapshots_np.shape}")
        return artifacts, summary

    # -- fwi ---------------------------------------------------------------

    def _run_fwi(self, spec: FWISpec, task_dir: Path):
        # Dispatch the OBN 3-D multisource supershot path early — it has
        # its own per-iter SEG-Y reads + UTM→model rotation + source-
        # encoded loop that don't fit the static-(sources, obs)
        # assumptions of the main ``_fwi_train_step`` machinery. Triggered
        # by setting ``obs.plan.sampling`` (a PlanSamplingConfig) on a
        # ``grouping='crg'`` plan.
        _obs_plan = getattr(spec.obs, "plan", None) if spec.obs is not None else None
        if _obs_plan is not None and getattr(_obs_plan, "sampling", None) is not None:
            return self._run_fwi_multisource(spec, task_dir)

        import torch

        from sweep_runner import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        if dist_info.is_distributed and spec.optimizer.kind == "lbfgs":
            raise ValueError(
                "LBFGS optimiser is not supported under torchrun in Phase 1 "
                "(its line-search closure semantics conflict with shot-parallel "
                "all-reduce). Use adam or sgd."
            )

        _apply_seed(spec.seed)
        dev = _dist.resolve_dist_device(spec.device, dist_info.local_rank)
        equation_cls = _get_equation_class(spec.physics.equation)

        # Graceful ctrl-c / SIGTERM: install the handler *before* the heavy
        # setup phase (obs synth, solver build, etc.) so that a signal
        # delivered early just arms the stop flag instead of crashing the
        # process with KeyboardInterrupt. The stage loop checks the flag
        # at each iter boundary; if armed before training even starts,
        # we fall straight through to the final-outputs path.
        stopper = _GracefulStopper()
        stopper.install(label="fwi")
        interrupted = False
        interrupted_at_epoch: int | None = None

        # 1) Init models: accept either init_model (single) or init_models (list).
        init_models = _normalize_fwi_init_models(spec)

        # 2) Shape: grid.shape > first model file shape > first model constant shape.
        if spec.grid.shape is not None:
            shape = tuple(int(v) for v in spec.grid.shape)
        elif init_models[0].constant is not None:
            shape = tuple(int(v) for v in init_models[0].shape)
        else:
            shape = tuple(np.load(init_models[0].path, mmap_mode="r").shape)

        # 3) Build solver, wavelet, geometry.
        #    Gap 1: when data_plan.dt_target_s is set, the obs gets resampled
        #    to that dt — so the solver must run at the same dt or syn/obs
        #    will mismatch. Auto-sync here (preserves total time = dt * nt).
        effective_dt = float(spec.time.dt)
        effective_nt = int(spec.time.nt)
        if spec.data_plan is not None and spec.data_plan.dt_target_s is not None:
            target_dt = float(spec.data_plan.dt_target_s)
            if abs(target_dt - effective_dt) > 1e-12:
                new_nt = int(round(effective_dt * effective_nt / target_dt))
                if dist_info.is_root:
                    print(f"[fwi] data_plan.dt_target_s={target_dt}s → auto-sync "
                          f"solver: dt {effective_dt} -> {target_dt}, nt "
                          f"{effective_nt} -> {new_nt}")
                effective_dt = target_dt
                effective_nt = new_nt

        # CFL pre-check from the init_model vmax (cheap mmap peek).
        try:
            ref = init_models[0]
            if ref.path is not None:
                vmax_estimate = float(np.load(ref.path, mmap_mode="r").max())
            elif ref.constant is not None:
                vmax_estimate = float(ref.constant)
            else:
                vmax_estimate = 0.0
            if vmax_estimate > 0 and dist_info.is_root:
                _cfl_check(vmax_estimate, float(spec.grid.dh), effective_dt)
        except FileNotFoundError:
            pass

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, effective_dt, effective_nt, dev
        )
        wavelet = _build_wavelet(
            spec.wavelet, spec.time,
            override_dt=effective_dt, override_nt=effective_nt,
        )
        # Shared cache so SEG-Y-backed geometry + obs don't scan the file twice.
        segy_cache: dict = {}
        sources, receivers = _build_geometry_2d(
            spec.geometry, shape, dh=spec.grid.dh, segy_cache=segy_cache,
        )
        nshots = int(sources.shape[0])

        # 3c) Gap 2 — apply model_plan if set: crop every loaded model array,
        # drop out-of-window sources, rebase geometry indices to the crop.
        model_plan_cropped: dict | None = None
        model_plan_keep: "np.ndarray | None" = None
        if spec.model_plan is not None:
            loaded_originals = []
            for ref in init_models:
                if ref.constant is not None:
                    loaded_originals.append(
                        (ref.name, np.full(tuple(ref.shape), float(ref.constant), dtype=np.float32))
                    )
                else:
                    loaded_originals.append(
                        (ref.name, np.load(ref.path).astype(np.float32))
                    )
            model_plan_cropped, sources, receivers, model_plan_keep = _apply_model_plan_to_fwi(
                spec, loaded_originals, sources, receivers, spec.grid.dh,
            )
            new_shape = next(iter(model_plan_cropped.values())).shape
            if dist_info.is_root:
                print(f"[model_plan] vp shape {shape} -> {new_shape}, "
                      f"sources kept: {int((model_plan_keep is None) or model_plan_keep.sum())}"
                      f"/{nshots}")
            shape = tuple(new_shape)
            nshots = int(sources.shape[0])
            # Rebuild solver at the cropped grid.
            solver = _build_solver(
                spec.physics, spec.backend, shape, spec.grid.dh,
                effective_dt, effective_nt, dev,
            )

        # 4) Build inv_tensors (ordered to match equation.MODEL_SPECS).
        inv_in_order, inv_by_name, required_names = _build_inv_tensors(
            init_models, dev, equation_cls,
            overrides=model_plan_cropped,
        )
        if dist_info.is_root:
            print(f"[fwi] shape={shape} nshots={nshots} inverted_models={required_names} "
                  f"world_size={dist_info.world_size}")

        # 4b) Optional NN reparameterization of vp (sweep-nn).
        # When active, vp is rendered each forward by a hash-encoded SIREN
        # whose parameters become the optimization variables. The vp leaf
        # tensor in inv_by_name is repurposed as a buffer holding the
        # current rendered vp (refreshed after each optimizer step) so
        # snapshots / checkpoints see the right values.
        reparam_net = None
        pristine_base_vp = None
        if spec.reparam is not None and "vp" in inv_by_name:
            reparam_net = _build_reparam_net(
                spec.reparam, inv_by_name["vp"], _effective_bound(spec.model_bounds, "vp"),
            )
            # Keep the pristine base around for stage transitions — at each
            # new stage we bilinear-resample THIS to the new shape and feed
            # the network. The network's learned parameters carry over.
            pristine_base_vp = inv_by_name["vp"].detach().clone()
            # Replace inv_by_name["vp"] with the rendered velocity — and
            # turn off requires_grad on it so the optimizer never sees it.
            with torch.no_grad():
                rendered = reparam_net().detach()
            inv_by_name["vp"] = rendered
            inv_in_order = [
                rendered if n == "vp" else inv_by_name[n] for n in required_names
            ]
            if dist_info.is_root:
                n_params = sum(p.numel() for p in reparam_net.parameters())
                print(f"[fwi] reparam=velocity_inr  net_params={n_params:,}  "
                      f"lr={spec.reparam.lr:.3e}  hash_enabled={spec.reparam.hash.enabled}")

        # 5) Generate or load obs. Every rank does this independently (deterministic).
        # When model_plan dropped shots, the obs loader still produces the
        # ORIGINAL shot count for npy / SEG-Y paths (they don't know about the
        # crop). Generate obs at the original nshots first, then mask.
        if model_plan_keep is not None and (
            spec.obs.npy_path is not None or spec.obs.segy is not None
            or spec.obs.segy_index is not None
        ):
            original_nshots = int(model_plan_keep.size)
            obs = self._fwi_generate_obs(
                spec, equation_cls, solver, wavelet,
                sources, receivers, shape, dev, original_nshots,
                segy_cache=segy_cache,
            )
            obs = obs[model_plan_keep]
        else:
            obs = self._fwi_generate_obs(
                spec, equation_cls, solver, wavelet,
                sources, receivers, shape, dev, nshots,
                segy_cache=segy_cache,
            )

        # 5b) Apply data_plan (and reject unimplemented model_plan) — see
        # _apply_data_plan_to_fwi for the supported subset. Updates
        # `nshots` if shot subsetting kicked in.
        sources, receivers, obs = _apply_data_plan_to_fwi(
            spec, sources, receivers, obs, spec.grid.dh, spec.time.dt, spec.time.nt, dev,
        )
        nshots = int(sources.shape[0])

        # 6) Optimizer + scheduler.
        total_epochs = (sum(s.epochs for s in spec.stages)
                        if spec.stages else int(spec.epochs))
        if reparam_net is not None:
            # In reparam mode the optimizer trains the network's parameters,
            # not the vp tensor. The top-level optimizer.lr is meaningful
            # for grid-FWI (~25) but completely wrong for SIREN/hash params
            # (~1e-4); we use spec.reparam.lr instead. The kind/betas/eps
            # still come from spec.optimizer.
            optimizer = _build_reparam_optimizer(
                spec.optimizer, reparam_net.parameters(), float(spec.reparam.lr),
            )
        else:
            optimizer = _build_optimizer(spec.optimizer, inv_by_name, required_names)
        scheduler = _build_scheduler(spec.scheduler, optimizer, total_epochs)
        initial_lrs = _remember_initial_lrs(optimizer)

        # 7) Resume from checkpoint if requested. Rank 0 loads + broadcasts state.
        # Two modes:
        #   * ``spec.resume_from``: load from a *different* task_id under
        #     output_dir (the original cross-run continuation path).
        #   * ``spec.resume``: auto-resume from THIS task_dir's checkpoint
        #     if one is on disk. Pairs with ``ctrl-c → re-run same yaml``.
        #     Silent no-op when no checkpoint is found.
        losses: list[float] = []
        start_epoch = 0
        resume_src: str | None = None
        if spec.resume_from:
            resume_src = spec.resume_from
            ckpt_dir = Path(spec.output_dir).expanduser() / spec.resume_from
        elif spec.resume and (task_dir / "checkpoint.pt").exists():
            resume_src = task_dir.name
            ckpt_dir = task_dir
        else:
            ckpt_dir = None
        if ckpt_dir is not None:
            if dist_info.is_root:
                ckpt = _load_checkpoint(ckpt_dir)
            else:
                ckpt = None
            ckpt = _dist.broadcast_object(ckpt, dist_info, src=0)
            for name, t in inv_by_name.items():
                t.data.copy_(ckpt["models"][name].to(dev))
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            losses = list(ckpt.get("losses", []))
            start_epoch = int(ckpt["epoch"]) + 1
            if dist_info.is_root:
                torch.set_rng_state(ckpt["torch_rng"])
                np.random.set_state(ckpt["numpy_rng"])
                print(f"[fwi] resumed from '{resume_src}' at epoch {start_epoch}")

        # 8) Multi-stage training loop.
        stages = _normalise_stage_list(spec)
        out_dir = task_dir / "output"
        snapshots_dir = out_dir / "epochs"
        if dist_info.is_root:
            snapshots_dir.mkdir(exist_ok=True)
        _dist.barrier(dist_info)
        epoch_global = start_epoch

        # Build the pristine snapshot used to re-derive each stage's obs
        # (so per-stage bandpass / dt / receiver-mask don't compound).
        # Canonical layout for both backends: (nshots, nt, nrec[, nfield]).
        # 4-D shapes carry the trailing channel; 3-D obs lacks it (rare,
        # e.g. user-supplied npy without channel axis).
        from sweep_io.geometry import PhysicalGeometry
        pristine_time_axis = -3 if obs.ndim >= 4 else -2
        pristine_recv_axis = -2 if obs.ndim >= 4 else -1
        pristine_obs_np = (obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor)
                           else np.asarray(obs))
        # Multi-rank memory share: when the runner is torchrun-launched
        # (world_size > 1), every rank otherwise holds its own ~per-shot
        # × per-trace × nt float32 copy of pristine_obs_np (no slicing
        # at this point). For Viking-scale 2-D obs this is ~2 GB/rank ×
        # 8 ranks = 16 GB JUST for the pristine snapshot, plus transient
        # bandpass/resample doubles during stage transitions, easily
        # OOMs at finer stages. Mitigate by writing the snapshot once
        # (rank 0) to a tmpfile under /dev/shm and re-opening it via
        # ``np.load(..., mmap_mode='r')`` everywhere — Linux shares the
        # file's pages across processes through the page cache, so the
        # per-rank RSS attribution stays tiny.
        if dist_info.is_distributed and pristine_obs_np.nbytes > 64 * 1024 * 1024:
            import os
            import tempfile

            jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
            shm_dir = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
            shared_path = shm_dir / f"sweep_tasks_pristine_obs_{jid}.npy"
            if dist_info.is_root:
                np.save(shared_path, pristine_obs_np)
            _dist.barrier(dist_info)
            try:
                pristine_obs_np = np.load(shared_path, mmap_mode="r")
                if dist_info.is_root:
                    print(f"[fwi] pristine_obs_np ({pristine_obs_np.nbytes / (1<<30):.2f} GB) "
                          f"shared via {shared_path} (mmap'd by every rank)")
            except Exception as err:  # noqa: BLE001
                if dist_info.is_root:
                    print(f"[fwi] pristine_obs_np shared-mmap fallback "
                          f"failed ({err}); each rank keeps its own copy.")
        # Pristine physical geometry: when SEG-Y was loaded, prefer the
        # ORIGINAL un-rounded physical positions (in meters from SEG-Y
        # headers) over the dh-quantized versions. This is critical for
        # ``dedupe=True`` workflows because at finer stages we want
        # ``pg.to_grid(new_dh)`` to use the original 25 m spacing — not
        # 75 m-quantized positions which would collapse identically at
        # every dh ≥ initial.
        pristine_physical_geom = None
        seg_geom_kinds = ("from_segy_headers", "from_segy_index")
        if getattr(spec.geometry, "kind", None) in seg_geom_kinds and segy_cache:
            # The geometry resolver populated segy_cache; grab the payload
            # back and use its original physical_geometry (un-rounded).
            for payload in segy_cache.values():
                pg_orig = payload.get("physical_geometry") if isinstance(payload, dict) else None
                if pg_orig is not None and hasattr(pg_orig, "sources_xyz_m"):
                    pristine_physical_geom = PhysicalGeometry(
                        sources_xyz_m=np.asarray(pg_orig.sources_xyz_m, dtype="float64"),
                        receivers_xyz_m=np.asarray(pg_orig.receivers_xyz_m, dtype="float64"),
                        dt=effective_dt, nt=effective_nt,
                    )
                    break
        if pristine_physical_geom is None:
            pristine_physical_geom = PhysicalGeometry(
                sources_xyz_m=sources.astype("float64") * float(spec.grid.dh),
                receivers_xyz_m=receivers.astype("float64") * float(spec.grid.dh),
                dt=effective_dt, nt=effective_nt,
            )

        # ``pristine_dt`` is the sampling interval of the obs we just snapped
        # — NOT the solver's dt. For SEG-Y obs the runner did NOT resample
        # at load time (per-stage helpers do the dt-sync), so pristine_dt
        # must reflect the SEG-Y file's native dt; otherwise the first
        # stage's ``_resample_obs_time(obs, pristine_dt, new_dt)`` mis-
        # interprets the time axis and either decimates valid samples or
        # smears the trace via fake up-sampling. ``data_plan.dt_target_s``
        # already resampled the obs to ``effective_dt`` upstream, so in
        # that path we keep the effective_dt label.
        obs_native_dt = float(effective_dt)
        if (spec.data_plan is None or spec.data_plan.dt_target_s is None) and (
            (getattr(spec.obs, "segy", None) is not None
             or getattr(spec.obs, "segy_index", None) is not None
             or getattr(spec.obs, "plan", None) is not None)
        ) and segy_cache:
            for payload in segy_cache.values():
                if isinstance(payload, dict) and "dt_s" in payload:
                    obs_native_dt = float(payload["dt_s"])
                    break
            if abs(obs_native_dt - float(effective_dt)) > 1e-12 and dist_info.is_root:
                print(f"[fwi] obs native dt={obs_native_dt}s differs from solver "
                      f"dt={effective_dt}s; pristine_dt set to SEG-Y native so "
                      f"per-stage resample lines up. Set spec.data_plan.dt_target_s "
                      f"to force an upfront resample.")

        # Per-stage mutable state. Seeded with the post-step-7 baseline.
        # Per-stage helpers consult / mutate this dict; everything the
        # train step needs lives in here.
        state: dict = {
            "dh": float(spec.grid.dh), "pristine_dh": float(spec.grid.dh),
            "dt": float(effective_dt),
            "nt": int(effective_nt),
            "shape": tuple(shape), "pristine_shape": tuple(shape),
            "sources": sources, "receivers": receivers, "nshots": int(nshots),
            "inv_in_order": inv_in_order, "inv_by_name": inv_by_name,
            "solver": solver, "wavelet": wavelet,
            "optimizer": optimizer, "initial_lrs": initial_lrs,
            # Reparam (sweep-nn): the network + pristine base for stage
            # transitions. Both None when reparam is not configured.
            "reparam_net": reparam_net,
            "pristine_base_vp": pristine_base_vp,
            "obs": obs, "batchsize": int(spec.batchsize),
            # pristine sources for re-derivation
            "pristine_obs_np": pristine_obs_np,
            "pristine_dt": obs_native_dt,
            "pristine_obs_time_axis": pristine_time_axis,
            "pristine_obs_receiver_axis": pristine_recv_axis,
            "pristine_physical_geom": pristine_physical_geom,
            # Inherit dedupe / dedup_method from geometry spec when available.
            # The default (True / nearest) matches the top-level resolver.
            "dedupe_grid_snap": getattr(spec.geometry, "dedupe", True),
            "dedup_method": getattr(spec.geometry, "dedup_method", "nearest"),
            # QC: snapshot the initial vp at coarsest grid for Δvp plots.
            # The QC orchestrator resamples on demand to the current stage.
            "qc_initial_vp": inv_by_name["vp"].detach().cpu().clone()
                if "vp" in inv_by_name else None,
        }

        # QC setup. The qc_dir + cadence guard live outside the stage loop
        # so the per-epoch hook is just a single function call.
        qc_dir = task_dir / "qc"
        qc_enabled = spec.qc is not None and spec.qc.every_n_epochs > 0
        stage_epoch_boundaries: list[int] = []  # cumulative epoch indices

        # If a signal landed during setup, skip training entirely and
        # fall through to final-outputs (which will write whatever state
        # we have — typically just the init model).
        if stopper.should_stop(dist_info):
            interrupted = True
            interrupted_at_epoch = epoch_global - 1 if epoch_global > 0 else None
            if dist_info.is_root:
                print(f"[fwi] signal received during setup; skipping training loop.")

        for stage_idx, stage in enumerate(stages):
            if interrupted:
                break
            stage_offset = sum(s.epochs for s in stages[:stage_idx])
            already_done_in_stage = max(0, epoch_global - stage_offset)
            remaining = stage.epochs - already_done_in_stage
            if remaining <= 0:
                continue

            # Free GPU memory held by the previous stage's solver / state
            # before allocating the next one — boundary buffers / wavefield
            # caches can hold hundreds of MB to many GB.
            if stage_idx > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            _prepare_stage(
                spec=spec, stage=stage, state=state,
                equation_cls=equation_cls, required_names=required_names,
                dev=dev, dist_info=dist_info, stage_idx=stage_idx,
            )

            # Record stage-end epoch index for the final loss-curve QC
            # (shading boundaries). Use the absolute epoch counter.
            stage_epoch_boundaries.append(epoch_global + remaining)

            if dist_info.is_root:
                print(f"[fwi] stage {stage_idx} ({remaining}/{stage.epochs} epochs) "
                      f"lr_scale={stage.lr_scale} wavelet_overridden={stage.wavelet is not None}")
            # Build the per-batch local-window context once per stage and
            # pass it through; the chunk dispatcher inside _fwi_train_step
            # uses it to compute fresh windows + cached solvers per chunk.
            local_window_ctx = None
            if (spec.local_model_window is not None
                    and spec.local_model_window.enabled):
                local_window_ctx = {
                    "spec": spec.local_model_window,
                    "shape": state["shape"],
                    "dh": state["dh"],
                    "dt": state["dt"],
                    "nt": state["nt"],
                    "solver_cache": state.setdefault("local_solver_cache", {}),
                }
            # Per-stage syn bandpass: matches obs's per-stage bandpass so the
            # loss compares like-for-like. fwi_workflow-dev does the same via
            # ``_apply_torch_filter`` on syn (sweep_fwi.py:1040). When the
            # stage's bandpass ``target == 'wavelet'``, the wavelet was
            # pre-filtered at stage entry and syn is naturally band-limited;
            # we skip the per-iteration syn filter to save autograd cost.
            stage_bandpass = stage.bandpass
            if stage_bandpass is not None \
                    and getattr(stage_bandpass, "target", "syn") == "wavelet":
                stage_bandpass = None  # already applied to wavelet
            for _ in range(remaining):
                # Only save the obs/syn snapshot for QC when QC actually fires
                # this iter. Without this gate every iter pays a ~250 ms D2H
                # copy of syn+obs_chunk (~300 MB) into a dict that's discarded
                # next iter.
                _take_qc_snap = qc_enabled and dist_info.is_root and (
                    epoch_global % spec.qc.every_n_epochs == 0
                    or epoch_global == total_epochs - 1
                )
                loss_value = self._fwi_train_step(
                    spec, state["solver"], state["wavelet"],
                    state["sources"], state["receivers"],
                    state["inv_in_order"], state["inv_by_name"],
                    state["obs"], state["optimizer"],
                    state["nshots"], dev,
                    dist_info=dist_info,
                    stage_batchsize=state["batchsize"],
                    reparam_net=state.get("reparam_net"),
                    local_window_ctx=local_window_ctx,
                    syn_bandpass=stage_bandpass,
                    stage_dt=state["dt"],
                    state_for_dump=state,
                    take_qc_snapshot=_take_qc_snap,
                )
                losses.append(loss_value)
                if scheduler is not None:
                    scheduler.step()
                # In reparam mode the network's render-time clamp already
                # enforces vp bounds; only clamp the non-reparam tensors.
                reparam_skip = {"vp"} if state.get("reparam_net") is not None else set()
                _apply_bounds(state["inv_by_name"], spec.model_bounds, skip_names=reparam_skip)
                # Refresh inv_by_name["vp"] from the network so snapshots /
                # checkpoints save the actual current rendered velocity.
                if state.get("reparam_net") is not None:
                    with torch.no_grad():
                        rendered = state["reparam_net"]().detach()
                    state["inv_by_name"]["vp"] = rendered
                    state["inv_in_order"] = [
                        rendered if n == "vp" else state["inv_by_name"][n]
                        for n in required_names
                    ]
                if dist_info.is_root:
                    print(f"[fwi] stage {stage_idx} epoch {epoch_global:04d} loss={loss_value:.6e}")

                snapshot_now = (epoch_global % spec.show_every == 0
                                or epoch_global == total_epochs - 1)
                if snapshot_now and dist_info.is_root:
                    for name, t in state["inv_by_name"].items():
                        np.save(snapshots_dir / f"{name}_epoch_{epoch_global:04d}.npy",
                                t.detach().cpu().numpy())
                    if spec.save_illumination:
                        _save_illumination(state["solver"], snapshots_dir, epoch_global)

                # QC artefacts (vp / Δvp / gradient / shot gather) — only
                # at the configured cadence and only on the rank 0 process.
                if qc_enabled and dist_info.is_root and (
                    epoch_global % spec.qc.every_n_epochs == 0
                    or epoch_global == total_epochs - 1
                ):
                    self._run_epoch_qc_safe(
                        spec=spec, state=state, qc_dir=qc_dir,
                        epoch=epoch_global, dev=dev,
                    )

                if dist_info.is_root:
                    _save_checkpoint(task_dir, {
                        "epoch": epoch_global,
                        "models": {name: t.detach().cpu().clone()
                                   for name, t in state["inv_by_name"].items()},
                        "optimizer": state["optimizer"].state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "losses": losses,
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                    })
                # Honor graceful-stop request now that the checkpoint for the
                # just-completed epoch is on disk. The next ``sweep-tasks run``
                # with ``resume: true`` will pick up at epoch_global + 1.
                if stopper.should_stop(dist_info):
                    interrupted = True
                    interrupted_at_epoch = epoch_global
                    if dist_info.is_root:
                        print(f"[fwi] stopping after epoch {epoch_global} "
                              f"(checkpoint.pt saved). Re-run the same YAML "
                              f"with `resume: true` to continue.")
                    epoch_global += 1
                    break
                epoch_global += 1
            if interrupted:
                break

        stopper.uninstall()

        # Stage loop done — keep references aligned for the final-outputs step.
        inv_by_name = state["inv_by_name"]
        inv_in_order = state["inv_in_order"]
        solver = state["solver"]
        sources = state["sources"]
        receivers = state["receivers"]

        # 9) Final outputs (rank 0 only).
        artifacts: list[Path] = []
        if dist_info.is_root:
            for name, t in inv_by_name.items():
                p = out_dir / f"inverted_{name}.npy"
                np.save(p, t.detach().cpu().numpy())
                artifacts.append(p)
            loss_path = out_dir / "loss.npy"
            np.save(loss_path, np.array(losses, dtype=np.float64))
            artifacts.append(loss_path)
            try:
                artifacts.append(_plot_loss_curve(losses, out_dir / "loss.png", title="FWI Loss"))
            except Exception as plot_err:  # noqa: BLE001
                print(f"[fwi] loss plot skipped: {plot_err}")
            # QC final products (loss curve with stage shading etc.)
            if spec.qc is not None:
                try:
                    from sweep_tasks.qc import save_final_qc
                    final_qc = save_final_qc(
                        qc_spec=spec.qc, qc_dir=qc_dir, losses=losses,
                        stage_epoch_boundaries=stage_epoch_boundaries,
                    )
                    artifacts.extend(final_qc)
                except Exception as qc_err:  # noqa: BLE001
                    print(f"[fwi] final QC skipped: {qc_err}")
        _dist.barrier(dist_info)

        summary = {
            "epochs": total_epochs,
            "epochs_completed": len(losses),
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "models_inverted": list(required_names),
            "num_stages": len(stages),
            "resumed_from": resume_src,
            "interrupted": interrupted,
            "interrupted_at_epoch": interrupted_at_epoch,
            "world_size": dist_info.world_size,
        }
        return artifacts, summary

    def _fwi_generate_obs(self, spec, equation_cls, solver, wavelet,
                          sources, receivers, shape, dev, nshots,
                          *, segy_cache: dict | None = None):
        """Return obs tensor in canonical layout (nshots, nt, nrec, 1)
        for both backends (geophyai 21041c5 aligned the c-backend record
        output). Always CPU-resident.

        For SEG-Y sources, when the same file / index was already scanned by
        the geometry resolver, the cached obs is reused without a second read.
        """

        import torch

        obs_spec = spec.obs
        if obs_spec.npy_path is not None:
            obs_np = np.load(obs_spec.npy_path).astype(np.float32)
            obs = torch.from_numpy(obs_np)
            if obs.shape[0] != nshots:
                raise ValueError(
                    f"obs.npy_path first-axis size {obs.shape[0]} != nshots {nshots}."
                )
            return obs

        def _adapt_segy_obs_to_backend(obs_nrec_nt: np.ndarray) -> "torch.Tensor":
            """SEGYReader always returns (nshots, nrec, nt). Match sweep's
            canonical syn layout shared by both backends after geophyai
            commit 21041c5: (nshots, nt, nrec, 1).
            """
            arr = obs_nrec_nt.astype(np.float32, copy=False)
            arr = np.ascontiguousarray(arr.transpose(0, 2, 1))[..., None]
            return torch.from_numpy(arr)

        if obs_spec.segy is not None:
            cfg = obs_spec.segy
            payload = _load_segy_single_file_payload(
                cfg.path,
                byte_map=cfg.byte_map,
                source_depth_m_override=cfg.source_depth_m_override,
                receiver_depth_m_override=cfg.receiver_depth_m_override,
                coord_scalar_override=cfg.coord_scalar_override,
                shot_ids=cfg.shot_ids,
                cache=segy_cache,
            )
            # Initial parse: keep ALL receivers (dedupe=False). Runner-side
            # per-stage dedupe (when geometry.dedupe=True) operates on the
            # pristine 120-receiver obs, allowing finer stages to access the
            # full receiver count even if a coarser stage drops some.
            _, _, obs_aligned = _segy_geometry_to_grid_indices(
                payload, float(spec.grid.dh), dedupe=False, dedup_method="nearest",
            )
            if obs_aligned.shape[0] != nshots:
                raise ValueError(
                    f"obs.segy gave {obs_aligned.shape[0]} shots but geometry-derived nshots={nshots}."
                )
            return _adapt_segy_obs_to_backend(obs_aligned)

        if obs_spec.segy_index is not None:
            cfg = obs_spec.segy_index
            payload = _load_segy_index_payload(
                cfg.index_path,
                shot_ids=cfg.shot_ids,
                cache=segy_cache,
                lazy=cfg.lazy,
                coalesce_gap=cfg.coalesce_gap,
            )
            # Initial parse: keep ALL receivers (per-stage dedupe handled
            # later by ``_prepare_stage``).
            _, _, obs_aligned = _segy_geometry_to_grid_indices(
                payload, float(spec.grid.dh), dedupe=False, dedup_method="nearest",
            )
            if obs_aligned.shape[0] != nshots:
                raise ValueError(
                    f"obs.segy_index gave {obs_aligned.shape[0]} shots but "
                    f"geometry-derived nshots={nshots}."
                )
            return _adapt_segy_obs_to_backend(obs_aligned)

        if obs_spec.plan is not None:
            cfg = obs_spec.plan
            grid_ndim = len(spec.grid.shape) if spec.grid.shape is not None else None
            payload = _load_seismic_plan_payload(
                cfg.plan_path,
                cache_all=bool(cfg.cache_all),
                grid_ndim=grid_ndim,
                cache=segy_cache,
            )
            _, _, obs_aligned = _segy_geometry_to_grid_indices(
                payload, float(spec.grid.dh), dedupe=False, dedup_method="nearest",
            )
            if obs_aligned.shape[0] != nshots:
                raise ValueError(
                    f"obs.plan gave {obs_aligned.shape[0]} shots but "
                    f"geometry-derived nshots={nshots}."
                )
            return _adapt_segy_obs_to_backend(obs_aligned)

        # synthetic: collect true models in equation MODEL_SPECS order
        if obs_spec.synthetic_from_models is not None:
            true_refs = list(obs_spec.synthetic_from_models)
        else:
            true_refs = [obs_spec.synthetic_from]
        required = _model_names_for_equation(equation_cls)
        by_name = {m.name: m for m in true_refs}
        if set(by_name) != set(required):
            raise ValueError(
                f"obs.synthetic_from* model names {sorted(by_name)} must match "
                f"equation '{equation_cls.__name__}' models {required}."
            )
        # When model_plan is active, crop the synthetic-from "true" tensors
        # the same way the init vp was cropped, so the solver (built at the
        # cropped shape) gets matching inputs.
        if spec.model_plan is not None:
            from sweep_io.plan import ModelPlan, apply_model_plan
            mp = ModelPlan(
                x_window_m=spec.model_plan.x_window_m,
                y_window_m=spec.model_plan.y_window_m,
                z_window_m=spec.model_plan.z_window_m,
                drop_outside_sources=False,
                drop_outside_receivers=False,
            )
            true_in_order = []
            for n in required:
                arr = np.load(by_name[n].path).astype(np.float32) if by_name[n].path is not None \
                      else np.full(by_name[n].shape, float(by_name[n].constant), dtype=np.float32)
                dh_tuple = (float(spec.grid.dh),) * arr.ndim
                cropped, _, _ = apply_model_plan(mp, arr, dh=dh_tuple, geom=None)
                true_in_order.append(torch.from_numpy(np.ascontiguousarray(cropped)).to(dev))
        else:
            true_in_order = [_load_model_tensor(by_name[n]).to(dev) for n in required]

        mod_wavelet, mod_sources, mod_receivers, used_override = _resolve_modeling_inputs(
            spec, wavelet, sources, receivers, shape
        )
        if used_override:
            print(f"[fwi] modeling_override applied (wavelet={spec.modeling_override.wavelet is not None}, "
                  f"geometry={spec.modeling_override.geometry is not None})")
        with torch.no_grad():
            # Chunk the obs-generation forward over shots so peak solver
            # workspace stays O(chunk) instead of O(nshots). Generating all
            # shots in one call allocates an adjoint workspace for the whole
            # batch and OOMs on large OBN surveys; the per-shot obs are
            # independent, so concatenating chunk outputs is exact. Chunk by
            # ``batchsize`` (the training shot batch); wavelet is shared.
            _n_src = int(mod_sources.shape[0])
            _cn = max(1, int(getattr(spec, "batchsize", 0) or _n_src))
            if _cn >= _n_src:
                obs = solver(mod_wavelet, mod_sources, mod_receivers,
                             models=true_in_order).detach().cpu()
            else:
                _obs_parts = []
                for _s0 in range(0, _n_src, _cn):
                    _s1 = min(_s0 + _cn, _n_src)
                    _part = solver(mod_wavelet, mod_sources[_s0:_s1],
                                   mod_receivers[_s0:_s1],
                                   models=true_in_order).detach().cpu()
                    _obs_parts.append(_part)
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                obs = torch.cat(_obs_parts, dim=0)
        del true_in_order
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return obs

    def _run_epoch_qc_safe(self, *, spec, state, qc_dir: Path, epoch: int, dev) -> None:
        """Run all enabled QC products; never let plotting break the inversion.

        Optionally captures a forward-modelled syn for shot-gather QC when
        ``spec.qc.shot_gather`` is on. The shot indices used are spread
        evenly across the current obs to give a representative cross-section
        of acquisition geometry.
        """
        import torch

        try:
            from sweep_tasks import qc as qc_mod
        except Exception as err:  # noqa: BLE001
            print(f"[qc] import failed at epoch {epoch}: {err}")
            return

        extract = None
        if spec.qc.shot_gather:
            # Reuse the syn / obs snapshot captured at the end of the most
            # recent train step's last chunk. This avoids running an extra
            # solver forward inside the QC path (which doubles per-iter
            # GPU work) AND guarantees we visualise exactly what the loss
            # compared — including the syn bandpass applied in the train
            # step.
            snap = getattr(self, "_last_qc_snapshot", None)
            if snap is None:
                # First QC fires before the first train step on rare cold
                # paths; nothing to plot. Skip silently.
                extract = None
            else:
                n_pick = max(1, min(int(spec.qc.shot_gather_n_shots),
                                    int(snap["syn"].shape[0])))
                # Evenly subsample within the last chunk to give a
                # representative sample at low cost.
                snap_n = int(snap["syn"].shape[0])
                pick_within = np.linspace(0, snap_n - 1, n_pick, dtype=int)
                shot_ids = [int(snap["chunk_indices"][int(i)]) for i in pick_within]

            def _extract():
                syn_np = snap["syn"][pick_within].numpy()
                obs_chunk = snap["obs"][pick_within].numpy()
                # Both backends emit canonical (n_shots, nt, nrec, nchan).
                # Squeeze the trailing channel for the QC plotter which
                # expects (n_shots, nt, nrec) with time on axis -2.
                if syn_np.ndim == 4 and syn_np.shape[-1] == 1:
                    syn_np = syn_np[..., 0]
                if obs_chunk.ndim == 4 and obs_chunk.shape[-1] == 1:
                    obs_chunk = obs_chunk[..., 0]

                # Per-shot dedupe: at coarse dh, multiple physical receivers
                # snap to the same grid cell, producing identical syn traces.
                # The loss already gets these as duplicates; the QC should
                # display only ONE trace per unique grid cell so the user
                # sees what the inversion actually "sees".
                dh_m = float(state.get("dh", 1.0))
                src_grid = np.asarray(state["sources"][shot_ids], dtype=np.int64)
                rec_grid = np.asarray(state["receivers"][shot_ids], dtype=np.int64)
                src_xy_m = src_grid.astype(np.float64) * dh_m       # (n_shots, 2)
                rec_xy_m = rec_grid.astype(np.float64) * dh_m       # (n_shots, nrec, 2)
                # Build per-shot unique-cell indices.
                unique_idx_per_shot = []
                for s in range(len(shot_ids)):
                    seen: dict = {}
                    keep_in_order: list[int] = []
                    for r in range(rec_grid.shape[1]):
                        key = (int(rec_grid[s, r, 0]), int(rec_grid[s, r, 1]))
                        if key not in seen:
                            seen[key] = r
                            keep_in_order.append(r)
                    unique_idx_per_shot.append(np.asarray(keep_in_order, dtype=np.int64))
                return {
                    "obs": obs_chunk,                # (n_shots, nt, nrec) raw
                    "syn": syn_np,                   # (n_shots, nt, nrec) raw
                    "shot_ids": shot_ids,
                    "src_xy_m": src_xy_m,            # (n_shots, 2) in meters
                    "rec_xy_m": rec_xy_m,            # (n_shots, nrec, 2) in meters
                    "unique_idx_per_shot": unique_idx_per_shot,
                    "dh_m": dh_m,
                    "dt_s": float(state.get("dt", 1.0)),
                }

            extract = _extract

        try:
            qc_mod.run_epoch_qc(
                qc_spec=spec.qc, qc_dir=qc_dir, epoch=epoch,
                state=state, spec=spec, extract_obs_syn_panels=extract,
            )
        except Exception as err:  # noqa: BLE001
            print(f"[qc] epoch {epoch} failed: {err}")

    def _maybe_dump_loss_inputs(self, obs_chunk, syn, chunk, dist_info,
                                syn_bandpass, stage_dt,
                                state: dict | None = None) -> None:
        """One-shot debug dump of the exact obs/syn pair the loss consumed.

        Activated by env ``DUMP_LOSS_INPUTS=<path.npz>``. Fires once per
        process — saves the last chunk's CPU-side obs and syn (post
        bandpass), the chunk shot indices, and the current stage's dt /
        bandpass metadata. With ``DUMP_LOSS_INPUTS_EXIT=1`` the runner
        exits cleanly right after the dump, so users can capture a
        single iter's loss inputs with one sbatch and read them back
        offline.
        """
        if not dist_info.is_root:
            return
        dump_path = os.environ.get("DUMP_LOSS_INPUTS")
        if not dump_path or getattr(self, "_loss_inputs_dumped", False):
            return
        payload: dict = {
            "obs": obs_chunk.detach().cpu().numpy(),
            "syn": syn.detach().cpu().numpy(),
            "chunk_indices": np.asarray(chunk, dtype=np.int64),
            "stage_dt_s": float(stage_dt) if stage_dt is not None else float("nan"),
        }
        if syn_bandpass is not None:
            payload["bandpass_lo_hz"] = float(syn_bandpass.lo_hz)
            payload["bandpass_hi_hz"] = float(syn_bandpass.hi_hz)
            payload["bandpass_order"] = int(syn_bandpass.order)
        # Optional: snapshot the runner-side per-shot positions + pre-bandpass
        # pristine obs slice for the same chunk, so the user can verify
        # exactly what raw data the bandpass was applied to.
        if state is not None:
            try:
                src = np.asarray(state.get("sources"))[np.asarray(chunk, dtype=np.int64)]
                payload["source_grid_idx"] = np.asarray(src, dtype=np.int64)
                rec = np.asarray(state.get("receivers"))[np.asarray(chunk, dtype=np.int64)]
                payload["receiver_grid_idx"] = np.asarray(rec, dtype=np.int64)
                payload["grid_dh_m"] = float(state.get("dh", float("nan")))
                pristine = state.get("pristine_obs_np")
                if pristine is not None:
                    payload["pristine_obs_for_chunk"] = (
                        np.asarray(pristine)[np.asarray(chunk, dtype=np.int64)]
                        .astype(np.float32, copy=False)
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[debug] dump-state extras failed: {exc}", flush=True)
        out_dir = os.path.dirname(dump_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(dump_path, **payload)
        self._loss_inputs_dumped = True
        print(f"[debug] dumped loss inputs to {dump_path} "
              f"(obs={payload['obs'].shape}, syn={payload['syn'].shape})",
              flush=True)
        if os.environ.get("DUMP_LOSS_INPUTS_EXIT"):
            print("[debug] DUMP_LOSS_INPUTS_EXIT set; exiting cleanly",
                  flush=True)
            raise SystemExit(0)

    def _get_data_mask(self, spec, obs, dev):
        """Optional per-sample DATA mute mask (the on/off switch). Returns a CPU
        float tensor broadcastable to ``obs`` (nshots, nt, nrec[, nchan]) loaded
        from ``spec.loss.data_mask_path``, or None when unset -> the misfit runs
        exactly as before. Cached after first load. Indexed per-chunk like obs."""
        import torch
        path = getattr(getattr(spec, "loss", None), "data_mask_path", None)
        if not path:
            return None
        cached = getattr(self, "_data_mask_cache", None)
        if cached is not None and cached[0] == path:
            return cached[1]
        m = torch.as_tensor(np.asarray(np.load(path)), dtype=torch.float32)
        while m.ndim < obs.ndim:
            m = m.unsqueeze(-1)          # per-trace (nshots,nt,nrec) -> add channel axis
        try:
            torch.broadcast_shapes(tuple(m.shape), tuple(obs.shape))
        except RuntimeError as e:
            raise ValueError(
                f"data_mask_path shape {tuple(m.shape)} not broadcastable to "
                f"obs {tuple(obs.shape)}: {e}"
            )
        self._data_mask_cache = (path, m)
        return m

    def _fwi_train_step(self, spec, solver, wavelet, sources, receivers,
                        inv_in_order, inv_by_name, obs, optimizer, nshots, dev,
                        *, dist_info=None, stage_batchsize: int | None = None,
                        reparam_net=None, local_window_ctx=None,
                        syn_bandpass=None, stage_dt: float | None = None,
                        state_for_dump: dict | None = None,
                        take_qc_snapshot: bool = True) -> float:
        """One outer optimizer step.

        In single-process mode the rank picks a `batchsize` shot batch, breaks
        it into `train_shot_batchsize` chunks, runs forward+backward on each
        chunk, and steps. In distributed mode (world_size > 1) the **global**
        batchsize is split round-robin across ranks (rank 0 picks + broadcasts
        the global indices, then `shot_idx[rank::world_size]` is local). The
        loss is normalised by the GLOBAL element count so that
        ``all_reduce(SUM)`` of gradients yields the gradient of the global
        per-element mean loss.

        ``stage_batchsize`` (if given) overrides ``spec.batchsize`` — used
        by the multi-stage loop so each stage can pick its own shot count
        (Gap 4).
        """

        import torch

        from sweep_runner import distributed as _dist

        if dist_info is None:
            dist_info = getattr(self, "_dist", None) or _dist.init_distributed_if_needed()

        global_batchsize = min(
            int(stage_batchsize) if stage_batchsize is not None else int(spec.batchsize),
            nshots,
        )

        # Rank 0 picks the global shot indices, then broadcasts.
        if dist_info.is_root:
            shot_idx_global = np.random.choice(nshots, size=global_batchsize, replace=False)
        else:
            shot_idx_global = None
        shot_idx_global = _dist.broadcast_shot_indices(
            shot_idx_global, global_batchsize, dist_info, src=0
        )

        local_shots = _dist.split_for_rank(shot_idx_global, dist_info)
        chunk_size = spec.train_shot_batchsize or max(len(local_shots), 1)
        if len(local_shots) > 0:
            chunks = [local_shots[i:i + chunk_size]
                      for i in range(0, len(local_shots), chunk_size)]
        else:
            chunks = []

        # Global element count used for loss normalisation. Pulled from a sample
        # obs slice so any inferred dimension (nt, nrec) stays in sync.
        sample = obs[:1]
        per_shot_numel = int(sample.numel())
        global_norm = float(per_shot_numel * global_batchsize)
        data_mask = self._get_data_mask(spec, obs, dev)
        if data_mask is not None:
            global_norm = float(global_norm * max(float(data_mask.float().mean()), 1.0e-6))

        # Per-chunk forward input. In reparam mode we render a fresh vp
        # each chunk so the autograd graph from solver -> loss -> backward
        # flows into the network's parameters. The other inv tensors (if
        # any) pass through unchanged.
        def _models_for_chunk():
            if reparam_net is None:
                return inv_in_order
            rendered = reparam_net()
            return [
                rendered if name == "vp" else inv_by_name[name]
                for name in [m for m in inv_by_name]
            ] if len(inv_by_name) > 1 else [rendered]

        # Per-chunk local-window dispatcher. When ``local_window_ctx`` is
        # set, each chunk gets a (z0, z1, x0, x1) crop of the model + a
        # shape-keyed cached solver + window-local geometry indices.
        # Otherwise we fall back to the full-model path (above).
        def _chunk_inputs(chunk_idx):
            chunk_src = sources[chunk_idx]
            chunk_rec = receivers[chunk_idx]
            if local_window_ctx is None:
                return _models_for_chunk(), solver, chunk_src, chunk_rec
            win_spec = local_window_ctx["spec"]
            full_shape = local_window_ctx["shape"]
            dh = local_window_ctx["dh"]
            window = _compute_local_window(
                chunk_src, chunk_rec, full_shape, dh, win_spec,
            )
            if len(window) == 4:
                z0, z1, x0, x1 = window
                local_shape = (z1 - z0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0}
                vp_slice = (slice(z0, z1), slice(x0, x1))
            else:
                z0, z1, y0, y1, x0, x1 = window
                local_shape = (z1 - z0, y1 - y0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0, "y0": y0}
                vp_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            cache = local_window_ctx["solver_cache"]
            cache_key = local_shape
            if cache_key not in cache:
                # Bound the cache so randomly varying windows don't accumulate
                # solvers indefinitely (each c-backend solver pins GPU memory
                # for its wavefield + PML buffers). Evict the least-recently-
                # used entry when over the cap.
                _LOCAL_SOLVER_CACHE_CAP = 4
                if len(cache) >= _LOCAL_SOLVER_CACHE_CAP:
                    # dict insertion-ordered: drop the oldest entry.
                    oldest = next(iter(cache))
                    del cache[oldest]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                cache[cache_key] = _build_solver(
                    spec.physics, spec.backend, local_shape,
                    dh, local_window_ctx["dt"], local_window_ctx["nt"], dev,
                )
            else:
                # Move-to-end so this entry is "most recently used".
                cache[cache_key] = cache.pop(cache_key)
            local_solver = cache[cache_key]
            local_src, local_rec = _rebase_geometry_to_window(chunk_src, chunk_rec, **rebase_kwargs)
            # Build the model list. Only vp is currently supported as
            # the windowed model; non-vp models pass through unmodified.
            if reparam_net is not None:
                if len(window) != 4:
                    raise NotImplementedError(
                        "Reparam render_window is only implemented for 2-D "
                        "grids; disable spec.reparam for 3-D local windowing."
                    )
                local_vp = reparam_net.render_window(z0, z1, x0, x1)
            else:
                # Slicing a leaf tensor yields a view; PyTorch's autograd
                # scatters the gradient back to the leaf at vp_slice.
                local_vp = inv_by_name["vp"][vp_slice]
            local_vp = local_vp.contiguous()
            ordered_names = list(inv_by_name.keys())
            models = [
                local_vp if name == "vp" else inv_by_name[name]
                for name in ordered_names
            ] if len(inv_by_name) > 1 else [local_vp]
            # perf/acoustic-bwd-skip-illum (3d6fe97): the c-backend now makes
            # backward illumination OPT-IN (default off for speed). Enable it on
            # each (cached) solver when illumination_precondition is active, else
            # solver.source_illumination stays None and the precond silently
            # no-ops. Harmless on older cores (plain attr, ignored).
            if getattr(getattr(spec, "illumination_precondition", None), "enabled", False):
                try:
                    local_solver.compute_illumination = True
                except Exception:
                    pass
            return models, local_solver, local_src, local_rec

        if spec.optimizer.kind == "lbfgs":
            # LBFGS in single-process only (the runner blocks dist + LBFGS earlier).
            def _closure():
                optimizer.zero_grad()
                acc_loss = 0.0
                for chunk in chunks:
                    models, chunk_solver, chunk_src, chunk_rec = _chunk_inputs(chunk)
                    syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                    if syn_bandpass is not None and stage_dt is not None:
                        syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                                  stage_dt, order=syn_bandpass.order)
                    obs_chunk = obs[chunk].to(dev)
                    loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev))
                    (loss_t / global_norm).backward()
                    acc_loss += float(loss_t.detach().cpu())
                if reparam_net is None:
                    _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                return torch.tensor(acc_loss / global_norm if global_norm else 0.0)
            result = optimizer.step(_closure)
            return float(result) if result is not None else 0.0

        # Adam / SGD path. Two top-level branches:
        #   (A) Single-step (no reparam, OR reparam with backward_mode='single_step'):
        #       one autograd graph through net -> solver -> loss -> backward.
        #   (B) Two-pass reparam (default, mirrors fwi_workflow-dev): render
        #       net under no_grad to a leaf, solver backward into leaf.grad,
        #       then a second pass pushes leaf.grad through net in either
        #       'two_pass_full' (one big backward) or 'two_pass_chunked'
        #       (row-chunked re-render via VelocityINR.backward_velocity_gradient).
        reparam_mode = (
            spec.reparam.backward_mode
            if (reparam_net is not None and spec.reparam is not None)
            else "single_step"
        )

        illum_spec = getattr(spec, "illumination_precondition", None)
        illum_on = bool(illum_spec is not None and illum_spec.enabled)
        if illum_on and local_window_ctx is not None:
            raise NotImplementedError(
                "illumination_precondition is not supported together with "
                "local_model_window: the per-window illumination tensor "
                "shape varies per chunk, so per-window accumulation is "
                "not implemented. Disable one of the two."
            )
        # The eager (autograd PyTorch) backend does not populate
        # ``solver.source_illumination`` / ``receiver_illumination`` — those
        # are a feature of the compiled C kernel. Warn once per train step
        # so the user isn't silently running a no-op precondition.
        if illum_on and dist_info.is_root and getattr(spec.backend, "impl", "eager") == "eager":
            if not getattr(self, "_illum_eager_warned", False):
                print(
                    "[illum] WARN: illumination_precondition is enabled but "
                    "spec.backend.impl='eager' — the eager autograd backend "
                    "does not populate solver.source_illumination / "
                    "receiver_illumination, so the precondition is a no-op. "
                    "Switch to backend.impl='c' for the production path."
                )
                self._illum_eager_warned = True
        if reparam_net is None or reparam_mode == "single_step":
            # ---- (A) single-step: everything inside one autograd graph ----
            # Optional per-iter profiling: set ``SWEEP_TASKS_TPROF=1`` in
            # the environment to print a millisecond-level breakdown of
            # each step (zero_grad / forward / bandpass / obs-H2D / loss /
            # backward / item / snapshot-D2H / reduce+clamp / opt.step).
            # Each phase boundary is followed by ``torch.cuda.synchronize()``
            # so the numbers reflect real GPU time, not kernel launch latency.
            import time as _t
            _TPROF = os.environ.get("SWEEP_TASKS_TPROF") == "1"
            def _sync():
                if _TPROF and torch.cuda.is_available(): torch.cuda.synchronize()
            if _TPROF: _sync(); _t0 = _t.perf_counter()
            optimizer.zero_grad()
            if _TPROF: _sync(); _t_zero = _t.perf_counter()
            acc_loss_local = 0.0
            sill_sum = None
            rill_sum = None
            for chunk_idx_in_iter, chunk in enumerate(chunks):
                models, chunk_solver, chunk_src, chunk_rec = _chunk_inputs(chunk)
                if _TPROF: _sync(); _t_pre = _t.perf_counter()
                syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                if _TPROF: _sync(); _t_fwd = _t.perf_counter()
                if syn_bandpass is not None and stage_dt is not None:
                    syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                              stage_dt, order=syn_bandpass.order)
                if _TPROF: _sync(); _t_bp = _t.perf_counter()
                obs_chunk = obs[chunk].to(dev)
                if _TPROF: _sync(); _t_obs = _t.perf_counter()
                loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev))
                if _TPROF: _sync(); _t_loss = _t.perf_counter()
                (loss_t / global_norm).backward()
                if _TPROF: _sync(); _t_bwd = _t.perf_counter()
                acc_loss_local += float(loss_t.detach().cpu())
                if _TPROF: _sync(); _t_item = _t.perf_counter()
                if illum_on:
                    sill_sum, rill_sum = _accumulate_illumination(
                        chunk_solver, sill_sum, rill_sum,
                    )
                # Snapshot last chunk for QC reuse (avoid running a fresh
                # forward inside ``_run_epoch_qc_safe._extract``). Only the
                # final chunk's syn/obs survive across iters; QC at every
                # show_every iter will see the most recent train batch.
                if (take_qc_snapshot
                        and dist_info.is_root
                        and chunk_idx_in_iter == len(chunks) - 1):
                    self._last_qc_snapshot = {
                        "chunk_indices": np.asarray(chunk, dtype=np.int64).copy(),
                        "syn": syn.detach().to("cpu"),
                        "obs": obs_chunk.detach().to("cpu"),
                    }
                    self._maybe_dump_loss_inputs(
                        obs_chunk, syn, chunk, dist_info, syn_bandpass, stage_dt,
                        state=state_for_dump,
                    )
                if _TPROF: _sync(); _t_snap = _t.perf_counter()
            if reparam_net is None:
                _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                _dist.all_reduce_grad_sum(inv_in_order, dist_info)
                if illum_on:
                    # Match the gradient's all-reduce so multi-rank accumulation
                    # is consistent. We only support per-cell vp illumination
                    # here (the only inverted model in OBN FWI).
                    if sill_sum is not None and rill_sum is not None:
                        _dist.all_reduce_sum_inplace(sill_sum, dist_info)
                        _dist.all_reduce_sum_inplace(rill_sum, dist_info)
                    vp_grad = inv_by_name.get("vp").grad if "vp" in inv_by_name else None
                    _apply_illumination_precond(
                        vp_grad, sill_sum, rill_sum,
                        eps=illum_spec.epsilon, exponent=illum_spec.exponent,
                    )
            else:
                _dist.all_reduce_grad_sum(
                    [p for p in reparam_net.parameters() if p.grad is not None],
                    dist_info,
                )
                if illum_on and dist_info.is_root:
                    # Reparam + single_step: net params don't have a per-cell
                    # illumination correspondence. Use two_pass_full instead
                    # to get illumination preconditioning on the leaf gradient
                    # before the network backward.
                    print("[illum] WARN: illumination_precondition is "
                          "a no-op under reparam.backward_mode='single_step'. "
                          "Switch to 'two_pass_full' to enable it.")
            if _TPROF: _sync(); _t_reduce = _t.perf_counter()
            optimizer.step()
            if _TPROF:
                _sync(); _t_step = _t.perf_counter()
                _ms = lambda a, b: (b - a) * 1000.0
                print(f"[tprof] zero={_ms(_t0,_t_zero):5.1f} "
                      f"fwd={_ms(_t_pre,_t_fwd):6.1f} "
                      f"bp={_ms(_t_fwd,_t_bp):4.1f} "
                      f"obs_h2d={_ms(_t_bp,_t_obs):5.1f} "
                      f"loss={_ms(_t_obs,_t_loss):4.1f} "
                      f"bwd={_ms(_t_loss,_t_bwd):6.1f} "
                      f"item={_ms(_t_bwd,_t_item):4.1f} "
                      f"snap_d2h={_ms(_t_item,_t_snap):5.1f} "
                      f"reduce={_ms(_t_snap,_t_reduce):4.1f} "
                      f"step={_ms(_t_reduce,_t_step):5.1f} "
                      f"TOTAL={_ms(_t0,_t_step):6.1f}ms")
        else:
            # ---- (B) two-pass: solver phase against a leaf, then net phase ----
            optimizer.zero_grad()
            acc_loss_local = 0.0
            # Optional SYNCED per-phase profile (SWEEP_TASKS_TPROF=1). The
            # two-pass timing the QC line prints is NOT cuda-synced, so the
            # solver-adjoint async tail bleeds into reparam_bwd. This block
            # measures real wall-time per phase.
            import time as _tm
            _TPROF = os.environ.get("SWEEP_TASKS_TPROF") == "1"
            def _sync():
                if _TPROF and torch.cuda.is_available(): torch.cuda.synchronize()
            _pf = {"render": 0.0, "fwd": 0.0, "loss": 0.0, "bwd": 0.0, "reparam": 0.0}
            if _TPROF: _sync(); _t_a = _tm.perf_counter()

            # Render the full-grid base velocity once, detached. The leaf is
            # the autograd boundary; solver backward accumulates onto leaf.grad.
            with torch.no_grad():
                base_leaf = reparam_net().detach().clone()
            base_leaf = base_leaf.requires_grad_(True)
            if _TPROF: _sync(); _pf["render"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()

            # Per-chunk forward + backward (autograd graph from solver to leaf).
            # ``_chunk_inputs`` already handles local-window slicing if enabled;
            # we just substitute the leaf for the network output and reuse the
            # window logic.
            def _two_pass_chunk_inputs(chunk_idx, leaf):
                chunk_src = sources[chunk_idx]
                chunk_rec = receivers[chunk_idx]
                if local_window_ctx is None:
                    models = [leaf]
                    return models, solver, chunk_src, chunk_rec
                win_spec = local_window_ctx["spec"]
                full_shape = local_window_ctx["shape"]
                dh = local_window_ctx["dh"]
                window = _compute_local_window(
                    chunk_src, chunk_rec, full_shape, dh, win_spec,
                )
                if len(window) == 4:
                    z0, z1, x0, x1 = window
                    local_shape = (z1 - z0, x1 - x0)
                    rebase_kwargs = {"z0": z0, "x0": x0}
                    leaf_slice = (slice(z0, z1), slice(x0, x1))
                else:
                    z0, z1, y0, y1, x0, x1 = window
                    local_shape = (z1 - z0, y1 - y0, x1 - x0)
                    rebase_kwargs = {"z0": z0, "x0": x0, "y0": y0}
                    leaf_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                cache = local_window_ctx["solver_cache"]
                cache_key = local_shape
                if cache_key not in cache:
                    # See note above _chunk_inputs: bound the cache to 4
                    # entries (LRU) so varying batches don't accumulate
                    # solvers indefinitely.
                    if len(cache) >= 4:
                        oldest = next(iter(cache))
                        del cache[oldest]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    cache[cache_key] = _build_solver(
                        spec.physics, spec.backend, local_shape,
                        dh, local_window_ctx["dt"], local_window_ctx["nt"], dev,
                    )
                else:
                    cache[cache_key] = cache.pop(cache_key)  # touch (LRU)
                local_solver = cache[cache_key]
                local_src, local_rec = _rebase_geometry_to_window(chunk_src, chunk_rec, **rebase_kwargs)
                # View into leaf — gradient scatters back to leaf.grad on backward.
                models = [leaf[leaf_slice].contiguous()]
                return models, local_solver, local_src, local_rec

            sill_sum = None
            rill_sum = None
            for chunk_idx_in_iter, chunk in enumerate(chunks):
                models, chunk_solver, chunk_src, chunk_rec = _two_pass_chunk_inputs(chunk, base_leaf)
                if _TPROF: _sync(); _t_a = _tm.perf_counter()
                syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                if _TPROF: _sync(); _pf["fwd"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()
                if syn_bandpass is not None and stage_dt is not None:
                    syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                              stage_dt, order=syn_bandpass.order)
                obs_chunk = obs[chunk].to(dev)
                loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev))
                if _TPROF: _sync(); _pf["loss"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()
                (loss_t / global_norm).backward()
                if _TPROF: _sync(); _pf["bwd"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()
                acc_loss_local += float(loss_t.detach().cpu())
                if illum_on:
                    sill_sum, rill_sum = _accumulate_illumination(
                        chunk_solver, sill_sum, rill_sum,
                    )
                # Snapshot last chunk for QC reuse (see single-step branch).
                if dist_info.is_root and chunk_idx_in_iter == len(chunks) - 1:
                    self._last_qc_snapshot = {
                        "chunk_indices": np.asarray(chunk, dtype=np.int64).copy(),
                        "syn": syn.detach().to("cpu"),
                        "obs": obs_chunk.detach().to("cpu"),
                    }
                    self._maybe_dump_loss_inputs(
                        obs_chunk, syn, chunk, dist_info, syn_bandpass, stage_dt,
                        state=state_for_dump,
                    )

            # leaf.grad now holds the full-grid FWI gradient ∂L/∂v.
            # Push it through the network in a second pass.
            v_grad = base_leaf.grad
            # Illumination precondition the leaf gradient BEFORE the
            # network backward so the network sees the preconditioned
            # velocity gradient. The all-reduce of the leaf gradient
            # happens implicitly via the network backward (each rank
            # computes its own contribution and gradients on net params
            # are all_reduced after).
            if illum_on and v_grad is not None:
                if sill_sum is not None and rill_sum is not None:
                    _dist.all_reduce_sum_inplace(sill_sum, dist_info)
                    _dist.all_reduce_sum_inplace(rill_sum, dist_info)
                _apply_illumination_precond(
                    v_grad, sill_sum, rill_sum,
                    eps=illum_spec.epsilon, exponent=illum_spec.exponent,
                )
            if v_grad is None:
                # Nothing actually backpropped (e.g. all chunks were empty);
                # nothing to do.
                pass
            elif reparam_mode == "two_pass_full":
                # One full-graph backward through the network with v_grad as
                # the gradient at the (re-rendered) output.
                rendered = reparam_net()
                # Make sure shapes agree (they always do — net renders to
                # base_velocity.shape and leaf came from net()).
                rendered.backward(v_grad)
            elif reparam_mode == "two_pass_chunked":
                # Chunked re-render — memory-conscious for large grids.
                reparam_net.backward_velocity_gradient(
                    v_grad, chunk_rows=int(spec.reparam.backward_chunk_rows),
                )
            else:
                raise ValueError(f"unknown reparam backward_mode {reparam_mode!r}")

            if _TPROF: _sync(); _pf["reparam"] += _tm.perf_counter() - _t_a
            if _TPROF and dist_info.is_root:
                _tot = sum(_pf.values()) * 1e3
                print(f"[TPROF-B synced] render={_pf['render']*1e3:6.0f} solver_fwd={_pf['fwd']*1e3:6.0f} "
                      f"loss={_pf['loss']*1e3:5.0f} solver_bwd_adjoint={_pf['bwd']*1e3:6.0f} "
                      f"reparam={_pf['reparam']*1e3:6.0f}  sum={_tot:6.0f} ms", flush=True)

            _dist.all_reduce_grad_sum(
                [p for p in reparam_net.parameters() if p.grad is not None],
                dist_info,
            )
            optimizer.step()
            # Two-pass leaks references to the leaf's grad tensor across
            # Adam steps unless we explicitly drop the leaf. Without this,
            # GPU memory grows ~1 GB / step. ``empty_cache`` is cheap and
            # gives the allocator a chance to defragment between steps.
            del base_leaf
            if v_grad is not None:
                del v_grad
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        acc_loss_global = _dist.all_reduce_scalar_sum(acc_loss_local, dist_info)
        return acc_loss_global / global_norm if global_norm else 0.0

    # -- lsrtm -------------------------------------------------------------

    # ------------------------------------------------------------------
    # OBN 3-D CRG-plan FWI (source-encoded supershot path)
    # ------------------------------------------------------------------

    def _run_fwi_multisource(self, spec: FWISpec, task_dir: Path):
        """3-D OBN FWI driver — unified plan + UTM rotation + source encoding.

        Implements the production OBN 1-GPU path on the unified plan stack:
        every iteration draws a shared-shots batch from a
        ``grouping='crg'`` :class:`sweep_io.seismic_plan.SeismicPlan`,
        encodes it as a ±1-signed supershot, runs one ``Acoustic3D``
        forward + adjoint with ``source_encoding=True``, and steps the
        optimizer. Per-shot multi-GPU CRG mode (without encoding) is not
        yet wired and will raise a clear error.

        Setup deviations from :meth:`_run_fwi` (the static-obs CSG path):

        * Geometry comes from :class:`FromPlanGeometry` configured with
          ``rotation_metadata`` + (optionally) ``dh_xyz_m`` /
          ``grid_origin_xyz_m`` / ``auto_origin_pad_cells``. UTM source /
          shot positions get rotated via
          :class:`sweep_io.geometry.RotatedFrame` into the model frame,
          then quantised to grid indices.
        * Obs is streamed per iter via
          :class:`sweep_io.seismic_plan.PlanReader.read_rows` (one SEG-Y
          read pass per iter). No giant ``(nshots, nrec, nt)`` tensor
          materialised at task start.
        * Source encoding flips ``signs ∈ {-1, +1}^B`` per iter; the
          wavelet becomes ``(1, B, nt)`` and obs becomes
          ``Σ_i sign_i · obs_i`` of shape ``(1, n_shared, nt)`` — one
          forward call per iter regardless of B.

        Activated via the ``obs.plan.sampling`` block — when that's set on
        a ``grouping='crg'`` plan the runner dispatches here; otherwise it
        runs the static-obs CSG loop in :meth:`_run_fwi`.
        """
        import torch

        from sweep_io.geometry import load_rotation_metadata
        from sweep_io.seismic_plan import (
            PlanReader,
            SeismicPlan,
            build_shotkey_to_nodes_index,
            precompute_group_unique_keys,
            sample_shared_shots_from_plan,
            sample_shared_shots_receiver_first,
        )
        from sweep_runner import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        if spec.obs.plan is None or spec.obs.plan.sampling is None:
            raise ValueError(
                "_run_fwi_multisource requires obs.plan.sampling to be set "
                "(PlanSamplingConfig — see ObsPlanConfig.sampling)."
            )
        sampling_cfg = spec.obs.plan.sampling
        if Path(spec.geometry.plan_path) != Path(spec.obs.plan.plan_path):
            raise ValueError(
                "geometry.plan_path and obs.plan.plan_path must point at "
                f"the same file; got {spec.geometry.plan_path!r} vs "
                f"{spec.obs.plan.plan_path!r}."
            )

        _apply_seed(spec.seed)
        dev = _resolve_device(spec.device)

        encoding_spec = spec.source_encoding
        encoding_on = bool(encoding_spec is not None and encoding_spec.enabled)
        # B.2 (2026-05-22): per-shot multi-GPU CRG path. When
        # ``source_encoding.enabled=false`` (or absent), the forward loop
        # below loops over the B OBN groups as a per-shot batch
        # (``solver(..., source_encoding=False)`` with wavelet shape
        # ``(B_local, 1, nt)``) instead of collapsing them into a ±1
        # supershot. Under torchrun, each rank gets a contiguous slice of
        # the B batch — ``B_local = ceil(B / world_size)`` shots per rank
        # — and the vp gradient is all-reduce'd after backward.
        if encoding_on and dist_info.is_distributed:
            raise NotImplementedError(
                "Encoded supershot FWI is 1-GPU by construction. Run with "
                "source_encoding.enabled=false to use the per-shot multi-"
                "GPU path, or run on a single GPU (no torchrun)."
            )
        if int(sampling_cfg.shared_shots_per_iter) == 0:
            raise ValueError(
                "obs.plan.sampling.shared_shots_per_iter > 0 is required "
                "for both the encoded supershot path and the per-shot path "
                "(every batch group needs a shared-receiver set)."
            )

        # Setup-phase timer: each major step prints its wall time so the
        # iter-loop profile isn't bottlenecked by setup costs we can't see.
        _t_setup0 = time.perf_counter()
        def _stage(label: str, t_start: float) -> float:
            now = time.perf_counter()
            print(f"[multisource] setup {label}: {now - t_start:.2f}s "
                  f"(cumulative {now - _t_setup0:.2f}s)")
            return now

        # --- 1) Load plan + rotation, apply min_coverage filter.
        _t = time.perf_counter()
        plan = SeismicPlan.load(spec.geometry.plan_path)
        _t = _stage(f"SeismicPlan.load (n_rows={plan.n_rows:,}, n_groups={plan.n_groups})", _t)
        if plan.grouping != "crg":
            raise ValueError(
                "_run_fwi_multisource requires a grouping='crg' SeismicPlan "
                f"(got {plan.grouping!r}). Use 'sweep-tasks build-plan "
                "--grouping crg --receiver-quantize-m ...' to build one."
            )
        if spec.geometry.rotation_metadata is None:
            raise ValueError(
                "Multisource FWI on a CRG plan requires "
                "geometry.rotation_metadata (UTM → model-frame transform). "
                "Set FromPlanGeometry.rotation_metadata to the dataset's "
                "rotation_metadata.json."
            )
        min_cov = int(sampling_cfg.min_coverage)
        if encoding_spec is not None:
            min_cov = max(min_cov, int(encoding_spec.min_coverage))
        # NOTE: we do NOT pre-filter the plan by min_coverage at this point.
        # `sample_shared_shots_from_plan` accepts min_coverage as a sampling
        # parameter and applies it as a cheap per-group-row-counts mask
        # each iter (microseconds). Pre-filtering via plan.filter_groups
        # rewrites the entire multi-GB plan even when only a few
        # groups are dropped — costs minutes of setup on a large survey
        # and saves zero solver compute.
        if min_cov > 0:
            n_below = int((plan.per_group_row_counts() < min_cov).sum())
            print(f"[multisource] min_coverage={min_cov}: deferred to sampler "
                  f"(would drop {n_below}/{plan.n_groups} groups; pre-filtering "
                  "the plan is too expensive on a multi-GB CRG layout).")
        frame = load_rotation_metadata(spec.geometry.rotation_metadata)
        _t = _stage("load_rotation_metadata", _t)

        # --- 2) Init model + shape + dh.
        init_models = _normalize_fwi_init_models(spec)
        if init_models[0].path is None:
            raise ValueError(
                "Multisource FWI requires init_model.path (a 3-D vp "
                "npy); ModelRef.constant is not supported on this path."
            )
        init_vp_np = np.load(init_models[0].path).astype(np.float32)
        if init_vp_np.ndim != 3:
            raise ValueError(
                f"CRG-plan FWI expects a 3-D init_model (nz, ny, nx); "
                f"got shape {init_vp_np.shape}"
            )
        dh_xyz = spec.geometry.dh_xyz_m
        if dh_xyz is None:
            dh = float(spec.grid.dh)
            dh_xyz = (dh, dh, dh)
        dz_m, dy_m, dx_m = float(dh_xyz[0]), float(dh_xyz[1]), float(dh_xyz[2])

        # --- 3) Pre-project UTM → model frame xy.
        # ONE rotation pass per array. Rotation is a numpy matmul on every
        # point — was previously called 2-3 times (auto-origin, initial
        # projection, post-filter re-projection); each call took tens of seconds
        # on a large plan, even Lustre-warm, summing to minutes of dead
        # setup time. Now we project once, reuse via fancy-indexing.
        # MOVED UP (was post-crop): the auto-origin is now computed
        # before the model_plan crop so the crop can be origin-aware
        # (see the fix below). Same data, same single rotation pass —
        # just relocated.
        plan_model_xy = frame.to_model(plan.row_source_xyz[:, :2])
        _t = _stage(f"frame.to_model on plan.row_source_xy ({plan.n_rows:,} pts)", _t)
        src_model_xy = frame.to_model(plan.group_xyz[:, :2])
        _t = _stage(f"frame.to_model on plan.group_xy ({plan.n_groups} pts)", _t)

        # --- 3b) Grid origin (auto from rotated bbox if not given). The
        # init_vp_np grid uses THIS same origin (it's built with the same
        # auto-pad rule); the model_plan crop below converts user-space
        # x/y/z windows in model meters → cell indices via this origin.
        origin = spec.geometry.grid_origin_xyz_m
        pad_z, pad_y, pad_x = (int(p) for p in spec.geometry.auto_origin_pad_cells)
        if origin is None:
            x_min = float(min(src_model_xy[:, 0].min(), plan_model_xy[:, 0].min()))
            y_min = float(min(src_model_xy[:, 1].min(), plan_model_xy[:, 1].min()))
            # z-origin: the model top IS the free surface (sea surface), which
            # by convention is the first layer (index 0 = 0 m datum). Do NOT
            # pad above it — there is nothing to model above the sea surface,
            # and a z-pad would shift the model_plan z-crop downward, silently
            # chopping the top of the water column and dropping the free
            # surface / receivers ~pad_z*dz below where they belong. Only the
            # lateral x/y axes get the PML buffer pad.
            origin = (
                0.0,
                y_min - pad_y * dy_m,
                x_min - pad_x * dx_m,
            )
            print(f"[multisource] auto grid_origin_xyz_m = "
                  f"({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}) "
                  f"(z=0 at sea surface; x/y padded by auto_origin_pad_cells)")
            _t = _stage("auto grid_origin (z=0 surface; x/y bbox.min - pad)", _t)
        init_origin_z = float(origin[0])
        init_origin_y = float(origin[1])
        init_origin_x = float(origin[2])

        # --- 2b) Optional ModelPlan: crop vp to inversion window. The
        # crop is ORIGIN-AWARE: ``x_lo = floor((x_window - origin_x) / dh)``
        # so the data slice corresponds to the requested model-meter
        # window. Earlier versions computed ``x_lo = floor(x_window / dh)``
        # (no origin shift) — with a negative auto-origin (the edge pad) this
        # shifted the actual grid coverage left by the pad,
        # silently chopping the OBN nodes on the right edge of every run.
        nz_pre, ny_pre, nx_pre = init_vp_np.shape
        z_lo, z_hi = 0, nz_pre
        y_lo, y_hi = 0, ny_pre
        x_lo, x_hi = 0, nx_pre
        crop_origin_offset = (0.0, 0.0, 0.0)
        if spec.model_plan is not None:
            mp = spec.model_plan
            z_lo = int(np.floor((mp.z_window_m[0] - init_origin_z) / dz_m)) if mp.z_window_m else 0
            z_hi = int(np.ceil((mp.z_window_m[1] - init_origin_z) / dz_m)) + 1 if mp.z_window_m else nz_pre
            y_lo = int(np.floor((mp.y_window_m[0] - init_origin_y) / dy_m)) if mp.y_window_m else 0
            y_hi = int(np.ceil((mp.y_window_m[1] - init_origin_y) / dy_m)) + 1 if mp.y_window_m else ny_pre
            x_lo = int(np.floor((mp.x_window_m[0] - init_origin_x) / dx_m)) if mp.x_window_m else 0
            x_hi = int(np.ceil((mp.x_window_m[1] - init_origin_x) / dx_m)) + 1 if mp.x_window_m else nx_pre
            z_lo = max(0, z_lo); z_hi = min(nz_pre, z_hi)
            y_lo = max(0, y_lo); y_hi = min(ny_pre, y_hi)
            x_lo = max(0, x_lo); x_hi = min(nx_pre, x_hi)
            if z_hi <= z_lo or y_hi <= y_lo or x_hi <= x_lo:
                raise ValueError(
                    f"model_plan window collapsed: "
                    f"z=[{z_lo},{z_hi}) y=[{y_lo},{y_hi}) x=[{x_lo},{x_hi})"
                )
            init_vp_np = init_vp_np[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].copy()
            crop_origin_offset = (z_lo * dz_m, y_lo * dy_m, x_lo * dx_m)
            print(f"[multisource] model_plan crop: vp shape "
                  f"({nz_pre},{ny_pre},{nx_pre}) -> {init_vp_np.shape}, "
                  f"crop_origin_offset += ({crop_origin_offset[0]:.1f}, "
                  f"{crop_origin_offset[1]:.1f}, {crop_origin_offset[2]:.1f}) m "
                  f"(model-x window covered: "
                  f"[{init_origin_x + crop_origin_offset[2]:.1f}, "
                  f"{init_origin_x + crop_origin_offset[2] + init_vp_np.shape[-1] * dx_m:.1f}) m)")
        shape = tuple(int(v) for v in init_vp_np.shape)
        nz, ny, nx = shape

        # Final cropped-frame origin (used by all grid-index projections).
        origin_z = init_origin_z + crop_origin_offset[0]
        origin_y = init_origin_y + crop_origin_offset[1]
        origin_x = init_origin_x + crop_origin_offset[2]

        # --- 4) Compute integer grid indices (no more rotations).
        plan_grid_x = np.rint((plan_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64)
        plan_grid_y = np.rint((plan_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64)
        plan_grid_z = np.rint(
            (plan.row_source_xyz[:, 2] - origin_z) / dz_m
        ).astype(np.int64)
        src_grid_x = np.rint((src_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64)
        src_grid_y = np.rint((src_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64)
        src_grid_z = np.rint(
            (plan.group_xyz[:, 2] - origin_z) / dz_m
        ).astype(np.int64)
        # Sweep's source/receiver coord convention is (x, y, z) along the
        # last axis (matches Acoustic3D's grid).
        plan_grid_xyz = np.stack([plan_grid_x, plan_grid_y, plan_grid_z], axis=-1)
        src_grid_xyz = np.stack([src_grid_x, src_grid_y, src_grid_z], axis=-1)
        _t = _stage("grid index projection (rint + stack)", _t)

        # When the grid is smaller than the rotated survey extent, drop
        # rows/groups whose source position falls outside.
        row_in = (
            (plan_grid_xyz[:, 0] >= 0) & (plan_grid_xyz[:, 0] < nx)
            & (plan_grid_xyz[:, 1] >= 0) & (plan_grid_xyz[:, 1] < ny)
            & (plan_grid_xyz[:, 2] >= 0) & (plan_grid_xyz[:, 2] < nz)
        )
        slot_in = (
            (src_grid_xyz[:, 0] >= 0) & (src_grid_xyz[:, 0] < nx)
            & (src_grid_xyz[:, 1] >= 0) & (src_grid_xyz[:, 1] < ny)
            & (src_grid_xyz[:, 2] >= 0) & (src_grid_xyz[:, 2] < nz)
        )
        _t = _stage(f"bounds check (row_in={row_in.sum()}/{row_in.size}, "
                    f"slot_in={slot_in.sum()}/{slot_in.size})", _t)

        # --- 4a) Setup-stage in-window filter (ALL modes). Drop ONCE, here,
        # every shot row whose receiver grid index is out of bounds AND every
        # row belonging to an out-of-grid source node, then drop nodes left
        # empty. Unified-plan equivalent of fwi_workflow Step-6b
        # ``filter_*_to_rotated_box``: the sampler (random OR receiver-first)
        # then only ever sees in-window shots/nodes, so no out-of-window shot
        # or trace can reach the solver — where the C kernel would silently
        # zero it (record_kernel range-guard), diluting the gradient with
        # zero-synthetic-vs-real-obs residuals. This REPLACES the old per-iter
        # ``row_in_keep``, which was computed, printed, and NEVER applied —
        # out-of-window receivers leaked straight through to the solver.
        if not (row_in.all() and slot_in.all()):
            # Combined row keep: in-window receiver AND its source node in-window.
            row_group = np.repeat(
                np.arange(plan.n_groups, dtype=np.int64),
                np.diff(plan.group_offsets),
            )
            row_keep = row_in & slot_in[row_group]
            n_oob_rows = int((~row_in).sum())
            n_oob_groups = int((~slot_in).sum())
            n_rows0, n_groups0 = int(plan.n_rows), int(plan.n_groups)
            plan = plan.filter_rows(row_keep).drop_empty_groups()
            # Re-derive every plan-indexed geometry array on the filtered plan
            # (group renumbering after drop_empty_groups makes the old arrays
            # stale). Same origin / dh / grid as steps 3-4 above.
            plan_model_xy = frame.to_model(plan.row_source_xyz[:, :2])
            src_model_xy = frame.to_model(plan.group_xyz[:, :2])
            plan_grid_xyz = np.stack([
                np.rint((plan_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64),
                np.rint((plan_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64),
                np.rint((plan.row_source_xyz[:, 2] - origin_z) / dz_m).astype(np.int64),
            ], axis=-1)
            src_grid_xyz = np.stack([
                np.rint((src_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64),
                np.rint((src_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64),
                np.rint((plan.group_xyz[:, 2] - origin_z) / dz_m).astype(np.int64),
            ], axis=-1)
            # Everything left is in-window by construction.
            row_in = np.ones(plan.n_rows, dtype=bool)
            slot_in = np.ones(plan.n_groups, dtype=bool)
            print(
                f"[multisource] setup in-window filter: dropped {n_oob_rows} "
                f"({n_oob_rows / max(n_rows0, 1):.1%}) OOB shot rows + "
                f"{n_oob_groups} OOB nodes -> plan {n_rows0}->{plan.n_rows} rows, "
                f"{n_groups0}->{plan.n_groups} nodes "
                f"(grid nz,ny,nx={nz},{ny},{nx})"
            )
            _t = _stage("setup in-window plan filter (filter_rows+drop_empty)", _t)
        eligible_groups = np.flatnonzero(slot_in).astype(np.int64)

        # --- 4b) QC: dump the receiver layout in BOTH UTM (raw) and model
        # frame (post-rotation, origin-shifted) so the user can confirm the
        # rotation_metadata.json gives the same survey layout as the
        # legacy pipeline. Side effect: writes
        # ``qc/receiver_layout_rotation.png`` + ``qc/receiver_layout.npz``.
        try:
            _dump_receiver_rotation_qc(
                task_dir=task_dir,
                recv_model_xy=src_model_xy,                # (n_groups, 2) rotated
                recv_utm_xy=plan.group_xyz[:, :2],         # (n_groups, 2) raw UTM
                recv_z=plan.group_xyz[:, 2],               # (n_groups,)  z (vertical, no rotation)
                grid_origin_xyz=(origin_x, origin_y, origin_z),
                grid_shape=(nx, ny, nz),
                dh_xyz=(dx_m, dy_m, dz_m),
            )
        except Exception as qerr:  # noqa: BLE001
            print(f"[multisource] receiver layout QC skipped: {qerr}")
        _t = _stage("receiver-layout QC dump", _t)

        # --- 5) Build solver, wavelet, vp tensor (+ optional reparam_net).
        effective_dt = float(spec.time.dt)
        effective_nt = int(spec.time.nt)
        _cfl_check(
            float(init_vp_np.max()), float(min(dx_m, dy_m, dz_m)), effective_dt,
        )
        solver = _build_solver(
            spec.physics, spec.backend, shape, dx_m, effective_dt, effective_nt, dev,
        )
        _t = _stage(f"_build_solver ({spec.physics.equation}, shape={shape})", _t)
        wavelet_np = _build_wavelet(
            spec.wavelet, spec.time,
            override_dt=effective_dt, override_nt=effective_nt,
        )
        wavelet_t = torch.as_tensor(wavelet_np, dtype=torch.float32, device=dev)
        # SIREN-pipeline wavelet has a zero-prepad (``source_delay_s``)
        # at the start of the wavelet array — main bang sits at sample
        # ``round(source_delay_s / dt)``. SIREN training also pre-padded
        # ``observed`` by the same amount, so to reuse the SIREN wavelet
        # as-is we mirror that frame: shift the prefetched obs forward
        # in time by the same number of samples each iter. Without this,
        # syn and obs are off by ``source_delay_s`` (a systematic
        # cycle-skip once the delay is a sizeable part of a period).
        obs_prepad_s = _get_wavelet_source_delay_s(spec.wavelet)
        obs_prepad_samples = (
            int(round(obs_prepad_s / float(effective_dt)))
            if obs_prepad_s > 0.0 else 0
        )
        if obs_prepad_samples > 0:
            print(
                f"[multisource] wavelet source_delay_s={obs_prepad_s*1000:.1f} ms "
                f"→ obs left-shifted by {obs_prepad_samples} samples "
                f"(@ effective_dt={effective_dt*1000:.2f} ms) each iter to "
                f"align syn/obs in the SIREN frame"
            )
        # Multi-stage frequency continuation. ``_normalise_stage_list``
        # returns the stage list (single-stage fallback when spec.stages is
        # None, using spec.bandpass + spec.epochs). The encoded path
        # bandpasses BOTH the source wavelet and the per-iter obs from the
        # PRISTINE wavelet at each stage entry ("target=wavelet" regime: syn
        # stays naturally bandlimited, no per-iter syn filter). The reparam
        # net / optimizer carry over across stages — one plan load + one
        # solver build for the whole sweep, no per-band resume.
        stage_list = _normalise_stage_list(spec)
        for _si, _st in enumerate(stage_list):
            _unsup = [_f for _f in ("dh_m", "dt_s", "nt", "wavelet")
                      if getattr(_st, _f, None) is not None]
            if abs(float(_st.lr_scale) - 1.0) > 1e-12:
                _unsup.append("lr_scale")
            if abs(float(getattr(_st, "inr_lr_scale", 1.0)) - 1.0) > 1e-12:
                _unsup.append("inr_lr_scale")
            if getattr(_st, "batch_size", None) is not None:
                _unsup.append("batch_size")
            if _unsup:
                raise NotImplementedError(
                    f"stage {_si}: the encoded OBN path supports only "
                    f"bandpass / epochs / optimizer_reset per stage; "
                    f"unsupported here: {_unsup}"
                )
        stage_epochs = [int(_st.epochs) for _st in stage_list]
        stage_starts = [int(s) for s in np.cumsum([0] + stage_epochs[:-1])]
        wavelet_orig = wavelet_t.detach().clone()  # pristine, pre-bandpass

        def _stage_bandpass(_st):
            return (_st.bandpass if _st.bandpass is not None
                    else getattr(spec, "bandpass", None))
        # bandpass_spec is (re)assigned per stage inside the loop; seed it
        # with stage 0 so the run-meta / wavelet build reflect band 1.
        bandpass_spec = _stage_bandpass(stage_list[0])
        if len(stage_list) > 1:
            _bands = [(b.lo_hz, b.hi_hz) if (b := _stage_bandpass(s)) is not None
                      else None for s in stage_list]
            print(f"[crg] multi-stage FWI: {len(stage_list)} stages, "
                  f"epochs={stage_epochs} (total {sum(stage_epochs)}), "
                  f"bandpass/stage={_bands}")
        vp_leaf = torch.from_numpy(init_vp_np).to(dev).requires_grad_(True)
        inv_by_name = {"vp": vp_leaf}
        reparam_net = None
        # Pre-build the SIREN water mask from a 2-D seabed-depth map
        # (preferred — geological data) before constructing the reparam
        # net. The full-extent npz is sliced with the SAME (y_lo:y_hi,
        # x_lo:x_hi) window that cropped init_vp_np above, so the
        # broadcast 3-D mask lines up voxel-for-voxel with vp_leaf.
        water_mask_override = None
        if (spec.reparam is not None
                and bool(getattr(spec.reparam, "mask_water_layer", False))
                and getattr(spec.reparam, "seabed_depth_path", None) is not None):
            from sweep_tasks.bathymetry import (
                load_seabed_depth_npz, water_mask_from_seabed_depth,
            )
            full_sd = load_seabed_depth_npz(spec.reparam.seabed_depth_path)
            if full_sd.ndim != 2:
                raise ValueError(
                    f"reparam.seabed_depth_path: expected 2-D (ny, nx) "
                    f"seabed_depth for 3-D FWI; got shape {full_sd.shape}"
                )
            if (full_sd.shape[0] != ny_pre or full_sd.shape[1] != nx_pre):
                raise ValueError(
                    f"reparam.seabed_depth_path shape {full_sd.shape} "
                    f"!= init_vp pre-crop (ny, nx) ({ny_pre}, {nx_pre}); "
                    "regenerate the bathymetry from the same init_vp."
                )
            cropped_sd = full_sd[y_lo:y_hi, x_lo:x_hi]
            mask_np = water_mask_from_seabed_depth(
                cropped_sd, nz=int(init_vp_np.shape[0]), dh_z_m=float(dz_m),
            )
            water_mask_override = torch.from_numpy(mask_np)
            print(
                f"[crg] reparam water_mask from {spec.reparam.seabed_depth_path}: "
                f"shape={tuple(mask_np.shape)}, "
                f"voxels={int(mask_np.sum()):,}/{mask_np.size:,} "
                f"({100*mask_np.sum()/mask_np.size:.2f}%), "
                f"depth range [{cropped_sd.min():.1f}, {cropped_sd.max():.1f}] m"
            )
        if spec.reparam is not None:
            reparam_net = _build_reparam_net(
                spec.reparam, vp_leaf, _effective_bound(spec.model_bounds, "vp"),
                water_mask_override=water_mask_override,
            )
            with torch.no_grad():
                rendered = reparam_net().detach()
            inv_by_name["vp"] = rendered
            n_params = sum(p.numel() for p in reparam_net.parameters())
            print(f"[crg] reparam=velocity_inr  net_params={n_params:,}  "
                  f"lr={spec.reparam.lr:.3e}")

        _t = _stage(f"build wavelet + bandpass + vp_leaf + reparam_net "
                    f"({'SIREN' if reparam_net is not None else 'no reparam'})", _t)

        # --- 6) Optimizer.
        if reparam_net is not None:
            optimizer = _build_reparam_optimizer(
                spec.optimizer, reparam_net.parameters(), float(spec.reparam.lr),
            )
        else:
            optimizer = _build_optimizer(spec.optimizer, inv_by_name, ["vp"])
        _t = _stage("optimizer build", _t)

        # --- 7) Plan reader + RNG for the per-iter sampler.
        # PlanReader serves SEG-Y trace bytes through MultiFileSEGYReader
        # under the hood. cache_all=False keeps memory bounded (about a GB
        # even on a large survey) at the cost of decoding bytes per call;
        # cache_all=True is fine for ≤ ~50 GB plans that fit in RAM.
        # Trace cache + coalesce gap: legacy OBN-supershot IO optimisations
        # that absorb cold-Lustre re-reads (hot traces in RAM) and merge
        # adjacent intra-file ``pread``s into one big read. Without these
        # wait_io dominated every iteration on a large survey; with them it is a fraction.
        _trace_stride_bytes = int(
            plan.trace_size_per_file[0]
        ) if plan.trace_size_per_file.size else (
            240 + int(plan.samples_per_trace) * 4
        )
        # 4× trace stride: coalesces any two traces within 4 strides of
        # each other in the same file (typical for shared-shot sampling
        # where the CDF of intra-file gaps clusters near the stride).
        _coalesce_gap = max(0, 4 * _trace_stride_bytes)
        reader = PlanReader(
            plan, mmap=True,
            cache_all=bool(spec.obs.plan.cache_all),
            trace_cache_bytes=int(getattr(sampling_cfg, "trace_cache_bytes", 0)),
            coalesce_gap=_coalesce_gap,
        )
        if bool(spec.obs.plan.cache_all):
            print(f"[multisource] PlanReader: cache_all=True "
                  f"(eagerly loaded {plan.n_rows} traces into RAM)")
        else:
            _tcb = int(getattr(sampling_cfg, "trace_cache_bytes", 0))
            if _tcb < 0:
                cache_desc = "per-trace LRU, unbounded"
            elif _tcb > 0:
                cache_desc = f"per-trace LRU, budget {_tcb / 1e9:.2f} GB"
            else:
                cache_desc = "DISABLED — wait_io may dominate on cold-Lustre"
            print(f"[multisource] PlanReader: trace_cache={cache_desc}, "
                  f"coalesce_gap={_coalesce_gap} bytes "
                  f"(= 4× stride {_trace_stride_bytes})")
        # ``sign_rng`` is only used by the encoded supershot path. In the
        # per-shot path it stays None and the forward loop simply skips
        # the sign sampling.
        if encoding_on:
            sign_rng = np.random.default_rng(encoding_spec.sign_seed)
        else:
            sign_rng = None
        sample_rng = np.random.default_rng(spec.seed)
        _t = _stage(f"PlanReader init (cache_all={bool(spec.obs.plan.cache_all)})", _t)

        # Pre-compute sorted-unique (sx, sy) keys for every CRG group once
        # at setup. The shared-shot sampler then skips ``np.unique`` per
        # group per iter (saves O(B × N log N) Python work × 1000 iters).
        # On a large survey: seconds ONCE here save seconds per iter in the sampler.
        group_unique_keys = precompute_group_unique_keys(plan)
        _t = _stage(
            f"precompute_group_unique_keys ({plan.n_groups} groups)", _t,
        )

        # Receiver-first reverse index (shot_key -> covering nodes), built
        # ONCE when sampling.receiver_first is on. It lets the per-iter
        # target shot sweep the whole survey (pick a target, then the nodes
        # that recorded it) instead of the random-group intersection
        # collapsing toward the survey centre on partial-coverage OBN data.
        shotkey_to_nodes = None
        shotkey_keys_arr = None
        if bool(getattr(sampling_cfg, "receiver_first", False)):
            # min_cov (= max(sampling, source_encoding) coverage) and
            # eligible_groups (= the in-grid slot_in nodes) must match the
            # random-sampler path exactly, else receiver-first surfaces nodes
            # the encoded path would reject (out-of-grid source / below the
            # effective coverage threshold).
            shotkey_to_nodes = build_shotkey_to_nodes_index(
                plan, min_coverage=int(min_cov),
                eligible_groups=eligible_groups,
                verbose=bool(dist_info.is_root),
            )
            shotkey_keys_arr = np.fromiter(
                shotkey_to_nodes.keys(), dtype=np.int64,
                count=len(shotkey_to_nodes),
            )
            _t = _stage(
                f"build_shotkey_to_nodes_index "
                f"({len(shotkey_to_nodes)} shot keys)", _t,
            )

        # Survey-wide sub-sample of every physical shot's (sx, sy) — used
        # as the faint background in the per-iter QC map. The full
        # set kills matplotlib; ~20k random hits give a clear footprint.
        # NOTE: the QC map is drawn in MODEL frame (post-rotation,
        # origin-shifted), not UTM — that's the frame the inversion
        # actually operates in. We capture the bg-row indices now and
        # do the rotation lookup once via the already-computed
        # plan_model_xy / src_model_xy arrays (no extra rotation pass).
        _SURVEY_MAP_MAX_SHOTS = 20_000
        if int(plan.n_rows) > _SURVEY_MAP_MAX_SHOTS:
            _rng_map = np.random.default_rng(0)
            _bg_idx = _rng_map.choice(
                int(plan.n_rows), size=_SURVEY_MAP_MAX_SHOTS, replace=False,
            )
        else:
            _bg_idx = np.arange(int(plan.n_rows), dtype=np.int64)
        all_shots_model_bg = plan_model_xy[_bg_idx].astype(np.float64)
        all_groups_model_bg = src_model_xy.astype(np.float64)
        # Inversion-active extent (model-frame meters) — what the solver
        # actually sees. ymin/xmin = grid origin (post-crop, post-pad);
        # max = origin + n*dh. Drawn as a dashed rectangle on the QC map
        # so the user immediately sees which picked OBN / used shots
        # land INSIDE the active inversion window.
        inversion_extent_xy_m = (
            float(origin_x), float(origin_x + nx * dx_m),
            float(origin_y), float(origin_y + ny * dy_m),
        )
        _t = _stage(
            f"survey-map background sample ({all_shots_model_bg.shape[0]} "
            f"shots / {all_groups_model_bg.shape[0]} OBN nodes; model frame; "
            f"grid {nx}×{ny} cells "
            f"= {nx*dx_m/1000:.1f}×{ny*dy_m/1000:.1f} km)", _t,
        )

        # Well-log QC positions: 6 pseudo-wells on a 3×2 (x × y) grid
        # spaced across the cropped inversion extent. (x_km, y_km) are
        # in MODEL frame meters; converted to grid (iy, ix) via the
        # current origin / dh. Wells outside the grid render as
        # "OUT OF GRID" stubs (the helper handles it cleanly).
        _well_x_m = np.linspace(
            float(origin_x) + 0.15 * nx * dx_m,
            float(origin_x) + 0.85 * nx * dx_m, 3,
        )
        _well_y_m = np.linspace(
            float(origin_y) + 0.25 * ny * dy_m,
            float(origin_y) + 0.75 * ny * dy_m, 2,
        )
        well_xy_m = np.array(
            [(x, y) for y in _well_y_m for x in _well_x_m],
            dtype=np.float64,
        )
        well_grid_idx = np.stack([
            np.rint((well_xy_m[:, 1] - origin_y) / dy_m).astype(np.int64),
            np.rint((well_xy_m[:, 0] - origin_x) / dx_m).astype(np.int64),
        ], axis=-1)
        well_labels = [
            f"x={xy[0]/1000:.1f} y={xy[1]/1000:.1f} km"
            for xy in well_xy_m
        ]
        _t = _stage(
            f"well-log QC wells ({len(well_labels)} pseudo-wells: "
            f"{', '.join(well_labels)})", _t,
        )

        # Cross-iter SEG-Y prefetcher. Each iter's batch sampling + SEG-Y
        # read runs in a background thread so the GPU can compute on iter
        # N while we fetch iter N+1's traces. Without this, random batches
        # past the OS pagecache working set blow up wait_io from ~0.5s to
        # tens of seconds per iter (a survey larger than the available RAM).
        #
        # Inside the loader we also parallelise the per-slot SEG-Y reads
        # across an inner thread pool (``num_workers`` from the config) —
        # the SEG-Y reader releases the GIL during mmap reads so threads
        # see real concurrency on NVMe storage.
        from concurrent.futures import ThreadPoolExecutor

        n_io_workers = max(1, int(sampling_cfg.num_workers))
        # Outer pool: cross-iter pre-load (1 worker is enough — we only
        # need one prefetched batch at a time).
        prefetch_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sharedshot-prefetch",
        )
        # Inner pool: per-group SEG-Y reads inside one iter's load.
        io_pool = (
            ThreadPoolExecutor(max_workers=n_io_workers,
                                thread_name_prefix="sharedshot-io")
            if n_io_workers > 1 else None
        )

        # Per-iter timing accumulator for the prefetch worker so we can
        # break down wait_io into (sample / read / resample / trim) — the
        # main loop's wait_io only tells us the OUTER prefetcher cost.
        # Each entry is one iter's dict; reset by main loop after read.
        prefetch_timings: list[dict] = []

        def _load_iter_payload(rng_state):
            """Sample shared-shots + read all per-group traces. Returns
            ``(batch, traces_per_group)`` where ``traces_per_group`` is a
            ``(B, n_shared, nt_segy)`` ``float32`` numpy array.

            ``rng_state`` seeds the sampler for determinism — the caller
            (the main loop) advances the master ``sample_rng`` state so
            re-runs with the same seed produce the same iter sequence.
            """
            tstats = {"sample": 0.0, "alloc": 0.0, "read": 0.0,
                      "resample": 0.0, "trim": 0.0}
            t_sample0 = time.perf_counter()
            local_rng = np.random.default_rng(rng_state)
            # Deterministic group ENUMERATION (SWEEP_ENUM_GROUPS=1): treat
            # ``rng_state`` as the group ordinal and process exactly that
            # ONE CRG (B=1) with its OWN shots. Iterating ordinals 0..N-1
            # over N=epochs dumps the init-model sum/mean of every CRG's
            # gradient — a true full-survey gradient, vs the random
            # shared-shot supershot draws.
            if os.environ.get("SWEEP_ENUM_GROUPS") == "1":
                _elig_all = (eligible_groups if eligible_groups is not None
                             else np.arange(int(plan.n_groups), dtype=np.int64))
                _gi = int(rng_state) % int(_elig_all.size)
                _elig_iter = _elig_all[_gi:_gi + 1]
                _bs_iter = 1
            else:
                _elig_iter = eligible_groups
                _bs_iter = int(spec.batchsize)
            if shotkey_to_nodes is not None and _bs_iter > 1:
                # Receiver-first: pick a target shot then the nodes that
                # recorded it, so the per-iter supershot sweeps the whole
                # survey. Falls back internally to the random sampler if no
                # target reaches batch_size within the retry budget.
                b = sample_shared_shots_receiver_first(
                    plan, local_rng,
                    batch_size=_bs_iter,
                    source_lines_per_group=int(sampling_cfg.source_lines_per_group),
                    max_traces_per_sourceline=int(sampling_cfg.max_traces_per_sourceline),
                    min_coverage=int(min_cov),
                    max_retries=int(sampling_cfg.receiver_first_max_retries),
                    eligible_groups=eligible_groups,
                    shotkey_to_nodes=shotkey_to_nodes,
                    shotkey_keys_arr=shotkey_keys_arr,
                    precomputed_group_unique_keys=group_unique_keys,
                )
            else:
                b = sample_shared_shots_from_plan(
                    plan, local_rng,
                    batch_size=_bs_iter,
                    source_lines_per_group=int(sampling_cfg.source_lines_per_group),
                    max_traces_per_sourceline=int(sampling_cfg.max_traces_per_sourceline),
                    # min_coverage + eligible_groups applied per-iter at sample
                    # time (instead of pre-filtering the plan via filter_groups
                    # / filter_rows) — see _run_fwi_multisource setup.
                    min_coverage=int(min_cov),
                    eligible_groups=_elig_iter,
                    precomputed_group_unique_keys=group_unique_keys,
                )
            tstats["sample"] = time.perf_counter() - t_sample0
            t_alloc0 = time.perf_counter()
            B = int(b.group_indices.size)
            n_shared = int(b.n_shared)
            t_np = np.empty(
                (B, n_shared, plan.samples_per_trace), dtype=np.float32,
            )
            tstats["alloc"] = time.perf_counter() - t_alloc0

            def _read_one(i):
                # b.rows_per_group[i] holds ABSOLUTE plan-row indices —
                # PlanReader.read_rows takes those directly.
                t_np[i] = reader.read_rows(b.rows_per_group[i])

            t_read0 = time.perf_counter()
            if io_pool is None:
                for i in range(B):
                    _read_one(i)
            else:
                list(io_pool.map(_read_one, range(B)))
            tstats["read"] = time.perf_counter() - t_read0

            t_resample0 = time.perf_counter()
            # Resample obs time axis on the prefetcher thread too so the
            # main loop's wait_io covers resample as well.
            if abs(plan.dt_s - effective_dt) > 1.0e-12:
                from sweep_preproc.resample import resample_time
                t_np = resample_time(t_np, plan.dt_s, effective_dt, axis=-1)
            tstats["resample"] = time.perf_counter() - t_resample0

            t_trim0 = time.perf_counter()
            # Trim / zero-pad to match effective_nt — unconditionally,
            # since SEG-Y trace length can differ from spec.time.nt even
            # when dt matches (field SEG-Y often stores a longer record than the
            # YAML asks for). Previously this trim was nested
            # inside the dt-mismatch branch and missed nt-only mismatches.
            cur_nt = t_np.shape[-1]
            if cur_nt > effective_nt:
                t_np = t_np[..., :effective_nt]
            elif cur_nt < effective_nt:
                pad = [(0, 0)] * t_np.ndim
                pad[-1] = (0, effective_nt - cur_nt)
                t_np = np.pad(t_np, pad)
            t_np = t_np.astype(np.float32, copy=False)
            tstats["trim"] = time.perf_counter() - t_trim0
            tstats["B"] = B
            tstats["n_shared"] = n_shared
            prefetch_timings.append(tstats)
            return b, t_np

        def _next_rng_state():
            """Draw + return a fresh seed for the next iter's sampler,
            keeping ``sample_rng`` as the single source of randomness."""
            return int(sample_rng.integers(0, 2**31 - 1))

        def _iter_seed(iter_idx):
            """Sampler arg for iter ``iter_idx``: the group ordinal in
            enumerate mode (SWEEP_ENUM_GROUPS=1), else a fresh random seed."""
            if os.environ.get("SWEEP_ENUM_GROUPS") == "1":
                return int(iter_idx)
            return _next_rng_state()

        # --- 7b) Optional priors: TVPrior + SeabedFreezeMask.
        tv_prior = None
        smooth_spec = getattr(spec, "smooth_regularization", None)
        if smooth_spec is not None and float(smooth_spec.weight) > 0.0:
            from sweep_nn import TVPrior

            tv_prior = TVPrior(
                order=smooth_spec.order,
                x_weight=float(smooth_spec.x_weight),
                y_weight=float(smooth_spec.y_weight),
                z_weight=float(smooth_spec.z_weight),
                velocity_scale_m_s=float(smooth_spec.velocity_scale_m_s),
            )
        seabed_mask = None
        freeze_spec = getattr(spec, "freeze_water_layer", None)
        if freeze_spec is not None and freeze_spec.enabled:
            from sweep_nn import SeabedFreezeMask

            sb = np.load(freeze_spec.seabed_depth_path)
            if isinstance(sb, np.lib.npyio.NpzFile):
                sb = sb["seabed_depth"]
            seabed_mask = SeabedFreezeMask(
                np.asarray(sb), dz_m=dz_m,
                buffer_cells=int(freeze_spec.buffer_cells),
            )

        # --- 8) Outputs / QC dirs (rank-0 / single-rank).
        out_dir = task_dir / "output"
        out_dir.mkdir(exist_ok=True)
        snapshots_dir = out_dir / "epochs"
        snapshots_dir.mkdir(exist_ok=True)
        qc_dir = task_dir / "qc"
        qc_enabled = spec.qc is not None and spec.qc.every_n_epochs > 0

        # --- 9) Per-iter loop (global epoch counter across all stages).
        total_epochs = int(sum(stage_epochs))
        losses: list[float] = []
        illum_spec = getattr(spec, "illumination_precondition", None)
        illum_on = bool(illum_spec is not None and illum_spec.enabled)
        if illum_on and getattr(spec.backend, "impl", "eager") == "eager":
            print("[illum] WARN: eager backend does not populate solver "
                  "illumination buffers; precondition is a no-op.")
        # perf/acoustic-bwd-skip-illum (3d6fe97): c-backend illumination is now
        # opt-in (default off). Re-enable on the solver when precond is active so
        # the backend populates source/receiver_illumination. No-op on old cores.
        if illum_on and getattr(spec.backend, "impl", "eager") == "c":
            try:
                solver.compute_illumination = True
            except Exception:
                pass
        loss_fn = lambda syn, obs: _compute_loss(syn, obs, spec.loss)
        def _csync():
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        _t = _stage("priors + output dirs + loss_fn ready (setup DONE)", _t)

        # Dump the resolved config + comprehensive runtime metadata at
        # task_dir root. Captures everything a future reader needs to
        # interpret this run: full pydantic-expanded YAML, host / GPU
        # / package versions / git SHAs, plus derived setup quantities
        # (shape, origin, n_groups, water_mask voxel count, obs prepad
        # samples, sampler config, ...). Best-effort — wrapped so any
        # I/O hiccup never blocks the run.
        try:
            _runtime_extras = {
                "device": str(dev),
                "epochs": int(total_epochs),
                "batchsize": int(spec.batchsize),
                "effective_dt_s": float(effective_dt),
                "effective_nt": int(effective_nt),
                "grid_shape_zyx": list(shape),
                "dh_xyz_m": [float(dz_m), float(dy_m), float(dx_m)],
                "origin_xyz_m": [
                    float(origin_z), float(origin_y), float(origin_x),
                ],
                "crop_origin_offset_m": list(crop_origin_offset),
                "init_vp_min_mean_max": [
                    float(init_vp_np.min()),
                    float(init_vp_np.mean()),
                    float(init_vp_np.max()),
                ],
                "plan_n_rows": int(plan.n_rows),
                "plan_n_groups": int(plan.n_groups),
                "plan_samples_per_trace": int(plan.samples_per_trace),
                "eligible_groups": int(eligible_groups.size),
                "min_coverage": int(min_cov),
                "n_picked_per_iter_B": int(spec.batchsize),
                "obs_prepad_s": float(obs_prepad_s),
                "obs_prepad_samples": int(obs_prepad_samples),
                "wavelet_kind": getattr(spec.wavelet, "kind", None),
                "loss_kind": getattr(spec.loss, "kind", None),
                "reparam_kind": (
                    getattr(spec.reparam, "kind", None)
                    if spec.reparam is not None else None
                ),
                "reparam_net_params": (
                    int(sum(p.numel() for p in reparam_net.parameters()))
                    if reparam_net is not None else None
                ),
                "reparam_lr": (
                    float(spec.reparam.lr) if spec.reparam is not None else None
                ),
                "reparam_vp_std": (
                    float(spec.reparam.vp_std) if spec.reparam is not None else None
                ),
                "reparam_vp_mean": (
                    float(spec.reparam.vp_mean) if spec.reparam is not None else None
                ),
                "reparam_mask_water_layer": (
                    bool(getattr(spec.reparam, "mask_water_layer", False))
                    if spec.reparam is not None else False
                ),
                "reparam_water_vp_m_s": (
                    float(getattr(spec.reparam, "water_vp_m_s", 1500.0))
                    if spec.reparam is not None else None
                ),
                "reparam_seabed_depth_path": (
                    str(getattr(spec.reparam, "seabed_depth_path", None))
                    if (spec.reparam is not None
                        and getattr(spec.reparam, "seabed_depth_path", None)
                        is not None)
                    else None
                ),
                "water_mask_voxels_true": (
                    int(water_mask_override.sum().item())
                    if water_mask_override is not None else None
                ),
                "water_mask_voxels_total": (
                    int(water_mask_override.numel())
                    if water_mask_override is not None else None
                ),
                "bandpass_hz": (
                    [float(bandpass_spec.lo_hz),
                     float(bandpass_spec.hi_hz),
                     int(bandpass_spec.order)]
                    if bandpass_spec is not None else None
                ),
                "sampling": {
                    "shared_shots_per_iter": int(sampling_cfg.shared_shots_per_iter),
                    "source_lines_per_group": int(sampling_cfg.source_lines_per_group),
                    "max_traces_per_sourceline": int(sampling_cfg.max_traces_per_sourceline),
                    "min_coverage": int(sampling_cfg.min_coverage),
                    "num_workers": int(sampling_cfg.num_workers),
                    "prefetch_factor": int(sampling_cfg.prefetch_factor),
                    "trace_cache_bytes": int(
                        getattr(sampling_cfg, "trace_cache_bytes", 0)
                    ),
                    "dedup_mode": str(getattr(sampling_cfg, "dedup_mode", "none")),
                },
                "qc_every_n_epochs": (
                    int(spec.qc.every_n_epochs) if spec.qc is not None else None
                ),
                "show_every": int(spec.show_every),
                "setup_wall_s": float(time.perf_counter() - _t_setup0),
            }
            _dump_run_metadata(spec, task_dir, extras=_runtime_extras)
            print(f"[run-meta] wrote {task_dir/'config_resolved.yaml'} "
                  f"+ {task_dir/'run_meta.json'}")
        except Exception as meta_err:  # noqa: BLE001
            print(f"[run-meta] dump skipped: {meta_err}")

        # Prime the prefetcher with iter 0's load so the first iter's
        # ``wait_io`` is also overlapped (with setup work above, ideally).
        next_future = prefetch_pool.submit(_load_iter_payload, _iter_seed(0))
        for epoch in range(total_epochs):
            # Stage entry: re-bandpass the PRISTINE wavelet + switch the obs
            # bandpass spec when crossing into a new frequency-continuation
            # stage. Net/optimizer state carries over (reset only on request).
            if epoch in stage_starts:
                _si = stage_starts.index(epoch)
                _st = stage_list[_si]
                bandpass_spec = _stage_bandpass(_st)
                wavelet_t = (_bandpass_torch_fft(
                    wavelet_orig, lo=float(bandpass_spec.lo_hz),
                    hi=float(bandpass_spec.hi_hz), dt=float(effective_dt),
                    order=int(bandpass_spec.order), axis=-1).detach()
                    if bandpass_spec is not None else wavelet_orig)
                if bool(getattr(_st, "optimizer_reset", False)) and _si > 0:
                    optimizer.state.clear()
                _bp_txt = (f"{bandpass_spec.lo_hz}-{bandpass_spec.hi_hz}Hz"
                           if bandpass_spec is not None else "none")
                print(f"[crg] === STAGE {_si + 1}/{len(stage_list)} "
                      f"epoch[{epoch}:{epoch + int(_st.epochs)}] "
                      f"bandpass={_bp_txt} ===", flush=True)
            t_iter = time.perf_counter()
            # Wait for the prefetched iter's payload (already in flight).
            t = time.perf_counter()
            batch, traces_per_group = next_future.result()
            t_wait = time.perf_counter() - t
            # Kick off iter N+1's load BEFORE doing iter N's GPU work so
            # the I/O overlaps with the solver fwd/bwd.
            if epoch + 1 < total_epochs:
                next_future = prefetch_pool.submit(
                    _load_iter_payload, _iter_seed(epoch + 1),
                )
            B = int(batch.group_indices.size)
            n_shared = int(batch.n_shared)
            t = time.perf_counter()
            obs_t = torch.as_tensor(traces_per_group, dtype=torch.float32, device=dev)
            # SIREN-frame alignment: shift obs forward in time by the
            # wavelet's ``source_delay_s`` (computed once at setup) so
            # the main wavelet bang at sample ``obs_prepad_samples`` in
            # syn lines up with the actual event in obs. ``torch.roll``
            # is cyclic, then we zero the wrapped tail to be safe
            # (legacy ``_run_fwi_3d_crg.py:1715-1719`` behaviour).
            if obs_prepad_samples > 0:
                obs_t = torch.roll(obs_t, shifts=obs_prepad_samples, dims=-1)
                obs_t[..., :obs_prepad_samples] = 0.0
            t_h2d = time.perf_counter() - t
            # The prefetcher absorbed the SEG-Y read AND the dt resample;
            # ``t_wait`` is the blocking portion (~0 when compute > io).
            t_sample = 0.0  # sampling now happens inside the prefetch worker
            t_io = t_wait + t_h2d
            t_resample = 0.0  # resample now happens inside the prefetch worker

            # Grid-indexed sources / receivers for this iter.
            sources_grid = src_grid_xyz[batch.group_indices]  # (B, 3)
            recv_grid = plan_grid_xyz[batch.rows_per_group[0]]  # (n_shared, 3)
            # Track the post-dedup plan-row indices in lock-step with
            # obs_t / recv_grid so downstream QC can join back to the
            # plan (file_id, sx_utm, etc.) for sorting / labelling.
            rows0_used = batch.rows_per_group[0]

            # Optional per-iter dedupe: collapse duplicate (gx,gy,gz) cells.
            # ``"first"`` keeps the first hit; ``"nearest"`` keeps the
            # receiver whose pre-quantisation model-xy was closest to the
            # cell center. Apply the same row mask to obs traces across
            # all groups so the encoded supershot stays receiver-consistent.
            dedup_mode = getattr(sampling_cfg, "dedup_mode", "none")
            if dedup_mode != "none" and n_shared > 1:
                if dedup_mode == "first":
                    _, keep_idx = np.unique(
                        recv_grid, axis=0, return_index=True,
                    )
                    keep_idx = np.sort(keep_idx)
                else:  # "nearest"
                    # Compute model-frame xy of each shared shot (already
                    # available via rows_per_group[0] + frame.to_model).
                    rows0 = batch.rows_per_group[0]
                    mxy = frame.to_model(plan.row_source_xyz[rows0, :2])
                    mz = plan.row_source_xyz[rows0, 2]
                    cell_center_x = recv_grid[:, 0] * dx_m + origin_x
                    cell_center_y = recv_grid[:, 1] * dy_m + origin_y
                    cell_center_z = recv_grid[:, 2] * dz_m + origin_z
                    d2 = ((mxy[:, 0] - cell_center_x) ** 2
                          + (mxy[:, 1] - cell_center_y) ** 2
                          + (mz - cell_center_z) ** 2)
                    # Bucket rows by their (gx,gy,gz) cell key.
                    key = (
                        (recv_grid[:, 2].astype(np.int64) * (1 << 22))
                        | (recv_grid[:, 1].astype(np.int64) * (1 << 11))
                        | recv_grid[:, 0].astype(np.int64)
                    )
                    order = np.argsort(key, kind="stable")
                    key_s = key[order]
                    starts = np.concatenate(
                        [[0], np.flatnonzero(np.diff(key_s) != 0) + 1]
                    )
                    ends = np.concatenate([starts[1:], [key.size]])
                    keep_idx = []
                    for s_i, e_i in zip(starts, ends):
                        local = order[s_i:e_i]
                        keep_idx.append(int(local[int(np.argmin(d2[local]))]))
                    keep_idx = np.sort(np.asarray(keep_idx, dtype=np.int64))
                if keep_idx.size < n_shared:
                    obs_t = obs_t[:, keep_idx, :].contiguous()
                    recv_grid = recv_grid[keep_idx]
                    rows0_used = rows0_used[keep_idx]
                    n_shared = int(keep_idx.size)
            if encoding_on:
                # ----- encoded supershot path (1 GPU, B=1) -----
                # source-encoding (sweep IO contract geophyai 24e91c9): sources
                # (1, nsrc, 3), receivers batch 1, wavelet (nsrc, nt) [per-source signed].
                sources_super = sources_grid[None, :, :].astype(np.int64)
                receivers_super = recv_grid[None, :, :].astype(np.int64)
                signs_np = sign_rng.choice([-1.0, 1.0], size=B).astype(np.float32)
                signs_t = torch.as_tensor(signs_np, device=dev)
                wavelet_super = (wavelet_t[None, :] * signs_t[:, None])              # (B, nt)
                obs_super = (obs_t * signs_t[:, None, None]).sum(dim=0, keepdim=True)
                # Bandpass the encoded obs supershot to match the (already
                # bandpassed) wavelet. Linearity: filtering each obs_i then
                # summing == filtering the sum (so equivalent to per-slot
                # pre-filtering, with one filter call per iter).
                if bandpass_spec is not None:
                    obs_super = _bandpass_torch_fft(
                        obs_super, lo=float(bandpass_spec.lo_hz),
                        hi=float(bandpass_spec.hi_hz),
                        dt=float(effective_dt),
                        order=int(bandpass_spec.order),
                        axis=-1,
                    )
                # local slice metadata for the encoded path: all on rank 0.
                local_start, local_end, dummy_slice = 0, B, False
            else:
                # ----- per-shot path (multi-GPU via DDP, B_local per rank) -----
                # Slice the B-batch contiguously across ranks; the rest of
                # the loop runs ONLY on the local slice. Gradients (and
                # illuminations) are all-reduce'd after backward so each
                # rank sees the full-batch update before optimizer.step.
                if dist_info.is_distributed:
                    rank = int(dist_info.rank)
                    world_size = int(dist_info.world_size)
                else:
                    rank, world_size = 0, 1
                per_rank = int(math.ceil(B / world_size))
                local_start = rank * per_rank
                local_end = min(local_start + per_rank, B)
                if local_end <= local_start:
                    # This rank drew an empty slice (B < world_size). Run
                    # ONE dummy shot with a near-zero loss multiplier so the
                    # backward + all_reduce participation stays consistent
                    # across ranks (collective ops must be called every iter
                    # by every rank). ``dummy_slice`` flag suppresses any
                    # contribution to the synced gradient.
                    local_start, local_end = 0, 1
                    dummy_slice = True
                else:
                    dummy_slice = False
                B_local = local_end - local_start
                sources_local = sources_grid[local_start:local_end].astype(np.int64)
                # sweep IO contract (geophyai 24e91c9): per-shot multi-shot uses
                # 2-D sources (nshots, ndim) + 2-D per-shot wavelet (nshots, nt);
                # 3-D (1, nsrc, ndim) is reserved for source-encoding mode.
                sources_super = sources_local                                      # (B_local, 3)
                receivers_super = np.broadcast_to(
                    recv_grid[None, :, :], (B_local, n_shared, 3),
                ).astype(np.int64).copy()                                          # (B_local, n_shared, 3)
                wavelet_super = (
                    wavelet_t[None, :].expand(B_local, -1).contiguous()
                )                                                                  # (B_local, nt)
                obs_super = obs_t[local_start:local_end].contiguous()              # (B_local, n_shared, nt)
                if bandpass_spec is not None:
                    obs_super = _bandpass_torch_fft(
                        obs_super, lo=float(bandpass_spec.lo_hz),
                        hi=float(bandpass_spec.hi_hz),
                        dt=float(effective_dt),
                        order=int(bandpass_spec.order),
                        axis=-1,
                    )

            optimizer.zero_grad()
            if reparam_net is None:
                models = [vp_leaf]
                v_leaf_for_illum = vp_leaf
            elif spec.reparam.backward_mode == "single_step":
                models = [reparam_net()]
                v_leaf_for_illum = None  # net params; illum precond no-op
            else:
                with torch.no_grad():
                    base_leaf = reparam_net().detach().clone()
                base_leaf = base_leaf.requires_grad_(True)
                models = [base_leaf]
                v_leaf_for_illum = base_leaf

            t = time.perf_counter()
            syn = solver(
                wavelet_super, sources_super, receivers_super,
                models=models, source_encoding=encoding_on,
            )
            _csync()
            t_fwd = time.perf_counter() - t
            # Adapt obs to syn's canonical layout (n, nt, nrec, 1).
            # ``obs_super`` from the CRG prefetcher is (n, nrec, nt), so we
            # always permute + unsqueeze. After geophyai 21041c5 both
            # backends emit syn as 4-D canonical, so this is unconditional.
            obs_match = obs_super.permute(0, 2, 1).unsqueeze(-1).contiguous()
            t = time.perf_counter()
            loss_t = loss_fn(syn, obs_match).sum()
            global_norm = float(syn.numel())
            # Dummy-slice ranks contribute zero loss / zero grad so the
            # all_reduce stays consistent across ranks; non-dummy ranks
            # use their full per-rank loss.
            loss_scale = 0.0 if (not encoding_on and dummy_slice) else 1.0
            (loss_t / global_norm * loss_scale).backward()
            _csync()
            t_bwd = time.perf_counter() - t

            # DDP gradient sync (per-shot path only). Each rank's
            # vp_leaf.grad holds B_local shots' partial contribution; sum
            # across ranks reproduces the full-batch gradient. The encoded
            # path stays 1-GPU and skips this entirely.
            if (not encoding_on
                    and dist_info.is_distributed
                    and v_leaf_for_illum is not None
                    and v_leaf_for_illum.grad is not None):
                import torch.distributed as _td
                _td.all_reduce(v_leaf_for_illum.grad, op=_td.ReduceOp.SUM)
                # Also sync illuminations stored on the solver for the
                # precond step below (per-rank partial sums → global sum).
                for _attr in ("source_illumination", "receiver_illumination"):
                    _ill = getattr(solver, _attr, None)
                    if isinstance(_ill, torch.Tensor):
                        _td.all_reduce(_ill, op=_td.ReduceOp.SUM)
            losses.append(float(loss_t.detach().cpu()) / global_norm)

            # --- Illumination precond (single_step + grid; or two_pass leaf).
            if illum_on:
                sill = getattr(solver, "source_illumination", None)
                rill = getattr(solver, "receiver_illumination", None)
                if reparam_net is None or v_leaf_for_illum is not None:
                    target_grad = v_leaf_for_illum.grad if v_leaf_for_illum is not None else None
                    # DUMP grad before/after illum precond. SWEEP_DUMP_GRAD=1
                    # writes the iter-0 raw/precond/illum once. Additionally
                    # SWEEP_DUMP_GRAD_NMEAN>0 accumulates the RAW grad (and
                    # S*R illum) over EVERY iter and writes a running MEAN.
                    # With the model frozen (lr=0) this mean is the full-batch
                    # gradient estimate: E over the per-iter shared-shot draws
                    # = the all-shots gradient the stochastic mini-batches
                    # sample (the supershot geometry forbids all groups at
                    # once, so we average instead).
                    _DG = (os.environ.get("SWEEP_DUMP_GRAD") == "1"
                           and dist_info.is_root and target_grad is not None)
                    _nmean = int(os.environ.get("SWEEP_DUMP_GRAD_NMEAN", "0") or 0)
                    if _DG:
                        import numpy as _np
                        _dd = os.environ.get("SWEEP_DUMP_DIR", "/tmp")
                        _graw = target_grad.detach().cpu().numpy()
                        _illsr = ((sill * rill).detach().cpu().numpy()
                                  if (sill is not None and rill is not None)
                                  else None)
                        if epoch == 0:
                            _np.save(_dd + "/grad_raw.npy", _graw)
                            if _illsr is not None:
                                _np.save(_dd + "/illum_sr.npy", _illsr)
                        if _nmean > 0:
                            _acc = getattr(self, "_graddump_acc", None)
                            if _acc is None:
                                _acc = {"g": _np.zeros_like(_graw),
                                        "i": (None if _illsr is None
                                              else _np.zeros_like(_illsr)),
                                        "n": 0}
                                self._graddump_acc = _acc
                            _acc["g"] += _graw
                            if _illsr is not None and _acc["i"] is not None:
                                _acc["i"] += _illsr
                            _acc["n"] += 1
                            _np.save(_dd + "/grad_raw_mean.npy",
                                     _acc["g"] / _acc["n"])
                            if _acc["i"] is not None:
                                _np.save(_dd + "/illum_sr_mean.npy",
                                         _acc["i"] / _acc["n"])
                            print(f"[graddump] mean over n={_acc['n']} iters",
                                  flush=True)
                    _apply_illumination_precond(
                        target_grad, sill, rill,
                        eps=illum_spec.epsilon, exponent=illum_spec.exponent,
                    )
                    if _DG and epoch == 0:
                        _np.save(_dd + "/grad_precond.npy",
                                 target_grad.detach().cpu().numpy())

            # --- Smooth regularization (TVPrior on the current vp/leaf).
            # SYNCED profile (SWEEP_TASKS_TPROF=1): time smooth-reg separately
            # and sync before the reparam timer so t_reparam_bwd is PURE reparam
            # (otherwise the smooth-reg backward's async tail bleeds into it).
            _tprof = os.environ.get("SWEEP_TASKS_TPROF") == "1"
            if _tprof and dev.type == "cuda": torch.cuda.synchronize(dev)
            _t_reg0 = time.perf_counter()
            if tv_prior is not None and v_leaf_for_illum is not None:
                reg_loss = float(smooth_spec.weight) * tv_prior(v_leaf_for_illum)
                # Accumulate into v_leaf_for_illum.grad (.backward adds).
                reg_loss.backward(retain_graph=False)
            if _tprof and dev.type == "cuda": torch.cuda.synchronize(dev)
            t_smoothreg = time.perf_counter() - _t_reg0

            # --- Seabed-freeze mask on the leaf gradient (post illum +
            # post smooth-reg so any mask zeros take precedence).
            if seabed_mask is not None and v_leaf_for_illum is not None:
                seabed_mask.apply_to(v_leaf_for_illum.grad)

            # --- Reparam two-pass full: push leaf grad through the net.
            # Drain any in-flight solver-adjoint async tail BEFORE starting the
            # reparam timer — otherwise that tail (which is large relative to a
            # fast/optimized reparam) bleeds into t_reparam_bwd and makes it
            # jitter wildly even though the reparam compute itself is steady.
            if _tprof and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t = time.perf_counter()
            if (reparam_net is not None
                    and spec.reparam.backward_mode == "two_pass_full"):
                v_grad = v_leaf_for_illum.grad
                if v_grad is not None:
                    rendered = reparam_net()
                    rendered.backward(v_grad)
            elif (reparam_net is not None
                  and spec.reparam.backward_mode == "two_pass_chunked"):
                v_grad = v_leaf_for_illum.grad
                if v_grad is not None:
                    reparam_net.backward_velocity_gradient(
                        v_grad, chunk_rows=int(spec.reparam.backward_chunk_rows),
                    )
            _csync()
            t_reparam_bwd = time.perf_counter() - t

            t = time.perf_counter()
            optimizer.step()
            _csync()
            t_opt = time.perf_counter() - t
            if reparam_net is None:
                bound = _effective_bound(spec.model_bounds, "vp")
                if bound is not None:
                    vp_leaf.data.clamp_(min=bound.min, max=bound.max)
            else:
                with torch.no_grad():
                    inv_by_name["vp"] = reparam_net().detach()

            iter_s = time.perf_counter() - t_iter
            cache_str = ""
            if hasattr(reader, "trace_cache_stats"):
                cs = reader.trace_cache_stats
                if cs is not None:
                    total = cs["hits"] + cs["misses"]
                    hit_rate = cs["hits"] / total if total > 0 else 0.0
                    cache_str = (f"  cache:[hit_rate={hit_rate:.1%} "
                                 f"entries={cs['entries']:,} "
                                 f"bytes={cs['bytes']/1e9:.1f}GB]")
            # Prefetch-worker per-stage breakdown for THIS iter.
            # prefetch_timings[-1] = the iter we just consumed (FIFO).
            pf_str = ""
            if prefetch_timings:
                pf = prefetch_timings[-2] if len(prefetch_timings) >= 2 else prefetch_timings[-1]
                # NOTE: the timing belongs to the iter we just finished
                # waiting on (= iter N); the next-submitted future is iter
                # N+1. With prime_pool[0] = iter 0, the pool of timings
                # contains [iter 0, iter 1, ...] in order; len at this
                # point equals number of FINISHED prefetch loads.
                pf_str = (f"  prefetch:[sample={pf['sample']:.2f} "
                          f"alloc={pf['alloc']:.2f} read={pf['read']:.2f} "
                          f"resample={pf['resample']:.2f} trim={pf['trim']:.2f}]")
            print(f"[multisource] epoch {epoch:04d} loss={losses[-1]:.6e} "
                  f"B={B} n_shared={n_shared} iter_s={iter_s:.2f}  "
                  f"[wait_io={t_wait:.2f} h2d={t_h2d:.2f} resample={t_resample:.2f} "
                  f"fwd={t_fwd:.2f} bwd={t_bwd:.2f} smoothreg={t_smoothreg:.2f} "
                  f"reparam_bwd={t_reparam_bwd:.2f} "
                  f"opt={t_opt:.2f}]{cache_str}{pf_str}")

            # --- Snapshots + QC.
            snapshot_now = (epoch % spec.show_every == 0
                            or epoch == total_epochs - 1)
            if snapshot_now:
                vp_now = inv_by_name["vp"]
                np.save(
                    snapshots_dir / f"vp_epoch_{epoch:04d}.npy",
                    vp_now.detach().cpu().numpy(),
                )
            if qc_enabled and (epoch % spec.qc.every_n_epochs == 0
                               or epoch == total_epochs - 1):
                state_for_qc = {
                    "inv_by_name": inv_by_name,
                    "dh": float(dx_m),
                    "dh_xyz": (float(dz_m), float(dy_m), float(dx_m)),
                    "dt": float(effective_dt),
                    "qc_initial_vp": torch.from_numpy(init_vp_np),
                }
                self._run_epoch_qc_safe(
                    spec=spec, state=state_for_qc, qc_dir=qc_dir,
                    epoch=epoch, dev=dev,
                )

                # Multisource supershot QC: obs/syn interleave gather,
                # amplitude spectrum, and the per-iter survey footprint
                # (picked OBN nodes + used physical shots in the FULL
                # acquisition context). Snapshot the CPU tensors from
                # this iter — obs_super / syn are still alive here. Use
                # rows0_used (post-dedup) so length matches obs/syn.
                # Gated on ``qc.supershot_panel`` (independent of the
                # legacy 2D ``shot_gather`` knob).
                # QC panel: encoded path always dumps from the single rank.
                # In per-shot DDP, only rank 0 dumps (and only its first
                # local source as a representative slice — the full-batch
                # acquisition footprint is still shown via
                # ``picked_group_model_xy`` from ``batch.group_indices``).
                _qc_dump_panel = (
                    bool(getattr(spec.qc, "supershot_panel", True))
                    and (encoding_on
                         or not dist_info.is_distributed
                         or int(dist_info.rank) == 0)
                )
                if _qc_dump_panel:
                    try:
                        from sweep_tasks.qc import save_supershot_qc_panel
                        # All coords in MODEL frame (post-rotation, in meters)
                        # so the QC map matches the frame the inversion
                        # runs in, and the dashed grid-extent rectangle
                        # (xmin/xmax, ymin/ymax in model meters) is
                        # meaningful on the same axes.
                        used_shot_model_xy = plan_model_xy[
                            rows0_used
                        ].astype(np.float64)
                        picked_group_model_xy = src_model_xy[
                            batch.group_indices
                        ].astype(np.float64)
                        f_hi = (float(bandpass_spec.hi_hz)
                                if bandpass_spec is not None else None)
                        # Source-line id per physical-shot column — lets
                        # the plotter sort gather columns by line so the
                        # eye sees along-line moveout cleanly (sampler
                        # natural order is (sx, sy)-key, which scatters
                        # lines randomly). Secondary key = absolute plan-
                        # row index: within a source line, rows are
                        # stored in SEG-Y file order (= along-sail-line
                        # order) regardless of survey orientation.
                        # Sorting by model_x as secondary collapses to
                        # chaos for sail lines that run along the y axis
                        # (common once a survey is rotated into the model frame).
                        used_sourceline_ids = plan.row_file_id[
                            rows0_used
                        ].astype(np.int64)
                        # In per-shot mode, take the first local source so
                        # the panel's obs/syn shapes match the encoded
                        # path's (1, n_shared, *) convention.
                        _obs_for_panel = obs_super.detach().cpu()
                        _syn_for_panel = syn.detach().cpu()
                        if not encoding_on:
                            _obs_for_panel = _obs_for_panel[:1]
                            _syn_for_panel = _syn_for_panel[:1]
                        save_supershot_qc_panel(
                            _obs_for_panel,
                            _syn_for_panel,
                            picked_group_utm_xy=picked_group_model_xy,
                            used_shot_utm_xy=used_shot_model_xy,
                            all_groups_utm_xy=all_groups_model_bg,
                            all_shots_utm_xy=all_shots_model_bg,
                            sourceline_ids=used_sourceline_ids,
                            within_sourceline_sort_key=np.asarray(
                                rows0_used, dtype=np.int64,
                            ),
                            frame_label="model",
                            inversion_extent_xy_m=inversion_extent_xy_m,
                            dt=float(effective_dt),
                            out_path=qc_dir / "supershot" / f"iter_{epoch:04d}.png",
                            epoch=epoch,
                            f_hi_hz=f_hi,
                        )
                    except Exception as ss_err:  # noqa: BLE001
                        print(f"[multisource] supershot QC skipped: {ss_err}")

                # Well-log QC: vp(z) at the 6 pseudo-wells set up at
                # session start. Quick sanity probe on the water layer,
                # seabed jump, and shallow gradient — flat 1500 m/s
                # across the water column is the strong expected signal.
                if bool(getattr(spec.qc, "well_logs", True)):
                    try:
                        from sweep_tasks.qc import save_vp_well_logs_png
                        vp_now_np = inv_by_name["vp"].detach().cpu().numpy()
                        save_vp_well_logs_png(
                            vp_now_np, init_vp_np,
                            well_grid_idx=well_grid_idx,
                            well_labels=well_labels,
                            dz_m=float(dz_m),
                            out_path=qc_dir / "well_logs" / f"iter_{epoch:04d}.png",
                            epoch=epoch,
                            vmin=(_effective_bound(spec.model_bounds, "vp").min
                                  if _effective_bound(spec.model_bounds, "vp") else None),
                            vmax=(_effective_bound(spec.model_bounds, "vp").max
                                  if _effective_bound(spec.model_bounds, "vp") else None),
                        )
                    except Exception as wl_err:  # noqa: BLE001
                        print(f"[multisource] well-log QC skipped: {wl_err}")

                # Gradient ortho-slice QC: dump the per-voxel velocity
                # gradient (post illumination precond + smooth-reg +
                # seabed mask, just before SIREN's reparam_bwd push)
                # next to the vp ortho slices. Uses sweep_image cmap +
                # symmetric percentile clipping so the sign structure is
                # visible. Snapshot ``v_leaf_for_illum.grad`` if it
                # exists (reparam two-pass path); fall back to
                # ``vp_leaf.grad`` for the direct-velocity path.
                # Gated on ``qc.gradient_png`` (shared with the 2D path).
                if bool(getattr(spec.qc, "gradient_png", False)):
                    try:
                        from sweep_tasks.qc import save_gradient_ortho_slices_png
                        grad_now = None
                        if (v_leaf_for_illum is not None
                                and v_leaf_for_illum.grad is not None):
                            grad_now = (
                                v_leaf_for_illum.grad.detach().cpu().numpy()
                            )
                        elif vp_leaf.grad is not None:
                            grad_now = vp_leaf.grad.detach().cpu().numpy()
                        if grad_now is not None:
                            save_gradient_ortho_slices_png(
                                grad_now,
                                dh=(float(dz_m), float(dy_m), float(dx_m)),
                                out_path=qc_dir / "gradient" / f"iter_{epoch:04d}.png",
                                epoch=epoch,
                            )
                    except Exception as g_err:  # noqa: BLE001
                        print(f"[multisource] gradient QC skipped: {g_err}")

                # Loss-curve QC: refresh ``qc/loss_curve.png`` (overwrites
                # in place each QC epoch) + dump the raw ``losses[]`` to
                # ``output/loss.npy`` so the user can monitor convergence
                # without waiting for the final-outputs block. Cheap —
                # one matplotlib call on a few-hundred-point list.
                # Gated on ``qc.loss_curve`` (shared with the 2D path).
                if bool(getattr(spec.qc, "loss_curve", True)):
                    try:
                        if losses:
                            _plot_loss_curve(
                                losses,
                                qc_dir / "loss_curve.png",
                                title=(
                                    f"OBN multisource FWI loss "
                                    f"(epoch {epoch}/{total_epochs - 1})"
                                ),
                            )
                            np.save(
                                out_dir / "loss.npy",
                                np.array(losses, dtype=np.float64),
                            )
                    except Exception as lc_err:  # noqa: BLE001
                        print(f"[multisource] loss-curve QC skipped: {lc_err}")

        # --- 10) Final outputs.
        prefetch_pool.shutdown(wait=False, cancel_futures=True)
        if io_pool is not None:
            io_pool.shutdown(wait=False, cancel_futures=True)
        reader.close()
        artifacts: list[Path] = []
        final_vp_path = out_dir / "inverted_vp.npy"
        np.save(final_vp_path, inv_by_name["vp"].detach().cpu().numpy())
        artifacts.append(final_vp_path)
        loss_path = out_dir / "loss.npy"
        np.save(loss_path, np.array(losses, dtype=np.float64))
        artifacts.append(loss_path)
        try:
            artifacts.append(_plot_loss_curve(losses, out_dir / "loss.png",
                                              title="OBN multisource FWI Loss"))
        except Exception as plot_err:  # noqa: BLE001
            print(f"[multisource] loss plot skipped: {plot_err}")

        summary = {
            "epochs": total_epochs,
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "n_virtual_sources": plan.n_groups,
            "batchsize": int(spec.batchsize),
            "encoding": True,
            "shape": list(shape),
        }
        return artifacts, summary

    # -- rtm ---------------------------------------------------------------

    def _run_rtm(self, spec: RTMSpec, task_dir: Path):
        """Post-FWI Reverse Time Migration: one pass over all shots, no iteration.

        Per batch: forward, bandpass syn/obs, backward → the gradient image
        IS the RTM cross-correlation under the chosen matching loss. The
        c-backend forward call (re)populates ``solver.source_illumination``
        and ``solver.receiver_illumination`` during backward, so the two
        illumination maps come for free; no separate ``solver.rtm`` call.
        Accumulate sums across all shots, then normalise by
        ``sqrt(S * R + eps)`` and save raw + normalised products.

        Legacy reference: :func:`fwi_workflow.imaging.rtm.run_sweep_imaging`
        (which DID call ``solver.rtm`` separately because the older sweep
        backend did not expose illumination via the regular forward path).
        """
        import torch

        from sweep_runner import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        _apply_seed(spec.seed)
        dev = _dist.resolve_dist_device(spec.device, dist_info.local_rank)
        equation_cls = _get_equation_class(spec.physics.equation)
        _validate_single_model(equation_cls, spec.velocity_model)

        # ---- 1) Shape, solver, wavelet, geometry, obs ------------------------
        # Mirror _run_fwi's data_plan.dt_target_s sync: when obs is resampled
        # via data_plan, the solver runs at the matching dt so syn/obs align.
        effective_dt = float(spec.time.dt)
        effective_nt = int(spec.time.nt)
        if spec.data_plan is not None and spec.data_plan.dt_target_s is not None:
            target_dt = float(spec.data_plan.dt_target_s)
            if abs(target_dt - effective_dt) > 1e-12:
                new_nt = int(round(effective_dt * effective_nt / target_dt))
                if dist_info.is_root:
                    print(f"[rtm] data_plan.dt_target_s={target_dt}s -> auto-sync "
                          f"solver: dt {effective_dt} -> {target_dt}, nt "
                          f"{effective_nt} -> {new_nt}")
                effective_dt = target_dt
                effective_nt = new_nt

        vm_ref = spec.velocity_model
        if spec.grid.shape is not None:
            shape = tuple(int(v) for v in spec.grid.shape)
        elif vm_ref.constant is not None:
            shape = tuple(int(v) for v in vm_ref.shape)
        else:
            shape = tuple(np.load(vm_ref.path, mmap_mode="r").shape)

        try:
            if vm_ref.path is not None:
                vmax_estimate = float(np.load(vm_ref.path, mmap_mode="r").max())
            elif vm_ref.constant is not None:
                vmax_estimate = float(vm_ref.constant)
            else:
                vmax_estimate = 0.0
            if vmax_estimate > 0 and dist_info.is_root:
                _cfl_check(vmax_estimate, float(spec.grid.dh), effective_dt)
        except FileNotFoundError:
            pass

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh,
            effective_dt, effective_nt, dev,
        )
        wavelet = _build_wavelet(
            spec.wavelet, spec.time,
            override_dt=effective_dt, override_nt=effective_nt,
        )

        segy_cache: dict = {}
        sources, receivers = _build_geometry_2d(
            spec.geometry, shape, dh=spec.grid.dh, segy_cache=segy_cache,
        )
        nshots = int(sources.shape[0])

        # Load velocity (as a fresh leaf each batch — we need grad on vp to
        # derive the FWI gradient image; the input itself is constant).
        vp_base = _load_model_tensor(vm_ref).to(dev)

        # ---- Obs source decision: lazy PlanReader vs eager fallback ----------
        # When ``spec.obs.plan`` is set we use a PlanReader directly so each
        # batch's traces are read on demand (cache_all=True keeps them hot in
        # RAM after the first pass; cache_all=False keeps memory bounded to
        # one batch — required for production OBN-scale obs). Per-batch resample +
        # bandpass move into the batch loop so the lazy path doesn't have to
        # materialise a full (nshots, nt, nrec, 1) tensor.
        #
        # All other obs sources (synthetic, npy, segy, segy_index) fall back
        # to the eager `_fwi_load_obs` route. They could be lazy-fied later
        # by adding similar reader objects; for now they stay one-shot.
        from sweep_io.seismic_plan import PlanReader, SeismicPlan

        plan_reader: "PlanReader | None" = None
        plan_obj: "SeismicPlan | None" = None
        obs_np = None
        if getattr(spec.obs, "plan", None) is not None:
            plan_obj = SeismicPlan.load(spec.obs.plan.plan_path)
            if plan_obj.grouping != "csg":
                raise NotImplementedError(
                    f"[rtm] obs.plan lazy reader currently supports "
                    f"grouping='csg' only (got {plan_obj.grouping!r}); "
                    "use the legacy CRG path for OBN datasets."
                )
            counts = plan_obj.per_group_row_counts()
            if not (counts == counts[0]).all():
                raise ValueError(
                    "[rtm] obs.plan with non-uniform receiver count per shot; "
                    "rebuild the plan with a filter that yields constant nrec."
                )
            if int(plan_obj.n_groups) != nshots:
                raise ValueError(
                    f"[rtm] obs.plan n_groups={plan_obj.n_groups} != "
                    f"geometry-derived nshots={nshots}. The plan must match "
                    "the geometry source (use geometry.kind=from_plan with "
                    "the same plan_path)."
                )
            plan_reader = PlanReader(plan_obj, cache_all=bool(spec.obs.plan.cache_all))
            if dist_info.is_root:
                cache_label = ("cache_all=True (pre-loaded to RAM)"
                               if spec.obs.plan.cache_all
                               else "cache_all=False (per-batch disk read)")
                print(f"[rtm] obs.plan lazy reader open: {plan_obj.n_groups} "
                      f"groups, {int(counts[0])} rec/group, dt={plan_obj.dt_s:.4g}s, "
                      f"{cache_label}")
        else:
            # Eager fallback for non-plan obs (synthetic / npy / segy / segy_index).
            obs = self._fwi_generate_obs(
                spec, equation_cls, solver, wavelet,
                sources, receivers, shape, dev, nshots,
                segy_cache=segy_cache,
            )
            sources, receivers, obs = _apply_data_plan_to_fwi(
                spec, sources, receivers, obs, spec.grid.dh,
                spec.time.dt, spec.time.nt, dev,
            )
            nshots = int(sources.shape[0])
            obs_np = (obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor)
                      else np.asarray(obs))
            if obs_np.ndim != 4:
                raise ValueError(
                    f"[rtm] eager obs.ndim={obs_np.ndim} not supported; "
                    "expected canonical 4-D (nshots, nt, nrec, 1)."
                )
            obs_np = np.ascontiguousarray(obs_np.astype(np.float32, copy=False))

        # ---- 2) pristine_dt fix (mirror _run_fwi exactly) --------------------
        # SEG-Y / plan-backed obs is read at the file's native dt; we resample
        # per-batch below. Don't let a mis-labelled pristine_dt corrupt the
        # resample (the same bug that bit the Viking benchmark).
        obs_native_dt = float(effective_dt)
        if spec.data_plan is None or spec.data_plan.dt_target_s is None:
            if plan_obj is not None:
                obs_native_dt = float(plan_obj.dt_s)
            elif segy_cache and (
                getattr(spec.obs, "segy", None) is not None
                or getattr(spec.obs, "segy_index", None) is not None
            ):
                for payload in segy_cache.values():
                    if isinstance(payload, dict) and "dt_s" in payload:
                        obs_native_dt = float(payload["dt_s"])
                        break
        if abs(obs_native_dt - float(effective_dt)) > 1e-12 and dist_info.is_root:
            print(f"[rtm] obs native dt={obs_native_dt}s differs from solver "
                  f"dt={effective_dt}s; per-batch resample will sync.")

        imaging = spec.imaging
        lo_hz = imaging.filter_lowcut_hz
        hi_hz = imaging.filter_highcut_hz
        bandpass_obs = (lo_hz is not None and hi_hz is not None)

        # ---- 2c) Optional wavelet bandpass (filter_target='wavelet') ---------
        # Mirrors the FWI ``stage.bandpass.target == 'wavelet'`` path: pre-
        # filter the source wavelet ONCE so the solver outputs naturally band-
        # limited syn, and we then skip the per-batch syn bandpass entirely.
        # Useful when the user wants to keep syn out of the autograd-filter
        # path (cheaper) or to mirror legacy FWI imaging that applied filter
        # on obs + wavelet rather than on syn.
        filter_target = getattr(imaging, "filter_target", "syn")
        if bandpass_obs and filter_target == "wavelet":
            pad_arg = imaging.filter_padtype
            if pad_arg == "none":
                pad_arg = None
            is_torch = hasattr(wavelet, "detach")
            if is_torch:
                wav_np = wavelet.detach().cpu().numpy()
                wav_dtype = wav_np.dtype
                wav_device = wavelet.device
            else:
                wav_np = np.asarray(wavelet)
                wav_dtype = wav_np.dtype
                wav_device = None
            wav_filt = _bandpass_cpu(
                wav_np, lo=float(lo_hz), hi=float(hi_hz), dt=effective_dt,
                order=int(imaging.filter_order), axis=-1, padtype=pad_arg,
            ).astype(wav_dtype, copy=False)
            wav_filt = np.ascontiguousarray(wav_filt)
            if is_torch:
                wavelet = torch.from_numpy(wav_filt).to(wav_device)
            else:
                wavelet = wav_filt
            if dist_info.is_root:
                peak = float(np.abs(wav_filt).max())
                print(f"[rtm] wavelet bandpass {lo_hz}-{hi_hz} Hz applied "
                      f"(peak now {peak:.3f}) -> syn naturally band-limited; "
                      "syn filter SKIPPED per-batch")
        elif bandpass_obs and filter_target not in ("syn", "wavelet"):
            raise ValueError(
                f"[rtm] imaging.filter_target must be 'syn' or 'wavelet'; "
                f"got {filter_target!r}"
            )

        # ---- 3) Distribute shot batches across DDP ranks ---------------------
        shots_per_batch = max(1, int(imaging.shots_per_batch))
        shot_indices_all = np.arange(nshots, dtype=np.int64)
        all_batches = [
            shot_indices_all[start:start + shots_per_batch]
            for start in range(0, nshots, shots_per_batch)
        ]
        # Round-robin batches across ranks (each rank processes its slice; the
        # final all_reduce SUMs the four accumulators across ranks).
        local_batches = (
            all_batches[dist_info.rank::dist_info.world_size]
            if dist_info.is_distributed else all_batches
        )

        if dist_info.is_root:
            print(f"[rtm] shape={shape} nshots={nshots} shots_per_batch={shots_per_batch} "
                  f"n_batches={len(all_batches)} world_size={dist_info.world_size} "
                  f"normalize_by_illumination={imaging.normalize_by_illumination}")

        # ---- 4) Per-batch local-window context (optional) --------------------
        local_window_ctx = None
        if (spec.local_model_window is not None
                and spec.local_model_window.enabled):
            local_window_ctx = {
                "spec": spec.local_model_window,
                "shape": tuple(shape),
                "dh": float(spec.grid.dh),
                "dt": float(effective_dt),
                "nt": int(effective_nt),
                "solver_cache": {},
            }

        # ---- 5) Output dirs + history CSV ------------------------------------
        out_dir = task_dir / "output"
        qc_dir = task_dir / "qc"
        per_shot_dir = out_dir / "per_shot"
        if dist_info.is_root:
            out_dir.mkdir(exist_ok=True)
            qc_dir.mkdir(parents=True, exist_ok=True)
            if imaging.save_per_shot:
                per_shot_dir.mkdir(parents=True, exist_ok=True)
        _dist.barrier(dist_info)

        # ---- 6) Accumulators -------------------------------------------------
        full_shape = tuple(int(v) for v in shape)
        gradient_sum = np.zeros(full_shape, dtype=np.float32)
        rtm_sum = np.zeros(full_shape, dtype=np.float32)
        source_illum_sum = np.zeros(full_shape, dtype=np.float32)
        receiver_illum_sum = np.zeros(full_shape, dtype=np.float32)
        # Per-shot illumination-normalised accumulators: each batch's
        # gradient / RTM image gets divided by its OWN illumination before
        # the sum, so deep / weakly-illuminated cells are not drowned by
        # high-illumination shallow regions. This is what the legacy Viking
        # imaging step ships as the canonical product
        # (illumination_normalized_rtm_per_shot.npy + the matching FWI
        # gradient variant). Mirrors fwi_workflow.imaging.rtm lines 563-576.
        gradient_per_shot_norm_sum = np.zeros(full_shape, dtype=np.float32)
        rtm_per_shot_norm_sum = np.zeros(full_shape, dtype=np.float32)
        history: list[dict] = []

        history_path = task_dir / "history.csv"
        history_file = None
        history_writer = None
        if dist_info.is_root:
            import csv as _csv
            history_file = history_path.open("w", newline="")
            history_writer = _csv.DictWriter(
                history_file,
                fieldnames=[
                    "batch_index", "shot_count", "shot_id_min", "shot_id_max",
                    "loss", "gradient_rms", "gradient_abs_max",
                    "rtm_abs_max", "local_z0", "local_z1",
                    "local_x0", "local_x1",
                ],
            )
            history_writer.writeheader()
            history_file.flush()

        # ---- 7) Main one-pass loop -------------------------------------------
        # Derive the per-batch loss spec. ``imaging.loss_kind`` overrides
        # ``spec.loss.kind`` (the imaging block is the authoritative source for
        # post-FWI products), but secondary knobs (huber_delta, trace_cosine_eps)
        # still come from ``spec.loss``.
        from sweep_tasks.schemas import LossSpec as _LossSpec
        loss_spec_for_grad = _LossSpec(
            kind=imaging.loss_kind,
            huber_delta=float(spec.loss.huber_delta),
            trace_cosine_demean=bool(imaging.trace_cosine_demean),
            trace_cosine_eps=float(spec.loss.trace_cosine_eps),
        )

        # Bandpass kwargs for the per-batch obs prep (shared by both lazy
        # and eager paths). Wavelet was already pre-filtered above when
        # filter_target == 'wavelet', so we still bandpass obs here either
        # way — only the syn bandpass inside _rtm_process_batch is gated.
        if bandpass_obs:
            _bp_pad = imaging.filter_padtype
            if _bp_pad == "none":
                _bp_pad = None
            bandpass_kwargs = dict(
                lo=float(lo_hz), hi=float(hi_hz),
                dt=float(effective_dt),
                order=int(imaging.filter_order),
                padtype=_bp_pad,
            )
        else:
            bandpass_kwargs = None

        for local_idx, batch_indices in enumerate(local_batches):
            global_batch_index = (
                (local_idx * dist_info.world_size + dist_info.rank)
                if dist_info.is_distributed else local_idx
            )
            try:
                obs_chunk = self._rtm_prep_obs_batch(
                    batch_indices=batch_indices,
                    plan_reader=plan_reader,
                    obs_np=obs_np,
                    obs_native_dt=obs_native_dt,
                    effective_dt=float(effective_dt),
                    effective_nt=int(effective_nt),
                    bandpass_kwargs=bandpass_kwargs,
                    dev=dev,
                )
                self._rtm_process_batch(
                    spec=spec,
                    solver=solver,
                    wavelet=wavelet,
                    sources=sources,
                    receivers=receivers,
                    obs_chunk=obs_chunk,
                    vp_base=vp_base,
                    full_shape=full_shape,
                    dev=dev,
                    batch_indices=batch_indices,
                    global_batch_index=global_batch_index,
                    loss_spec=loss_spec_for_grad,
                    imaging=imaging,
                    bandpass_obs=bandpass_obs,
                    filter_target=filter_target,
                    effective_dt=float(effective_dt),
                    local_window_ctx=local_window_ctx,
                    gradient_sum=gradient_sum,
                    rtm_sum=rtm_sum,
                    source_illum_sum=source_illum_sum,
                    receiver_illum_sum=receiver_illum_sum,
                    gradient_per_shot_norm_sum=gradient_per_shot_norm_sum,
                    rtm_per_shot_norm_sum=rtm_per_shot_norm_sum,
                    history=history,
                    history_writer=history_writer,
                    history_file=history_file,
                    per_shot_dir=per_shot_dir,
                    qc_dir=qc_dir,
                    dist_info=dist_info,
                )
            except Exception as berr:  # noqa: BLE001
                # One bad batch (e.g. boundary saving issue at extreme window
                # shape) should not nuke the whole RTM run. Log + continue.
                import traceback as _tb
                print(f"[rtm] batch {global_batch_index} failed: "
                      f"{type(berr).__name__}: {berr}")
                _tb.print_exc()

            # Periodic live QC re-render (rank 0 only — uses local accumulators
            # which differ between ranks, but the user just wants something to
            # eyeball; the final reduced result lives in qc/ after the loop).
            if (dist_info.is_root
                    and (local_idx + 1) % int(imaging.live_update_every_batches) == 0):
                try:
                    _save_rtm_qc_pngs(
                        gradient_sum=gradient_sum,
                        rtm_sum=rtm_sum,
                        source_illum_sum=source_illum_sum,
                        receiver_illum_sum=receiver_illum_sum,
                        gradient_per_shot_norm_sum=gradient_per_shot_norm_sum,
                        rtm_per_shot_norm_sum=rtm_per_shot_norm_sum,
                        qc_dir=qc_dir,
                        suffix="_latest",
                        eps=float(imaging.illumination_epsilon),
                        normalize=bool(imaging.normalize_by_illumination),
                    )
                except Exception as qerr:  # noqa: BLE001
                    print(f"[rtm] live QC skipped at batch {local_idx + 1}: {qerr}")

        if history_file is not None:
            history_file.close()

        # Close the lazy plan reader (closes SEG-Y handles + frees the cache
        # tensor when cache_all=True). Eager path has nothing to close.
        if plan_reader is not None:
            plan_reader.close()

        # ---- 8) All-reduce across ranks --------------------------------------
        if dist_info.is_distributed:
            for buf in (gradient_sum, rtm_sum,
                        source_illum_sum, receiver_illum_sum,
                        gradient_per_shot_norm_sum, rtm_per_shot_norm_sum):
                t = torch.from_numpy(buf)
                if dev.type == "cuda":
                    t = t.to(dev)
                _dist.all_reduce_sum_inplace(t, dist_info)
                buf[...] = t.detach().cpu().numpy()

        # ---- 9) Save outputs (rank 0 only) -----------------------------------
        artifacts: list[Path] = []
        if dist_info.is_root:
            np.save(out_dir / "fwi_gradient_image.npy", gradient_sum)
            np.save(out_dir / "rtm_image.npy", rtm_sum)
            np.save(out_dir / "source_illumination.npy", source_illum_sum)
            np.save(out_dir / "receiver_illumination.npy", receiver_illum_sum)
            eps = float(imaging.illumination_epsilon)
            denom = np.sqrt(np.maximum(source_illum_sum * receiver_illum_sum, 0.0) + eps)
            grad_norm = (gradient_sum / denom).astype(np.float32)
            rtm_norm = (rtm_sum / denom).astype(np.float32)
            np.save(out_dir / "fwi_gradient_image_normalised.npy", grad_norm)
            np.save(out_dir / "rtm_image_normalised.npy", rtm_norm)
            # Per-shot illumination-normalised products: each shot's image was
            # divided by its OWN sqrt(S*R+eps) BEFORE summation, so deep /
            # weakly-illuminated cells survive the stack. Legacy Viking's
            # canonical RTM display is this variant — strongly recommended
            # over the global-normalised one for plotting.
            np.save(out_dir / "fwi_gradient_image_per_shot_normalised.npy",
                    gradient_per_shot_norm_sum)
            np.save(out_dir / "rtm_image_per_shot_normalised.npy",
                    rtm_per_shot_norm_sum)
            np.savez_compressed(
                out_dir / "rtm_result.npz",
                fwi_gradient_image=gradient_sum,
                rtm_image=rtm_sum,
                source_illumination=source_illum_sum,
                receiver_illumination=receiver_illum_sum,
                fwi_gradient_image_normalised=grad_norm,
                rtm_image_normalised=rtm_norm,
                fwi_gradient_image_per_shot_normalised=gradient_per_shot_norm_sum,
                rtm_image_per_shot_normalised=rtm_per_shot_norm_sum,
                illumination_epsilon=np.float32(eps),
                dh=np.float32(spec.grid.dh),
                dt=np.float32(effective_dt),
                nt=np.int64(effective_nt),
                shape=np.asarray(full_shape, dtype=np.int64),
                n_shots=np.int64(nshots),
                n_batches=np.int64(len(all_batches)),
            )
            artifacts.extend([
                out_dir / "fwi_gradient_image.npy",
                out_dir / "rtm_image.npy",
                out_dir / "source_illumination.npy",
                out_dir / "receiver_illumination.npy",
                out_dir / "fwi_gradient_image_normalised.npy",
                out_dir / "rtm_image_normalised.npy",
                out_dir / "fwi_gradient_image_per_shot_normalised.npy",
                out_dir / "rtm_image_per_shot_normalised.npy",
                out_dir / "rtm_result.npz",
                history_path,
            ])
            try:
                qc_artefacts = _save_rtm_qc_pngs(
                    gradient_sum=gradient_sum,
                    rtm_sum=rtm_sum,
                    source_illum_sum=source_illum_sum,
                    receiver_illum_sum=receiver_illum_sum,
                    gradient_per_shot_norm_sum=gradient_per_shot_norm_sum,
                    rtm_per_shot_norm_sum=rtm_per_shot_norm_sum,
                    qc_dir=qc_dir,
                    suffix="",
                    eps=eps,
                    normalize=bool(imaging.normalize_by_illumination),
                )
                artifacts.extend(qc_artefacts)
            except Exception as qerr:  # noqa: BLE001
                print(f"[rtm] final QC PNG skipped: {qerr}")

            # ---- 9b) Optional post-filter (depth-tapered z-low-cut) ---------
            # Mirrors legacy ``07_filter_imaging.py``. Runs only when the user
            # opts in via ``imaging.post_filter``; outputs land next to the raw
            # npy files as ``<stem>_shallow_zlowcut.{npy,png}``. The same
            # algorithm is exposed standalone as ``sweep-tasks filter-image``
            # so users can iterate on params without re-running the RTM.
            post_filter = getattr(imaging, "post_filter", None)
            if post_filter is not None and bool(post_filter.enabled):
                from sweep_tasks.postproc.filter_image import filter_image_file

                all_targets = [
                    "fwi_gradient_image",
                    "rtm_image",
                    "fwi_gradient_image_normalised",
                    "rtm_image_normalised",
                    "fwi_gradient_image_per_shot_normalised",
                    "rtm_image_per_shot_normalised",
                ]
                if post_filter.targets == "all":
                    pf_targets = all_targets
                else:
                    pf_targets = [str(t) for t in post_filter.targets]
                dh = float(spec.grid.dh)
                nx_full = int(full_shape[1])
                x_max_m = float((nx_full - 1) * dh)
                for stem in pf_targets:
                    src = out_dir / f"{stem}.npy"
                    if not src.is_file():
                        print(f"[rtm] post_filter skip {stem}: {src} not found")
                        continue
                    try:
                        meta = filter_image_file(
                            src,
                            output_dir=out_dir,
                            output_name=f"{stem}_shallow_zlowcut",
                            dz_m=dh,
                            dx_m=dh,
                            wavelength_m=float(post_filter.wavelength_m),
                            depth_m=float(post_filter.depth_m),
                            taper_m=float(post_filter.taper_m),
                            clip_percentile=float(post_filter.clip_percentile),
                            display_scale=float(post_filter.display_scale),
                            x_origin_m=0.0,
                            z_origin_m=0.0,
                            x_max_m=x_max_m,
                            cmap=str(post_filter.cmap),
                            save_png=bool(post_filter.save_png),
                        )
                    except Exception as ferr:  # noqa: BLE001
                        print(f"[rtm] post_filter {stem} skipped: "
                              f"{type(ferr).__name__}: {ferr}")
                        continue
                    out_stem = out_dir / f"{stem}_shallow_zlowcut"
                    artifacts.append(out_stem.with_suffix(".npy"))
                    artifacts.append(Path(meta["removed"]))
                    artifacts.append(Path(meta["z_taper"]))
                    artifacts.append(out_dir / f"{stem}_shallow_zlowcut_metadata.json")
                    if bool(post_filter.save_png):
                        artifacts.append(out_stem.with_suffix(".png"))
                        artifacts.append(out_dir / f"{stem}_shallow_zlowcut_comparison.png")
                    print(f"[rtm] post_filter -> {out_stem.name}.npy "
                          f"(wavelength={post_filter.wavelength_m}m, "
                          f"depth={post_filter.depth_m}m, "
                          f"taper={post_filter.taper_m}m)")

        _dist.barrier(dist_info)

        summary = {
            "n_shots": int(nshots),
            "n_batches": int(len(all_batches)),
            "shots_per_batch": int(shots_per_batch),
            "image_abs_max": float(np.abs(rtm_sum).max()),
            "image_per_shot_norm_abs_max": float(np.abs(rtm_per_shot_norm_sum).max()),
            "gradient_abs_max": float(np.abs(gradient_sum).max()),
            "gradient_per_shot_norm_abs_max": float(np.abs(gradient_per_shot_norm_sum).max()),
            "source_illum_max": float(source_illum_sum.max()),
            "receiver_illum_max": float(receiver_illum_sum.max()),
            "world_size": dist_info.world_size,
            "normalize_by_illumination": bool(imaging.normalize_by_illumination),
        }
        return artifacts, summary

    # -- rtm per-batch obs prep (lazy / eager unified) --------------------

    def _rtm_prep_obs_batch(
        self,
        *,
        batch_indices,
        plan_reader,
        obs_np,
        obs_native_dt: float,
        effective_dt: float,
        effective_nt: int,
        bandpass_kwargs: dict | None,
        dev,
    ):
        """Materialise one shot batch's obs as a torch ``(B, nt, nrec, 1)`` tensor.

        Two routes selected by argument presence:

        * ``plan_reader`` set → **lazy**: read each shot's traces via
          :meth:`PlanReader.read_group`. With ``cache_all=True`` on the
          reader, traces are already in RAM (slice is cheap); with
          ``cache_all=False`` it's a per-batch SEG-Y read.
        * ``obs_np`` set → **eager**: slice the in-RAM
          ``(nshots, nt, nrec, 1)`` tensor produced by ``_fwi_load_obs``.

        Per-batch ops (resample → trim/pad → bandpass) are applied here so
        the lazy path doesn't have to materialise the full obs. Returns a
        contiguous ``torch.float32`` tensor on ``dev``.
        """
        import torch

        batch_arr = np.asarray(batch_indices, dtype=np.int64)
        if plan_reader is not None:
            # (B, nrec, nt_native) from PlanReader
            chunks = [plan_reader.read_group(int(s)) for s in batch_arr]
            raw = np.stack(chunks, axis=0).astype(np.float32, copy=False)
            # Canonical layout: (B, nt, nrec, 1)
            raw = np.ascontiguousarray(raw.transpose(0, 2, 1))[..., None]
        elif obs_np is not None:
            raw = np.take(obs_np, batch_arr, axis=0)
        else:
            raise RuntimeError(
                "[rtm] _rtm_prep_obs_batch: neither plan_reader nor obs_np "
                "provided. This is a runner-internal bug."
            )

        # time_axis on canonical 4-D is axis=1; resample/trim/pad/bandpass
        # helpers default to axis=-1, so pass time_axis explicitly.
        time_axis = 1
        if abs(obs_native_dt - effective_dt) > 1.0e-12:
            raw = _resample_obs_time(raw, obs_native_dt, effective_dt,
                                     time_axis=time_axis)
        raw = _trim_or_pad_time(raw, effective_nt, time_axis=time_axis)

        if bandpass_kwargs is not None:
            raw = _bandpass_cpu(raw, axis=time_axis, **bandpass_kwargs)

        raw = np.ascontiguousarray(raw.astype(np.float32, copy=False))
        return torch.as_tensor(raw, dtype=torch.float32, device=dev)

    # -- rtm per-batch kernel ---------------------------------------------

    def _rtm_process_batch(
        self,
        *,
        spec: "RTMSpec",
        solver,
        wavelet,
        sources,
        receivers,
        obs_chunk,
        vp_base,
        full_shape: tuple,
        dev,
        batch_indices,
        global_batch_index: int,
        loss_spec,
        imaging,
        bandpass_obs: bool,
        filter_target: str = "syn",
        effective_dt: float,
        local_window_ctx,
        gradient_sum,
        rtm_sum,
        source_illum_sum,
        receiver_illum_sum,
        gradient_per_shot_norm_sum=None,
        rtm_per_shot_norm_sum=None,
        history,
        history_writer,
        history_file,
        per_shot_dir: Path,
        qc_dir: Path,
        dist_info,
    ) -> None:
        """Run one shot batch: forward + backward populates both the
        gradient image (= RTM cross-correlation under the matching loss) and
        the solver-side ``source_illumination`` / ``receiver_illumination``
        attributes. No separate ``solver.rtm`` invocation is needed — the
        c-backend's regular adjoint pass owns illumination since the layout
        unification (sweep commit 21041c5).

        Mutates the four accumulator buffers (gradient_sum / rtm_sum /
        *_illum_sum) in-place, writes one history row, and (when
        ``imaging.save_per_shot``) dumps a per-shot npz. The ``rtm_sum`` and
        ``gradient_sum`` accumulators receive the same per-batch image; the
        duplication is kept so downstream consumers expecting the legacy
        filename split (rtm_image.npy vs fwi_gradient_image.npy) still get
        both outputs without surprises.
        """
        import torch

        # ---- Resolve solver + window slice for this batch --------------------
        ndim = len(full_shape)
        if local_window_ctx is None:
            chunk_solver = solver
            if ndim == 3:
                z0, z1 = 0, int(full_shape[0])
                y0, y1 = 0, int(full_shape[1])
                x0, x1 = 0, int(full_shape[2])
                vp_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            else:
                z0, z1 = 0, int(full_shape[0])
                x0, x1 = 0, int(full_shape[1])
                y0, y1 = 0, 0
                vp_slice = (slice(z0, z1), slice(x0, x1))
            chunk_src = sources[batch_indices]
            chunk_rec = receivers[batch_indices]
            chunk_vp_base = vp_base
        else:
            win_spec = local_window_ctx["spec"]
            dh = local_window_ctx["dh"]
            window = _compute_local_window(
                sources[batch_indices], receivers[batch_indices],
                full_shape, dh, win_spec,
            )
            if len(window) == 4:
                z0, z1, x0, x1 = window
                y0, y1 = 0, 0
                local_shape = (z1 - z0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0}
                vp_slice = (slice(z0, z1), slice(x0, x1))
            else:
                z0, z1, y0, y1, x0, x1 = window
                local_shape = (z1 - z0, y1 - y0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0, "y0": y0}
                vp_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            cache = local_window_ctx["solver_cache"]
            if local_shape not in cache:
                # Mirror _fwi_train_step's cache cap.
                if len(cache) >= 4:
                    oldest = next(iter(cache))
                    del cache[oldest]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                cache[local_shape] = _build_solver(
                    spec.physics, spec.backend, local_shape, dh,
                    local_window_ctx["dt"], local_window_ctx["nt"], dev,
                )
            else:
                cache[local_shape] = cache.pop(local_shape)  # LRU touch
            chunk_solver = cache[local_shape]
            chunk_src, chunk_rec = _rebase_geometry_to_window(
                sources[batch_indices], receivers[batch_indices], **rebase_kwargs,
            )
            chunk_vp_base = vp_base[vp_slice].contiguous()

        # ---- Forward + bandpass syn + loss + backward → FWI grad image ------
        # Fresh leaf each batch (we discard grads between batches; the RTM
        # accumulator on the caller side does the summation).
        local_vp = chunk_vp_base.detach().clone().requires_grad_(True)
        syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=[local_vp])

        # Syn bandpass is needed ONLY when ``filter_target='syn'``: with
        # ``filter_target='wavelet'`` the wavelet itself was pre-filtered in
        # the driver, so the solver output is already band-limited and an
        # extra autograd-aware filter on syn would just double-apply the
        # response.
        if bandpass_obs and filter_target == "syn":
            syn_filt = _bandpass_syn_torch(
                syn, float(imaging.filter_lowcut_hz),
                float(imaging.filter_highcut_hz),
                effective_dt, order=int(imaging.filter_order),
            )
        else:
            syn_filt = syn

        # ``obs_chunk`` arrives pre-prepared (sliced + resampled + bandpassed
        # + on-device) from the caller's ``_rtm_prep_obs_batch`` — that helper
        # serves both the lazy PlanReader and the eager full-RAM paths so we
        # do not have to materialise a production OBN-scale obs tensor in this loop.

        loss = _compute_loss(syn_filt, obs_chunk, loss_spec).sum()
        loss.backward()
        if local_vp.grad is None:
            raise RuntimeError(
                f"[rtm] batch {global_batch_index}: vp.grad is None — backward "
                "did not populate the gradient image."
            )
        gradient_local = (
            local_vp.grad.detach().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .cpu().numpy().astype(np.float32)
        )
        loss_value = float(loss.detach().cpu())

        # ---- Illumination snapshot ------------------------------------------
        # The c-backend forward call zeros + (re)populates
        # ``solver.source_illumination`` and ``solver.receiver_illumination``
        # during the backward pass.  No separate ``solver.rtm`` invocation is
        # required — the gradient we just took IS the imaging-condition
        # cross-correlation (gradient = RTM image under the chosen loss).
        sill_attr = getattr(chunk_solver, "source_illumination", None)
        rill_attr = getattr(chunk_solver, "receiver_illumination", None)
        if sill_attr is None or rill_attr is None:
            raise RuntimeError(
                "[rtm] backend did not expose solver.source_illumination / "
                "receiver_illumination after backward. Confirm backend.impl='c' "
                "(the eager autograd backend does not populate these)."
            )

        target_shape = tuple(int(v) for v in local_vp.shape)
        src_illum = _crop_padded_volume_to_model(
            sill_attr.detach(), target_shape, spec.physics,
        )
        rec_illum = _crop_padded_volume_to_model(
            rill_attr.detach(), target_shape, spec.physics,
        )

        # RTM image == FWI gradient under the chosen matching loss.  Keep the
        # two output arrays identical so downstream consumers that expect the
        # legacy filename split (rtm_image.npy vs fwi_gradient_image.npy) keep
        # working without surprises.
        rtm_image = gradient_local

        # ---- Scatter local batch products into the full-grid accumulators ----
        gradient_sum[vp_slice] += gradient_local
        rtm_sum[vp_slice] += rtm_image
        source_illum_sum[vp_slice] += src_illum
        receiver_illum_sum[vp_slice] += rec_illum

        # Per-shot illumination-normalised contribution: divide this batch's
        # image by its OWN sqrt(S*R+eps) BEFORE adding to the stack. Cleans
        # deep / weakly-illuminated cells that the global-normalised stack
        # buries under shallow energy (legacy
        # ``illumination_*_per_shot.npy``). Optional buffers — only filled
        # when the caller passes them.
        if gradient_per_shot_norm_sum is not None or rtm_per_shot_norm_sum is not None:
            eps_ps = float(imaging.illumination_epsilon)
            denom_local = np.sqrt(
                np.maximum(src_illum.astype(np.float64) * rec_illum.astype(np.float64), 0.0)
                + eps_ps
            ).astype(np.float32)
            if gradient_per_shot_norm_sum is not None:
                gradient_per_shot_norm_sum[vp_slice] += (
                    gradient_local / denom_local
                ).astype(np.float32)
            if rtm_per_shot_norm_sum is not None:
                rtm_per_shot_norm_sum[vp_slice] += (
                    rtm_image / denom_local
                ).astype(np.float32)

        # ---- History row -----------------------------------------------------
        grad_abs_max = float(np.abs(gradient_local).max())
        grad_rms = float(np.sqrt(np.mean(gradient_local.astype(np.float64) ** 2)))
        rtm_abs_max = float(np.abs(rtm_image).max())
        entry = {
            "batch_index": int(global_batch_index),
            "shot_count": int(np.asarray(batch_indices).size),
            "shot_id_min": int(np.min(batch_indices)),
            "shot_id_max": int(np.max(batch_indices)),
            "loss": loss_value,
            "gradient_rms": grad_rms,
            "gradient_abs_max": grad_abs_max,
            "rtm_abs_max": rtm_abs_max,
            "local_z0": int(z0), "local_z1": int(z1),
            "local_x0": int(x0), "local_x1": int(x1),
        }
        history.append(entry)
        if history_writer is not None and dist_info.is_root:
            history_writer.writerow(entry)
            if history_file is not None:
                history_file.flush()

        # ---- Optional per-shot npz dump -------------------------------------
        if imaging.save_per_shot and dist_info.is_root:
            for k, shot_id in enumerate(np.asarray(batch_indices)):
                np.savez_compressed(
                    per_shot_dir / f"shot_{int(shot_id):05d}.npz",
                    gradient=gradient_local,
                    rtm_image=rtm_image,
                    source_illumination=src_illum,
                    receiver_illumination=rec_illum,
                    shot_id=np.int64(int(shot_id)),
                    z0=np.int32(z0), z1=np.int32(z1),
                    x0=np.int32(x0), x1=np.int32(x1),
                    loss=np.float32(loss_value),
                )

        if ndim == 3:
            win_str = f"z[{z0},{z1})y[{y0},{y1})x[{x0},{x1})"
        else:
            win_str = f"z[{z0},{z1})x[{x0},{x1})"
        print(f"[rtm] batch {global_batch_index:04d} "
              f"shots={batch_indices.size} loss={loss_value:.6e} "
              f"grad_rms={grad_rms:.3e} rtm_max={rtm_abs_max:.3e} "
              f"window={win_str}", flush=True)

        # Free per-batch tensors so subsequent batches see a clean allocator.
        del syn, syn_filt, obs_chunk, local_vp
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def _run_lsrtm(self, spec: LSRTMSpec, task_dir: Path):
        import torch

        from sweep_runner import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        if dist_info.is_distributed and spec.optimizer.kind == "lbfgs":
            raise ValueError(
                "LBFGS optimiser is not supported under torchrun in Phase 1. "
                "Use adam or sgd."
            )

        _apply_seed(spec.seed)
        dev = _dist.resolve_dist_device(spec.device, dist_info.local_rank)
        lsrtm_cls = _get_equation_class(spec.physics.equation)

        bg_equation_name = _lsrtm_background_equation(spec.physics.equation)
        bg_cls = _get_equation_class(bg_equation_name)
        _validate_single_model(bg_cls, spec.background_model)

        if spec.grid.shape is not None:
            shape = tuple(int(v) for v in spec.grid.shape)
        elif spec.background_model.constant is not None:
            shape = tuple(int(v) for v in spec.background_model.shape)
        else:
            shape = tuple(np.load(spec.background_model.path, mmap_mode="r").shape)

        # Background Acoustic solver uses the plain ["h1"] receiver type.
        bg_physics = PhysicsSpec(
            equation=bg_equation_name,
            spatial_order=spec.physics.spatial_order,
            abcn=spec.physics.abcn,
            free_surface=spec.physics.free_surface,
            pml_type=spec.physics.pml_type,
            source_type=spec.physics.source_type,
            receiver_type=["h1"],
        )
        acoustic_solver = _build_solver(
            bg_physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )
        lsrtm_solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )

        wavelet = _build_wavelet(spec.wavelet, spec.time)
        sources, receivers = _build_geometry_2d(spec.geometry, shape)
        nshots = int(sources.shape[0])
        if dist_info.is_root:
            print(f"[lsrtm] shape={shape} nshots={nshots} epochs={spec.epochs} "
                  f"world_size={dist_info.world_size}")

        true_tensor = _load_model_tensor(spec.true_model).to(dev)
        bg_tensor = _load_model_tensor(spec.background_model).to(dev)

        mod_wavelet, mod_sources, mod_receivers, used_override = _resolve_modeling_inputs(
            spec, wavelet, sources, receivers, shape
        )
        if used_override and dist_info.is_root:
            print(f"[lsrtm] modeling_override applied (wavelet={spec.modeling_override.wavelet is not None}, "
                  f"geometry={spec.modeling_override.geometry is not None})")

        # scattered observed data = forward(true) - forward(background) — same on every rank.
        with torch.no_grad():
            obs_true = acoustic_solver(mod_wavelet, mod_sources, mod_receivers,
                                       models=[true_tensor]).detach().clone()
            obs_bg = acoustic_solver(mod_wavelet, mod_sources, mod_receivers,
                                     models=[bg_tensor]).detach().clone()
        obs = (obs_true - obs_bg).detach().cpu()
        del obs_true, obs_bg, true_tensor
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # Reflectivity is the only inverted parameter; vp stays as background.
        vp = bg_tensor
        ref = torch.zeros_like(vp, requires_grad=True)
        inv_by_name = {"reflectivity": ref}
        inv_in_order = [ref]
        required_names = ["reflectivity"]

        total_epochs = (sum(s.epochs for s in spec.stages)
                        if spec.stages else int(spec.epochs))
        optimizer = _build_optimizer(spec.optimizer, inv_by_name, required_names)
        scheduler = _build_scheduler(spec.scheduler, optimizer, total_epochs)
        initial_lrs = _remember_initial_lrs(optimizer)

        # Resume: same two modes as FWI — see ``_run_fwi`` for semantics.
        losses: list[float] = []
        start_epoch = 0
        resume_src: str | None = None
        if spec.resume_from:
            resume_src = spec.resume_from
            ckpt_dir = Path(spec.output_dir).expanduser() / spec.resume_from
        elif spec.resume and (task_dir / "checkpoint.pt").exists():
            resume_src = task_dir.name
            ckpt_dir = task_dir
        else:
            ckpt_dir = None
        if ckpt_dir is not None:
            if dist_info.is_root:
                ckpt = _load_checkpoint(ckpt_dir)
            else:
                ckpt = None
            ckpt = _dist.broadcast_object(ckpt, dist_info, src=0)
            ref.data.copy_(ckpt["models"]["reflectivity"].to(dev))
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            losses = list(ckpt.get("losses", []))
            start_epoch = int(ckpt["epoch"]) + 1
            if dist_info.is_root:
                torch.set_rng_state(ckpt["torch_rng"])
                np.random.set_state(ckpt["numpy_rng"])
                print(f"[lsrtm] resumed from '{resume_src}' at epoch {start_epoch}")

        # Bounds keyed by "reflectivity" to reuse the FWI helper.
        bounds_by_name: dict = {}
        if spec.reflectivity_bounds is not None:
            bounds_by_name["reflectivity"] = spec.reflectivity_bounds

        stages = _normalise_stage_list(spec)
        out_dir = task_dir / "output"
        snapshots_dir = out_dir / "epochs"
        if dist_info.is_root:
            snapshots_dir.mkdir(exist_ok=True)
        _dist.barrier(dist_info)
        epoch_global = start_epoch

        # Graceful ctrl-c — see ``_run_fwi`` for the full rationale.
        stopper = _GracefulStopper()
        stopper.install(label="lsrtm")
        interrupted = False
        interrupted_at_epoch: int | None = None

        for stage_idx, stage in enumerate(stages):
            stage_offset = sum(s.epochs for s in stages[:stage_idx])
            already_done_in_stage = max(0, epoch_global - stage_offset)
            remaining = stage.epochs - already_done_in_stage
            if remaining <= 0:
                continue
            stage_wavelet = (_build_wavelet(stage.wavelet, spec.time)
                             if stage.wavelet is not None else wavelet)
            _apply_stage_lr_scale(optimizer, initial_lrs, stage.lr_scale)
            if dist_info.is_root:
                print(f"[lsrtm] stage {stage_idx} ({remaining}/{stage.epochs} epochs) "
                      f"lr_scale={stage.lr_scale} wavelet_overridden={stage.wavelet is not None}")
            for _ in range(remaining):
                loss_value = self._lsrtm_train_step(
                    spec, lsrtm_solver, stage_wavelet, sources, receivers,
                    vp, ref, obs, optimizer, nshots, dev,
                    dist_info=dist_info,
                )
                losses.append(loss_value)
                if scheduler is not None:
                    scheduler.step()
                _apply_bounds(inv_by_name, bounds_by_name)
                if dist_info.is_root:
                    print(f"[lsrtm] stage {stage_idx} epoch {epoch_global:04d} loss={loss_value:.6e}")

                snapshot_now = (epoch_global % spec.show_every == 0
                                or epoch_global == total_epochs - 1)
                if snapshot_now and dist_info.is_root:
                    np.save(snapshots_dir / f"reflectivity_epoch_{epoch_global:04d}.npy",
                            ref.detach().cpu().numpy())
                    if spec.save_illumination:
                        _save_illumination(lsrtm_solver, snapshots_dir, epoch_global)

                if dist_info.is_root:
                    _save_checkpoint(task_dir, {
                        "epoch": epoch_global,
                        "models": {"reflectivity": ref.detach().cpu().clone()},
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "losses": losses,
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                    })
                if stopper.should_stop(dist_info):
                    interrupted = True
                    interrupted_at_epoch = epoch_global
                    if dist_info.is_root:
                        print(f"[lsrtm] stopping after epoch {epoch_global} "
                              f"(checkpoint.pt saved). Re-run the same YAML "
                              f"with `resume: true` to continue.")
                    epoch_global += 1
                    break
                epoch_global += 1
            if interrupted:
                break

        stopper.uninstall()

        # Final outputs (rank 0 only).
        artifacts: list[Path] = []
        if dist_info.is_root:
            final_ref_path = out_dir / "reflectivity.npy"
            np.save(final_ref_path, ref.detach().cpu().numpy())
            loss_path = out_dir / "loss.npy"
            np.save(loss_path, np.array(losses, dtype=np.float64))
            artifacts.extend([final_ref_path, loss_path])
            try:
                artifacts.append(_plot_loss_curve(losses, out_dir / "loss.png", title="LSRTM Loss"))
            except Exception as plot_err:  # noqa: BLE001
                print(f"[lsrtm] loss plot skipped: {plot_err}")
        _dist.barrier(dist_info)

        summary = {
            "epochs": total_epochs,
            "epochs_completed": len(losses),
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "num_stages": len(stages),
            "resumed_from": resume_src,
            "interrupted": interrupted,
            "interrupted_at_epoch": interrupted_at_epoch,
            "world_size": dist_info.world_size,
        }
        return artifacts, summary

    def _lsrtm_train_step(self, spec, lsrtm_solver, wavelet, sources, receivers,
                          vp, ref, obs, optimizer, nshots, dev,
                          *, dist_info=None) -> float:
        import torch

        from sweep_runner import distributed as _dist

        if dist_info is None:
            dist_info = getattr(self, "_dist", None) or _dist.init_distributed_if_needed()

        inv_in_order = [ref]
        global_batchsize = min(spec.batchsize, nshots)

        if dist_info.is_root:
            shot_idx_global = np.random.choice(nshots, size=global_batchsize, replace=False)
        else:
            shot_idx_global = None
        shot_idx_global = _dist.broadcast_shot_indices(
            shot_idx_global, global_batchsize, dist_info, src=0
        )
        local_shots = _dist.split_for_rank(shot_idx_global, dist_info)
        chunk_size = spec.train_shot_batchsize or max(len(local_shots), 1)
        chunks = ([local_shots[i:i + chunk_size]
                   for i in range(0, len(local_shots), chunk_size)]
                  if len(local_shots) > 0 else [])

        sample = obs[:1]
        per_shot_numel = int(sample.numel())
        global_norm = float(per_shot_numel * global_batchsize)
        data_mask = self._get_data_mask(spec, obs, dev)
        if data_mask is not None:
            global_norm = float(global_norm * max(float(data_mask.float().mean()), 1.0e-6))

        if spec.optimizer.kind == "lbfgs":
            def _closure():
                optimizer.zero_grad()
                acc_loss = 0.0
                for chunk in chunks:
                    syn = lsrtm_solver(wavelet, sources[chunk], receivers[chunk],
                                       models=[vp, ref])
                    obs_chunk = obs[chunk].to(dev)
                    loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev))
                    (loss_t / global_norm).backward()
                    acc_loss += float(loss_t.detach().cpu())
                _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                return torch.tensor(acc_loss / global_norm if global_norm else 0.0)
            result = optimizer.step(_closure)
            return float(result) if result is not None else 0.0

        optimizer.zero_grad()
        acc_loss_local = 0.0
        for chunk in chunks:
            syn = lsrtm_solver(wavelet, sources[chunk], receivers[chunk],
                               models=[vp, ref])
            obs_chunk = obs[chunk].to(dev)
            loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev))
            (loss_t / global_norm).backward()
            acc_loss_local += float(loss_t.detach().cpu())
        _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
        _dist.all_reduce_grad_sum(inv_in_order, dist_info)
        optimizer.step()
        acc_loss_global = _dist.all_reduce_scalar_sum(acc_loss_local, dist_info)
        return acc_loss_global / global_norm if global_norm else 0.0


# ---------- equation-name helpers ----------------------------------------

def _lsrtm_background_equation(lsrtm_name: str) -> str:
    mapping = {
        "AcousticLSRTM": "Acoustic",
        "AcousticLSRTM3D": "Acoustic3D",
    }
    if lsrtm_name not in mapping:
        raise ValueError(
            f"No background equation known for LSRTM variant '{lsrtm_name}'. "
            f"Known: {sorted(mapping)}"
        )
    return mapping[lsrtm_name]


# ---------- lightweight plotting helpers ---------------------------------

def _crop_padded_volume_to_model(volume, target_shape: tuple, physics) -> "np.ndarray":
    """Crop a sweep c-backend RTM/illumination volume back to the model shape.

    The c-backend pads each model side by ``M = spatial_order // 2`` (for FD
    stencil) and ``abcn`` (for the absorbing layer). For free-surface runs,
    the top z-side is padded only by ``M`` instead of ``M + abcn``.

    Mirrors ``fwi_workflow.imaging.rtm._crop_solver_volume_2d``. 2-D-only;
    3-D RTM cropping is not yet implemented (raise a clear error).
    """
    import torch as _torch

    if isinstance(volume, _torch.Tensor):
        values = volume.detach().cpu()
        while values.ndim > 2:
            # Squeeze leading shot / channel axes (typical c-backend output is
            # (B, 1, nz, nx) for 2-D acoustic; sum over the batch axis to merge
            # contributions from multiple shots in the same batch).
            if int(values.shape[0]) == 1:
                values = values.squeeze(0)
            else:
                values = values.sum(dim=0)
        array = values.numpy().astype(np.float32, copy=False)
    else:
        array = np.asarray(volume, dtype=np.float32)
        while array.ndim > 2:
            if int(array.shape[0]) == 1:
                array = array[0]
            else:
                array = array.sum(axis=0)

    if tuple(array.shape) == tuple(target_shape):
        return np.ascontiguousarray(array)
    if array.ndim != 2:
        raise NotImplementedError(
            f"_crop_padded_volume_to_model: only 2-D crops are supported; "
            f"got array shape {tuple(array.shape)} (3-D RTM cropping not yet "
            "implemented)."
        )
    margin = max(0, int(physics.spatial_order) // 2)
    pad = max(0, int(physics.abcn)) + margin
    nz_pad, nx_pad = int(array.shape[0]), int(array.shape[1])
    nz_target, nx_target = int(target_shape[0]), int(target_shape[1])
    x_slice = slice(pad, nx_pad - pad if pad > 0 else None)
    z_slice = slice(
        margin if physics.free_surface else pad,
        nz_pad - pad if pad > 0 else None,
    )
    cropped = array[z_slice, x_slice]
    # Some backends return a slightly larger volume than the strict crop;
    # if the strict slice is wrong by 1-2 cells, take the leading sub-array
    # that matches the target shape (matches the legacy fallback path).
    if cropped.shape != (nz_target, nx_target):
        if (cropped.shape[0] >= nz_target and cropped.shape[1] >= nx_target):
            cropped = cropped[:nz_target, :nx_target]
        else:
            raise ValueError(
                f"_crop_padded_volume_to_model: cannot crop from {array.shape} "
                f"to {target_shape}; got {cropped.shape}."
            )
    return np.ascontiguousarray(cropped, dtype=np.float32)


def _save_rtm_qc_pngs(
    *,
    gradient_sum,
    rtm_sum,
    source_illum_sum,
    receiver_illum_sum,
    gradient_per_shot_norm_sum=None,
    rtm_per_shot_norm_sum=None,
    qc_dir: Path,
    suffix: str = "",
    eps: float = 1.0e-6,
    normalize: bool = True,
) -> list[Path]:
    """Write the standard RTM QC PNGs to ``qc_dir``.

    ``suffix`` lets the live-update path emit ``*_latest.png`` without
    overwriting the final products.

    Returns the list of written paths (so the caller can register them as
    artifacts in the task status).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    # Side-effect import: registers ``sweep_image`` / ``sweep_vp`` so the
    # RTM panels below resolve. Tolerate sweep_viz being missing — caller
    # gets a clear matplotlib error in that case.
    try:
        from sweep_viz.colormaps import IMAGE_CMAP as _RTM_CMAP
    except Exception:  # noqa: BLE001
        _RTM_CMAP = "seismic"

    qc_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def _imshow(ax, array, title, *, symmetric: bool, cmap: str, perc=(1.0, 99.0)):
        finite = array[np.isfinite(array)]
        if finite.size:
            vmin, vmax = np.percentile(finite, list(perc))
            if symmetric:
                m = max(abs(float(vmin)), abs(float(vmax)))
                vmin, vmax = -m, m
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None
        im = ax.imshow(array, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        return im

    def _single_png(array, title, path, *, symmetric, cmap):
        fig, ax = plt.subplots(1, 1, figsize=(11, 3.8))
        try:
            im = _imshow(ax, array, title, symmetric=symmetric, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        return path

    # Raw RTM image
    paths.append(_single_png(
        rtm_sum, f"RTM image{suffix}", qc_dir / f"rtm_image{suffix}.png",
        symmetric=True, cmap=_RTM_CMAP,
    ))
    # Raw FWI gradient image
    paths.append(_single_png(
        gradient_sum, f"FWI gradient image{suffix}",
        qc_dir / f"gradient_image{suffix}.png",
        symmetric=True, cmap=_RTM_CMAP,
    ))

    # Normalised image (only when normalize_by_illumination is on; otherwise
    # the raw RTM is already the deliverable)
    if normalize:
        denom = np.sqrt(np.maximum(source_illum_sum * receiver_illum_sum, 0.0) + float(eps))
        rtm_norm = (rtm_sum / denom).astype(np.float32)
        paths.append(_single_png(
            rtm_norm,
            f"RTM image (global illumination-normalised){suffix}",
            qc_dir / f"rtm_image_normalised{suffix}.png",
            symmetric=True, cmap=_RTM_CMAP,
        ))

    # Per-shot illumination-normalised image — the canonical Viking RTM
    # display (legacy `illumination_normalized_rtm_per_shot.png`). Each
    # shot's contribution was divided by its own illumination BEFORE the
    # stack, so deep / weakly-illuminated cells survive instead of being
    # drowned by the strong shallow energy in the global-normalised view.
    if rtm_per_shot_norm_sum is not None:
        paths.append(_single_png(
            np.asarray(rtm_per_shot_norm_sum),
            f"RTM image (per-shot illumination-normalised){suffix}",
            qc_dir / f"rtm_image_per_shot_normalised{suffix}.png",
            symmetric=True, cmap=_RTM_CMAP,
        ))
    if gradient_per_shot_norm_sum is not None:
        paths.append(_single_png(
            np.asarray(gradient_per_shot_norm_sum),
            f"FWI gradient (per-shot illumination-normalised){suffix}",
            qc_dir / f"gradient_image_per_shot_normalised{suffix}.png",
            symmetric=True, cmap=_RTM_CMAP,
        ))

    # Illumination panel: source / receiver / product
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), squeeze=False)
        _imshow(axes[0, 0], source_illum_sum, "Source illumination", symmetric=False, cmap="magma")
        _imshow(axes[0, 1], receiver_illum_sum, "Receiver illumination", symmetric=False, cmap="magma")
        prod = source_illum_sum * receiver_illum_sum
        _imshow(axes[0, 2], prod, "S * R (illumination product)", symmetric=False, cmap="magma")
        fig.tight_layout()
        ill_path = qc_dir / f"illumination{suffix}.png"
        fig.savefig(ill_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(ill_path)
    except Exception:  # noqa: BLE001
        plt.close("all")

    return paths


def _dump_receiver_rotation_qc(
    *,
    task_dir: Path,
    recv_model_xy: np.ndarray,
    recv_utm_xy: np.ndarray,
    recv_z: np.ndarray,
    grid_origin_xyz: tuple[float, float, float],
    grid_shape: tuple[int, int, int],
    dh_xyz: tuple[float, float, float],
) -> Path:
    """Save a receiver-layout PNG + npz showing the UTM→model rotation.

    Two-panel PNG (model frame on left, raw UTM on right) for visual
    confirmation that the ``rotation_metadata.json`` projects the OBN
    nodes the same way the legacy pipeline did. Companion npz holds the
    underlying float64 arrays so the user can do ``np.allclose`` against
    a legacy reference.

    Inputs are POST-filter (after any ModelPlan crop) — exactly the
    receivers the FWI run will actually see.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    qc_dir = task_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, _nz = (int(v) for v in grid_shape)
    dx_m, dy_m, _dz_m = (float(v) for v in dh_xyz)
    origin_x, origin_y, _origin_z = (float(v) for v in grid_origin_xyz)
    recv_model_xy = np.asarray(recv_model_xy, dtype=np.float64)
    recv_utm_xy = np.asarray(recv_utm_xy, dtype=np.float64)
    recv_z = np.asarray(recv_z, dtype=np.float64)

    # Origin-shifted model coords so (0, 0) = grid corner; matches the
    # propagator's local frame. Helpful when overlaying the grid bbox.
    shifted = recv_model_xy - np.asarray([origin_x, origin_y])[None, :]
    bbox_x = [0.0, nx * dx_m, nx * dx_m, 0.0, 0.0]
    bbox_y = [0.0, 0.0, ny * dy_m, ny * dy_m, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Left — model frame, origin-shifted, grid bbox overlay.
    sc0 = axes[0].scatter(
        shifted[:, 0], shifted[:, 1], c=recv_z, cmap="viridis",
        s=18, edgecolors="k", linewidths=0.4,
    )
    axes[0].plot(bbox_x, bbox_y, "k--", lw=1, label="grid bbox")
    axes[0].set_xlabel("x (m, model frame; origin at 0)")
    axes[0].set_ylabel("y (m, model frame; origin at 0)")
    axes[0].set_title(
        f"Receivers in model frame (n={len(shifted)})  "
        f"grid={nx}×{ny} @ ({dx_m:.1f}, {dy_m:.1f}) m"
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.04, pad=0.02)
    cb0.set_label("receiver z (m)")

    # Right — raw UTM (pre-rotation) for direct comparison with legacy outputs.
    sc1 = axes[1].scatter(
        recv_utm_xy[:, 0], recv_utm_xy[:, 1], c=recv_z, cmap="viridis",
        s=18, edgecolors="k", linewidths=0.4,
    )
    axes[1].set_xlabel("UTM x (m)")
    axes[1].set_ylabel("UTM y (m)")
    axes[1].set_title(f"Receivers in UTM (raw, n={len(recv_utm_xy)})")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(alpha=0.3)
    cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.04, pad=0.02)
    cb1.set_label("receiver z (m)")

    fig.suptitle(
        "Receiver layout — rotation QC "
        "(left = post-rotation model frame, right = raw UTM)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = qc_dir / "receiver_layout_rotation.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Companion npz — full float64 arrays for np.allclose vs. legacy.
    out_npz = qc_dir / "receiver_layout.npz"
    np.savez(
        out_npz,
        recv_model_xy=recv_model_xy.astype(np.float64, copy=False),
        recv_utm_xy=recv_utm_xy.astype(np.float64, copy=False),
        recv_z=recv_z.astype(np.float64, copy=False),
        grid_origin_xyz=np.asarray(grid_origin_xyz, dtype=np.float64),
        grid_shape=np.asarray(grid_shape, dtype=np.int64),
        dh_xyz=np.asarray(dh_xyz, dtype=np.float64),
    )
    print(f"[multisource] receiver layout QC -> {out_png}")
    print(f"[multisource] receiver layout npz -> {out_npz}")
    return out_png


def _plot_loss_curve(losses, path: Path, title: str) -> Path:
    """Loss curve via `sweep_viz.convergence.plot_loss`."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sweep_viz.convergence import plot_loss

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss(list(losses), ax=ax, logy=True, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_wavefield_snapshots(snapshots_np, snapshot_times, abcn, shape, path,
                              free_surface) -> Path:
    """Wavefield snapshot grid via `sweep_viz.wavefield.plot_snapshot` (one per panel).

    The PML / absorbing-boundary cropping logic (and the `(nsnap, 1, 1, 1, ...)`
    sweep-binding tensor layout) is sweep-tasks-specific, so it stays here;
    each cropped panel is then rendered by sweep_viz.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sweep_viz.wavefield import plot_snapshot

    nz, nx = int(shape[0]), int(shape[1])
    nsnap = snapshots_np.shape[0]
    fig, axes = plt.subplots(1, nsnap, figsize=(4 * nsnap, 4), squeeze=False)
    for i in range(nsnap):
        panel = snapshots_np[i, 0, 0, 0]
        if free_surface:
            panel = panel[:nz, abcn: abcn + nx]
        else:
            panel = panel[abcn: abcn + nz, abcn: abcn + nx]
        plot_snapshot(panel, ax=axes[0, i], perc=100.0,
                      cmap="seismic", title=f"t-step {snapshot_times[i]}")
        axes[0, i].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
