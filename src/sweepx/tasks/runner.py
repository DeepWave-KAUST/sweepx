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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sweep
import sweep.equations as eq_mod
from sweep.signal import ricker
from sweep_tasks.schemas import (
    BaseTaskSpec,
    ForwardSpec,
    FWISpec,
    IntrospectSpec,
    LSRTMSpec,
    LineSet,
    ModelRef,
    PhysicsSpec,
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


def _build_geometry_2d(
    geometry,
    shape: tuple[int, ...],
    *,
    dh: float | None = None,
    segy_cache: dict | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Dispatch on geometry.kind and return (sources, receivers) numpy arrays.

    Output shapes are always sources=(nshots, ndim) and receivers=(nshots, nrec, ndim).

    ``dh`` and ``segy_cache`` are only used by the SEG-Y-backed kinds
    (``from_segy_headers`` / ``from_segy_index``). The cache lets the runner
    share a single SEG-Y scan between the geometry resolver and the obs
    loader when both point at the same file / same index.
    """

    kind = getattr(geometry, "kind", None)
    if kind == "line":
        nx = int(shape[-1])
        sources = _line_array(geometry.sources, nx)
        rec = _line_array(geometry.receivers, nx)
        receivers = rec[None, ...].repeat(sources.shape[0], axis=0)
        return sources, receivers
    if kind == "explicit":
        return _explicit_geometry_arrays(geometry)
    if kind == "from_file":
        return _from_file_geometry_arrays(geometry)
    if kind == "from_segy_headers":
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
    raise ValueError(f"Unknown geometry.kind '{kind}'.")


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
    raise ValueError(f"Unknown wavelet.kind '{kind}'.")


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

    # obs layout depends on the propagator backend:
    #   eager (PropTorch+autograd):  (nshots, nt, nrec[, 1])
    #   c     (compiled CUDA bind.): (nshots, nrec, nt)
    # Pick the right (receiver_axis, time_axis) so DataPlan's offset filter +
    # receiver stride + time resample / window touch the intended axis.
    obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
    if spec.backend.impl == "eager":
        time_axis = -3 if obs_np.ndim >= 4 else -2
        receiver_axis = -2 if obs_np.ndim >= 4 else -1
    else:  # "c" or any future binding using the (nshots, nrec, nt) layout
        receiver_axis = 1
        time_axis = -2 if obs_np.ndim >= 4 else -1

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
        # Dispatch to sweep-loss's GlobalCorrelationLoss with demean=True.
        # sweep-loss expects canonical (ns, nt, nrec, nchan); the c backend
        # gives us (ns, nrec, nt) so we permute first, run the loss to get
        # a per-trace tensor of shape (N,), then broadcast it back to the
        # syn shape so the caller's .sum() / global_norm yields mean(1-cos)
        # per trace.
        from sweep_loss import global_correlation_loss
        eps = float(getattr(loss_spec, "trace_cosine_eps", 1.0e-8))
        demean = bool(getattr(loss_spec, "trace_cosine_demean", True))

        if syn.ndim == 4:
            # eager: (ns, nt, nrec, 1) — already canonical
            syn_can, obs_can = syn, obs
            permuted = False
        elif syn.ndim == 3 and syn.shape[-1] != 1:
            # c backend: (ns, nrec, nt) → permute → (ns, nt, nrec); add chan.
            syn_can = syn.permute(0, 2, 1).unsqueeze(-1)  # (ns, nt, nrec, 1)
            obs_can = obs.permute(0, 2, 1).unsqueeze(-1)
            permuted = True
        elif syn.ndim == 3 and syn.shape[-1] == 1:
            # eager 3D (nt, nrec, 1) — single-shot synthetic path; sweep-loss
            # promotes to (1, nt, nrec, 1) via to_canonical.
            syn_can, obs_can = syn, obs
            permuted = False
        else:
            raise ValueError(f"trace_cosine: unsupported syn.ndim {syn.ndim}")

        per_trace = global_correlation_loss(
            syn_can, obs_can,
            offset_one=True, demean=demean, eps=eps, reduction="none",
        )  # (N,) where N = ns * nrec * nchan after sweep-loss flatten

        # Reshape back to per-trace canonical and broadcast across time so
        # the caller's .sum() / global_norm yields mean(1-cos) over traces.
        # sweep_loss.base.flatten_traces uses order (ns, nr, nc) so:
        if syn_can.ndim == 4:
            ns_c, nt_c, nr_c, nc_c = syn_can.shape
        else:  # (nt, nrec, 1) → canonical promoted shape (1, nt, nrec, 1)
            ns_c, nt_c, nr_c, nc_c = 1, syn_can.shape[0], syn_can.shape[1], syn_can.shape[2]
        pt_can = per_trace.view(ns_c, 1, nr_c, nc_c).expand(ns_c, nt_c, nr_c, nc_c)

        if permuted:
            # back to (ns, nrec, nt)
            return pt_can.squeeze(-1).permute(0, 2, 1).contiguous()
        if syn.ndim == 4:
            return pt_can
        # 3D eager (nt, nrec, 1)
        return pt_can.squeeze(0)
    raise ValueError(f"Unknown loss kind '{kind}'.")


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
        bound = bounds_by_name.get(name)
        if bound is None:
            continue
        t.data.clamp_(min=bound.min, max=bound.max)


def _compute_local_window(sources_chunk, receivers_chunk, full_shape, dh, win_spec):
    """Return ``(z0, z1, x0, x1)`` tightly enclosing the batch's sources +
    receivers, expanded by ``padding_x_m`` / ``padding_z_m``.

    ``sources_chunk`` has shape ``(B, 2)`` and ``receivers_chunk`` has shape
    ``(B, nrec, 2)``. Convention (matches the rest of the runner): the
    last axis is ``[x, z]`` — i.e. ``[lateral, depth]``. The result is
    clamped to the full-model shape ``(nz, nx)``.
    """
    import numpy as _np
    nz, nx = int(full_shape[0]), int(full_shape[1])
    pad_x = int(float(win_spec.padding_x_m) / float(dh))
    pad_z = int(float(win_spec.padding_z_m) / float(dh))
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
    min_w = int(float(win_spec.min_width_m) / float(dh))
    if min_w > 0 and (x1 - x0) < min_w:
        extra = min_w - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(nx, x0 + min_w)
        if (x1 - x0) < min_w:  # right edge clipping
            x0 = max(0, x1 - min_w)
    return int(z0), int(z1), int(x0), int(x1)


def _rebase_geometry_to_window(sources_chunk, receivers_chunk, z0, x0):
    """Shift grid-index source / receiver coordinates to window-local origin.

    Returns fresh arrays (not views); callers can pass to solver freely.
    Axis convention matches the runner: ``[x, z]`` on the last axis.
    """
    import numpy as _np
    s = _np.asarray(sources_chunk, dtype=_np.int64).copy()
    r = _np.asarray(receivers_chunk, dtype=_np.int64).copy()
    s[:, 0] -= int(x0); s[:, 1] -= int(z0)
    r[..., 0] -= int(x0); r[..., 1] -= int(z0)
    return s, r


def _build_reparam_net(spec, base_vp, bounds):
    """Construct a :class:`sweep_nn.VelocityINR` from a ReparamSpec.

    ``base_vp`` is the initial vp tensor at the first stage's grid. The net
    keeps it as a buffer (its forward returns ``base + delta``). ``bounds``
    is the optional :class:`ModelBounds` for vp — if provided, the network
    clamps its render output to those limits.
    """
    from sweep_nn import VelocityINR

    bounds_tuple = None
    if bounds is not None and bounds.min is not None and bounds.max is not None:
        bounds_tuple = (float(bounds.min), float(bounds.max))
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
    """Bilinear resample of a 2-D vp tensor between grid resolutions.

    Returns a fresh leaf tensor (requires_grad=True) — caller is responsible
    for re-initialising the optimizer because Adam state is shape-bound.
    """
    import torch
    import torch.nn.functional as F

    if tuple(vp.shape) == tuple(new_shape):
        return vp.detach().clone().requires_grad_(True)
    src = vp.detach().unsqueeze(0).unsqueeze(0)
    dst = F.interpolate(src, size=tuple(new_shape), mode="bilinear", align_corners=True)
    return dst.squeeze(0).squeeze(0).contiguous().clone().requires_grad_(True)


def _resample_obs_time(obs_np: "np.ndarray", dt_old: float, dt_new: float,
                       *, time_axis: int = -1) -> "np.ndarray":
    """Resample obs along the time axis via sweep_preproc.resample.resample_time."""
    if abs(dt_old - dt_new) < 1e-12:
        return obs_np
    from sweep_preproc.resample import resample_time
    return resample_time(obs_np, dt_old, dt_new, axis=time_axis)


def _bandpass_obs(obs_np: "np.ndarray", lo: float, hi: float, dt: float,
                  *, order: int, time_axis: int = -1,
                  padtype: str | None = "odd") -> "np.ndarray":
    """Per-stage bandpass on obs (zero-phase Butterworth).

    ``order`` is the prototype Butterworth order (matches
    ``fwi_workflow-dev``'s ``filter_order``); ``padtype`` controls
    edge handling (default ``"odd"`` reflective padding; ``None``
    matches ``torchaudio.functional.filtfilt``'s no-padding behavior).
    """
    from sweep_preproc.filter import bandpass
    return bandpass(obs_np, lo=lo, hi=hi, dt=dt, order=order,
                    axis=time_axis, padtype=padtype)


def _bandpass_syn_torch(syn: "torch.Tensor", lo: float, hi: float, dt: float,
                        *, order: int) -> "torch.Tensor":
    """Differentiable bandpass on a synthetic torch tensor.

    Mirrors ``fwi_workflow-dev``'s ``_apply_torch_filter`` (which routes
    through ``torchaudio.functional.filtfilt``). Without this, FWI loss
    compares un-filtered syn (containing frequencies above the stage's
    ``hi_hz``) against bandpassed obs — producing spurious high-frequency
    residuals that drive bad gradient updates. The result is what users
    see as "dispersion" in the synthetic output.

    The time axis is auto-detected from tensor layout:
        * c backend, 3-D ``(n, nrec, nt)``       → time at axis -1
        * eager 4-D ``(n, nt, nrec, nchan)``     → time at axis 1
        * eager 3-D ``(nt, nrec, nchan=1)``      → time at axis 0
    """
    import torch
    from scipy.signal import butter
    from torchaudio.functional import filtfilt as _ta_filtfilt

    # Detect time axis + plan permutations to bring time to last.
    if syn.ndim == 4:
        # eager: (n, nt, nrec, nchan). Move nt to last.
        permute_to_last = (0, 2, 3, 1)
        inverse_permute = (0, 3, 1, 2)
    elif syn.ndim == 3 and syn.shape[-1] == 1:
        # eager 3-D: (nt, nrec, 1). Move nt to last.
        permute_to_last = (1, 2, 0)
        inverse_permute = (2, 0, 1)
    elif syn.ndim == 3:
        # c backend: (n, nrec, nt). Time already last.
        permute_to_last = None
        inverse_permute = None
    else:
        raise ValueError(f"_bandpass_syn_torch: unsupported syn.ndim {syn.ndim}")

    syn_perm = syn.permute(*permute_to_last).contiguous() if permute_to_last else syn

    nyq = 0.5 / float(dt)
    wn = [float(lo) / nyq, float(hi) / nyq]
    b, a = butter(int(order), wn, btype="bandpass")
    # ``(b, a)`` form of order >= 4 is numerically ill-conditioned in float32
    # — cast to float64 for the filtfilt and cast back. Matches
    # ``fwi_workflow-dev``'s ``data.double()`` pattern (sweep_torch.py:237).
    a_t = torch.as_tensor(a, dtype=torch.float64, device=syn_perm.device)
    b_t = torch.as_tensor(b, dtype=torch.float64, device=syn_perm.device)
    out = _ta_filtfilt(syn_perm.double(), a_t, b_t, clamp=False).to(dtype=syn.dtype)
    if inverse_permute is not None:
        out = out.permute(*inverse_permute).contiguous()
    return out


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
            dh=(new_dh, new_dh),
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
    rebuilt_wavelet = False
    if time_changed or stage.wavelet is not None or "wavelet" not in state:
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
        from sweep_preproc.filter import bandpass as _bp_cpu
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
        wav_filt = _bp_cpu(
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
        # already-rectangular ``(nshots, n_common, ...)`` tensor.
        if obs_np.ndim == 3:
            # c-backend layout: (nshots, nrec, nt). receiver_axis_pristine == 1.
            nshots, nrec_pristine, nt_pristine = obs_np.shape
            new = np.empty(
                (nshots, per_shot_keep_idx.shape[1], nt_pristine),
                dtype=obs_np.dtype,
            )
            for s in range(nshots):
                new[s] = obs_np[s, per_shot_keep_idx[s], :]
            obs_np = new
        elif obs_np.ndim == 4:
            # eager layout: (nshots, nt, nrec, nchan). axis -2 = receiver.
            nshots, nt_pristine, nrec_pristine, nchan = obs_np.shape
            new = np.empty(
                (nshots, nt_pristine, per_shot_keep_idx.shape[1], nchan),
                dtype=obs_np.dtype,
            )
            for s in range(nshots):
                new[s] = obs_np[s, :, per_shot_keep_idx[s], :]
            obs_np = new
        else:
            raise NotImplementedError(
                f"per-shot keep on obs.ndim={obs_np.ndim} not implemented"
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

    # 4) bandpass (per-stage; uses the new dt, so it's correctly normalised)
    if stage.bandpass is not None:
        obs_np = _bandpass_obs(
            obs_np, lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
            dt=new_dt, order=stage.bandpass.order,
            time_axis=time_axis_pristine,
            padtype=stage.bandpass.padtype,
        )
        if dist_info.is_root:
            pad_tag = stage.bandpass.padtype or "none"
            print(f"[stage {stage_idx}] bandpass {stage.bandpass.lo_hz}-{stage.bandpass.hi_hz} Hz "
                  f"order={stage.bandpass.order} padtype={pad_tag}")
    state["_active_bandpass"] = stage.bandpass

    obs_np = np.ascontiguousarray(obs_np.astype(np.float32, copy=False))
    state["obs"] = torch.from_numpy(obs_np)
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
    # so DO NOT rebuild — preserving Adam moments across stages is the
    # main multi-scale benefit of network reparameterization.
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
    elif grid_changed and state.get("reparam_net") is None:
        state["optimizer"] = _build_optimizer(
            spec.optimizer, state["inv_by_name"], required_names,
        )
        state["initial_lrs"] = _remember_initial_lrs(state["optimizer"])

    _apply_stage_lr_scale(state["optimizer"], state["initial_lrs"], stage.lr_scale)

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
            status.state = "failed"
            status.error = f"{type(e).__name__}: {e}"
        finally:
            status.finished_at = _now_iso()
            if dist_info.is_root:
                status.write(task_dir / "status.json")
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
                spec.reparam, inv_by_name["vp"], spec.model_bounds.get("vp"),
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
        losses: list[float] = []
        start_epoch = 0
        if spec.resume_from:
            if dist_info.is_root:
                prev_dir = Path(spec.output_dir).expanduser() / spec.resume_from
                ckpt = _load_checkpoint(prev_dir)
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
                print(f"[fwi] resumed from '{spec.resume_from}' at epoch {start_epoch}")

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
        # Layout depends on backend (see _adapt_segy_obs_to_backend above
        # and _apply_data_plan_to_fwi).
        from sweep_io.geometry import PhysicalGeometry
        if spec.backend.impl == "eager":
            pristine_time_axis = -3 if obs.ndim >= 4 else -2
            pristine_recv_axis = -2 if obs.ndim >= 4 else -1
        else:
            pristine_recv_axis = 1
            pristine_time_axis = -2 if obs.ndim >= 4 else -1
        pristine_obs_np = (obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor)
                           else np.asarray(obs))
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
            "pristine_dt": float(effective_dt),
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

        for stage_idx, stage in enumerate(stages):
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
                epoch_global += 1

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
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "models_inverted": list(required_names),
            "num_stages": len(stages),
            "resumed_from": spec.resume_from,
            "world_size": dist_info.world_size,
        }
        return artifacts, summary

    def _fwi_generate_obs(self, spec, equation_cls, solver, wavelet,
                          sources, receivers, shape, dev, nshots,
                          *, segy_cache: dict | None = None):
        """Return obs tensor — shape depends on backend:
            eager:  (nshots, nt, nrec, 1)
            c:      (nshots, nrec, nt)
        Always CPU-resident.

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
            """SEGYReader always returns (nshots, nrec, nt). Match sweep's syn:
                eager → (nshots, nt, nrec, 1)
                c     → (nshots, nrec, nt)
            """
            arr = obs_nrec_nt.astype(np.float32, copy=False)
            if spec.backend.impl == "eager":
                # transpose to (nshots, nt, nrec) and add the trailing channel.
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
            obs = solver(mod_wavelet, mod_sources, mod_receivers,
                         models=true_in_order).detach().cpu()
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
            n_obs = int(state["obs"].shape[0])
            n_pick = max(1, min(int(spec.qc.shot_gather_n_shots), n_obs))
            shot_ids = np.linspace(0, n_obs - 1, n_pick, dtype=int).tolist()

            def _extract():
                # One extra forward over the picked shots. Build a list of
                # ``models`` matching the train step's chunk forward — use
                # the current vp (rendered if reparam, leaf otherwise).
                with torch.no_grad():
                    if state.get("reparam_net") is not None:
                        models = [state["reparam_net"]().detach()]
                    else:
                        models = state["inv_in_order"]
                    src = state["sources"][shot_ids]
                    rec = state["receivers"][shot_ids]
                    syn = state["solver"](state["wavelet"], src, rec, models=models)
                syn_np = syn.detach().cpu().numpy()
                obs_chunk = state["obs"][shot_ids].detach().cpu().numpy() \
                    if hasattr(state["obs"][shot_ids], "detach") \
                    else np.asarray(state["obs"][shot_ids])
                # Normalise both to (n_shots, nt, nrec) — same as canonical sweep_loss
                # layout, so the QC plotter can assume time is on axis -2.
                if spec.backend.impl == "eager":
                    if syn_np.ndim == 4 and syn_np.shape[-1] == 1:
                        syn_np = syn_np[..., 0]
                    if obs_chunk.ndim == 4 and obs_chunk.shape[-1] == 1:
                        obs_chunk = obs_chunk[..., 0]
                else:  # c backend: (n, nrec, nt) → transpose to (n, nt, nrec)
                    syn_np = np.transpose(syn_np, (0, 2, 1))
                    obs_chunk = np.transpose(obs_chunk, (0, 2, 1))

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

    def _fwi_train_step(self, spec, solver, wavelet, sources, receivers,
                        inv_in_order, inv_by_name, obs, optimizer, nshots, dev,
                        *, dist_info=None, stage_batchsize: int | None = None,
                        reparam_net=None, local_window_ctx=None,
                        syn_bandpass=None, stage_dt: float | None = None) -> float:
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
            z0, z1, x0, x1 = _compute_local_window(
                chunk_src, chunk_rec, full_shape, dh, win_spec,
            )
            local_shape = (z1 - z0, x1 - x0)
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
            local_src, local_rec = _rebase_geometry_to_window(chunk_src, chunk_rec, z0, x0)
            # Build the model list. Only vp is currently supported as
            # the windowed model; non-vp models pass through unmodified.
            if reparam_net is not None:
                local_vp = reparam_net.render_window(z0, z1, x0, x1)
            else:
                # Slicing a leaf tensor yields a view; PyTorch's autograd
                # scatters the gradient back to the leaf at [z0:z1, x0:x1].
                local_vp = inv_by_name["vp"][z0:z1, x0:x1]
            local_vp = local_vp.contiguous()
            ordered_names = list(inv_by_name.keys())
            models = [
                local_vp if name == "vp" else inv_by_name[name]
                for name in ordered_names
            ] if len(inv_by_name) > 1 else [local_vp]
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
                    loss_t = _compute_loss(syn, obs_chunk, spec.loss).sum()
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

        if reparam_net is None or reparam_mode == "single_step":
            # ---- (A) single-step: everything inside one autograd graph ----
            optimizer.zero_grad()
            acc_loss_local = 0.0
            for chunk in chunks:
                models, chunk_solver, chunk_src, chunk_rec = _chunk_inputs(chunk)
                syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                if syn_bandpass is not None and stage_dt is not None:
                    syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                              stage_dt, order=syn_bandpass.order)
                obs_chunk = obs[chunk].to(dev)
                loss_t = _compute_loss(syn, obs_chunk, spec.loss).sum()
                (loss_t / global_norm).backward()
                acc_loss_local += float(loss_t.detach().cpu())
            if reparam_net is None:
                _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                _dist.all_reduce_grad_sum(inv_in_order, dist_info)
            else:
                _dist.all_reduce_grad_sum(
                    [p for p in reparam_net.parameters() if p.grad is not None],
                    dist_info,
                )
            optimizer.step()
        else:
            # ---- (B) two-pass: solver phase against a leaf, then net phase ----
            optimizer.zero_grad()
            acc_loss_local = 0.0

            # Render the full-grid base velocity once, detached. The leaf is
            # the autograd boundary; solver backward accumulates onto leaf.grad.
            with torch.no_grad():
                base_leaf = reparam_net().detach().clone()
            base_leaf = base_leaf.requires_grad_(True)

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
                z0, z1, x0, x1 = _compute_local_window(
                    chunk_src, chunk_rec, full_shape, dh, win_spec,
                )
                local_shape = (z1 - z0, x1 - x0)
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
                local_src, local_rec = _rebase_geometry_to_window(chunk_src, chunk_rec, z0, x0)
                # View into leaf — gradient scatters back to leaf.grad on backward.
                models = [leaf[z0:z1, x0:x1].contiguous()]
                return models, local_solver, local_src, local_rec

            for chunk in chunks:
                models, chunk_solver, chunk_src, chunk_rec = _two_pass_chunk_inputs(chunk, base_leaf)
                syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                if syn_bandpass is not None and stage_dt is not None:
                    syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                              stage_dt, order=syn_bandpass.order)
                obs_chunk = obs[chunk].to(dev)
                loss_t = _compute_loss(syn, obs_chunk, spec.loss).sum()
                (loss_t / global_norm).backward()
                acc_loss_local += float(loss_t.detach().cpu())

            # leaf.grad now holds the full-grid FWI gradient ∂L/∂v.
            # Push it through the network in a second pass.
            v_grad = base_leaf.grad
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

        losses: list[float] = []
        start_epoch = 0
        if spec.resume_from:
            if dist_info.is_root:
                prev_dir = Path(spec.output_dir).expanduser() / spec.resume_from
                ckpt = _load_checkpoint(prev_dir)
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
                print(f"[lsrtm] resumed from '{spec.resume_from}' at epoch {start_epoch}")

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
                epoch_global += 1

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
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "num_stages": len(stages),
            "resumed_from": spec.resume_from,
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

        if spec.optimizer.kind == "lbfgs":
            def _closure():
                optimizer.zero_grad()
                acc_loss = 0.0
                for chunk in chunks:
                    syn = lsrtm_solver(wavelet, sources[chunk], receivers[chunk],
                                       models=[vp, ref])
                    obs_chunk = obs[chunk].to(dev)
                    loss_t = _compute_loss(syn, obs_chunk, spec.loss).sum()
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
            loss_t = _compute_loss(syn, obs_chunk, spec.loss).sum()
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
