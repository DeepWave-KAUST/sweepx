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
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=geometry.dedupe, dedup_method=geometry.dedup_method,
        )
        return sources_idx, receivers_idx
    if kind == "from_segy_index":
        if dh is None:
            raise ValueError("from_segy_index requires `dh` (spec.grid.dh).")
        payload = _load_segy_index_payload(
            geometry.index_path, shot_ids=geometry.shot_ids,
            cache=segy_cache,
        )
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=geometry.dedupe, dedup_method=geometry.dedup_method,
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

    ``lazy=False`` (default): read every shot eagerly and stack into a single
    ``(nshots, nrec, nt)`` tensor — same shape as ``ObsSpec.npy_path``.
    ``lazy=True``: return an :class:`IndexedShotGatherDataset` for incremental
    consumption. Not yet wired through the runner's training loop — see
    Gap 3 in the Viking REPORT — so for now ``lazy=True`` raises.
    """
    from sweep_io.segy_index import SEGYIndex, IndexedShotGatherDataset

    key = (str(Path(index_path).resolve()), tuple(shot_ids) if shot_ids else None,
           bool(lazy), int(coalesce_gap))
    if cache is not None and key in cache:
        return cache[key]

    idx = SEGYIndex.load(index_path)
    sids = (
        np.asarray(shot_ids, dtype=np.int64)
        if shot_ids is not None else np.asarray(idx.shot_ids, dtype=np.int64)
    )
    if lazy:
        raise NotImplementedError(
            "ObsSegyIndexConfig.lazy=True is not yet wired into the runner's "
            "training loop. Set lazy=False (eager load) for now. See Gap 3 in "
            "the Viking integration REPORT."
        )

    ds = IndexedShotGatherDataset(idx, shot_ids=sids, coalesce_gap=coalesce_gap)
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

    Returns (sources_idx, receivers_idx, obs_aligned). When dedupe drops
    receivers, the obs tensor is sliced to keep only the surviving ones
    (works only when the receiver mask is uniform across shots — the
    streamer-typical case). Non-uniform masks raise NotImplementedError.
    """
    pg = payload["physical_geometry"]
    obs = payload["obs"]
    gg, mask = pg.to_grid(dh=(dh, dh), dedupe=dedupe, dedup_method=dedup_method)
    uniform = bool(np.all(mask == mask[0:1]))
    if not uniform:
        raise NotImplementedError(
            "SEG-Y geometry produced a non-uniform per-shot receiver mask "
            "after to_grid dedupe. Set dedupe=false in the spec, or upgrade "
            "the runner to thread per-shot masks through the loss."
        )
    keep = np.flatnonzero(mask[0])
    if keep.size != obs.shape[1]:
        obs = np.ascontiguousarray(obs[:, keep, :])
        receivers_idx = gg.receivers[:, keep, :]
    else:
        receivers_idx = gg.receivers
    return gg.sources.astype(np.int64), receivers_idx.astype(np.int64), obs


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


def _build_wavelet(wavelet_spec, time_spec) -> "np.ndarray":
    """Dispatch on wavelet.kind. Returned array has length time.nt."""

    nt = int(time_spec.nt)
    dt = float(time_spec.dt)
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

def _apply_data_plan_to_fwi(
    spec, sources_idx, receivers_idx, obs, dh: float, dt: float, nt: int, dev
):
    """Apply ``spec.data_plan`` (and validate / reject ``spec.model_plan``).

    Subsets shots / receivers / time *after* obs has been generated. Lifts
    the grid-index geometry into ``sweep_io.PhysicalGeometry`` so the
    offset filter works in real meters, then drops back to grid indices.

    ModelPlan is intentionally not handled in the runner today — apply it
    upstream (crop your vp .npy and trim your geometry YAML) and re-run.
    A future commit will fold it into the runner so the resume / checkpoint
    metadata records the cropped domain. For now, a non-None ``model_plan``
    raises a clear error.

    Receiver-axis subsetting requires a uniform mask across shots (typical
    for streamers). A non-uniform mask (offset filter where each shot picks
    a different receiver set) raises a clear error.
    """
    if spec.model_plan is not None:
        raise NotImplementedError(
            "FWISpec.model_plan is accepted in the schema but not yet "
            "applied by TaskRunner — preprocess your vp arrays and "
            "geometry YAML upstream, or wait for a follow-up patch."
        )
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


def _build_inv_tensors(init_models: list, dev, equation_cls):
    """Load each ModelRef as a requires_grad=True tensor and return both
    ordered list (per equation MODEL_SPECS) and name-keyed dict."""

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
        t = _load_model_tensor(by_name[name]).to(dev).requires_grad_(True)
        in_order.append(t)
        by_tensor[name] = t
    return in_order, by_tensor, required


def _compute_loss(syn, obs, loss_spec):
    """Elementwise misfit via `sweep_loss` functional API.

    Returns the pointwise loss tensor; the caller is responsible for the
    final reduction (so multi-rank averaging stays in the runner's hands).
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


def _apply_bounds(inv_tensors_by_name, bounds_by_name) -> None:
    if not bounds_by_name:
        return
    for name, t in inv_tensors_by_name.items():
        bound = bounds_by_name.get(name)
        if bound is None:
            continue
        t.data.clamp_(min=bound.min, max=bound.max)


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
        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )
        wavelet = _build_wavelet(spec.wavelet, spec.time)
        # Shared cache so SEG-Y-backed geometry + obs don't scan the file twice.
        segy_cache: dict = {}
        sources, receivers = _build_geometry_2d(
            spec.geometry, shape, dh=spec.grid.dh, segy_cache=segy_cache,
        )
        nshots = int(sources.shape[0])

        # 4) Build inv_tensors (ordered to match equation.MODEL_SPECS).
        inv_in_order, inv_by_name, required_names = _build_inv_tensors(
            init_models, dev, equation_cls
        )
        if dist_info.is_root:
            print(f"[fwi] shape={shape} nshots={nshots} inverted_models={required_names} "
                  f"world_size={dist_info.world_size}")

        # 5) Generate or load obs. Every rank does this independently (deterministic).
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
                print(f"[fwi] stage {stage_idx} ({remaining}/{stage.epochs} epochs) "
                      f"lr_scale={stage.lr_scale} wavelet_overridden={stage.wavelet is not None}")
            for _ in range(remaining):
                loss_value = self._fwi_train_step(
                    spec, solver, stage_wavelet, sources, receivers,
                    inv_in_order, inv_by_name, obs, optimizer, nshots, dev,
                    dist_info=dist_info,
                )
                losses.append(loss_value)
                if scheduler is not None:
                    scheduler.step()
                _apply_bounds(inv_by_name, spec.model_bounds)
                if dist_info.is_root:
                    print(f"[fwi] stage {stage_idx} epoch {epoch_global:04d} loss={loss_value:.6e}")

                snapshot_now = (epoch_global % spec.show_every == 0
                                or epoch_global == total_epochs - 1)
                if snapshot_now and dist_info.is_root:
                    for name, t in inv_by_name.items():
                        np.save(snapshots_dir / f"{name}_epoch_{epoch_global:04d}.npy",
                                t.detach().cpu().numpy())
                    if spec.save_illumination:
                        _save_illumination(solver, snapshots_dir, epoch_global)

                if dist_info.is_root:
                    _save_checkpoint(task_dir, {
                        "epoch": epoch_global,
                        "models": {name: t.detach().cpu().clone()
                                   for name, t in inv_by_name.items()},
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "losses": losses,
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                    })
                epoch_global += 1

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
            # Align obs to the same dedup mask the geometry resolver used.
            # We re-snap to grid here only to recover the receiver-keep mask;
            # the cache means this is cheap on the second call.
            geom_kind = getattr(spec.geometry, "kind", None)
            dedupe = geom_kind in ("from_segy_headers", "from_segy_index")
            dedup_method = "nearest"
            if geom_kind == "from_segy_headers":
                dedupe = spec.geometry.dedupe
                dedup_method = spec.geometry.dedup_method
            _, _, obs_aligned = _segy_geometry_to_grid_indices(
                payload, float(spec.grid.dh), dedupe=dedupe, dedup_method=dedup_method,
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
            geom_kind = getattr(spec.geometry, "kind", None)
            dedupe = True
            dedup_method = "nearest"
            if geom_kind == "from_segy_index":
                dedupe = spec.geometry.dedupe
                dedup_method = spec.geometry.dedup_method
            _, _, obs_aligned = _segy_geometry_to_grid_indices(
                payload, float(spec.grid.dh), dedupe=dedupe, dedup_method=dedup_method,
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

    def _fwi_train_step(self, spec, solver, wavelet, sources, receivers,
                        inv_in_order, inv_by_name, obs, optimizer, nshots, dev,
                        *, dist_info=None) -> float:
        """One outer optimizer step.

        In single-process mode the rank picks a `batchsize` shot batch, breaks
        it into `train_shot_batchsize` chunks, runs forward+backward on each
        chunk, and steps. In distributed mode (world_size > 1) the **global**
        batchsize is split round-robin across ranks (rank 0 picks + broadcasts
        the global indices, then `shot_idx[rank::world_size]` is local). The
        loss is normalised by the GLOBAL element count so that
        ``all_reduce(SUM)`` of gradients yields the gradient of the global
        per-element mean loss.
        """

        import torch

        from sweep_runner import distributed as _dist

        if dist_info is None:
            dist_info = getattr(self, "_dist", None) or _dist.init_distributed_if_needed()

        global_batchsize = min(spec.batchsize, nshots)

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

        if spec.optimizer.kind == "lbfgs":
            # LBFGS in single-process only (the runner blocks dist + LBFGS earlier).
            def _closure():
                optimizer.zero_grad()
                acc_loss = 0.0
                for chunk in chunks:
                    syn = solver(wavelet, sources[chunk], receivers[chunk],
                                 models=inv_in_order)
                    obs_chunk = obs[chunk].to(dev)
                    loss_t = _compute_loss(syn, obs_chunk, spec.loss).sum()
                    (loss_t / global_norm).backward()
                    acc_loss += float(loss_t.detach().cpu())
                _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                return torch.tensor(acc_loss / global_norm if global_norm else 0.0)
            result = optimizer.step(_closure)
            return float(result) if result is not None else 0.0

        # Adam / SGD path: zero grads, accumulate, all-reduce, step.
        optimizer.zero_grad()
        acc_loss_local = 0.0
        for chunk in chunks:
            syn = solver(wavelet, sources[chunk], receivers[chunk],
                         models=inv_in_order)
            obs_chunk = obs[chunk].to(dev)
            loss_t = _compute_loss(syn, obs_chunk, spec.loss).sum()
            (loss_t / global_norm).backward()
            acc_loss_local += float(loss_t.detach().cpu())
        _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
        _dist.all_reduce_grad_sum(inv_in_order, dist_info)
        optimizer.step()

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
