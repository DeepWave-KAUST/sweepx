"""Pydantic v2 schemas for serialisable sweep task specs.

These specs are the contract between the CLI / YAML files / future LLM
adapters and the local TaskRunner. They mirror `PropTorch` + `Equation` API
surface and conventions from `examples/_shared/configure_*.py`.

Discriminated unions are used for sub-structures that admit several shapes:

  Wavelet  := RickerWavelet | FromNpyWavelet           (tag: kind)
  Geometry := LineGeometry  | ExplicitGeometry
                            | FromFileGeometry          (tag: kind)
  TaskSpec := IntrospectSpec | ForwardSpec | WavefieldSpec
                             | FWISpec    | LSRTMSpec   (tag: task_type)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    model_validator,
)

from sweep.propagator.options import (
    BoundaryOptions,
    CUDAOptions,
    CkptOptions,
    EagerOptions,
    MemoryOptions,
)


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----- option mirrors -----------------------------------------------------

class EagerOptionsModel(_Forbid):
    use_compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = False
    compile_backend: str | None = None
    compile_fullgraph: bool = False
    store_last_wavefield: bool = False

    def to_dataclass(self) -> EagerOptions:
        return EagerOptions(**self.model_dump())


class BoundaryOptionsModel(_Forbid):
    storage: Literal["gpu", "cpu", "disk"] = "gpu"
    transfer_interval: int | None = None
    pinned_memory: bool | None = None
    disk_dir: str | None = None
    ring_buffers: int | None = None
    disk_async_read: bool = False

    def to_dataclass(self) -> BoundaryOptions:
        return BoundaryOptions(**self.model_dump())


class CkptOptionsModel(_Forbid):
    mode: Literal["chunk", "recursive"] = "chunk"
    chunks: int = 100
    count: int = 0
    storage: Literal["gpu", "cpu"] = "gpu"
    pinned_memory: bool | None = None

    def to_dataclass(self) -> CkptOptions:
        return CkptOptions(**self.model_dump())


class MemoryOptionsModel(_Forbid):
    strategy: Literal["boundary", "ckpt"] | None = None
    boundary: BoundaryOptionsModel | None = None
    ckpt: CkptOptionsModel | None = None

    def to_dataclass(self) -> MemoryOptions:
        return MemoryOptions(
            strategy=self.strategy,
            boundary=self.boundary.to_dataclass() if self.boundary else None,
            ckpt=self.ckpt.to_dataclass() if self.ckpt else None,
        )


class CUDAOptionsModel(_Forbid):
    memory: MemoryOptionsModel | None = None

    def to_dataclass(self) -> CUDAOptions:
        return CUDAOptions(memory=self.memory.to_dataclass() if self.memory else None)


# ----- model reference ---------------------------------------------------

class ModelRef(_Forbid):
    """Reference to a model tensor by name (must appear in equation.MODEL_SPECS)."""

    name: str
    path: Path | None = None
    constant: float | None = None
    shape: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self):
        has_path = self.path is not None
        has_const = self.constant is not None
        if has_path == has_const:
            raise ValueError(
                f"ModelRef '{self.name}': exactly one of `path` or `constant` must be set."
            )
        if has_const and self.shape is None:
            raise ValueError(
                f"ModelRef '{self.name}': `shape` is required when `constant` is set."
            )
        return self


# ----- wavelet -----------------------------------------------------------

class RickerWavelet(_Forbid):
    kind: Literal["ricker"] = "ricker"
    fm: float = Field(gt=0)
    delay: float = 0.0
    scale: float = 1.0


class FromNpyWavelet(_Forbid):
    """Load wavelet samples from a .npy file. Length must equal time.nt."""

    kind: Literal["from_npy"] = "from_npy"
    path: Path
    scale: float = 1.0


Wavelet = Annotated[Union[RickerWavelet, FromNpyWavelet], Discriminator("kind")]


# ----- geometry ----------------------------------------------------------

class LineSet(_Forbid):
    """A 2D line of source/receiver points: x = arange(start, stop, step), z = depth."""

    step: int = Field(ge=1)
    depth: int = Field(ge=0)
    start: int = 0
    stop: int | None = None  # None means "use grid extent (nx)"


class LineGeometry(_Forbid):
    """Regular line of sources and receivers, generated from start/stop/step rules."""

    kind: Literal["line"] = "line"
    sources: LineSet
    receivers: LineSet


class ExplicitGeometry(_Forbid):
    """Literal source / receiver coordinate lists.

    `sources` is `(nshots, ndim)`. `receivers` is either `(nrec, ndim)` (shared
    across all shots) or `(nshots, nrec, ndim)` (per-shot). The runner
    promotes the shared form to per-shot at use time.
    """

    kind: Literal["explicit"] = "explicit"
    sources: list[list[int]]
    receivers: list

    @model_validator(mode="after")
    def _validate_shapes(self):
        if not self.sources:
            raise ValueError("ExplicitGeometry.sources must be non-empty.")
        ndim = len(self.sources[0])
        for s in self.sources:
            if len(s) != ndim:
                raise ValueError(
                    f"ExplicitGeometry.sources entries must all have ndim={ndim}."
                )
        if not self.receivers:
            raise ValueError("ExplicitGeometry.receivers must be non-empty.")
        first = self.receivers[0]
        if not isinstance(first, list):
            raise ValueError("ExplicitGeometry.receivers must be a list of lists.")
        if first and isinstance(first[0], list):
            # per-shot: validate top-level length
            if len(self.receivers) != len(self.sources):
                raise ValueError(
                    f"Per-shot receivers length {len(self.receivers)} does not match "
                    f"nshots {len(self.sources)}."
                )
        return self


class FromFileGeometry(_Forbid):
    """Load source / receiver arrays from .npy files.

    Expected shapes:
      sources_file   -> (nshots, ndim)            int64
      receivers_file -> (nrec, ndim) shared   OR  (nshots, nrec, ndim) per-shot
    """

    kind: Literal["from_file"] = "from_file"
    sources_file: Path
    receivers_file: Path


class FromSegyGeometry(_Forbid):
    """Derive sources / receivers from a **single** SEG-Y file's trace headers.

    Header byte offsets default to SEG-Y rev1; override only the keys you
    need. ``source_depth_m_override`` / ``receiver_depth_m_override`` cover
    the common marine case where the depth bytes are zero.

    Physical positions read from the file (meters) are snapped to the FWI
    grid via :class:`sweep_io.geometry.PhysicalGeometry.to_grid` at
    ``spec.grid.dh`` (with optional dedupe for coarse grids).
    """

    kind: Literal["from_segy_headers"] = "from_segy_headers"
    path: Path
    byte_map: dict[str, int] | None = None      # rev1 defaults if None
    source_depth_m_override: float | None = None
    receiver_depth_m_override: float | None = None
    coord_scalar_override: float | None = None
    dedupe: bool = True
    dedup_method: Literal["nearest", "first"] = "nearest"


class FromSegyIndexGeometry(_Forbid):
    """Derive sources / receivers from a pre-built :class:`sweep_io.SEGYIndex`.

    Use this for multi-file projects (OBN-3D-scale) where the SEG-Y header
    catalogue has been built once via ``sweep_io.segy_index.build_segy_index``
    and saved as an ``.npz``. The same index can back :class:`SegyIndexObsSpec`
    for lazy obs loading — see the matching ``obs.kind="segy_index"``.

    Geometry snapping (to the FWI grid) happens at task-build time via
    :meth:`PhysicalGeometry.to_grid` using ``spec.grid.dh``.
    """

    kind: Literal["from_segy_index"] = "from_segy_index"
    index_path: Path
    shot_ids: list[int] | None = None             # subset; default = all
    dedupe: bool = True
    dedup_method: Literal["nearest", "first"] = "nearest"


Geometry = Annotated[
    Union[LineGeometry, ExplicitGeometry, FromFileGeometry,
          FromSegyGeometry, FromSegyIndexGeometry],
    Discriminator("kind"),
]


# ----- remaining sub-structures -----------------------------------------

class TimeSpec(_Forbid):
    dt: float = Field(gt=0)
    nt: int = Field(ge=1)


class GridSpec(_Forbid):
    dh: float = Field(gt=0)
    shape: tuple[int, ...] | None = None  # inferred from first ModelRef when None


class PhysicsSpec(_Forbid):
    equation: str
    spatial_order: int = 8
    abcn: int = 20
    free_surface: bool = False
    pml_type: str = "cpmlr"
    source_type: list[str] = Field(default_factory=lambda: ["h1"])
    receiver_type: list[str] = Field(default_factory=lambda: ["h1"])


class BackendSpec(_Forbid):
    impl: Literal["eager", "c"] = "eager"
    eager_options: EagerOptionsModel | None = None
    cuda_options: CUDAOptionsModel | None = None
    use_ckpt: bool = False
    ckpt_chunks: int = 100

    @model_validator(mode="after")
    def _impl_options_consistent(self):
        if self.impl == "eager" and self.cuda_options is not None:
            raise ValueError("cuda_options is only valid when impl='c'.")
        if self.impl == "c" and self.eager_options is not None:
            raise ValueError("eager_options is only valid when impl='eager'.")
        return self


# ----- loss, bounds, stages, optimizer, scheduler ------------------------

class LossSpec(_Forbid):
    """Misfit between synthetic and observed seismograms."""

    kind: Literal["mse", "l1", "huber"] = "mse"
    huber_delta: float = 1.0  # only used when kind="huber"


class DataPlanSpec(_Forbid):
    """Pre-FWI data selection: which shots / receivers / time samples to use.

    All fields are optional; omitted ones inherit the full dataset. Maps 1:1
    to :class:`sweep_io.plan.DataPlan`. Pure positional / sampling logic;
    signal-processing knobs (filter / mute / wavelet) stay in sweep-preproc.
    """

    shot_start: int = 0
    shot_stop: int | None = None
    shot_stride: int = Field(ge=1, default=1)
    shot_indices: list[int] | None = None  # overrides start/stop/stride

    receiver_stride: int = Field(ge=1, default=1)
    offset_min_m: float | None = None
    offset_max_m: float | None = None
    abs_offset: bool = True

    # Mutually exclusive time-axis subsetters (validated in the runner).
    dt_target_s: float | None = None
    time_decimate: int | None = Field(default=None, ge=1)

    t_start_s: float | None = None
    t_end_s: float | None = None


class ModelPlanSpec(_Forbid):
    """Crop the velocity model to a region of interest; drop out-of-window shots.

    Maps 1:1 to :class:`sweep_io.plan.ModelPlan`. Receivers outside the
    window are kept by default (sweep's PML absorbs them) — set
    ``drop_outside_receivers=True`` only if your acquisition truly stops
    at the model edge.
    """

    x_window_m: tuple[float, float] | None = None
    y_window_m: tuple[float, float] | None = None
    z_window_m: tuple[float, float] | None = None
    drop_outside_sources: bool = True
    drop_outside_receivers: bool = False


class ModelBounds(_Forbid):
    """Hard bounds applied via in-place clamp after every optimizer step."""

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _at_least_one(self):
        if self.min is None and self.max is None:
            raise ValueError("ModelBounds: at least one of min/max must be set.")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(f"ModelBounds: min ({self.min}) must be < max ({self.max}).")
        return self


class StageBandpass(_Forbid):
    """Per-stage bandpass applied to obs at stage entry (Gap 5).

    ``sweep_preproc.filter.bandpass`` is invoked on the *pristine* obs each
    time a new stage starts, so stages don't compose their filters.
    """

    lo_hz: float = Field(gt=0)
    hi_hz: float = Field(gt=0)
    order: int = Field(ge=1, default=4)


class StageSpec(_Forbid):
    """One leg of a multi-stage FWI run (frequency continuation pattern).

    Fields beyond ``epochs`` are all optional and override the top-level
    defaults when present:

    - ``wavelet``: replace the source signature for this stage
    - ``lr_scale``: scale the optimizer's initial lr (multiplicative)
    - ``dh_m``: rebuild solver + resample vp to this grid spacing
    - ``dt_s`` / ``nt``: rebuild solver at a different time grid
    - ``batch_size``: per-stage shot batch (overrides FWISpec.batchsize)
    - ``bandpass``: filter obs before this stage runs (uses sweep-preproc)
    """

    epochs: int = Field(ge=1)
    wavelet: "Wavelet | None" = None
    lr_scale: float = Field(gt=0, default=1.0)
    dh_m: float | None = None
    dt_s: float | None = None
    nt: int | None = None
    batch_size: int | None = Field(default=None, ge=1)
    bandpass: StageBandpass | None = None


class OptimizerAdam(_Forbid):
    kind: Literal["adam"] = "adam"
    # Scalar lr, or per-model-name lr (dict keys must match the inverted ModelRef names).
    lr: float | dict[str, float]
    eps: float = 1e-22
    betas: tuple[float, float] = (0.9, 0.999)


class OptimizerSGD(_Forbid):
    kind: Literal["sgd"] = "sgd"
    lr: float | dict[str, float]
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0


class OptimizerLBFGS(_Forbid):
    kind: Literal["lbfgs"] = "lbfgs"
    lr: float = 1.0  # LBFGS uses a single shared lr (no per-model splits)
    max_iter: int = 20
    history_size: int = 10
    line_search_fn: Literal["strong_wolfe"] | None = None


Optimizer = Annotated[
    Union[OptimizerAdam, OptimizerSGD, OptimizerLBFGS],
    Discriminator("kind"),
]

# Backward-compat alias for code that imported OptimizerSpec before discriminated union.
OptimizerSpec = OptimizerAdam


class SchedulerConstant(_Forbid):
    kind: Literal["constant"] = "constant"


class SchedulerStep(_Forbid):
    kind: Literal["step"] = "step"
    step_size: int = Field(ge=1)
    gamma: float = Field(gt=0, default=0.5)


class SchedulerExp(_Forbid):
    kind: Literal["exp"] = "exp"
    gamma: float = Field(gt=0, lt=1.0)


class SchedulerCosine(_Forbid):
    kind: Literal["cosine"] = "cosine"
    eta_min: float = 0.0
    t_max: int | None = None  # default = total epochs


Scheduler = Annotated[
    Union[SchedulerConstant, SchedulerStep, SchedulerExp, SchedulerCosine],
    Discriminator("kind"),
]


class ObsSegyConfig(_Forbid):
    """Single-file SEG-Y observed-data loader (Option A).

    The runner opens the file once, scans all trace headers in-process, builds
    a :class:`sweep_io.geometry.PhysicalGeometry` (meters) snapped to
    ``spec.grid.dh``, and reads the trace payloads in one coalesced pass via
    :class:`sweep_io.segy.SEGYReader`. Suitable for single-SEG-Y datasets up
    to ~few GB (Viking line 12: 750 MB → ~8 s scan + read).

    For multi-file or TB-scale data, use :class:`ObsSegyIndexConfig` instead.
    """

    path: Path
    byte_map: dict[str, int] | None = None
    source_depth_m_override: float | None = None
    receiver_depth_m_override: float | None = None
    coord_scalar_override: float | None = None
    shot_ids: list[int] | None = None       # subset; default = all


class ObsSegyIndexConfig(_Forbid):
    """Multi-file lazy SEG-Y observed-data loader (Option B).

    Backed by a pre-built :class:`sweep_io.segy_index.SEGYIndex` (.npz).
    The runner loads the index, and either materialises obs eagerly (small
    surveys) or wraps an :class:`IndexedShotGatherDataset` with prefetch
    (when ``lazy=True``).
    """

    index_path: Path
    shot_ids: list[int] | None = None
    coalesce_gap: int = 0
    lazy: bool = False
    # NOTE: when ``lazy=True``, the runner uses an IndexedShotGatherDataset
    # with a Prefetcher. The eager path materialises a single (nshots, nrec,
    # nt) tensor at task start — easier on small datasets, OOM-risky on huge.


class ObsSpec(_Forbid):
    """How to get observed data. Pick exactly one source.

    Available sources:
      - ``synthetic_from`` — re-run forward modeling on a "true" vp.
      - ``synthetic_from_models`` — same, for multi-model equations.
      - ``npy_path`` — a pre-saved ``(nshots, nrec, nt)`` ``.npy``.
      - ``segy`` — load straight from a single SEG-Y file (Option A).
      - ``segy_index`` — load from a multi-file SEG-Y index (Option B).
    """

    synthetic_from: ModelRef | None = None
    synthetic_from_models: list[ModelRef] | None = None
    npy_path: Path | None = None
    segy: ObsSegyConfig | None = None
    segy_index: ObsSegyIndexConfig | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self):
        choices = [
            ("synthetic_from", self.synthetic_from),
            ("synthetic_from_models", self.synthetic_from_models),
            ("npy_path", self.npy_path),
            ("segy", self.segy),
            ("segy_index", self.segy_index),
        ]
        set_choices = [name for name, value in choices if value is not None]
        if len(set_choices) != 1:
            raise ValueError(
                "ObsSpec: exactly one of "
                "synthetic_from / synthetic_from_models / npy_path / "
                f"segy / segy_index must be set; got {set_choices}."
            )
        return self


class ModelingOverride(_Forbid):
    """Selective overrides applied only when synthesising obs (FWI/LSRTM).

    Useful for "same true model, different modeling than inversion" workflows:
      * low-pass / shaped wavelet for obs
      * slightly different source / receiver positions (must keep nshots and
        nrec_per_shot identical to the inversion geometry)

    Overriding grid / time / physics is not supported in Phase 1: those
    changes require model / data resampling that the runner does not perform.
    """

    wavelet: Wavelet | None = None
    geometry: Geometry | None = None

    @model_validator(mode="after")
    def _at_least_one_field(self):
        if self.wavelet is None and self.geometry is None:
            raise ValueError(
                "ModelingOverride: must override at least one of wavelet / geometry."
            )
        return self


# ----- task specs ---------------------------------------------------------

class BaseTaskSpec(_Forbid):
    task_id: str | None = None  # auto-generated from timestamp if missing
    # NOTE: default is `./sweep_runs`, NOT `./sweep_tasks`. The latter would
    # collide with this package's import name `sweep_tasks` and turn run-output
    # directories into PEP-420 namespace packages that shadow real imports
    # (`python -c "import sweep_tasks"` would pick up the rogue dir before the
    # installed package). Pick any other name when overriding.
    output_dir: Path = Path("./sweep_runs")
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N"


class IntrospectSpec(BaseTaskSpec):
    task_type: Literal["introspect"] = "introspect"
    action: Literal[
        "list_equations",
        "describe_equation",
        "describe_field",
        "describe_model",
        "list_supported_bindings",
    ]
    target: str | None = None  # equation name for describe_*
    field_or_model: str | None = None  # field or model name


class ForwardSpec(BaseTaskSpec):
    task_type: Literal["forward"] = "forward"
    grid: GridSpec
    time: TimeSpec
    wavelet: Wavelet
    geometry: Geometry
    physics: PhysicsSpec
    backend: BackendSpec = Field(default_factory=BackendSpec)
    models: list[ModelRef]


class WavefieldSpec(BaseTaskSpec):
    task_type: Literal["wavefield"] = "wavefield"
    grid: GridSpec
    time: TimeSpec
    wavelet: Wavelet
    geometry: Geometry
    physics: PhysicsSpec
    backend: BackendSpec = Field(default_factory=BackendSpec)
    models: list[ModelRef]
    snapshot_times: list[int] = Field(min_length=1)
    plot: bool = True


class FWISpec(BaseTaskSpec):
    task_type: Literal["fwi"] = "fwi"
    grid: GridSpec
    time: TimeSpec
    wavelet: Wavelet
    geometry: Geometry
    physics: PhysicsSpec
    backend: BackendSpec = Field(default_factory=BackendSpec)

    # Acoustic FWI uses init_model (single ModelRef). Multi-model equations
    # (Elastic, etc.) use init_models — a list ordered to match the equation's
    # MODEL_SPECS. Exactly one must be set; the runner normalises to a list.
    init_model: ModelRef | None = None
    init_models: list[ModelRef] | None = None

    obs: ObsSpec
    optimizer: Optimizer
    scheduler: Scheduler = Field(default_factory=SchedulerConstant)
    loss: LossSpec = Field(default_factory=LossSpec)
    epochs: int = Field(ge=1)
    batchsize: int = Field(ge=1, default=1)
    train_shot_batchsize: int | None = None  # default = batchsize (no accumulation)
    show_every: int = 10

    # Phase-1 patch fields:
    model_bounds: dict[str, ModelBounds] = Field(default_factory=dict)
    freeze_top_n_rows: int = Field(ge=0, default=0)
    stages: list[StageSpec] | None = None
    resume_from: str | None = None  # task_id under output_dir to resume from
    save_illumination: bool = False

    modeling_override: ModelingOverride | None = None

    # Pre-FWI data selection + model cropping (TASK 009).
    # `data_plan` subsets shots/receivers/time before the inversion sees obs.
    # `model_plan` crops the velocity model to a region of interest and
    # (by default) drops sources whose physical positions land outside.
    # When omitted, the full dataset / full model is used (current behaviour).
    data_plan: DataPlanSpec | None = None
    model_plan: ModelPlanSpec | None = None

    @model_validator(mode="after")
    def _exactly_one_init(self):
        has_single = self.init_model is not None
        has_list = self.init_models is not None
        if has_single == has_list:
            raise ValueError(
                "FWISpec: exactly one of init_model / init_models must be set."
            )
        if has_list and not self.init_models:
            raise ValueError("FWISpec.init_models must be non-empty.")
        if self.stages is not None and len(self.stages) == 0:
            raise ValueError("FWISpec.stages must be non-empty when set.")
        return self


class LSRTMSpec(BaseTaskSpec):
    """LSRTM uses two solvers: AcousticLSRTM (foreground) and Acoustic (background).

    `physics.equation` names the LSRTM variant. The runner derives the matching
    background equation (e.g. AcousticLSRTM -> Acoustic) automatically.
    """

    task_type: Literal["lsrtm"] = "lsrtm"
    grid: GridSpec
    time: TimeSpec
    wavelet: Wavelet
    geometry: Geometry
    physics: PhysicsSpec
    backend: BackendSpec = Field(default_factory=BackendSpec)
    background_model: ModelRef
    true_model: ModelRef
    optimizer: Optimizer
    scheduler: Scheduler = Field(default_factory=SchedulerConstant)
    loss: LossSpec = Field(default_factory=LossSpec)
    epochs: int = Field(ge=1)
    batchsize: int = Field(ge=1, default=1)
    train_shot_batchsize: int | None = None
    show_every: int = 10

    # Phase-1 patch fields (the inverted parameter is reflectivity, a single tensor):
    reflectivity_bounds: ModelBounds | None = None
    freeze_top_n_rows: int = Field(ge=0, default=0)
    stages: list[StageSpec] | None = None
    resume_from: str | None = None
    save_illumination: bool = False

    modeling_override: ModelingOverride | None = None

    @model_validator(mode="after")
    def _validate_stages(self):
        if self.stages is not None and len(self.stages) == 0:
            raise ValueError("LSRTMSpec.stages must be non-empty when set.")
        return self


TaskSpec = Annotated[
    Union[IntrospectSpec, ForwardSpec, WavefieldSpec, FWISpec, LSRTMSpec],
    Discriminator("task_type"),
]
