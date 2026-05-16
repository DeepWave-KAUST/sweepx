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
    field_validator,
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

    kind: Literal["mse", "l1", "huber", "trace_cosine"] = "mse"
    huber_delta: float = 1.0  # only used when kind="huber"
    # trace_cosine: per-trace amplitude-normalised correlation misfit, equivalent
    # to ``1 - <s_unit, o_unit>`` after demeaning. Matches the loss used in
    # `fwi_workflow-dev`. Insensitive to per-trace amplitude scaling, so it's
    # robust to source-wavelet errors. The optional ``trace_cosine_demean`` flag
    # controls whether each trace's mean is subtracted before normalisation
    # (matches fwi_workflow-dev's behavior).
    trace_cosine_demean: bool = True
    trace_cosine_eps: float = 1.0e-8


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


class ReparamHashSpec(_Forbid):
    """Multi-resolution hash-grid encoder hyperparameters (Instant-NGP)."""

    enabled: bool = True
    levels: int = Field(ge=1, default=16)
    features_per_level: int = Field(ge=1, default=2)
    log2_size: int = Field(ge=1, default=15)
    base_resolution: int = Field(ge=1, default=4)
    finest_resolution: int = Field(ge=1, default=512)


class LocalModelWindowSpec(_Forbid):
    """Per-batch local-window FWI (Engquist-style domain decomposition).

    For each forward batch, compute a rectangular crop of the velocity
    model that tightly contains the batch's sources + receivers plus
    padding. The wave solver runs on this crop (typically 5-10× smaller
    than the full model in x for marine data), and PyTorch's view
    slicing automatically scatters the gradient back to the full vp
    tensor — no manual ``scatter_add`` needed.

    Cost: a brand-new solver is built for every distinct window shape.
    Solvers are cached by shape inside the runner state so repeated
    shapes (which is the common case at a given stage's dh) reuse
    the cached propagator.

    Mirrors ``use_local_model_windows`` / ``local_model_padding_x_m`` /
    ``local_model_padding_z_m`` / ``local_model_full_depth`` /
    ``local_model_min_width_m`` in ``fwi_workflow-dev``.
    """

    enabled: bool = True
    padding_x_m: float = Field(ge=0, default=1500.0)
    padding_z_m: float = Field(ge=0, default=0.0)
    full_depth: bool = True
    min_width_m: float = Field(ge=0, default=0.0)


class QCSpec(_Forbid):
    """Optional QC products generated during FWI inversion.

    All outputs land under ``<task_dir>/qc/<kind>/iter_NNNN.png``. By
    default everything is off (set ``every_n_epochs > 0`` to enable).

    ``every_n_epochs``
        Cadence: emit QC every N global epochs. Set to 0 to disable
        per-epoch QC entirely. The final-epoch QC is always emitted
        when any plot is enabled.

    ``vp_png``, ``vp_diff_png``
        Velocity model snapshot + Δvp vs initial. Cheap (no extra
        forward pass).

    ``gradient_png``
        FWI gradient ``∂loss/∂vp`` as a percentile-clipped diverging map.
        Grid mode only — in reparam mode the gradient lives on the
        network parameters, not the vp tensor, so this is skipped with
        a warning.

    ``shot_gather``
        Observed vs synthetic shot gathers, side-by-side. Triggers ONE
        extra forward pass per QC cadence to capture syn (cheap if
        ``shot_gather_n_shots`` is small). Set ``shot_gather_n_shots`` to
        control how many shots are rendered.

    ``loss_curve``
        Enhanced loss curve with per-stage shading + log-scale toggle.
        Emitted once at end-of-run only (cheap, but cadence-controlled
        loss snapshots are redundant with the runner's built-in loss.png).
    """

    every_n_epochs: int = Field(ge=0, default=10)
    vp_png: bool = True
    vp_diff_png: bool = True
    gradient_png: bool = False
    shot_gather: bool = False
    shot_gather_n_shots: int = Field(ge=1, default=1)
    shot_gather_perc: float = Field(gt=0, lt=100, default=99.0)
    # ``trace`` (default): divide each trace by its own RMS — obs and
    # syn visually comparable even when wavelet amplitudes don't match
    # (typical with trace_cosine + estimated wavelet). ``shot``: legacy
    # per-shot percentile (obs and syn independently scaled). ``joint``:
    # common percentile across both panels (quantitative; large-amplitude
    # obs may saturate while syn shows up faint).
    shot_gather_normalize: Literal["trace", "shot", "joint"] = "trace"
    # Interleaved-display block size for the rich shot_gather layout: each
    # block of N consecutive (unique-cell) traces alternates obs / syn /
    # obs / ..., so amplitudes can be eyeballed at matching x. Matches
    # ``fwi_workflow-dev``'s ``interleave_block`` (their default 64; here
    # smaller (12) because we plot one shot per row instead of multi-shot).
    shot_gather_interleave_block: int = Field(ge=1, default=12)
    loss_curve: bool = True


class ReparamSpec(_Forbid):
    """Neural-network reparameterization of the velocity model (sweep-nn).

    When set on an FWISpec, the runner replaces the raw vp tensor with a
    :class:`sweep_nn.VelocityINR` (hash-encoded SIREN by default). The
    optimizer is built on the network's parameters instead of the vp
    tensor; multi-stage transitions resample only the network's *base*
    velocity, preserving all learnable parameters (SIREN's multi-scale
    benefit).

    Currently only applies to the ``vp`` model; multi-parameter equations
    fall back to raw tensors for the non-vp models.
    """

    kind: Literal["velocity_inr"] = "velocity_inr"
    hidden_features: int = Field(ge=1, default=64)
    hidden_layers: int = Field(ge=1, default=3)
    first_omega0: float = Field(gt=0, default=30.0)
    hidden_omega0: float = Field(gt=0, default=30.0)
    use_bias: bool = False
    vp_mean: float = 0.0
    vp_std: float = Field(gt=0, default=50.0)
    direct_velocity: bool = False
    coord_min: float = 0.0
    coord_max: float = 1.0
    hash: ReparamHashSpec = Field(default_factory=ReparamHashSpec)
    # Optimizer lr override for the network (the top-level optimizer.lr is
    # ignored when reparam is active, since grid-FWI lr ~25 is wildly wrong
    # for SIREN/hash parameters ~1e-4).
    lr: float = Field(gt=0, default=1.0e-4)

    # Backward path for the reparam network. Matches fwi_workflow-dev's
    # ``inr_backward_mode`` config:
    #   * ``"two_pass_full"`` (default, matches the reference) — render
    #     under no_grad into a leaf tensor, run the wave-solver autograd
    #     forward/backward against the leaf, then push the leaf's
    #     gradient through the network in a *second* full-graph backward.
    #     Decouples the solver and network autograd graphs so the solver
    #     phase doesn't carry the network's activations.
    #   * ``"two_pass_chunked"`` — same two-pass split, but the second
    #     pass re-renders the network in row chunks of
    #     ``backward_chunk_rows`` and backwards each chunk separately,
    #     freeing the chunk's graph in between. Memory O(chunk_rows*nx),
    #     bigger compute overhead.
    #   * ``"single_step"`` — one combined autograd graph through
    #     solver+net (the original sweep-tasks behavior). Simplest but
    #     can OOM at large network or large grid because the solver's
    #     wavefield activations are pinned for the network backward.
    backward_mode: Literal["two_pass_full", "two_pass_chunked", "single_step"] = "two_pass_full"
    backward_chunk_rows: int = Field(ge=1, default=64)


class StageBandpass(_Forbid):
    """Per-stage bandpass applied to obs at stage entry (Gap 5).

    ``sweep_preproc.filter.bandpass`` is invoked on the *pristine* obs each
    time a new stage starts, so stages don't compose their filters.

    ``order``
        Prototype Butterworth order, matching ``filter_order`` in
        ``fwi_workflow-dev``. The zero-phase ``sosfiltfilt`` pass gives
        an effective magnitude response of ``|H_N(f)|²`` (``2N × 6 dB/oct``
        stop-band roll-off). Default ``4``.
    ``padtype``
        Padding policy passed to ``sosfiltfilt``. Default ``"odd"``
        (scipy reflective padding) keeps edge transients near machine
        precision. Set ``None`` to disable padding and match the
        ``torchaudio.functional.filtfilt`` behavior used inside
        ``fwi_workflow-dev``'s GPU path (expect visible edge transients
        across the whole trace). Allowed: ``"odd"``, ``"even"``,
        ``"constant"``, or ``None``.
    """

    lo_hz: float = Field(gt=0)
    hi_hz: float = Field(gt=0)
    order: int = Field(ge=1, default=4)
    padtype: Literal["odd", "even", "constant"] | None = "odd"
    # What to filter at this stage. ``"syn"`` (default, matches
    # ``fwi_workflow-dev``): bandpass obs at stage entry + bandpass syn
    # before the loss (via differentiable ``torchaudio.functional.filtfilt``).
    # ``"wavelet"``: bandpass obs + bandpass the source wavelet once at
    # stage entry — syn is naturally bandlimited (all energy comes from
    # the filtered source) and is NOT re-filtered. This skips the per-
    # iteration syn filter in the autograd path and is roughly
    # equivalent to ``"syn"`` if the solver is linear, but cheaper.
    target: Literal["syn", "wavelet"] = "syn"


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

    # Optional NN reparameterization of vp (sweep-nn VelocityINR).
    # When set, vp is rendered by a hash-encoded SIREN each forward; the
    # optimizer trains the network's parameters instead of the raw tensor.
    reparam: ReparamSpec | None = None

    # Optional QC artefacts (vp / gradient / shot gather PNGs etc.)
    # under <task_dir>/qc/. See :class:`QCSpec` for the catalog.
    qc: QCSpec | None = None

    # Optional per-batch local model windowing (Engquist-style). See
    # :class:`LocalModelWindowSpec`. When omitted, every forward runs on
    # the full stage grid (current behaviour).
    #
    # Three YAML shorthands are accepted:
    #   * omit or ``false``  -> off (default)
    #   * ``true``           -> on with all default parameters
    #   * dict / object      -> on, overriding selected fields
    local_model_window: LocalModelWindowSpec | None = None

    @field_validator("local_model_window", mode="before")
    @classmethod
    def _local_model_window_bool_shortcut(cls, v):
        # ``true`` / ``false`` shorthands for the most common toggle. Anything
        # else (None, dict, LocalModelWindowSpec instance) falls through
        # to the normal pydantic parsing.
        if v is True:
            return LocalModelWindowSpec()
        if v is False:
            return None
        return v

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
