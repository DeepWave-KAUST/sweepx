"""Pydantic v2 schemas for serialisable sweep task specs.

These specs are the contract between the CLI / YAML files / future LLM
adapters and the local TaskRunner. They mirror `PropTorch` + `Equation` API
surface and conventions from `examples/_shared/configure_*.py`.

Discriminated unions are used for sub-structures that admit several shapes:

  Wavelet  := RickerWavelet | FromNpyWavelet           (tag: kind)
  Geometry := LineGeometry  | ExplicitGeometry
                            | FromFileGeometry          (tag: kind)
  TaskSpec := IntrospectSpec | ForwardSpec | WavefieldSpec
                             | FWISpec    | LSRTMSpec
                             | RTMSpec                  (tag: task_type)
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
    # Boundary ring-buffer storage precision. fp16/bf16/int8 shrink the saved
    # boundary buffer (compute + seed frame stay fp32). Maps to
    # sweep BoundaryOptions.storage_dtype -> C boundary_cfg['storage_dtype'].
    storage_dtype: Literal["fp32", "fp16", "bf16", "int8"] = "fp32"
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


class FromSirenPipelineNpzWavelet(_Forbid):
    """Load wavelet samples from a SIREN-pipeline ``.npz`` container.

    Auto-detects the array key (one of ``"wavelet"``,
    ``"optimized_siren_wavelet"``, ``"direct_causal_wavelet"``,
    ``"initial_causal_wavelet"``) and the sample interval (``dt_s`` scalar
    or ``time_s`` array). The runner resamples to the solver's ``dt`` when
    they differ and truncates / zero-pads to length ``time.nt``. Mirrors
    the loader at :func:`sweep_io.wavelet.load_wavelet_npz`.

    ``explicit_key`` forces one of the listed keys (helpful when the
    container has both an initial and an optimised wavelet and you want
    the initial one explicitly).
    """

    kind: Literal["siren_pipeline_npz"] = "siren_pipeline_npz"
    path: Path
    explicit_key: str | None = None
    scale: float = 1.0


Wavelet = Annotated[
    Union[RickerWavelet, FromNpyWavelet, FromSirenPipelineNpzWavelet],
    Discriminator("kind"),
]


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

    Use this for multi-file projects (production OBN-scale) where the SEG-Y header
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


class FromPlanGeometry(_Forbid):
    """Geometry derived from a unified ``seismic_plan_v1`` cache.

    The plan (built via ``sweep-tasks build-plan``) declares the trace
    catalog + grouping (CSG for streamer-style shot gathers, CRG for
    OBN receiver gathers). The geometry resolver reads its
    ``row_source_xyz`` / ``row_receiver_xyz`` arrays and snaps them to
    the FWI grid via ``spec.grid.dh``.

    For ``grouping="csg"`` (typical 2-D / 3-D streamer): each shot's
    receiver layout is taken from the rows of that group; the runner
    requires a uniform receiver count per shot. Use
    ``geometry.dedupe=True`` to grid-snap-dedupe per shot when the
    layout varies across shots.

    For ``grouping="crg"`` (3-D OBN): pair with
    :class:`ObsPlanConfig.sampling` to drive the multisource supershot
    loop. ``rotation_metadata`` + ``dh_xyz_m`` + ``grid_origin_xyz_m`` /
    ``auto_origin_pad_cells`` then project UTM xyz into the propagator's
    axis-aligned model frame.
    """

    kind: Literal["from_plan"] = "from_plan"
    plan_path: Path
    dedupe: bool = True
    dedup_method: Literal["nearest", "first"] = "nearest"

    # --- 3-D / CRG fields (None on 2-D CSG, set on OBN runs) -----------
    # Path to the rotation_metadata.json that defines the UTM → model
    # frame transform. Only required when the underlying plan stores UTM
    # coordinates that need rotation into an axis-aligned grid.
    rotation_metadata: Path | None = None
    # Optional per-axis grid spacing in metres. When None (default) the
    # runner uses ``spec.grid.dh`` for every axis. Set on anisotropic
    # grids (e.g. ``(dz=25, dy=75, dx=75)``).
    dh_xyz_m: tuple[float, float, float] | None = None
    # The UTM coordinate that maps to grid index 0 on each axis. When
    # None (default), the runner picks the bounding box of the rotated
    # source + receiver positions with optional padding.
    grid_origin_xyz_m: tuple[float, float, float] | None = None
    # Padding around the snap origin in cells (z, y, x). Used only when
    # ``grid_origin_xyz_m`` is None.
    auto_origin_pad_cells: tuple[int, int, int] = (0, 0, 0)


Geometry = Annotated[
    Union[LineGeometry, ExplicitGeometry, FromFileGeometry,
          FromSegyGeometry, FromSegyIndexGeometry, FromPlanGeometry],
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
    # Irregular free-surface topography for curvilinear-grid equations
    # (AcousticCurvilinear / ElasticCurvilinear). Path to a 1-D .npy of
    # per-column surface row indices (length nx; topo[ix] = grid row of the
    # surface at column ix). PropTorch builds the boundary-fitted grid from it.
    # Distinct from the flat ``free_surface`` flag; None = no topo.
    topography: Path | None = None


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
    # Optional per-sample DATA mute mask (diving-wave window etc.). Path to a
    # .npy broadcastable to obs (nshots, nt, nrec[, nchan]). When set, the
    # data misfit is multiplied by it (window outside -> excluded); None = off
    # (the FWI/LSRTM misfit then runs exactly as before).
    data_mask_path: str | None = None


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
    """Hard bounds applied via in-place clamp after every optimizer step.

    Set ``enabled: false`` to keep the min/max values documented in the
    YAML but suppress the clamp at runtime (and also suppress consumers
    like the QC vp colormap range that read these bounds). Useful for
    A/B-ing bounded vs unbounded inversion without restructuring the
    config.
    """

    enabled: bool = True
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _at_least_one(self):
        if not self.enabled:
            return self
        if self.min is None and self.max is None:
            raise ValueError("ModelBounds: at least one of min/max must be set when enabled.")
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

    Works for both 2-D ``(nz, nx)`` and 3-D ``(nz, ny, nx)`` grids. In
    3-D the crop tightly encloses the batch in the lateral plane and is
    padded by ``padding_x_m`` on x and ``padding_y_m`` on y (falling
    back to ``padding_x_m`` when ``padding_y_m=None``); ``full_depth``
    keeps the entire z column by default. Set ``batchsize=1`` upstream
    to get a per-shot crop instead of per-batch.

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
    padding_y_m: float | None = Field(ge=0, default=None)
    padding_z_m: float = Field(ge=0, default=0.0)
    full_depth: bool = True
    min_width_m: float = Field(ge=0, default=0.0)


class SmoothRegSpec(_Forbid):
    """First / second-derivative smoothness penalty on vp (TV-style prior).

    Adds a scalar regularizer ``weight * TVPrior(order, x/y/z weights)(vp)``
    to the per-iter loss; the gradient flows back into the velocity (or
    network params in reparam mode). See :class:`sweep_nn.TVPrior` for
    the math and conventions (in particular ``velocity_scale_m_s``).

    The weight is in the same scale as the data misfit; values around
    ``1e-4`` to ``1e-2`` are typical for cosine misfit + vp ~2500 m/s.
    """

    weight: float = Field(ge=0, default=0.0)
    order: Literal["first", "second", "both", "mixed", "first_second"] = "first"
    x_weight: float = Field(ge=0, default=1.0)
    y_weight: float = Field(ge=0, default=1.0)
    z_weight: float = Field(ge=0, default=1.0)
    velocity_scale_m_s: float = Field(gt=0, default=1000.0)


class FreezeWaterLayerSpec(_Forbid):
    """Mask the FWI gradient above the seabed (water column).

    Reads a per-trace seabed-depth map (``(nx,)`` for 2-D or ``(ny, nx)``
    for 3-D) from ``seabed_depth_path``, builds the mask once at task
    setup, and multiplies it into the vp gradient after backward. See
    :class:`sweep_nn.SeabedFreezeMask`.

    The npz must contain a single array under the key ``seabed_depth``,
    in meters from grid index z=0. Use ``buffer_cells`` to freeze a few
    extra rows below the seabed when the source wavelet's rise time
    leaks across the interface.
    """

    enabled: bool = False
    seabed_depth_path: Path | None = None
    buffer_cells: int = Field(ge=0, default=0)

    @model_validator(mode="after")
    def _path_required_when_enabled(self):
        if self.enabled and self.seabed_depth_path is None:
            raise ValueError(
                "FreezeWaterLayerSpec: seabed_depth_path is required when enabled=True"
            )
        return self


class FreqSelectionSpec(_Forbid):
    """Frequency-selection (steady-state comb) encoding parameters.

    Every node of the active pool continuously emits ONE exclusive DFT-comb
    bin; the last ``probe_samples`` record samples form an integer-period
    window in which bins are orthogonal, so the window DFT separates nodes
    exactly (deterministic zero crosstalk). Wavelet-free: the GCN loss is
    invariant to any per-node complex scale, so no wavelet spec, no obs
    prepad, no bandpass filter and no shared-shots sampler exist in this
    mode. Observed data are pre-extracted DTFT coefficients
    (:mod:`sweep_tasks.freqsel`), fold-averaged onto grid cells — the
    inversion performs zero SEG-Y I/O.

    ``coeff_shards`` is a glob for the extraction npz shard(s). For
    synthetic tests set ``synthesize_from_true=True`` instead: the runner
    forward-models the shots through ``model_true`` once at setup, DTFTs
    them at the comb and writes the shard to the task dir (then inverts
    from it exactly like the field path).

    Comb: bins ``k_lo..k_hi`` of the window, i.e. frequencies
    ``k / (probe_samples * dt)``. Record length must be
    ``steady_samples + slack_samples + probe_samples``; the leading
    ``steady_samples`` let transients decay (validate with the two-window
    QC printed at setup), ``slack_samples`` gives the QC a second window.
    """

    coeff_shards: str | None = None
    synthesize_from_true: bool = False
    # synthetic-test layout (used only with synthesize_from_true): path of
    # the true model npy (target grid), number of nodes on the surface line
    # grid, receiver stride in cells, and node depth in cells.
    true_model_path: str | None = None
    synth_n_nodes: int = Field(gt=0, default=16)
    synth_rec_stride: int = Field(gt=0, default=2)
    synth_node_z: int = Field(ge=0, default=2)
    probe_samples: int = Field(gt=0, default=6000)
    k_lo: int = Field(gt=0, default=49)
    k_hi: int = Field(gt=0, default=96)
    steady_samples: int = Field(gt=0, default=2500)
    slack_samples: int = Field(ge=0, default=500)
    ramp_s: float = Field(gt=0, default=0.5)
    n_pools: int = Field(gt=0, default=14)
    eps: float = Field(gt=0, default=1.0e-12)

    @model_validator(mode="after")
    def _one_source(self):
        if bool(self.coeff_shards) == bool(self.synthesize_from_true):
            raise ValueError(
                "FreqSelectionSpec: set exactly one of coeff_shards / "
                "synthesize_from_true")
        return self


class SourceEncodingSpec(_Forbid):
    """Random ±1 source encoding for single-supershot OBN FWI.

    When enabled, each iteration:

    1. Picks ``batchsize`` virtual sources (= OBN nodes) that share a
       common set of physical-shot receivers (via the CRG dataset's
       shared-shots sampler).
    2. Draws a random ``signs ∈ {-1, +1}^batchsize`` vector.
    3. Builds an encoded supershot: wavelet shape becomes
       ``(1, batchsize, nt)`` (each virtual source carries the same
       wavelet times its sign), receivers shape ``(1, n_rec, 3)``, and
       obs is the signed sum ``Σ_i sign_i · obs_i`` of shape
       ``(1, n_rec, nt)``.
    4. Calls the propagator once with ``source_encoding=True``.

    One forward + adjoint per iter regardless of ``batchsize``: the OBN
    1-GPU production path. Re-seeding the signs every iter (the default)
    de-correlates the cross-talk and keeps the long-run gradient close
    to the per-shot expectation.

    Pairs with :class:`PlanSamplingConfig` on ``obs.plan.sampling``
    (``shared_shots_per_iter > 0``); has no effect under the per-shot
    CRG mode.
    """

    enabled: bool = False
    min_coverage: int = Field(ge=0, default=0)
    sign_seed: int | None = None
    reseed_every_iter: bool = True
    # "random": the ±1 path above. "frequency_selection": deterministic
    # frequency-division comb encoding (see FreqSelectionSpec); requires
    # ``frequency`` to be set. Default keeps existing YAMLs unchanged.
    mode: Literal["random", "frequency_selection"] = "random"
    frequency: FreqSelectionSpec | None = None

    @model_validator(mode="after")
    def _freqsel_requires_spec(self):
        if self.mode == "frequency_selection" and self.frequency is None:
            raise ValueError(
                "source_encoding.mode='frequency_selection' requires the "
                "'frequency' sub-spec")
        return self


class IlluminationPreconditionSpec(_Forbid):
    """Diagonal pseudo-Hessian illumination preconditioner applied per step.

    When enabled, after each ``_fwi_train_step``'s backward (and after the
    distributed all-reduce, if any), the FWI gradient ``∂loss/∂vp`` is
    divided by ``(S * R + eps) ** exponent`` where ``S`` and ``R`` are
    the per-iter source / receiver illumination tensors accumulated from
    the sweep propagator's ``solver.source_illumination`` /
    ``solver.receiver_illumination`` attrs.

    The default ``exponent=0.5`` is the OBN-FWI in-flight feature
    (sqrt-illumination preconditioning); ``exponent=1.0`` recovers the
    full pseudo-Hessian inverse (Plessix & Mulder 2004) and is closer
    to a "raw" preconditioner that can over-flatten weak deep events.

    Active for both grid-mode and reparam-mode FWI. In reparam mode the
    division is applied to the leaf gradient (``base_leaf.grad``) BEFORE
    the second-pass back-prop through the network — i.e. the network
    sees an illumination-preconditioned velocity gradient.
    """

    enabled: bool = False
    epsilon: float = Field(gt=0, default=1.0e-6)
    exponent: float = Field(gt=0, default=0.5)


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
    # 3-D multisource (CRG-plan + supershot) QC products. Independent
    # flags so the legacy ``shot_gather`` knob (per-physical-shot rich
    # 2-D panel) doesn't double-duty for the very different
    # encoded-supershot 3-panel. Defaults True because for the
    # multisource path these are the primary visualizations.
    supershot_panel: bool = True
    well_logs: bool = True


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
    # Pin the water column to a fixed velocity at render time. When
    # ``mask_water_layer`` is True the runner builds a 3-D boolean mask
    # and passes it to :class:`sweep_nn.VelocityINR`. The SIREN cannot
    # disturb those voxels — useful because SIREN's natural init has
    # ~std=0.08 raw output, so with ``vp_std=500`` the water layer
    # would otherwise sit at 1500±40 m/s of init noise from epoch 0.
    #
    # Mask construction: when ``seabed_depth_path`` is set (preferred),
    # the runner loads a 2-D ``seabed_depth`` array (same npz format
    # as :class:`FreezeWaterLayerSpec.seabed_depth_path` — key
    # ``"seabed_depth"``, depths in meters from z=0), crops it to the
    # post-model_plan window, and broadcasts via ``z*dh < depth[y,x]``
    # to a 3-D bool mask. This is the geologically-correct path —
    # per-column bathymetry, robust to smoothed init models that
    # don't have float-exact 1500 m/s in the water column.
    #
    # Fallback when ``seabed_depth_path`` is None: the runner infers
    # the mask via ``init_vp == water_vp_m_s``. Only works when the
    # init was stamped to exactly the water velocity.
    mask_water_layer: bool = False
    water_vp_m_s: float = Field(gt=0, default=1500.0)
    seabed_depth_path: Path | None = None
    hash: ReparamHashSpec = Field(default_factory=ReparamHashSpec)
    # Anisotropic lateral downsampling of the INR render: evaluate the
    # hash+MLP on a grid coarsened by this factor in x/y (full-res in z),
    # then upsample the delta back. lateral_ds**2 fewer coords → much faster
    # render + reparam backward, negligible error where the model is laterally
    # smooth. Param count UNCHANGED. 1 = off (default). int = same on x and y.
    lateral_downsample: int | tuple[int, int] = 1
    # torch.compile the (anisotropic) z-slab renderer — fuses the pure-torch
    # hash+MLP+interp into far fewer kernels (~4× render, large backward win).
    compile_render: bool = False
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
    - ``inr_lr_scale``: scale the reparam network's lr (multiplicative).
      Only meaningful when :class:`ReparamSpec` is active; ignored otherwise.
      Lets you slow down the SIREN/hash net on later stages without
      changing the grid-FWI lr (which is set via ``lr_scale``).
    - ``optimizer_reset``: when ``True``, rebuild the optimizer at stage
      entry — drops accumulated Adam first / second moments. Useful when
      a stage changes regime drastically (e.g. switching from low-freq
      sweep to high-freq inversion) and the old momentum is misleading.
      Defaults to ``False``; the runner already auto-resets the optimizer
      when ``dh_m`` changes (because Adam state is shape-bound).
    - ``dh_m``: rebuild solver + resample vp to this grid spacing
    - ``dt_s`` / ``nt``: rebuild solver at a different time grid
    - ``batch_size``: per-stage shot batch (overrides FWISpec.batchsize)
    - ``bandpass``: filter obs before this stage runs (uses sweep-preproc)
    """

    epochs: int = Field(ge=1)
    wavelet: "Wavelet | None" = None
    lr_scale: float = Field(gt=0, default=1.0)
    inr_lr_scale: float = Field(gt=0, default=1.0)
    optimizer_reset: bool = False
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


class PlanSamplingConfig(_Forbid):
    """Stochastic per-iter shared-shot sampler (CRG / OBN supershot mode).

    Attach to :class:`ObsPlanConfig.sampling` when the underlying plan is
    ``grouping='crg'`` and you want source-encoded supershot FWI (the
    canonical 3-D OBN training pattern). When unset (default), the runner
    treats ``obs.plan`` as static per-group obs (the CSG path).

    Grouping-agnostic naming: legacy CRG had ``source_lines_per_crg`` /
    ``min_shot_coverage`` — the unified path renames these to
    ``source_lines_per_group`` / ``min_coverage`` because the underlying
    sampler (:func:`sweep_io.seismic_plan.sample_shared_shots_from_plan`)
    operates on any grouping that admits a shared-shot intersection.

    Parameters
    ----------
    min_coverage
        Drop plan groups whose row count is below this before sampling.
        Mirrors legacy ``encoding_min_coverage``. ``0`` = keep all groups.
    shared_shots_per_iter
        Number of physical shots sampled per iter from the intersection of
        the chosen groups (= the n_shared dimension of the supershot).
        Must be ``> 0`` to enable shared-shot sampling.
    source_lines_per_group, max_traces_per_sourceline
        Hierarchical sub-sampling caps inside the shared-shot intersection.
        ``0`` disables the corresponding cap.
    num_workers, prefetch_factor, persistent_workers
        Forwarded to the prefetching DataLoader the runner builds for the
        per-iter SEG-Y reads. Defaults tuned for compute > I/O.
    dedup_mode
        Per-iter dedupe at the grid level. When multiple physical shots in
        an encoded supershot snap to the same ``(gx, gy, gz)`` cell, keep
        only one per cell. ``"first"`` keeps the first row, ``"nearest"``
        the row whose model-frame xy is closest to the cell centre.
        ``"none"`` disables.
    trace_cache_bytes
        Optional in-RAM trace cache shared across iters (bytes). Mirrors
        legacy ``--trace-cache-bytes``: keeps hot SEG-Y traces so repeated
        picks avoid cold-Lustre reads. ``0`` disables, ``-1`` = unbounded.
    receiver_first
        Use the receiver-first shared-shot sampler
        (:func:`sweep_io.seismic_plan.sample_shared_shots_receiver_first`)
        instead of the default random-group intersection. Picks ONE target
        shot per iter then the nodes that recorded it, so the per-iter
        supershot sweeps the WHOLE survey instead of collapsing toward the
        centre — fixes the periphery under-sampling of partial-coverage OBN
        surveys. A shot_key->nodes reverse index is built once at setup.
    receiver_first_max_retries
        Re-roll the target shot up to this many times to find one covered by
        ``>= shared_shots_per_iter`` nodes before falling back to the random
        sampler. Only used when ``receiver_first=True``.
    """

    min_coverage: int = Field(ge=0, default=0)
    shared_shots_per_iter: int = Field(ge=1, default=1)
    source_lines_per_group: int = Field(ge=0, default=0)
    max_traces_per_sourceline: int = Field(ge=0, default=0)
    num_workers: int = Field(ge=0, default=4)
    prefetch_factor: int = Field(ge=1, default=2)
    persistent_workers: bool = True
    dedup_mode: Literal["none", "first", "nearest"] = "none"
    receiver_first: bool = False
    receiver_first_max_retries: int = Field(ge=1, default=20)
    trace_cache_bytes: int = 0


class ObsPlanConfig(_Forbid):
    """Unified obs loader backed by a ``seismic_plan_v1`` plan.

    Reads the SEG-Y trace bytes through :class:`sweep_io.seismic_plan.PlanReader`
    using the file-id / byte-offset records the plan already stores.
    Pair with :class:`FromPlanGeometry` (the two normally share the same
    plan_path so the geometry layout matches what the reader returns).

    Parameters
    ----------
    plan_path
        Path to the ``seismic_plan_v1`` npz (built with ``sweep-tasks build-plan``).
        SEG-Y paths recorded inside the plan are env-var remappable via
        ``FWI_SEGY_ROOT`` / ``FWI_SEGY_REMAP``.
    cache_all
        When True, eagerly read every plan row into RAM at task start —
        ~700 MB for Viking-scale 2-D, fine; OOM-risky on production-OBN-scale.
        Default False (lazy per-iter reads).
    sampling
        Optional :class:`PlanSamplingConfig` that switches on stochastic
        per-iter shared-shot sampling (the OBN supershot pattern). Set
        only for ``grouping='crg'`` plans; CSG runs leave this ``None``
        and the runner treats the obs as static per-group tensors.
    """

    plan_path: Path
    cache_all: bool = False
    sampling: PlanSamplingConfig | None = None
    # --- Single-source materialization (sampling=None, conventional FWI) ----
    # When ``sampling`` is None the runner materialises ONE static
    # ``(nshots, nrec, nt)`` dataset from the plan at setup and runs the
    # standard single-source FWI loop (per-shot gradient accumulation via
    # ``train_shot_batchsize``). Works for BOTH 2-D/3-D, CSG and CRG plans:
    #   * CSG: group = air-gun shot (source = group_xyz), receivers = the
    #     recording nodes (row_receiver_xyz).
    #   * CRG: group = OBN node (virtual source = group_xyz, reciprocity),
    #     receivers = the air-gun positions it recorded (row_source_xyz).
    # Each group is reduced to a FIXED ``n_receivers_per_shot`` so the
    # dataset is rectangular (groups with fewer than ``min_receivers`` are
    # dropped; groups with more are sub-sampled per ``receiver_select``).
    n_receivers_per_shot: int | None = None
    receiver_select: Literal["random", "nearest", "all"] = "nearest"
    max_shots: int | None = None
    min_receivers: int = Field(ge=1, default=8)
    materialize_seed: int = 0


class ObsSpec(_Forbid):
    """How to get observed data. Pick exactly one source.

    Available sources:
      - ``synthetic_from`` — re-run forward modeling on a "true" vp.
      - ``synthetic_from_models`` — same, for multi-model equations.
      - ``npy_path`` — a pre-saved ``(nshots, nrec, nt)`` ``.npy``.
      - ``segy`` — load straight from a single SEG-Y file (Option A).
      - ``segy_index`` — load from a multi-file SEG-Y index (Option B).
      - ``plan`` — unified SeismicPlan reader (Option C — the canonical
        path for both 2-D CSG and 3-D CRG; set ``obs.plan.sampling`` to
        opt into the OBN multisource supershot loop).
    """

    synthetic_from: ModelRef | None = None
    synthetic_from_models: list[ModelRef] | None = None
    npy_path: Path | None = None
    segy: ObsSegyConfig | None = None
    segy_index: ObsSegyIndexConfig | None = None
    plan: ObsPlanConfig | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self):
        choices = [
            ("synthetic_from", self.synthetic_from),
            ("synthetic_from_models", self.synthetic_from_models),
            ("npy_path", self.npy_path),
            ("segy", self.segy),
            ("segy_index", self.segy_index),
            ("plan", self.plan),
        ]
        set_choices = [name for name, value in choices if value is not None]
        if len(set_choices) != 1:
            raise ValueError(
                "ObsSpec: exactly one of "
                "synthetic_from / synthetic_from_models / npy_path / "
                f"segy / segy_index / plan must be set; got {set_choices}."
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
    # wavelet/geometry/obs are optional ONLY under
    # source_encoding.mode == "frequency_selection" (wavelet-free, plan-free:
    # geometry and data live in the coefficient shards). The validator below
    # keeps them required for every other path, so existing YAMLs are
    # unaffected.
    wavelet: Wavelet | None = None
    geometry: Geometry | None = None
    physics: PhysicsSpec
    backend: BackendSpec = Field(default_factory=BackendSpec)

    # Acoustic FWI uses init_model (single ModelRef). Multi-model equations
    # (Elastic, etc.) use init_models — a list ordered to match the equation's
    # MODEL_SPECS. Exactly one must be set; the runner normalises to a list.
    init_model: ModelRef | None = None
    init_models: list[ModelRef] | None = None

    obs: ObsSpec | None = None
    optimizer: Optimizer
    scheduler: Scheduler = Field(default_factory=SchedulerConstant)
    loss: LossSpec = Field(default_factory=LossSpec)

    @model_validator(mode="after")
    def _freqsel_or_conventional(self):
        freq_on = (self.source_encoding is not None
                   and self.source_encoding.enabled
                   and self.source_encoding.mode == "frequency_selection")
        if not freq_on:
            missing = [n for n in ("wavelet", "geometry", "obs")
                       if getattr(self, n) is None]
            if missing:
                raise ValueError(
                    f"fwi: field(s) {missing} are required (they are "
                    "optional only under source_encoding.mode="
                    "'frequency_selection')")
        return self
    epochs: int = Field(ge=1)
    batchsize: int = Field(ge=1, default=1)
    train_shot_batchsize: int | None = None  # default = batchsize (no accumulation)
    show_every: int = 10

    # Phase-1 patch fields:
    model_bounds: dict[str, ModelBounds] = Field(default_factory=dict)
    freeze_top_n_rows: int = Field(ge=0, default=0)
    stages: list[StageSpec] | None = None
    resume_from: str | None = None  # task_id under output_dir to resume from
    # Auto-resume from the *same* task_dir's checkpoint.pt when it exists
    # (no need to set `resume_from` to the same task_id). Pair this with
    # a fixed ``task_id:`` in the YAML so the directory is deterministic;
    # ctrl-c during training will save a checkpoint at the next iter
    # boundary, and re-running the same YAML picks up where it left off.
    # When True but the file is absent, the run starts fresh silently.
    resume: bool = False
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

    # Optional diagonal pseudo-Hessian illumination preconditioner. See
    # :class:`IlluminationPreconditionSpec`. When omitted, gradients are
    # passed through to the optimizer unchanged (current behaviour).
    illumination_precondition: IlluminationPreconditionSpec | None = None

    # Optional source encoding: one encoded-supershot forward+adjoint per
    # iter regardless of ``batchsize``. Requires CRG mode + shared-shots
    # sampling. See :class:`SourceEncodingSpec`.
    source_encoding: SourceEncodingSpec | None = None

    # Optional zero-phase Butterworth bandpass applied to the wavelet
    # (once at setup) AND to the per-iter encoded obs supershot (via
    # the differentiable :func:`sweep_preproc.filter.bandpass_torch`).
    # Used by the OBN CRG path only; the multi-stage 2-D / 3-D path
    # uses :class:`StageBandpass` inside ``stages`` instead.
    bandpass: StageBandpass | None = None

    # Optional Sobolev / TV-style smoothness regularizer on vp. See
    # :class:`SmoothRegSpec`. The runner adds ``weight * TVPrior(vp)``
    # to the data misfit; the gradient propagates back into the vp
    # tensor (or net params in reparam mode).
    smooth_regularization: SmoothRegSpec | None = None

    # Optional water-column / seabed gradient freeze mask. See
    # :class:`FreezeWaterLayerSpec`. Built once at setup, applied to
    # the vp gradient after illumination preconditioning.
    freeze_water_layer: FreezeWaterLayerSpec | None = None

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


class PostFilterImageSpec(_Forbid):
    """Depth-tapered z-axis low-cut on RTM / FWI gradient images.

    When attached to :class:`RTMImagingSpec.post_filter`, the runner
    auto-applies :func:`sweep_tasks.postproc.filter_image.filter_image_file`
    to every saved imaging product (raw + globally normalised + per-shot
    normalised) right after the main RTM finishes. Same algorithm is
    available as a standalone CLI (``sweep-tasks filter-image <npy>``) for
    iterating on params without re-running the RTM itself.

    Defaults match the legacy ``07_filter_imaging.py`` Viking recipe.
    """
    enabled: bool = True
    wavelength_m: float = Field(gt=0, default=300.0)
    depth_m: float = Field(ge=0, default=600.0)
    taper_m: float = Field(ge=0, default=400.0)
    clip_percentile: float = Field(ge=0, le=49, default=1.0)
    display_scale: float = Field(gt=0, default=1.0)
    # ``sweep_image`` is the bundled diverging LUT from
    # ``sweep_viz.colormaps`` (registered as a matplotlib cmap at import
    # time). Tuned for percentile-clipped RTM / kernel images. Override
    # with any matplotlib cmap name (e.g. ``"seismic"``, ``"gray"``) for
    # legacy display.
    cmap: str = "sweep_image"
    # Which RTM outputs to filter. ``"all"`` covers raw + global-norm +
    # per-shot-norm; pass an explicit list of basenames (without ``.npy``)
    # to be selective. Output files land next to the inputs as
    # ``<stem>_shallow_zlowcut.npy`` + matching PNG.
    targets: Literal["all"] | list[str] = "all"
    save_png: bool = True


class RTMImagingSpec(_Forbid):
    """Per-batch RTM imaging knobs for :class:`RTMSpec`.

    Controls the post-FWI imaging loop: how many shots per batch, optional
    pre-loss bandpass on obs/wavelet/syn, illumination normalisation, QC
    cadence, and the misfit used to derive the gradient (= RTM) image.

    The RTM image equals the per-batch gradient under the chosen matching
    loss (the c-backend backward pass also populates
    ``solver.source_illumination`` / ``receiver_illumination`` for free —
    no separate ``solver.rtm`` invocation).
    """

    shots_per_batch: int = Field(ge=1, default=1)
    filter_lowcut_hz: float | None = None
    filter_highcut_hz: float | None = None
    filter_order: int = Field(ge=1, default=4)
    filter_padtype: Literal["odd", "even", "constant", "none"] | None = "odd"
    # Where the bandpass is applied. ``"syn"`` (default, FWI-style) filters
    # obs once at task start AND filters syn per-batch via a differentiable
    # torchaudio filtfilt — this is the safer numerical choice when the
    # solver naturally outputs broadband syn (the autograd-aware filter
    # ensures the residual is band-limited).  ``"wavelet"`` pre-filters the
    # source wavelet ONCE so the solver's syn is naturally band-limited;
    # the per-batch syn filter is then skipped (mirrors the legacy
    # ``stage.bandpass.target='wavelet'`` path on the FWI side).  Obs is
    # bandpassed once either way.
    filter_target: Literal["syn", "wavelet"] = "syn"
    illumination_epsilon: float = Field(gt=0, default=1.0e-6)
    normalize_by_illumination: bool = True
    save_per_shot: bool = False
    live_update_every_batches: int = Field(ge=1, default=10)
    # Loss kind used to derive the gradient image. Trace-cosine is the
    # default and matches the FWI benchmark.
    loss_kind: Literal["mse", "l1", "trace_cosine"] = "trace_cosine"
    trace_cosine_demean: bool = True

    # Optional post-processing: depth-tapered z-axis low-cut filter that
    # removes the slowly-varying-with-depth drift contaminating the
    # shallow part of the stacked image. Same algorithm is also exposed
    # standalone as ``sweep-tasks filter-image`` so users can iterate on
    # the filter without re-running the RTM.
    post_filter: PostFilterImageSpec | None = None


class RTMSpec(BaseTaskSpec):
    """Post-FWI Reverse Time Migration as a one-pass imaging task.

    For each shot batch we:
      1. Read obs traces, bandpass both obs and syn (when configured).
      2. Forward solve with the input ``velocity_model`` -> syn.
      3. Backward through the loss -> FWI gradient image (per cell).
      4. Call ``solver.rtm`` with residual ``(obs - syn)`` to get the RTM
         cross-correlation image + source / receiver illumination buffers.

    Accumulate four shape-(nz, nx) buffers across shots, then normalise
    by ``sqrt(S * R + eps)`` and save both raw and normalised products.

    Unlike FWI, there is no iteration / optimizer / stages: the input
    ``velocity_model`` is imaged as-is.
    """

    task_type: Literal["rtm"] = "rtm"

    grid: GridSpec
    time: TimeSpec
    wavelet: Wavelet
    geometry: Geometry
    physics: PhysicsSpec
    backend: BackendSpec = Field(default_factory=BackendSpec)

    # Velocity model to image (replaces FWI's ``init_model`` — no inversion).
    velocity_model: ModelRef

    obs: ObsSpec
    loss: LossSpec = Field(default_factory=LossSpec)

    # Per-name model bounds (only ``vp`` is currently used). Kept dict-shaped
    # for parity with FWISpec / future multi-parameter equations.
    model_bounds: dict[str, ModelBounds] | None = None

    # Optional per-batch local model windowing. Mirrors the FWI path.
    local_model_window: LocalModelWindowSpec | None = None

    imaging: RTMImagingSpec = Field(default_factory=RTMImagingSpec)

    qc: QCSpec | None = None
    data_plan: DataPlanSpec | None = None

    @field_validator("local_model_window", mode="before")
    @classmethod
    def _local_model_window_bool_shortcut(cls, v):
        if v is True:
            return LocalModelWindowSpec()
        if v is False:
            return None
        return v


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
    # See ``FWISpec.resume`` for semantics: auto-resume from the same
    # task_dir's checkpoint.pt when present; harmless when absent.
    resume: bool = False
    save_illumination: bool = False

    modeling_override: ModelingOverride | None = None

    @model_validator(mode="after")
    def _validate_stages(self):
        if self.stages is not None and len(self.stages) == 0:
            raise ValueError("LSRTMSpec.stages must be non-empty when set.")
        return self


TaskSpec = Annotated[
    Union[IntrospectSpec, ForwardSpec, WavefieldSpec, FWISpec, LSRTMSpec, RTMSpec],
    Discriminator("task_type"),
]
