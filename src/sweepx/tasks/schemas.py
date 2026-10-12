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

import math
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

    # Truncated backward: save boundary strips for (and reverse through) only
    # the LAST ``tail_steps`` of the record. The forward still runs full nt.
    # Effective reverse depth is ``tail_steps - 1`` (the restore at loop step
    # ``it`` consumes step ``it-1``'s strip), so callers must include that
    # one-step alignment tax in their margin. Only valid when the loss reads
    # nothing before the tail (freqsel's steady-window GCN); an impulsive
    # misfit genuinely needs the early adjoint correlation and will lose it.
    # On a sweep without this feature ``to_dataclass`` raises TypeError —
    # a free capability check instead of a silently ignored truncation.
    tail_steps: int | None = Field(default=None, gt=0)

    def to_dataclass(self) -> BoundaryOptions:
        dump = self.model_dump()
        # None means "feature not requested" — drop the key entirely so this
        # model still drives every sweep released before tail truncation
        # existed. Only an actually-set tail_steps reaches the dataclass, so
        # the TypeError-on-old-sweep capability check fires exactly when the
        # user asked for something their core cannot do, not always.
        if dump.get("tail_steps") is None:
            dump.pop("tail_steps", None)
        return BoundaryOptions(**dump)



class CkptOptionsModel(_Forbid):
    mode: Literal["chunk", "recursive"] = "chunk"
    chunks: int = 100
    count: int = 0
    storage: Literal["gpu", "cpu"] = "gpu"
    pinned_memory: bool | None = None

    def to_dataclass(self) -> CkptOptions:
        return CkptOptions(**self.model_dump())


class MemoryOptionsModel(_Forbid):
    """Gradient-memory mode for ``impl='c'``: how the backward gets the forward
    wavefield back.

    ``boundary`` (default, as in sweep itself) saves only the PML slab and
    reconstructs the interior exactly. ``full`` stores every time step —
    fastest, and on 2-D Marmousi at nt=10000 about 276 GiB, so only for small
    grids. ``ckpt`` re-runs the forward from checkpoints.
    """

    strategy: Literal["full", "boundary", "ckpt"] = "boundary"
    boundary: BoundaryOptionsModel | None = None
    ckpt: CkptOptionsModel | None = None

    @field_validator("strategy", mode="before")
    @classmethod
    def _null_is_default(cls, v):
        # `strategy: null` asked sweep for its backend default, which on
        # impl='c' is boundary saving; keep reading it that way.
        return "boundary" if v is None else v

    @model_validator(mode="after")
    def _options_match_strategy(self):
        if self.strategy == "boundary" and self.boundary is None:
            self.boundary = BoundaryOptionsModel()
        if self.strategy == "ckpt" and self.ckpt is None:
            self.ckpt = CkptOptionsModel()
        if self.strategy != "boundary" and self.boundary is not None:
            raise ValueError(f"memory.boundary is only read with strategy='boundary', "
                             f"not {self.strategy!r}.")
        if self.strategy != "ckpt" and self.ckpt is not None:
            raise ValueError(f"memory.ckpt is only read with strategy='ckpt', "
                             f"not {self.strategy!r}.")
        return self

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

class LinearGradientSpec(_Forbid):
    """A laterally-constant 1-D depth gradient, optionally over a water layer.

    The standard FWI cold start: no lateral structure at all, so nothing about
    the answer is smuggled into the starting model. Built in memory, so a task
    YAML stays self-contained.

    ``water_rows`` top rows are held at ``water_vp``; the remaining rows ramp
    linearly from ``vmin`` to ``vmax``. These rows only build the starting
    model -- they do not tell the inversion where the water is. To keep the
    inversion off the water column, whose velocity is a known constant rather
    than something to solve for, give it in ``water:`` (:class:`WaterSpec`).
    """

    vmin: float = Field(gt=0)
    vmax: float = Field(gt=0)
    water_rows: int = Field(ge=0, default=0)
    water_vp: float = Field(gt=0, default=1500.0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.vmax < self.vmin:
            raise ValueError(
                f"linear_gradient: vmax ({self.vmax}) must be >= vmin ({self.vmin})."
            )
        return self


class ModelRef(_Forbid):
    """Reference to a model tensor by name (must appear in equation.MODEL_SPECS).

    Exactly one SOURCE must be given:

    * ``path``     — an ``.npy`` file on disk.
    * ``constant`` — a uniform value (``shape`` then required).
    * ``dataset``  — a benchmark model from :mod:`sweep.datasets`, so a task
      YAML is self-contained and needs no pre-dumped ``.npy``. Embedded
      entries (``marmousi:2d-demo``, ``overthrust:2d-demo``) need no network;
      the rest download once into ``$SWEEP_DATASETS_CACHE`` (default
      ``~/.cache/sweep-datasets``). ``sweep datasets list`` shows the catalog.

    Any source may be post-processed by ``smooth_sigma_cells``.
    """

    name: str
    path: Path | None = None
    constant: float | None = None
    shape: tuple[int, ...] | None = None

    # ----- dataset source ------------------------------------------------
    # ``"<name>"`` or ``"<name>:<variant>"`` — e.g. "marmousi:2d-demo",
    # "overthrust" (resolves to its default variant), "overthrust:3d-acoustic".
    dataset: str | None = None
    # Which key of the loader's returned dict to take. Defaults to ``name``,
    # which is already the equation's model name ("vp" / "vs" / "rho"), so it
    # only needs setting for off-label picks.
    dataset_field: str | None = None
    # Preset selector for the embedded demo entries — forwarded as the
    # loader's ``name=`` kwarg ("vp_true" / "vp_smooth" / "vp_linear" for
    # marmousi:2d-demo, "true" / "smooth" for overthrust:2d-demo).
    preset: str | None = None
    # Decimation forwarded to the loader (full-size benchmarks only). Scalar
    # applies to every axis; a tuple is per-axis. Cuts both grid size and dh.
    downsample: int | tuple[int, ...] | None = None

    # ----- synthetic 1-D gradient source ---------------------------------
    # Built in memory from (vmin, vmax, water_rows); ``shape`` is required
    # because there is no file to take it from.
    linear_gradient: LinearGradientSpec | None = None

    # ----- post-processing (any source) ----------------------------------
    # Gaussian-smooth the loaded array by this sigma in CELLS. The standard
    # way to derive a starting model from a true one without shipping a
    # second file: ``dataset: overthrust`` + ``smooth_sigma_cells: 8``.
    smooth_sigma_cells: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _exactly_one_source(self):
        sources = [self.path is not None, self.constant is not None,
                   self.dataset is not None, self.linear_gradient is not None]
        if sum(sources) != 1:
            raise ValueError(
                f"ModelRef '{self.name}': exactly one of `path`, `constant`, "
                f"`dataset` or `linear_gradient` must be set (got "
                f"path={self.path!r}, constant={self.constant!r}, "
                f"dataset={self.dataset!r}, "
                f"linear_gradient={'set' if self.linear_gradient else None})."
            )
        if self.constant is not None and self.shape is None:
            raise ValueError(
                f"ModelRef '{self.name}': `shape` is required when `constant` is set."
            )
        if self.linear_gradient is not None and self.shape is None:
            raise ValueError(
                f"ModelRef '{self.name}': `shape` is required when "
                f"`linear_gradient` is set (nothing else defines the grid)."
            )
        if self.dataset is None:
            for f in ("dataset_field", "preset", "downsample"):
                if getattr(self, f) is not None:
                    raise ValueError(
                        f"ModelRef '{self.name}': `{f}` only applies together "
                        f"with `dataset`."
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
    """A 2D line of source/receiver points: x = arange(start, stop, step), z = depth.

    Points come out in sweep's (x, z) grid-index order.
    """

    step: int = Field(ge=1)
    depth: int = Field(ge=0)
    start: int = 0
    stop: int | None = None  # None means "use grid extent (nx)"


class LineGeometry(_Forbid):
    """Regular line of sources and receivers, generated from start/stop/step rules."""

    kind: Literal["line"] = "line"
    sources: LineSet
    receivers: LineSet


class AxisSpan(_Forbid):
    """``arange(start, stop, step)`` along one lateral axis of a 3-D grid.

    ``stop: null`` means "run to the grid extent on this axis" (nx for ``x``,
    ny for ``y``) — the same convention as :class:`LineSet`.
    """

    step: int = Field(ge=1)
    start: int = 0
    stop: int | None = None


class GridSet(_Forbid):
    """A 3-D rectangular patch of points: the ``x`` × ``y`` outer product at a
    single depth. Each point is (x, y, z); the points are enumerated x-major,
    y fastest-varying."""

    x: AxisSpan
    y: AxisSpan
    depth: int = Field(ge=0)


class GridGeometry(_Forbid):
    """The 3-D analogue of ``kind: line`` — a rectangular patch of sources and
    a rectangular patch of receivers, both from start/stop/step rules.

    Every shot sees the same receiver patch. 3-D grids only (``shape`` must be
    ``(nz, ny, nx)``); use ``kind: line`` in 2-D. It exists so a 3-D example can
    place a few hundred receivers without spelling out one YAML line each,
    which is what ``kind: explicit`` would need.
    """

    kind: Literal["grid"] = "grid"
    sources: GridSet
    receivers: GridSet


class ExplicitGeometry(_Forbid):
    """Literal source / receiver coordinate lists.

    `sources` is `(nshots, ndim)`. `receivers` is either `(nrec, ndim)` (shared
    across all shots) or `(nshots, nrec, ndim)` (per-shot). The runner
    promotes the shared form to per-shot at use time.

    Each point is a list of GRID INDICES in sweep's order: ``[x, z]`` on a
    2-D grid, ``[x, y, z]`` on a 3-D one — the order ``kind: line`` / ``grid``
    produce. They reach the solver as given, unreordered. (The MODEL arrays are
    the other way round, ``(nz, nx)`` / ``(nz, ny, nx)``.)
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

    The last axis is grid indices in sweep's order, (x, z) in 2-D and
    (x, y, z) in 3-D, the same as :class:`ExplicitGeometry`.
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
    :class:`ObsPlanConfig.sampling` to drive the plan-streaming supershot
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
    # NOTE: these three per-axis tuples (dh_xyz_m, grid_origin_xyz_m,
    # auto_origin_pad_cells) follow the MODEL axis order (z, y, x), despite
    # the ``_xyz_`` in two of the names. Source / receiver POINTS are (x, y, z).
    dh_xyz_m: tuple[float, float, float] | None = None
    # The coordinate (metres, in the rotated model frame) that maps to grid
    # index 0 on each axis, ordered (z, y, x). When None (default), the runner
    # picks the bounding box of the rotated source + receiver positions with
    # optional padding.
    grid_origin_xyz_m: tuple[float, float, float] | None = None
    # Padding around the snap origin in cells (z, y, x). Used only when
    # ``grid_origin_xyz_m`` is None.
    auto_origin_pad_cells: tuple[int, int, int] = (0, 0, 0)
    # When set (e.g. 1500.0 = water velocity), lift each CRG source down to the
    # water cell just above the seabed: src_iz = min(src_iz, seabed_iz - 1),
    # where seabed_iz is the first non-water cell (|vp - this| > 1) per (y, x)
    # in the (sharp-seabed) init model. Keeps sources off the discretized seabed
    # sediment cell, avoiding the source-on-interface near-field artifact. Uses
    # ``min`` so sources already in the water column are left untouched. None
    # (default) leaves snapped source depths unchanged.
    lift_source_to_water_vp: float | None = None


Geometry = Annotated[
    Union[LineGeometry, GridGeometry, ExplicitGeometry, FromFileGeometry,
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
    # None = the equation's own PML (sweep's ``default_pml_type``): every
    # equation ships one, and sweep refuses any other. Set it only to pick
    # Acoustic1st's alternative, 'spml'.
    pml_type: str | None = None
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
        # impl='c' with no memory block gets boundary saving, written into the
        # spec so config_resolved.yaml records what ran. Leaving it to the
        # solver ran store-all instead: the use_ckpt=False that solver_build
        # passes in that case reads to sweep as "no memory trick at all".
        # The legacy use_ckpt flag still selects checkpointing on its own.
        if self.impl == "c" and not self.use_ckpt:
            if self.cuda_options is None:
                self.cuda_options = CUDAOptionsModel()
            if self.cuda_options.memory is None:
                self.cuda_options.memory = MemoryOptionsModel()
        return self


# ----- loss, bounds, stages, optimizer, scheduler ------------------------

class LossSpec(_Forbid):
    """Misfit between synthetic and observed seismograms."""

    # steady_gcn: the complex-cosine GCN on steady-window DFT coefficients, the
    # one misfit of source_encoding.mode="frequency_selection" and one no other
    # path can evaluate. FWISpec resolves an unset kind to it in that mode and
    # rejects it everywhere else.
    kind: Literal["mse", "l1", "huber", "trace_cosine", "envelope", "ot",
                  "cc_traveltime", "steady_gcn"] = "mse"
    huber_delta: float = 1.0  # only used when kind="huber"
    # cc_traveltime (Luo & Schuster 1991): per-trace cross-correlation traveltime
    # misfit 0.5*dt^2, where dt is the syn->obs time shift measured by a
    # DIFFERENTIABLE soft-argmax of the (normalised) cross-correlation. Gives a
    # low-wavenumber (tomographic) gradient robust to LARGE traveltime shifts
    # (no cycle-skipping) — the right tool to update a smooth background
    # velocity (e.g. an LVZ) from reflection MOVEOUT, when the reflector is
    # already present in the model so its wavepath carries a transmission term.
    # cc_max_lag_samples: search window (+/- samples) for the shift; 0 -> nt//4.
    # cc_beta: soft-argmax sharpness over the normalised correlation (in [-1,1]);
    # larger -> closer to a hard argmax (peakier), smaller -> smoother/robuster.
    cc_max_lag_samples: int = 0
    cc_beta: float = 30.0
    # Early-time mute (samples): zero the first N time samples of syn AND obs
    # before the misfit, to remove the strong early diving-wave energy and let
    # the late wide-angle / reservoir-reflection event dominate. 0 = off.
    time_mute_samples: int = 0
    # Optional LATE mute (samples): zero time samples AFTER this index. Combined
    # with time_mute_samples this makes a time WINDOW [early, late] around a
    # specific event (e.g. isolate the reservoir reflection ~2.4-3.6 s and
    # exclude both the shallow diving first-arrival AND the far-offset diving
    # wave). Needed because a full-record cc/waveform misfit is dominated by the
    # strong first-arrival, which for an LVZ (velocity inversion) is turning-ray
    # BLIND — only the transmitted reservoir reflection carries the LVZ delay.
    time_mute_late_samples: int = 0
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
    # Apply ``data_mask_path`` MUTE-THEN-MISFIT (syn & obs multiplied by the
    # mask BEFORE the misfit) instead of as a post-hoc per-sample weight.
    # Required for kind="trace_cosine": its per-trace value is broadcast over
    # time, so a post-hoc weight rescales the trace but never windows it.
    # Default False keeps every existing data_mask_path run bit-identical.
    data_mask_window_mode: bool = False
    # Diving-wave time window computed ON-THE-FLY per batch from a per-node
    # first-arrival LUT (``diving_window_db`` npz: node_rec[Nn,3] + per-node
    # asinh moveout params node_asinh[Nn,3]=(t0,v0,k), from pick3d/FATT picks).
    # For each trace: center=asinh(offset), window=[center-pre, off/water_vel]
    # (bottom hugs the water direct), min width minwin, cosine taper. Offsets
    # come from the batch source/receiver grid geometry (CRG reciprocity: the
    # solver "shot" is the OBN node, its "receivers" are the sources).  When set
    # with kind="trace_cosine" the window is applied MUTE-THEN-CORRELATE (syn &
    # obs muted BEFORE the cosine) so it is a true window, not a per-trace
    # weight.  ``diving_obs_delay_s`` is added to the window times to align the
    # physical arrival axis with the recorded obs sample axis.  None = off.
    diving_window_db: str | None = None
    diving_pre_s: float = 0.30
    diving_minwin_s: float = 0.50
    diving_taper_s: float = 0.16
    diving_water_vel: float = 1500.0
    diving_obs_delay_s: float = 0.0


def _steady_gcn_needs_freqsel(loss: LossSpec, task: str) -> None:
    if loss.kind == "steady_gcn":
        raise ValueError(
            f"{task}: loss.kind='steady_gcn' is the misfit of fwi with "
            "source_encoding.mode='frequency_selection'; no other path can "
            "evaluate it")


def _plan_streaming_on(spec) -> bool:
    """Whether ``_run_fwi`` hands this FWI spec to the plan-streaming loop.

    The same test, in the same order, as its dispatch: frequency selection
    first, then ``obs.plan.sampling``.
    """
    se = spec.source_encoding
    if se is not None and se.enabled and se.mode == "frequency_selection":
        return False
    plan = spec.obs.plan if spec.obs is not None else None
    return plan is not None and plan.sampling is not None


def _leaf_gradient_features(spec) -> list[tuple[str, str]]:
    """The settings of an FWI ``spec`` that work on the velocity gradient,
    each with the value that turns it off.

    The plan-streaming loop skips them under domain decomposition, and with
    ``reparam.backward_mode: single_step``, which leaves it no velocity
    gradient. Each is listed only when it asks for something: a block with
    ``enabled: false`` or a zero weight asks for nothing, so the
    config_resolved.yaml of a run that left them off still loads.
    """
    out = []
    illum = spec.illumination_precondition
    if illum is not None and illum.enabled:
        out.append(("illumination_precondition", "enabled: false"))
    smooth = spec.smooth_regularization
    if smooth is not None and float(smooth.weight) > 0.0:
        out.append(("smooth_regularization", "weight: 0"))
    # `water:` is not one of them: a network's render pin holds the water
    # (padded to the DD grid, inside the single_step graph), and a grid model
    # is re-stamped after every step.
    return out


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


class ReparamHashC2FSpec(_Forbid):
    """Coarse-to-fine level unfreezing for the hash encoder (BARF-style).

    Only the ``base_levels`` coarsest levels are open at epoch 0; levels
    unfreeze coarse -> fine over ``[warmup, ramp_end]`` (fractions of the
    stage's epochs), the frontier level soft-weighted by ``ramp``. See
    :class:`sweep_nn.coarse_to_fine.CoarseToFineHashGrid`.
    """

    enabled: bool = False
    base_levels: int = Field(ge=1, default=2)
    ramp: Literal["cosine", "linear", "hard"] = "cosine"
    warmup: float = Field(ge=0.0, lt=1.0, default=0.0)
    ramp_end: float = Field(gt=0.0, le=1.0, default=1.0)
    # Cap the stage's unfreeze at this level count (None = all levels).
    # In a multiscale chain set e.g. 8/10/12/16 per band so the finest
    # levels stay frozen until the data actually carries high wavenumbers.
    final_levels: int | None = Field(ge=1, default=None)
    # Allocate fine levels ON DEMAND (sweep_nn.GrowingHashGrid) instead of
    # masking a fully-allocated grid — saves latent memory (finest levels
    # dominate in 3-D). Reuses base_levels -> final_levels over [warmup,
    # ramp_end] as the grow schedule; ``ramp`` is ignored (growth is
    # hard-stepped). Requires ``enabled: true``. A grown level joins the
    # optimizer at the lr the network runs at that epoch (reparam.lr x the
    # stage's scale x the scheduler's factor) and follows it from then on.
    growing: bool = False


class ReparamHashSpec(_Forbid):
    """Multi-resolution hash-grid encoder hyperparameters (Instant-NGP)."""

    enabled: bool = True
    levels: int = Field(ge=1, default=16)
    features_per_level: int = Field(ge=1, default=2)
    log2_size: int = Field(ge=1, default=15)
    # Scalar (isotropic) OR a per-axis list [n_z, n_y, n_x] (3-D) / [n_z, n_x]
    # (2-D) for an ANISOTROPIC hash — set proportional to each axis' physical
    # extent so every level resolves the same physical scale on all axes.
    # sweep_nn.MultiResHashGrid natively supports the per-axis list.
    base_resolution: int | list[int] = 4
    finest_resolution: int | list[int] = 512
    # "triton": fused GPU kernels (sweep_nn.triton_hash_encoding) — numerically
    # matches "pytorch" (cos≈1) with ~10-20x less encoder memory at large batch.
    # Requires triton + CUDA; incompatible with c2f.growing (lazy ParameterList).
    backend: Literal["pytorch", "triton"] = "pytorch"
    c2f: ReparamHashC2FSpec = Field(default_factory=ReparamHashC2FSpec)

    @field_validator("base_resolution", "finest_resolution")
    @classmethod
    def _resolution_positive(cls, v):
        vals = v if isinstance(v, list) else [v]
        if any(int(x) < 1 for x in vals):
            raise ValueError("base_resolution/finest_resolution entries must be >= 1")
        if isinstance(v, list) and len(v) not in (2, 3):
            raise ValueError(
                "per-axis base/finest_resolution must have 2 (2-D) or 3 (3-D) entries")
        return v

    @model_validator(mode="after")
    def _triton_incompatible_with_growing(self):
        if self.backend == "triton" and self.c2f.growing:
            raise ValueError(
                "hash.backend='triton' is incompatible with hash.c2f.growing=true "
                "(GrowingHashGrid's lazy per-level tables cannot be indexed by the "
                "fused kernel); use the c2f mask without growing, or backend='pytorch'")
        return self


class ReparamFourierSpec(_Forbid):
    """NeRF-style Fourier positional encoding (log-spaced sin/cos of coords).

    Mutually exclusive with the hash encoder. ``levels=L`` resolves up to
    ``2^(L-1)`` half-cycles across the normalized model extent — a smooth,
    global alternative to the hash grid (no local cells, so no blocky
    null-space texture; capacity between raw-coord SIREN and hash).
    """

    enabled: bool = False
    levels: int = Field(ge=1, default=6)
    include_input: bool = True


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


class DiffusionPriorSpec(_Forbid):
    """Plug-and-play DDPM diffusion prior added to the FWI objective.

    Wraps :class:`sweep_nn.diffusion.DiffusionVelocityPrior` — a DDPM trained on
    a velocity-model corpus (e.g. OpenFWI CurveVel-A, 64x64, 1500-4500 m/s)
    reused as a denoiser ``D(vp)`` — and adds a plug-and-play regularizer to the
    per-iter loss whose gradient flows into vp (or the reparam-net params):

    * ``kind="red"`` : ``0.5*mean((norm(vp) - D(vp).detach())**2)``
      (RED / proximal; deterministic DiffPIR/DDIM denoise, so DDP-safe added
      after the data-gradient all-reduce — identical on every rank).
    * ``kind="sds"`` : ``<score-distillation surrogate>`` (stochastic; the runner
      reseeds the RNG identically per rank each apply for DDP sync).

    ``weight_mode`` sets how ``weight`` scales that gradient before it is added
    to the data gradient:

    * ``"relative"`` (default): rescale the diffusion gradient so its norm is
      ``weight * ||accumulated data gradient||`` at each apply. ``weight`` is then
      a dimensionless fraction ("nudge the model 5% toward the prior") that is
      invariant to the misfit scale, grid size, and parametrization — the robust
      choice, since the raw RED gradient magnitude (normalised + mean-reduced)
      is tiny and dataset-dependent.
    * ``"absolute"``: add ``weight * grad(loss)`` directly. ``weight`` then lives
      on the (very small) raw scale — typically needs to be ``O(1e3-1e4)``.

    The denoiser bridges the size/distribution gap between the small generative
    model and the FWI grid via ``mode="patch"`` (slide a ``patch`` x ``patch``
    window at ``stride`` — native training size, in-distribution, preferred) or
    ``mode="resize"`` (bilinearly resize the whole field to the training size).
    ``strength`` sets the DiffPIR start time ``t_start = strength*T``; larger =
    stronger projection onto the prior.

    OUT-OF-DISTRIBUTION CAVEAT: a prior trained on synthetic models, applied
    in-loop to field data, can pull vp toward the training distribution. Keep
    ``weight`` small (~1e-4) and use ``every`` / ``start_band`` to apply it
    gently and only after the low frequencies have done the heavy lifting. The
    FWI is the main driver; the prior is a polish.
    """

    enabled: bool = True
    # sweep_nn-format DDPM checkpoint (a dict with ``config`` / ``ema`` (or
    # ``model``) / ``stats`` keys, as written by the sweep_nn diffusion trainer).
    ckpt_path: Path
    # In "relative" mode (default) this is the diffusion-to-data gradient-norm
    # ratio per apply (0.05 = a 5% nudge); in "absolute" mode it is the raw
    # loss weight (needs ~1e3-1e4). See ``weight_mode``.
    weight: float = Field(ge=0, default=0.05)
    weight_mode: Literal["relative", "absolute"] = "relative"
    kind: Literal["red", "sds"] = "red"
    mode: Literal["patch", "resize"] = "patch"
    strength: float = Field(gt=0, le=1.0, default=0.3)
    ddim_steps: int = Field(ge=1, default=10)
    patch: int = Field(ge=8, default=64)
    stride: int = Field(ge=1, default=32)
    vmin: float | None = None                     # None -> ckpt stats (typically 1500)
    vmax: float | None = None                     # None -> ckpt stats (typically 4500)
    use_ema: bool = True
    every: int = Field(ge=1, default=1)           # apply every N optimizer steps (epochs)
    start_band: int = Field(ge=0, default=0)      # apply only from this multiscale stage index onward
    sds_t_lo: float = Field(gt=0, lt=1.0, default=0.02)
    sds_t_hi: float = Field(gt=0, le=1.0, default=0.5)


class GradSmoothSpec(_Forbid):
    """Gaussian smoothing of the vp gradient before the optimizer step.

    The classic tomographic gradient preconditioner: a single-band
    finite-frequency traveltime (e.g. ``cc_traveltime``) kernel back-projects
    as an oscillatory, high-wavenumber "migration" pattern even though the
    physical sensitivity is smooth. Convolving the gradient with a Gaussian
    (separable, depthwise) each step removes that salt-and-pepper and leaves
    the low-wavenumber (background) update — turning a reflection-traveltime
    misfit into proper reflection-moveout tomography.

    ``sigma_z_cells`` / ``sigma_x_cells`` are the Gaussian stddevs in grid
    cells (z = depth axis, x = the fast lateral axis; ``sigma_y_cells`` used
    only in 3-D). A sigma of ~half the dominant wavelength in cells is a good
    starting point. Applied to the grid-mode vp gradient only (reparam nets
    are smooth by construction). ``every`` applies it on every N-th step.
    """

    enabled: bool = True
    sigma_z_cells: float = Field(ge=0, default=4.0)
    sigma_x_cells: float = Field(ge=0, default=4.0)
    sigma_y_cells: float = Field(ge=0, default=0.0)
    every: int = Field(ge=1, default=1)
    # Optional depth window (in grid rows) OUTSIDE which the vp gradient is
    # zeroed, applied after smoothing. Use to confine a reflection-traveltime
    # update to the overburden: a single reflector's finite-frequency kernel
    # leaks spurious high-wavenumber junk BELOW the reflector (nothing to
    # invert there) — masking rows below the reflector removes it without
    # injecting any structure. -1 = no limit on that side.
    mask_above_row: int = -1     # zero gradient for z-rows < this
    mask_below_row: int = -1     # zero gradient for z-rows > this
    # Linear taper width (rows) at the mask boundary. A hard cutoff makes the
    # gradient pile up at the boundary row (a spurious velocity band); a taper
    # spreads and removes it. 0 = hard cutoff.
    mask_taper_rows: int = 0


class WaterSpec(_Forbid):
    """The water layer, given once for every inversion path.

    WHERE the water is comes from exactly one of:

    * ``seabed`` -- an ``.npz`` with key ``"seabed_depth"``: the seabed depth
      per trace in metres below ``z = 0`` (the top of the native model grid),
      shape ``(nx,)`` for a 2-D model or ``(ny, nx)`` for 3-D, spanning the
      model's lateral extent at any resolution (first and last samples on the
      model's first and last columns). A plain number is a flat seabed at
      that depth, which keeps a synthetic task YAML self-contained. Every grid
      builds its own mask from it with
      :func:`sweep_tasks.bathymetry.water_mask_from_seabed_depth`, on that
      grid's own columns: a cell is water when its centre lies above
      ``seabed + buffer``.
    * ``mask`` -- a ``.npy`` boolean (or 0/1) array of the NATIVE model grid
      (``grid.dh``, the starting model's shape), True = water. A grid with
      another spacing gives each of its cells the value of the native cell
      nearest to that cell's position (``i * dh`` from the native origin; a tie
      goes to the cell with the larger index).

    The water is never inferred from velocities: no "cells at 1500 m/s", no
    thresholds. :func:`sweep_tasks.bathymetry.derive_seabed_depth_from_vp`
    turns a starting model into a seabed file once, as an explicit step.

    ``mode``:

    * ``pin`` (default): the water holds ``velocity`` for the whole run, on
      every stage -- the starting model is stamped with it, its gradient is
      zeroed, and a grid model is stamped again after every optimizer step; a
      reparam network renders ``velocity`` there. Any other model a grid
      inversion updates (vs, rho, ...) keeps its starting value in the water.
    * ``reset_each_stage``: the water is set to ``velocity`` at the start of
      every stage and inverted freely within the stage (a reparam network
      re-bakes its base there so its first render of the stage is
      ``velocity``).

    ``buffer_cells`` extends a seabed downward by that many cells of the
    native grid, converted to a depth (``buffer_cells * grid.dh``) so every
    stage grid marks the same physical layer. Seabed only: bake a buffer into
    a mask file instead.

    LSRTM inverts reflectivity, not velocity: there the water's reflectivity
    is held at zero (``pin``) or reset to zero at every stage, and
    ``velocity`` does not apply.

    ``water: false`` says "this model has no water layer". Only the
    frequency-selection path needs it: older versions pinned every cell of
    its starting model at exactly 1500 m/s without being asked, so that path
    refuses to start on such a model unless the water is given or ``false``.
    """

    seabed: float | Path | None = None
    mask: Path | None = None
    velocity: float = Field(gt=0, default=1500.0, allow_inf_nan=False)
    buffer_cells: int = Field(ge=0, default=0)
    mode: Literal["pin", "reset_each_stage"] = "pin"

    @model_validator(mode="after")
    def _one_source(self):
        if (self.seabed is None) == (self.mask is None):
            raise ValueError(
                "water: give exactly one of `seabed` (an .npz with key "
                "'seabed_depth') or `mask` (a boolean .npy of the native grid); "
                f"got seabed={self.seabed!r}, mask={self.mask!r}")
        if self.buffer_cells and self.mask is not None:
            raise ValueError(
                "water.buffer_cells extends a seabed downward; with `mask` it "
                "has nothing to extend -- bake the buffer into the mask file")
        if isinstance(self.seabed, float) and not (
                math.isfinite(self.seabed) and self.seabed >= 0.0):
            raise ValueError(
                f"water.seabed={self.seabed}: a flat seabed is a finite depth "
                ">= 0 in metres")
        return self


def water_of(spec) -> "WaterSpec | None":
    """The task's :class:`WaterSpec`, or ``None`` (``water`` unset or false)."""
    w = getattr(spec, "water", None)
    return w if isinstance(w, WaterSpec) else None


_RESOLVED_NOTE = (
    "A config_resolved.yaml from an older run records these keys at their "
    "defaults, which still load.")


def _is_unset(v, default) -> bool:
    if v is None:
        return True
    if isinstance(default, bool):
        return isinstance(v, (bool, int)) and not v
    try:
        return float(v) == float(default)
    except (TypeError, ValueError):
        return False


def _retired_top_level_water_keys(data, task: str):
    """Drop ``freeze_top_n_rows`` / ``freeze_water_layer`` at their old
    defaults (the shape an older run's config_resolved.yaml spells them in);
    reject any other value with the ``water:`` block that replaces it."""
    if not isinstance(data, dict):
        return data
    keys = [k for k in ("freeze_top_n_rows", "freeze_water_layer") if k in data]
    if not keys:
        return data
    data = dict(data)
    errors = []
    n = data.pop("freeze_top_n_rows", None)
    if not _is_unset(n, 0):
        try:
            last = f"0..{int(n) - 1}"
        except (TypeError, ValueError):
            last = f"0..{n}-1"
        if task == "lsrtm":
            how = ("    water:\n      mask: <boolean .npy of the native grid, "
                   f"rows {last} True>\n"
                   "LSRTM holds the water's reflectivity at zero there "
                   "(water.velocity does not apply).")
        else:
            how = ("    water:\n      mask: <boolean .npy of the native grid, "
                   f"rows {last} True>\n      velocity: <their velocity>\n"
                   "mode `pin` (the default) also stamps `velocity` into those "
                   "rows; freeze_top_n_rows kept whatever the starting model "
                   "held there.")
        errors.append(
            f"`freeze_top_n_rows: {n}` was retired -- the water layer is given "
            "once, in a top-level `water:` block, on every inversion path. "
            f"The same {n} native rows are\n{how}\n"
            "`sweep-tasks migrate-water <task.yaml>` writes that mask when the "
            "starting model holds one value in those rows.")
    fwl = data.pop("freeze_water_layer", None)
    if fwl is not None:
        if not isinstance(fwl, dict):
            errors.append("`freeze_water_layer` was retired; give the water "
                          "layer in `water:` (see sweep_tasks.schemas.WaterSpec).")
        elif fwl.get("enabled", False):
            sb = fwl.get("seabed_depth_path")
            buf = fwl.get("buffer_cells", 0) or 0
            errors.append(
                "`freeze_water_layer` was retired; its seabed goes in the "
                "top-level `water:` block:\n"
                f"    water:\n      seabed: {sb}\n"
                + (f"      buffer_cells: {buf}\n" if buf else "")
                + "      velocity: <water velocity>\n"
                "mode `pin` (the default) also stamps `velocity` into the water "
                "of the starting model. Unlike freeze_water_layer, `water:` is "
                "honoured by the conventional (static-obs) path as well -- it "
                "used to ignore the key -- and every path builds the mask with "
                "bathymetry.water_mask_from_seabed_depth (cell centre above the "
                "seabed), where plan-streaming rounded down to "
                "floor(depth / dz). The seabed must be an .npz with key "
                "'seabed_depth' (bathymetry.save_seabed_depth_npz).")
    if errors:
        raise ValueError(f"{task}: " + "\n".join(errors) + "\n" + _RESOLVED_NOTE)
    return data


_REPARAM_WATER_DEFAULTS = (("mask_water_layer", False), ("water_vp_m_s", 1500.0),
                           ("seabed_depth_path", None),
                           ("water_reset_each_stage", False))


def _retired_reparam_water_keys(data):
    """Same for ``reparam.mask_water_layer`` / ``water_vp_m_s`` /
    ``seabed_depth_path`` / ``water_reset_each_stage``."""
    if not isinstance(data, dict):
        return data
    if not any(k in data for k, _ in _REPARAM_WATER_DEFAULTS):
        return data
    data = dict(data)
    vals = {k: data.pop(k, d) for k, d in _REPARAM_WATER_DEFAULTS}
    bad = [k for k, d in _REPARAM_WATER_DEFAULTS if not _is_unset(vals[k], d)]
    if not bad:
        return data
    v = vals["water_vp_m_s"]
    vel = f"      velocity: {v}\n" if not _is_unset(v, 1500.0) else ""
    reset = ("      mode: reset_each_stage\n"
             if not _is_unset(vals["water_reset_each_stage"], False) else "")
    sb = vals["seabed_depth_path"]
    if not _is_unset(vals["mask_water_layer"], False) and sb is not None:
        how = (f"The same pin is the top-level block\n    water:\n      seabed: {sb}\n"
               f"{vel}{reset}"
               "(the conventional path used to ignore seabed_depth_path and pin "
               "the cells at exactly water_vp_m_s instead; it reads the seabed "
               "now).")
    elif not _is_unset(vals["mask_water_layer"], False) or reset:
        how = ("Without seabed_depth_path this pinned (or reset) every cell of "
               "the starting model at exactly water_vp_m_s; the water is no "
               "longer read off velocities. Give it explicitly in the top-level "
               "block\n    water:\n      seabed: <.npz, key 'seabed_depth'>   "
               "# or  mask: <boolean .npy of the native grid>\n"
               f"{vel}{reset}")
    elif sb is not None:
        how = ("seabed_depth_path did nothing without mask_water_layer. To pin "
               f"the water, write\n    water:\n      seabed: {sb}\n{vel}"
               "otherwise drop the key.")
    else:
        how = ("water_vp_m_s set the velocity the frequency-selection path "
               "pinned (every starting-model cell at exactly that value) and "
               "the VRZ Gardner water threshold; both now come from "
               f"`water.velocity` (water: {{seabed|mask: ..., velocity: {v}}}).")
    raise ValueError(
        "reparam: " + ", ".join(f"{k}={vals[k]!r}" for k in bad) + " -- the "
        "reparam water keys were retired; the water layer is given once, in a "
        "top-level `water:` block, on every inversion path. "
        + how.rstrip("\n") + "\n" + _RESOLVED_NOTE)


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
    # the true model npy (on the native grid, like the init), number of nodes
    # on the surface line, receiver stride in cells, and node depth in cells.
    # Cells are NATIVE (grid.dh) cells: a stage with its own dh_m places the
    # same physical nodes and receivers on its grid at round(x / dh_m).
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
    # When set, ignore the fixed interleaved pools and fire a fresh RANDOM
    # subset of this many nodes each iteration (matches the random ±1 path's
    # per-iter resampling). Must be <= number of comb bins (k_hi-k_lo+1).
    random_batch: int | None = Field(default=None, gt=0)
    # When set (e.g. 1500.0 = water velocity), lift each freqsel node source off
    # the seabed sediment cell into the water cell just above (per-node min):
    # node_iz = min(node_iz, seabed_iz - 1), where seabed_iz is the first
    # non-water cell (|vp - this| > 1) per (y, x) in the init vp. Keeps sources
    # off the discretized seabed (source-on-interface near-field artifact); uses
    # ``min`` so nodes already in water are left untouched. None (default) leaves
    # the shard node_grid_xyz z unchanged.
    lift_source_to_water_vp: float | None = None
    # Truncated backward (sweep BoundaryOptions.tail_steps): when set, each
    # stage's solver saves/reverses only the last ``probe_samples +
    # bwd_tail_margin`` steps. The margin buys (a) the adjoint field's decay
    # through the absorbing boundary after the probe window closes and (b) the
    # one-step restore alignment tax — so it should be >= 1, and the measured
    # gradient-vs-full cosine converges monotonically as it grows (0.992 ->
    # 1.000000 over margin 0 -> 800 on the validation model). None = exact
    # full-nt backward (default, bit-identical to before this field existed).
    bwd_tail_margin: int | None = Field(default=None, ge=0)
    # Same knob in SECONDS, converted per stage with that stage's dt. Prefer
    # this in a multi-rate cascade: the margin is a physical decay time, and a
    # flat step count silently rescales with dt — 2000 steps is 2 s at
    # dt=1 ms but 13.6 s at dt=6.8 ms, which can eat the entire saving.
    #
    # Production recipe (measured on a 3-D field-survey DD cascade): set
    # the margin on the EXPENSIVE rungs only -- the finer half of the ladder
    # low bands unset — most of the cascade saving comes from the last two
    # rungs, while a low band's long probe leaves little to skip. Per-band
    # control is native: the field lives on each stage's `frequency:` block,
    # unset = exact full-nt backward for that stage. The margin scales with
    # DOMAIN size (adjoint drain time), not with the model at hand's dt:
    # ~1-2 s sufficed on open 2-D marine lines, a ~20 km-scale 3-D volume
    # needed 6-8 s.
    bwd_tail_margin_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one_margin_convention(self):
        if self.bwd_tail_margin is not None and self.bwd_tail_margin_s is not None:
            raise ValueError(
                "set frequency.bwd_tail_margin (solver steps) or "
                "bwd_tail_margin_s (seconds), not both")
        return self

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
    4. Calls the propagator once; the ``(1, batchsize, 3)`` source array
       is what puts sweep in source-encoding mode.

    One forward + adjoint per iter regardless of ``batchsize``: the OBN
    1-GPU production path. The signs are re-drawn every iter, which
    de-correlates the cross-talk and keeps the long-run gradient close
    to the per-shot expectation.

    Pairs with :class:`PlanSamplingConfig` on ``obs.plan.sampling``, whose
    presence selects the plan-streaming loop. With ``enabled: false`` the
    block has no effect and that loop runs per shot (or per CRG). Its
    ``min_coverage`` other than 0 is then refused: it used to filter the
    receiver-first sampler's nodes even with the block off;
    ``obs.plan.sampling.min_coverage`` is that filter with or without
    encoding. With the block on, the larger of the two applies, read by
    the receiver-first sampler only (see ``PlanSamplingConfig.min_coverage``).
    """

    enabled: bool = False
    min_coverage: int = Field(ge=0, default=0)
    sign_seed: int | None = None
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


class ReceiverSmoothingSpec(_Forbid):
    """Lateral averaging over neighbouring receivers, applied to obs AND syn.

    At low frequency a marine record can be noise-dominated trace by trace
    while the signal stays laterally coherent, so averaging neighbouring
    receivers lifts the usable SNR. The operator is a row-normalised
    Gaussian over the receivers' *surface* positions (normalised
    convolution, so a varying receiver density introduces no amplitude
    bias) — see :func:`sweep_tasks.preproc.filter.receiver_smoothing_matrix`.

    ``target`` decides whether the synthetic is smoothed too.

    * ``"both"`` (default) applies the same linear operator to obs and syn,
      so the misfit stays a consistent comparison whatever the operator does
      to the signal. Autograd differentiates through it correctly.
    * ``"obs"`` smooths only the observed record, on the argument that the
      synthetic carries no noise to remove. That is only safe while the
      operator is near-identity on coherent signal — otherwise the inversion
      is asked to reproduce a laterally smeared wavefield that no model can
      generate, and it distorts the model trying.

    Whether ``"obs"`` is safe is a property of the band and sigma, and is
    measurable: smooth a noise-free synthetic and see how much it changes.
    On a field OBN dataset, ``corr(syn, S·syn)`` stays close to 1 in the
    lowest band with ``sigma_cells=2`` (so ``"obs"`` is defensible there),
    but drops clearly in the next band and with ``sigma_cells=3`` (where
    it is not).

    Trade-off on sigma: the operator averages across moveout, so the usable
    ``sigma_cells`` shrinks as frequency rises. Measured on the same data by
    comparing the amplitude a real gather retains against a phase-randomised
    control, the signal-to-noise gain peaks near ``sigma_cells=2-3`` at
    1.5-2 Hz and near ``1.5`` at 2-4 Hz; by ``3.0`` the 2-4 Hz signal is
    already being destroyed faster than the noise.
    """

    enabled: bool = False
    # Gaussian sigma in GRID CELLS (grid.dh), matching how the receiver
    # positions are stored.
    sigma_cells: float = Field(gt=0, default=2.0)
    # Truncate the kernel beyond this many sigmas.
    cutoff_sigmas: float = Field(gt=0, default=3.0)
    # Smooth both records (consistent misfit) or only the noisy observation.
    target: Literal["both", "obs"] = "both"


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
    # Relative water level: when set, the additive eps becomes
    # ``relative_epsilon * max(S*R)`` recomputed at every application and
    # the absolute ``epsilon`` above is ignored. Bounds the maximum
    # illumination boost to ``relative_epsilon ** -exponent`` (1e-3 with
    # exponent 0.5 caps at ~32x). An absolute eps is a silent no-op when
    # S*R spans many decades and its scale drifts with band/residual.
    relative_epsilon: float | None = Field(gt=0, default=None)


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
    # 3-D plan-streaming (CRG-plan + supershot) QC products. Independent
    # flags so the legacy ``shot_gather`` knob (per-physical-shot rich
    # 2-D panel) doesn't double-duty for the very different
    # encoded-supershot 3-panel. Defaults True because for the
    # plan-streaming path these are the primary visualizations.
    supershot_panel: bool = True
    well_logs: bool = True


class FreeParamSpec(_Forbid):
    """One output channel of a multi-parameter reparam (:class:`sweep_nn.MultiParamINR`).

    Each freed solver-model parameter (vp, z, and later vs, rho, eps …) gets its
    own affine scale + clamp; the init model comes from ``init_models`` matched
    by ``name``. List the entries in :attr:`ReparamSpec.free_params` in the SAME
    order as the equation's model specs — channel 0 (the first entry) is the
    PRIMARY parameter (vp), the one snapshots / DD tile render use.

    ``std`` is the per-parameter "scale": the channel's effective learning rate
    is ``reparam.lr * std``, so pick it for the parameter's magnitude (vp ~ 500,
    impedance z ~ 2). ``water_value`` is the value THIS channel holds in the
    water layer of ``water:`` (e.g. Gardner-water z -> 1.5); leave it null to
    leave the channel free there. The vp channel always holds
    ``water.velocity``, so its ``water_value`` is null or that same value.
    """

    name: str
    mean: float = 0.0
    std: float = Field(gt=0)
    bounds: ModelBounds | None = None
    water_value: float | None = None


class ReparamSpec(_Forbid):
    """Neural-network reparameterization of the velocity model (sweep-nn).

    When set on an FWISpec, the runner replaces the raw vp tensor with a
    :class:`sweep_nn.VelocityINR` (hash-encoded SIREN by default). The
    optimizer is built on the network's parameters instead of the vp
    tensor; multi-stage transitions resample only the network's *base*
    velocity, preserving all learnable parameters (SIREN's multi-scale
    benefit).

    Single-parameter by default (vp only); set :attr:`free_params` to invert
    several solver-model parameters JOINTLY from one shared-trunk network
    (:class:`sweep_nn.MultiParamINR`) — see that field.
    """

    kind: Literal["velocity_inr"] = "velocity_inr"

    @model_validator(mode="before")
    @classmethod
    def _retired_water_keys(cls, data):
        return _retired_reparam_water_keys(data)

    # Multi-parameter reparam: predict SEVERAL solver-model parameters jointly
    # from ONE shared-trunk network (sweep_nn.MultiParamINR), one output channel
    # per entry (channel 0 = vp = primary). When None (default) the reparam is
    # single-channel vp (VelocityINR) and non-vp models fall back to coupling
    # (VRZ option A: z = Gardner(vp)). When set — e.g. ``[{name: vp, std: 500,
    # bounds: {min: 1450, max: 5500}}, {name: z, std: 2.0, bounds: {min: 1.4,
    # max: 18}}]`` — vp AND z (and any further params) are inverted as free,
    # independent channels off a shared encoder + SIREN body (VRZ option C). The
    # shared-trunk hyperparameters (hidden_*, omega, hash, use_bias) below are
    # reused; each channel's affine/bounds/water come from its FreeParamSpec.
    # init models come from ``init_models`` matched by name.
    free_params: list[FreeParamSpec] | None = None
    # Warm-start: path to a saved reparam-net state_dict (a previous run's
    # ``reparam_net.pt``, dumped with ``save_net: true`` or SWEEP_SAVE_REPARAM_NET=1).
    # The runner loads it into the freshly-built network so a SECOND run continues
    # the SAME net across processes/bands (e.g. run 2-4Hz, then resume + extend at
    # 2-8Hz). With ``hash.c2f.growing`` the encoder auto-grows to the checkpoint's
    # level count on load; the hash config (levels/base/finest/log2/features) must
    # match the saved run.
    init_from: str | None = None
    # Dump the trained reparam-net state_dict to ``<task_dir>/reparam_net.pt`` at
    # the end of the run (declarative counterpart to ``init_from``): for offline
    # hash-feature analysis or warm-starting a later run. Honored by both the
    # freqsel and plan-streaming paths. The env var SWEEP_SAVE_REPARAM_NET=1 forces it
    # on regardless (backward compat). Off by default (large file).
    save_net: bool = False
    # Adam's moments are dumped alongside the weights. A resume that restores
    # only the network is NOT equivalent to a continuous run -- the first dozen
    # steps take a completely different step scale.
    save_optimizer: bool = True
    # Restore the optimizer state: a path, or "auto" to derive it from
    # ``init_from`` (``reparam_net_iter`` -> ``optim_iter``, else ``optim.pt``
    # beside it). A missing file is an error, never a silent zero state.
    init_optimizer_from: str | None = None
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
    # The water layer (pin / reset each stage) is NOT configured here: it is
    # the task's top-level ``water:`` block (WaterSpec), the same one every
    # inversion path reads. The network renders ``water.velocity`` there.
    hash: ReparamHashSpec = Field(default_factory=ReparamHashSpec)
    fourier: ReparamFourierSpec = Field(default_factory=ReparamFourierSpec)
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
    # The plan-streaming loop (obs.plan.sampling): single_step leaves it no
    # velocity gradient, so it refuses single_step together with
    # illumination_precondition / smooth_regularization;
    # under DD (SWEEP_DD_ENABLE=1) it reads neither field and refuses values
    # other than the defaults when the run starts.
    backward_mode: Literal["two_pass_full", "two_pass_chunked", "single_step"] = "two_pass_full"
    backward_chunk_rows: int = Field(ge=1, default=64)


class StageBandpass(_Forbid):
    """Per-stage bandpass applied to obs at stage entry (Gap 5).

    ``sweep_tasks.preproc.filter.bandpass`` is invoked on the *pristine* obs each
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
    # The plan-streaming loop (``obs.plan.sampling``) reads neither ``target``
    # nor ``padtype``: it filters the wavelet at each stage entry and the obs
    # every iteration with its own FFT Butterworth (no padding) and never the
    # syn, the default ``"syn"`` notwithstanding; it refuses other values.
    target: Literal["syn", "wavelet"] = "syn"


class StageSpec(_Forbid):
    """One leg of a multi-stage FWI run (frequency continuation pattern).

    Fields beyond ``epochs`` are all optional and override the top-level
    defaults when present. LSRTM reads only ``epochs``, ``lr_scale`` and
    ``wavelet`` and refuses the others (``LSRTMSpec``). An FWI path refuses
    the keys it never reads (``FWISpec``): the conventional path a stage's
    ``boundary`` and ``frequency``, frequency selection its ``wavelet``,
    ``nt``, ``batch_size`` and ``bandpass``, and both an ``inr_lr_scale``
    without ``reparam``. With stages the run is the stages' epochs, so the
    top-level ``epochs`` must equal their sum.

    - ``wavelet``: replace the source signature for this stage
    - ``lr_scale``: scale the optimizer's initial lr (multiplicative). The
      result is this stage's base lr. The top-level ``scheduler`` is one
      schedule over the whole run, evaluated on that base at the global
      epoch, so it continues across stages instead of restarting.
    - ``inr_lr_scale``: scale the reparam network's lr (multiplicative).
      Only meaningful when :class:`ReparamSpec` is active; rejected
      otherwise. Lets you slow down the SIREN/hash net on later stages
      without changing the grid-FWI lr (which is set via ``lr_scale``).
      Every param group of the network runs at it, the levels a
      ``hash.c2f.growing`` grid grows included: a level grown inside a stage
      joins at the network's lr of that moment.
      Under ``source_encoding.mode='frequency_selection'`` a stage's lr is
      ``optimizer.lr x lr_scale``, or ``reparam.lr x inr_lr_scale`` under
      reparam, where a non-default ``lr_scale`` is rejected; each stage
      starts at its own lr, never at the previous stage's.
    - ``optimizer_reset``: when ``True``, rebuild the optimizer at stage
      entry — drops accumulated Adam first / second moments. Useful when
      a stage changes regime drastically (e.g. switching from low-freq
      sweep to high-freq inversion) and the old momentum is misleading.
      Defaults to ``False``: the optimizer and its state carry over from the
      previous stage, except that a grid model's is rebuilt anyway when this
      stage's grid (``dh_m``) differs from the previous stage's, because
      Adam state is shape-bound; a reparam network's carries across a grid
      change. The rule is the same under
      ``source_encoding.mode='frequency_selection'``. Neither reset moves
      the ``scheduler``'s position. Both happen on entering the stage: a run
      resumed inside it keeps the checkpointed moments.
    - ``dh_m``: rebuild solver + resample vp to this grid spacing. The stage
      grid samples the native model (``grid.dh``) at ``i * dh_m`` from the
      native origin -- the positions geometry snaps to, ``round(x / dh_m)`` --
      and spans the whole native extent: ``ceil((n - 1) * grid.dh / dh_m) + 1``
      points per axis, the last one taking the native edge value when the
      extent is not a whole number of cells. The ``water:`` layer (the same
      physical layer, rebuilt on each stage grid), other masks and the reparam
      network's coordinates follow the same grid.
    - ``dt_s`` / ``nt``: rebuild solver at a different time grid
    - ``batch_size``: per-stage shot batch (overrides FWISpec.batchsize)
    - ``bandpass``: filter obs before this stage runs (uses sweep-preproc)
    - ``boundary``: per-stage override of
      ``backend.cuda_options.memory.boundary``. ONLY the fields you set are
      overridden; everything else inherits the top-level block, so a stage can
      move the strips to a different home without restating dtype / interval /
      ring buffers. The band cascade needs exactly this: the coarse bands'
      strips fit on the card (fastest home, and no host staging at all) while
      the fine bands do not and must be staged on the host. Read only under
      ``source_encoding.mode='frequency_selection'``.
    - ``frequency``: per-stage frequency-selection sub-spec (comb + coeff
      shards). Only used on the ``source_encoding.mode='frequency_selection'``
      path; overrides the top-level ``source_encoding.frequency`` for this
      stage so a single run can sweep bands (each with its own coeff_shards,
      k_lo/k_hi, probe/steady/slack). Pair with ``dh_m``/``dt_s`` to also
      refine the solver grid per band. The reparam network is carried across
      stages (its base is resampled to the new grid); only the comb / targets
      / pool scheduler / solver are rebuilt. That path takes a stage's record
      length (``nt``), batch (``batch_size``) and band (``bandpass``) from
      this block and models no ``wavelet``, so it rejects those four keys.
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
    frequency: "FreqSelectionSpec | None" = None
    boundary: BoundaryOptionsModel | None = None


class OptimizerAdam(_Forbid):
    """``torch.optim.Adam`` with this block's ``lr``, ``eps`` and ``betas``.

    Not under ``source_encoding.mode='frequency_selection'``: that path
    builds its own Adam at torch's defaults, ``eps=1e-8`` and ``betas=(0.9,
    0.999)``, takes only ``lr`` from here (``reparam.lr`` under reparam) and
    rejects any other kind, eps or betas. The ``eps: 1e-22`` a
    ``config_resolved.yaml`` of such a run records is this schema default,
    not what ran.
    """

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


# ``scheduler`` is ONE schedule over the whole run. Its epoch ``t`` is global:
# 0 .. total_epochs - 1 across all stages (total_epochs = the stages' epochs
# summed, or ``epochs`` without stages). Each stage has a base lr,
# ``optimizer.lr`` (``reparam.lr`` under reparam) x ``lr_scale`` (x
# ``inr_lr_scale`` under reparam), and during the stage the lr is the schedule
# at ``t`` applied to that base. Rebuilding the optimizer at a stage entry (a
# ``dh_m`` change, ``optimizer_reset``) drops Adam's moments but not the
# schedule's position, and a resumed run continues it from the checkpointed
# epoch. The conventional FWI loop and LSRTM apply it this way. The
# frequency-selection path has none: it rejects any kind but ``constant`` and
# runs each stage at its base lr.

class SchedulerConstant(_Forbid):
    """No schedule: each stage runs at its base lr."""

    kind: Literal["constant"] = "constant"


class SchedulerStep(_Forbid):
    """``lr = base * gamma ** (t // step_size)``, ``t`` the global epoch."""

    kind: Literal["step"] = "step"
    step_size: int = Field(ge=1)
    gamma: float = Field(gt=0, default=0.5)


class SchedulerExp(_Forbid):
    """``lr = base * gamma ** t``, ``t`` the global epoch."""

    kind: Literal["exp"] = "exp"
    gamma: float = Field(gt=0, lt=1.0)


class SchedulerCosine(_Forbid):
    """``lr = eta_min + (base - eta_min) * (1 + cos(pi * t / t_max)) / 2``.

    ``t`` is the global epoch. ``t_max`` defaults to the total epochs of all
    stages, so the default anneals once over the whole run; past ``t_max`` the
    cosine rises again. ``eta_min`` is an absolute lr (``lr_scale`` does not
    scale it).
    """

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
        Leave plan groups (nodes) with fewer rows than this out of the draw.
        Mirrors legacy ``encoding_min_coverage``. ``0`` = keep all groups.
        Only the receiver-first sampler applies it, through its shot -> nodes
        index (an iteration where it falls back to the random sampler draws
        from every node in the grid). The random shared-shot and per-CRG
        samplers draw from the in-grid nodes the loop hands them, and
        sweep-io's samplers ignore ``min_coverage`` when given that set, so a
        value other than 0 is refused unless ``receiver_first`` is on.
    shared_shots_per_iter
        At most this many shots per node gather and iteration (the
        ``n_shared`` dimension of the batch). When the sampler, after
        ``source_lines_per_group`` / ``max_traces_per_sourceline``, yields
        more, a random subset of this many is kept before the SEG-Y read;
        when it yields fewer, all are used and the run prints so once. With
        source encoding, the per-shot loop and ``receiver_first`` these are
        the shots every picked node shares (``dedup_mode`` then drops shots
        that share a grid cell, so an iteration can use fewer); with
        ``per_crg_independent``, each node's own receivers, one per grid
        cell. ``None`` (default) keeps them all, as every run did before
        this field was read. ``1`` is refused: it was the default then and
        capped nothing, so an older run's config_resolved.yaml records it.
    source_lines_per_group, max_traces_per_sourceline
        Hierarchical sub-sampling caps inside the shared-shot intersection
        (per node with ``per_crg_independent``). ``0`` disables the
        corresponding cap.
    num_workers
        Sizes the inner ThreadPoolExecutor that performs the per-iter
        SEG-Y reads (no torch DataLoader is involved). Defaults tuned
        for compute > I/O.
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
        Re-roll the target shot up to this many times to find one recorded
        by at least ``batchsize`` nodes before falling back to the random
        sampler. Only used when ``receiver_first=True``.
    """

    min_coverage: int = Field(ge=0, default=0)
    shared_shots_per_iter: int | None = Field(ge=1, default=None)
    source_lines_per_group: int = Field(ge=0, default=0)
    max_traces_per_sourceline: int = Field(ge=0, default=0)
    num_workers: int = Field(ge=0, default=4)
    prefetch_factor: int = Field(ge=1, default=2)
    dedup_mode: Literal["none", "first", "nearest"] = "none"
    receiver_first: bool = False
    receiver_first_max_retries: int = Field(ge=1, default=20)
    trace_cache_bytes: int = 0
    # Per-CRG independent coverage (per-shot / non-encoded path only). When
    # True, each iter draws B random nodes and gives EACH its own sub-sampled
    # rows (``sweep_io.seismic_plan.sample_percrg_independent``) instead of the
    # shared-shot intersection. Drops the wide-offset diving-wave bias of the
    # intersection. Each node is a separate solve, so it is refused with
    # source_encoding enabled (the supershot sums every node's obs over ONE
    # shared receiver set), and so is receiver_first, which it never uses: it
    # draws its nodes at random. Ragged rows are zero-padded to (B, max_nrec)
    # and the pad is masked out of the loss.
    per_crg_independent: bool = False

    @field_validator("shared_shots_per_iter")
    @classmethod
    def _one_was_the_old_default(cls, v):
        if v == 1:
            raise ValueError(
                "shared_shots_per_iter: 1 is refused. Until the plan-streaming "
                "loop read this field it defaulted to 1 and capped nothing, so "
                "the config_resolved.yaml of an older run records 1 for a run "
                "that used every shot the sampler drew; read now, 1 would cap "
                "every gather at one trace. Write null (or drop the key) to use "
                "them all, as such a run did, or a cap of 2 or more.")
        return v


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
        opt into the OBN plan-streaming supershot loop).
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

    @model_validator(mode="after")
    def _validate_wavefield(self):
        # Snapshot capture (return_wavefield + snapshot_times) exists on the
        # eager backend only. The C propagator takes no `snapshot_times` and
        # failed with an unexpected-keyword TypeError after the models were
        # already built — say what to change here instead.
        if self.backend.impl == "c":
            raise ValueError(
                "wavefield task: backend.impl='c' cannot capture snapshots — "
                "snapshot capture is eager-only; set backend: {impl: eager}. "
                "(forward / fwi / rtm / lsrtm tasks can keep impl: c.)"
            )
        # Eager chunk checkpointing keeps only the chunk boundaries and
        # recomputes the rest, so it has no interior field to hand back; sweep
        # refuses return_wavefield with it, again after the models are built.
        if self.backend.use_ckpt:
            raise ValueError(
                "wavefield task: backend.use_ckpt=true cannot return snapshots "
                "(checkpointing discards the interior field); set "
                "backend.use_ckpt: false."
            )
        nt = self.time.nt
        out_of_range = [t for t in self.snapshot_times if not 0 <= t < nt]
        if out_of_range:
            raise ValueError(
                f"wavefield task: snapshot_times {out_of_range} are outside "
                f"[0, {nt - 1}] (time.nt={nt}). They are time-step INDICES, "
                f"not seconds; use values below nt or raise time.nt."
            )
        # The solver returns snapshots sorted and de-duplicated, so an
        # unsorted or repeated list put snapshots.npy[i] and the figure title
        # "t-step snapshot_times[i]" on different time steps.
        times = list(self.snapshot_times)
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(
                f"wavefield task: snapshot_times {times} must be strictly "
                f"increasing with no repeats (e.g. {sorted(set(times))}): the "
                f"solver returns snapshots in increasing time order, so "
                f"snapshots.npy and the plot titles would not line up with "
                f"the list as written."
            )
        return self


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

    @model_validator(mode="before")
    @classmethod
    def _retired_water_keys(cls, data):
        return _retired_top_level_water_keys(data, "fwi")

    @model_validator(mode="after")
    def _water_values_need_water(self):
        # A channel's water_value is only read where `water:` says the water
        # is; without the block it would be dropped silently.
        w = water_of(self)
        fps = (self.reparam.free_params if self.reparam is not None else None) or ()
        for i, fp in enumerate(fps):
            if fp.water_value is None:
                continue
            if w is None:
                raise ValueError(
                    f"fwi: reparam.free_params[{i}] ({fp.name!r}) sets "
                    f"water_value={fp.water_value} but the task has no "
                    "`water:` block to say where the water is; add one, or "
                    "drop water_value")
            if fp.name == "vp" and float(fp.water_value) != float(w.velocity):
                raise ValueError(
                    f"fwi: reparam.free_params[{i}] (vp) water_value="
                    f"{fp.water_value} disagrees with water.velocity="
                    f"{w.velocity}; vp holds water.velocity in the water, so "
                    "drop the vp entry's water_value")
        return self

    @model_validator(mode="after")
    def _freqsel_loss(self):
        # The freqsel path builds its GCN misfit without reading ``loss``, so
        # left at the schema default the resolved config would record
        # ``kind: mse`` for a run that never evaluated it. Record the misfit
        # that runs, and reject every loss setting that path would drop.
        freq_on = (self.source_encoding is not None
                   and self.source_encoding.enabled
                   and self.source_encoding.mode == "frequency_selection")
        if not freq_on:
            _steady_gcn_needs_freqsel(self.loss, "fwi")
            return self
        # kind is checked as WRITTEN, the options by VALUE: a resolved config
        # spells out every option at its default and has to load again.
        ignored = []
        if ("kind" in self.loss.model_fields_set
                and self.loss.kind != "steady_gcn"):
            ignored.append(f"loss.kind={self.loss.kind!r}")
        ignored += [
            f"loss.{n}={getattr(self.loss, n)!r}"
            for n, f in LossSpec.model_fields.items()
            if n != "kind"
            and getattr(self.loss, n) != f.get_default(call_default_factory=True)]
        if ignored:
            raise ValueError(
                "fwi: source_encoding.mode='frequency_selection' only supports "
                "the steady-state GCN misfit (loss.kind: steady_gcn), which "
                f"takes no loss options, so {', '.join(ignored)} would be "
                "ignored. Drop the loss block or write loss: {kind: "
                "steady_gcn}; a config_resolved.yaml from an older run "
                "records the schema default kind: mse.")
        # A copy, so a LossSpec the caller passed in is not edited under them.
        self.loss = self.loss.model_copy(update={"kind": "steady_gcn"})
        return self

    @model_validator(mode="after")
    def _plan_streaming_keys(self):
        # The plan-streaming loop (tasks/fwi_plan_streaming.py) never reads
        # these, so a run that set them went ahead without them. Options are
        # judged by VALUE -- a config_resolved.yaml spells every one out at its
        # default and has to load again -- and a block by whether it asks for
        # anything (a block switched off with enabled: false asks for nothing).
        if not _plan_streaming_on(self):
            return self
        conventional = []
        if self.scheduler != SchedulerConstant():
            conventional.append(f"scheduler.kind={self.scheduler.kind!r}")
        if self.resume:
            conventional.append("resume=True")
        if self.resume_from is not None:
            conventional.append(f"resume_from={self.resume_from!r}")
        if self.train_shot_batchsize is not None:
            conventional.append(f"train_shot_batchsize={self.train_shot_batchsize}")
        if self.save_illumination:
            conventional.append("save_illumination=True")
        for name in ("grad_smooth", "diffusion_prior", "local_model_window"):
            block = getattr(self, name)
            if block is not None and block.enabled:
                conventional.append(name)
        # What shapes the dataset the conventional path materialises from the
        # plan (plan_materialize); here the sampler draws every iteration's
        # traces from it.
        plan = self.obs.plan
        conventional += [
            f"obs.plan.{n}={getattr(plan, n)!r}"
            for n in ("n_receivers_per_shot", "receiver_select", "max_shots",
                      "min_receivers", "materialize_seed")
            if getattr(plan, n) != ObsPlanConfig.model_fields[n].get_default()]
        notes = []
        if conventional:
            notes.append(
                f"To keep {', '.join(conventional)}, drop obs.plan.sampling: "
                "the conventional path (the plan materialised once) applies "
                "them.")
        if self.data_plan is not None:
            notes.append(
                "data_plan selects from a static obs tensor; here the "
                "obs.plan.sampling sampler picks the traces and model_plan "
                "crops the grid.")
        if self.modeling_override is not None:
            notes.append("modeling_override only shapes synthetic obs "
                         "(obs.synthetic_from).")
        staged = [f"stages[{i}].boundary"
                  for i, st in enumerate(self.stages or ()) if st.boundary is not None]
        if staged:
            notes.append(
                "Only source_encoding.mode='frequency_selection' reads a "
                "stage's boundary: override; set "
                "backend.cuda_options.memory.boundary for the whole run.")
        # A source_encoding block switched off asks for nothing, the frequency
        # block inside it included; a stage's is read by nothing here.
        encoded = self.source_encoding is not None and self.source_encoding.enabled
        freq = [f"stages[{i}].frequency"
                for i, st in enumerate(self.stages or ()) if st.frequency is not None]
        if encoded and self.source_encoding.frequency is not None:
            freq.insert(0, "source_encoding.frequency")
        if freq:
            notes.append(
                "Only source_encoding.mode='frequency_selection' reads a "
                "frequency: block.")
        sampling = plan.sampling
        loop = []
        if (sampling.prefetch_factor
                != PlanSamplingConfig.model_fields["prefetch_factor"].get_default()):
            loop.append(f"obs.plan.sampling.prefetch_factor={sampling.prefetch_factor}")
            notes.append(
                "The loop prefetches one iteration ahead whatever "
                "prefetch_factor says; obs.plan.sampling.num_workers sizes "
                "its read pool.")
        if encoded and sampling.per_crg_independent:
            loop.append("obs.plan.sampling.per_crg_independent=True")
            notes.append(
                "per_crg_independent is read only with source_encoding off: "
                "the encoded supershot needs the shared-shot intersection.")
        per_crg = sampling.per_crg_independent and not encoded
        if per_crg and sampling.receiver_first:
            loop.append("obs.plan.sampling.receiver_first=True")
            notes.append(
                "per_crg_independent draws its nodes at random and never uses "
                "the receiver-first sampler.")
        # min_coverage reaches the receiver-first sampler's node index only:
        # the random shared-shot and per-CRG samplers are handed the in-grid
        # nodes, and sweep-io's samplers ignore min_coverage given that set.
        # A source_encoding block switched off has no effect, its min_coverage
        # included, which used to reach that index all the same.
        receiver_first = sampling.receiver_first and not per_crg
        se = self.source_encoding
        cover = []
        if sampling.min_coverage and not receiver_first:
            cover.append(f"obs.plan.sampling.min_coverage={sampling.min_coverage}")
        if se is not None and se.min_coverage and not (se.enabled and receiver_first):
            cover.append(f"source_encoding.min_coverage={se.min_coverage}")
        if cover:
            loop += cover
            if not receiver_first:
                notes.append(
                    "min_coverage is applied only by the receiver-first sampler "
                    "(obs.plan.sampling.receiver_first: true, without "
                    "per_crg_independent): the random shared-shot and per-CRG "
                    "samplers draw from every node inside the grid, so earlier "
                    "runs did not apply it either.")
            if se is not None and se.min_coverage and not se.enabled:
                notes.append(
                    "A source_encoding block switched off has no effect. With "
                    "receiver_first its min_coverage still filtered that "
                    "sampler's nodes before; obs.plan.sampling.min_coverage is "
                    "that filter without encoding.")
        # From a bandpass the loop takes lo_hz, hi_hz and order: it filters the
        # wavelet and the observed data with its own FFT Butterworth, whatever
        # target and padtype say.
        bands = [("bandpass", self.bandpass)] + [
            (f"stages[{i}].bandpass", st.bandpass)
            for i, st in enumerate(self.stages or ())]
        band = [f"{where}.{n}={getattr(bp, n)!r}"
                for where, bp in bands if bp is not None
                for n in ("target", "padtype")
                if getattr(bp, n) != StageBandpass.model_fields[n].get_default()]
        if band:
            loop += band
            notes.append(
                "Of a bandpass this path reads lo_hz, hi_hz and order: it "
                "filters the source wavelet at each stage entry and the "
                "observed data every iteration with its own zero-phase FFT "
                "Butterworth, without padding, and never the synthetic, "
                "whatever target and padtype say (the conventional path's "
                "target: wavelet).")
        ignored = (conventional
                   + [n for n in ("data_plan", "modeling_override")
                      if getattr(self, n) is not None]
                   + staged + freq + loop)
        if ignored:
            raise ValueError(
                "fwi: the plan-streaming path (obs.plan.sampling) never reads "
                f"{', '.join(ignored)}, so they would be ignored. Remove "
                "them. " + " ".join(notes) + " A config_resolved.yaml records "
                "every key a run did not set at its default, which still "
                "loads.")
        return self

    @model_validator(mode="after")
    def _plan_streaming_single_step(self):
        # With backward_mode single_step the plan-streaming loop differentiates
        # through solver and network in one graph: no velocity gradient exists
        # for these to work on, and the loop skips them (as the DD loop
        # does, refused when a DD run starts). The config alone decides this,
        # so refuse it at load; a block switched off asks for nothing.
        if (not _plan_streaming_on(self) or self.reparam is None
                or self.reparam.backward_mode != "single_step"):
            return self
        skipped = _leaf_gradient_features(self)
        if skipped:
            it = "it" if len(skipped) == 1 else "them"
            raise ValueError(
                "fwi: the plan-streaming path (obs.plan.sampling) with "
                "reparam.backward_mode='single_step' does not support "
                f"{', '.join(n for n, _ in skipped)}: single_step sends the "
                "gradient straight to the network, so the loop has no velocity "
                f"gradient for {it} and skips {it}; the run would go ahead "
                f"without {it}. Use backward_mode: two_pass_full or "
                f"two_pass_chunked, which hand {it} that gradient, or turn {it} "
                "off (" + ", ".join(f"{n}.{off}" for n, off in skipped) + "). "
                "The config_resolved.yaml of an earlier such run records "
                f"{it} as set, though that run went without {it}.")
        return self

    @model_validator(mode="after")
    def _freqsel_optimizer(self):
        # The freqsel path builds its own torch Adam at torch's default eps and
        # betas, runs every stage at a constant lr, and never reads
        # ``scheduler``, ``optimizer.kind`` / ``eps`` / ``betas`` or, under
        # reparam, a stage's ``lr_scale``. Reject what it would drop, by the
        # rule of _freqsel_loss: a kind as written (it is the discriminator,
        # so a written block always names one), an option by VALUE.
        freq_on = (self.source_encoding is not None
                   and self.source_encoding.enabled
                   and self.source_encoding.mode == "frequency_selection")
        if not freq_on:
            return self
        errors = []
        if self.scheduler.kind != "constant":
            errors.append(
                f"scheduler.kind={self.scheduler.kind!r} would be ignored: "
                "there is no LR scheduler, every stage runs at a constant lr "
                "(optimizer.lr x lr_scale, or reparam.lr x inr_lr_scale under "
                "reparam). Drop the scheduler block or write scheduler: {kind: "
                "constant}; to lower the lr band by band, set the stages' "
                "scales.")
        if self.optimizer.kind != "adam":
            errors.append(
                f"optimizer.kind={self.optimizer.kind!r} would be ignored: it "
                "always runs Adam, at torch's default eps=1e-8 and betas=(0.9, "
                "0.999). Write optimizer: {kind: adam, lr: ...}.")
        else:
            opts = [f"optimizer.{n}={getattr(self.optimizer, n)!r}"
                    for n in ("eps", "betas")
                    if getattr(self.optimizer, n)
                    != OptimizerAdam.model_fields[n].get_default()]
            if opts:
                errors.append(
                    f"{', '.join(opts)} would be ignored: it runs Adam at "
                    "torch's default eps=1e-8 and betas=(0.9, 0.999) and reads "
                    "only optimizer.lr. Drop them; the schema default a "
                    "config_resolved.yaml records (eps: 1e-22) loads, but is "
                    "not what ran.")
        if self.reparam is not None:
            scaled = [f"stages[{i}].lr_scale={s.lr_scale!r}"
                      for i, s in enumerate(self.stages or ())
                      if s.lr_scale != StageSpec.model_fields["lr_scale"].default]
            if scaled:
                errors.append(
                    f"{', '.join(scaled)} would be ignored: under reparam each "
                    "stage runs at reparam.lr x inr_lr_scale, without lr_scale "
                    "(the conventional path multiplies both). Drop lr_scale to "
                    "keep what earlier runs of this config did, or move the "
                    "factor into inr_lr_scale.")
        if errors:
            raise ValueError(
                "fwi: source_encoding.mode='frequency_selection' builds its own "
                "optimizer, so " + " Also, ".join(errors))
        return self

    @model_validator(mode="after")
    def _stage_keys_the_path_reads(self):
        # The conventional loop (_helpers/stages._prepare_stage) and the
        # frequency-selection loop each read only some of a stage's keys;
        # reject a key the path that runs never reads rather than run the
        # stage without it. By VALUE against the schema default, as for
        # LSRTM: a config_resolved.yaml spells every key out at its default
        # and has to load again; for the blocks (default None) that is "the
        # block is there". The plan-streaming path checks its own stage keys.
        if not self.stages or _plan_streaming_on(self):
            return self
        freq_on = (self.source_encoding is not None
                   and self.source_encoding.enabled
                   and self.source_encoding.mode == "frequency_selection")
        if freq_on:
            path = "source_encoding.mode='frequency_selection'"
            instead = dict(_FREQSEL_STAGE_INSTEAD)
        else:
            path = ("the conventional path (no frequency selection, no "
                    "obs.plan.sampling)")
            instead = dict(_CONVENTIONAL_STAGE_INSTEAD)
        if self.reparam is None:
            instead["inr_lr_scale"] = _INR_LR_SCALE_WITHOUT_REPARAM
        for i, stage in enumerate(self.stages):
            ignored = [n for n in instead
                       if getattr(stage, n) != StageSpec.model_fields[n].get_default(
                           call_default_factory=True)]
            if not ignored:
                continue
            what = [f"a {n} block" if isinstance(getattr(stage, n), BaseModel)
                    else f"{n}={getattr(stage, n)!r}" for n in ignored]
            them = "them" if len(ignored) > 1 else "it"
            raise ValueError(
                f"fwi: stages[{i}] sets {', '.join(what)}, which {path} never "
                f"reads, so the stage would run without {them}. Instead -- "
                + " ".join(f"{n}: {instead[n]}." for n in ignored)
                + f" An older run's config_resolved.yaml hits this only if its "
                f"stages set {them}; that run ignored {them} too.")
        return self

    @model_validator(mode="after")
    def _epochs_are_the_stages(self):
        # With stages, the conventional and frequency-selection loops run the
        # stages' epochs and never read the top-level ``epochs`` the schema
        # requires. Accept it only as their sum, the number that runs; a
        # config_resolved.yaml records it as written, so a run's own record
        # loads again. The plan-streaming path checks its own keys.
        if not self.stages or _plan_streaming_on(self):
            return self
        total = sum(int(s.epochs) for s in self.stages)
        if int(self.epochs) != total:
            parts = " + ".join(str(int(s.epochs)) for s in self.stages)
            raise ValueError(
                f"fwi: epochs={self.epochs}, but a run with stages runs the "
                f"stages' epochs, {parts} = {total}, and never reads the "
                f"top-level epochs. Set epochs: {total}, or change the stages. "
                "An older run's config_resolved.yaml hits this only if its "
                "epochs differed from that sum; that run ran the stages' "
                f"{total} epochs.")
        return self

    @model_validator(mode="after")
    def _plan_materialisation_reads_dt_only(self):
        # obs.plan without sampling materialises the dataset straight from the
        # plan (plan_materialize) and never applies data_plan to it: of that
        # block only dt_target_s is read, as the solver dt the plan's traces
        # are resampled to. Reject the other fields by VALUE against the
        # schema default, so a config_resolved.yaml (every field written out)
        # loads again.
        freq_on = (self.source_encoding is not None
                   and self.source_encoding.enabled
                   and self.source_encoding.mode == "frequency_selection")
        plan = self.obs.plan if self.obs is not None else None
        if (freq_on or self.data_plan is None or plan is None
                or plan.sampling is not None):
            return self
        ignored = [f"data_plan.{n}={getattr(self.data_plan, n)!r}"
                   for n, f in DataPlanSpec.model_fields.items()
                   if n != "dt_target_s"
                   and getattr(self.data_plan, n) != f.get_default(
                       call_default_factory=True)]
        if ignored:
            raise ValueError(
                "fwi: obs.plan without sampling materialises the dataset from "
                "the plan and reads only data_plan.dt_target_s, so "
                f"{', '.join(ignored)} would be ignored. Select shots, "
                "receivers and offsets when building the plan (sweep-tasks "
                "build-plan) or cap them with obs.plan.max_shots / "
                "n_receivers_per_shot, and keep only dt_target_s in data_plan. "
                "An older run's config_resolved.yaml hits this only if its "
                "data_plan set them; that run ignored them too.")
        return self

    # With ``stages`` the run is the stages' epochs: the conventional and
    # frequency-selection paths require this to be their sum.
    epochs: int = Field(ge=1)
    batchsize: int = Field(ge=1, default=1)
    train_shot_batchsize: int | None = None  # default = batchsize (no accumulation)
    show_every: int = 10

    # Phase-1 patch fields:
    model_bounds: dict[str, ModelBounds] = Field(default_factory=dict)
    # The water layer, for every inversion path below (conventional, frequency
    # selection, plan streaming, reparam): see WaterSpec. ``false`` = this
    # model has no water layer.
    water: WaterSpec | Literal[False] | None = None
    stages: list[StageSpec] | None = None
    resume_from: str | None = None  # task_id under output_dir to resume from
    # Auto-resume from the *same* task_dir's checkpoint.pt when it exists
    # (no need to set `resume_from` to the same task_id). Pair this with
    # a fixed ``task_id:`` in the YAML so the directory is deterministic;
    # ctrl-c during training will save a checkpoint at the next iter
    # boundary, and re-running the same YAML picks up where it left off.
    # When True but the file is absent, the run starts fresh silently.
    # The checkpoint records its stage (the models are on that stage's grid)
    # and, under reparam, the network, so a stop anywhere -- inside a dh_m
    # stage included -- continues as the uninterrupted run would.
    resume: bool = False
    save_illumination: bool = False

    modeling_override: ModelingOverride | None = None

    # Pre-FWI data selection + model cropping (TASK 009).
    # `data_plan` subsets shots/receivers/time before the inversion sees obs.
    # With `obs.plan` (no sampling) only its `dt_target_s` applies: the
    # dataset is materialised from the plan, and the other fields are
    # rejected (select there, or with obs.plan's own caps).
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
    # the differentiable :func:`sweep_tasks.preproc.filter.bandpass_torch`).
    # Used by the OBN CRG path only, which reads lo_hz / hi_hz / order of it
    # (not target or padtype); the multi-stage 2-D / 3-D path
    # uses :class:`StageBandpass` inside ``stages`` instead.
    bandpass: StageBandpass | None = None

    # Optional lateral averaging of neighbouring receivers, applied to BOTH
    # obs and syn right before the loss. See :class:`ReceiverSmoothingSpec`.
    receiver_smoothing: ReceiverSmoothingSpec | None = None

    # Optional Sobolev / TV-style smoothness regularizer on vp. See
    # :class:`SmoothRegSpec`. The runner adds ``weight * TVPrior(vp)``
    # to the data misfit; the gradient propagates back into the vp
    # tensor (or net params in reparam mode).
    smooth_regularization: SmoothRegSpec | None = None

    # Optional plug-and-play DDPM diffusion prior on vp. See
    # :class:`DiffusionPriorSpec`. The runner builds a
    # ``sweep_nn.diffusion.DiffusionVelocityPrior`` from ``ckpt_path`` and adds
    # ``weight * red_loss(vp)`` (or ``sds_loss``) to the data misfit every
    # ``every`` steps from stage ``start_band`` on; the gradient propagates back
    # into vp (or net params in reparam mode), exactly like smooth_regularization.
    diffusion_prior: DiffusionPriorSpec | None = None

    # Optional Gaussian smoothing of the vp gradient before the optimizer
    # step (tomographic gradient preconditioner). See :class:`GradSmoothSpec`.
    # Turns an oscillatory single-band traveltime kernel into a smooth,
    # low-wavenumber background update. Grid-mode only.
    grad_smooth: GradSmoothSpec | None = None

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
    # ``sweep_tasks.viz.colormaps`` (registered as a matplotlib cmap at import
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

    @model_validator(mode="after")
    def _no_steady_gcn(self):
        _steady_gcn_needs_freqsel(self.loss, "rtm")
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
    # Where the water is (WaterSpec). LSRTM holds the water's reflectivity at
    # zero (mode pin) or resets it to zero at every stage; water.velocity
    # does not apply.
    water: WaterSpec | Literal[False] | None = None
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

    @model_validator(mode="before")
    @classmethod
    def _retired_water_keys(cls, data):
        return _retired_top_level_water_keys(data, "lsrtm")

    @model_validator(mode="after")
    def _water_velocity_unused(self):
        w = water_of(self)
        if w is not None and w.velocity != WaterSpec.model_fields["velocity"].default:
            raise ValueError(
                "lsrtm: the inverted field is reflectivity, which `water:` holds "
                "at zero in the water (or resets to zero each stage), so "
                f"water.velocity={w.velocity} does not apply; drop it")
        return self

    @model_validator(mode="after")
    def _stage_keys_lsrtm_reads(self):
        # The LSRTM stage loop reads only epochs, lr_scale and wavelet; reject
        # every other stage key rather than run the stage without it. Checked
        # by VALUE against the schema default: a config_resolved.yaml spells
        # each key out at its default and has to load again. The blocks
        # (bandpass, frequency, boundary) default to None, so for them this
        # is "the block is there".
        for i, stage in enumerate(self.stages or ()):
            ignored = [n for n, f in StageSpec.model_fields.items()
                       if n not in _LSRTM_STAGE_KEYS_READ
                       and getattr(stage, n) != f.get_default(call_default_factory=True)]
            if not ignored:
                continue
            what = [f"a {n} block" if isinstance(getattr(stage, n), BaseModel)
                    else f"{n}={getattr(stage, n)!r}" for n in ignored]
            instead = " ".join(
                f"{n}: {_LSRTM_STAGE_INSTEAD.get(n, 'remove it')}." for n in ignored)
            them = "them" if len(ignored) > 1 else "it"
            raise ValueError(
                f"lsrtm: stages[{i}] sets {', '.join(what)}, which LSRTM does "
                "not support: an LSRTM stage reads only epochs, lr_scale and "
                f"wavelet, so the stage would run without {them}. Instead -- "
                f"{instead} An older run's config_resolved.yaml hits this only "
                f"if its stages set {them}; that run ignored {them} too.")
        return self

    @model_validator(mode="after")
    def _no_steady_gcn(self):
        _steady_gcn_needs_freqsel(self.loss, "lsrtm")
        return self


# The stage keys the LSRTM loop reads, and what to do instead of each one it
# does not: one grid, time axis, shot batch, solver and optimizer serve every
# stage of an LSRTM run.
_LSRTM_STAGE_KEYS_READ = ("epochs", "lr_scale", "wavelet")
_LSRTM_STAGE_INSTEAD = {
    "dh_m": "set grid.dh (every stage runs on the top-level grid)",
    "dt_s": "set time.dt (every stage runs on the top-level time axis)",
    "nt": "set time.nt (every stage runs on the top-level time axis)",
    "batch_size": "set the top-level batchsize",
    "bandpass": ("remove it; to invert one band, band-limit the source wavelet "
                 "(wavelet, and modeling_override.wavelet if set, which models "
                 "the observed data)"),
    "optimizer_reset": "remove it (LSRTM keeps one optimizer for the whole run)",
    "inr_lr_scale": ("remove it (it scales a reparam network's lr; LSRTM has "
                     "none) and use lr_scale"),
    "boundary": "set backend.cuda_options.memory.boundary (one solver serves every stage)",
    "frequency": "remove it (frequency selection is an FWI source_encoding mode)",
}

# The stage keys an FWI path never reads, and what to do instead
# (FWISpec._stage_keys_the_path_reads). The conventional loop reads every
# other key; frequency selection reads epochs, dh_m, dt_s, lr_scale (a grid
# model) or inr_lr_scale (reparam), optimizer_reset, frequency and boundary.
_CONVENTIONAL_STAGE_INSTEAD = {
    "boundary": ("set backend.cuda_options.memory.boundary (one backend serves "
                 "every stage; only source_encoding.mode='frequency_selection' "
                 "reads a stage's boundary: override)"),
    "frequency": ("remove it (a stage's frequency: block is the comb of "
                  "source_encoding.mode='frequency_selection')"),
}
_FREQSEL_STAGE_INSTEAD = {
    "wavelet": ("remove it (no wavelet is modeled: each node emits its comb "
                "bins, turned on over frequency.ramp_s, and the misfit is blind "
                "to the source spectrum)"),
    "nt": ("set steady_samples, slack_samples and probe_samples in its "
           "frequency: block (the record length is their sum)"),
    "batch_size": ("set n_pools or random_batch in its frequency: block (the "
                   "pool is the batch)"),
    "bandpass": ("set k_lo and k_hi in its frequency: block (the comb is the "
                 "band; nothing is filtered)"),
}
_INR_LR_SCALE_WITHOUT_REPARAM = (
    "use lr_scale (inr_lr_scale scales a reparam network's lr, and there is no "
    "reparam: block)")


TaskSpec = Annotated[
    Union[IntrospectSpec, ForwardSpec, WavefieldSpec, FWISpec, LSRTMSpec, RTMSpec],
    Discriminator("task_type"),
]
