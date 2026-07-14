"""Unified Plan abstraction: ``seismic_plan_v1`` schema + lazy reader.

This module sits one layer above :class:`SEGYIndex` (which is a per-trace
catalog of every physical SEG-Y trace) and one layer below the FWI
runner. It exists so 2-D streamer (CSG) and 3-D OBN (CRG) workflows
share one storage format, one reader, and one set of filter primitives —
instead of the two separate paths the codebase historically grew.

Three-layer pipeline::

    SEG-Y files                          (raw bytes; never modified)
        │  build_segy_index (MPI scan)
        ▼
    SEGYIndex                            (per-trace catalog; no samples)
        │  build_seismic_plan(index, grouping=..., filter=...)
        ▼
    SeismicPlan                          (filter + grouping applied; npz)
        │  PlanReader(plan).read_group(g)
        ▼
    np.ndarray  (n_in_group, nt) f32     (lazy SEG-Y trace read)

A plan is "the user's declaration of which traces to use, organised by
grouping". Filtering, dedupe, offset windowing, source-cell mean-stacks
— anything that *selects* or *aggregates* trace metadata — happens at
plan-build time, never inside the FWI iteration loop. The runner just
calls :meth:`PlanReader.read_group` for whatever groups its batch
sampler picks.

Schema (``format = "seismic_plan_v1"``)::

    format                : "seismic_plan_v1"   (scalar str U32)
    grouping              : "csg" | "crg" | "supershot" (scalar str U16)

    # File registry — every per-trace byte_offset addresses one of these
    files                 : (n_files,)            U512    SEG-Y paths
    trace_size_per_file   : (n_files,)            int64   bytes per trace inc. header
    sample_format         : scalar                int32   SEG-Y rev1 format code
    samples_per_trace     : scalar                int32   nt of every trace
    dt_s                  : scalar                float64

    # Per-row pointers (n_rows total physical traces kept by the plan)
    row_file_id           : (n_rows,)             int64
    row_trace_offset      : (n_rows,)             int64
    row_source_xyz        : (n_rows, 3)           float64 (m, UTM)
    row_receiver_xyz      : (n_rows, 3)           float64 (m, UTM)

    # Group structure (CSR-style row pointer)
    group_id              : (n_groups,)           int64
    group_xyz             : (n_groups, 3)         float64
    group_offsets         : (n_groups + 1,)       int64
    group_member_count    : derived from offsets diff

    # Provenance JSON (build args, source SEGYIndex hash, etc.)
    build_meta            : scalar JSON-encoded str

``grouping`` semantics:

* ``"csg"``    — common-shot gather. ``group_id`` = source's FFID,
  ``group_xyz`` = source position, rows = receivers for that shot.
* ``"crg"``    — common-receiver gather. ``group_id`` = quantized
  receiver-cell index, ``group_xyz`` = receiver position, rows = shots
  recorded by that receiver.
* ``"supershot"`` — source-encoded supershot. ``group_xyz`` is the
  centroid of the encoded sources, rows are receiver traces.

Per-row arrays always store both source and receiver positions so the
"other endpoint" geometry is available regardless of grouping mode.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .segy import MultiFileSEGYReader


SCHEMA_FORMAT = "seismic_plan_v1"
VALID_GROUPINGS = ("csg", "crg", "supershot")


def _remap_segy_path(p: str | os.PathLike) -> str:
    """Apply ``FWI_SEGY_ROOT`` / ``FWI_SEGY_REMAP`` env-var remaps.

    Shared verbatim with the legacy ``crg_plan._remap_segy_path`` so
    plans built on one host work on another after a single env var.
    """
    s = str(p)
    root = os.environ.get("FWI_SEGY_ROOT", "").strip()
    if root:
        return os.path.join(root, os.path.basename(s))
    rules = os.environ.get("FWI_SEGY_REMAP", "").strip()
    if rules:
        for rule in rules.split(";"):
            rule = rule.strip()
            if not rule or ":" not in rule:
                continue
            old, new = rule.split(":", 1)
            if s.startswith(old):
                return new + s[len(old):]
    return s


# ============================================================================
# SeismicPlan
# ============================================================================
@dataclass(frozen=True)
class SeismicPlan:
    """A filter + grouping view over a :class:`SEGYIndex`, persisted to npz.

    Carries enough metadata to read the underlying SEG-Y traces lazily
    (via :class:`PlanReader`) but holds NO sample data itself —
    on-disk size is O(n_rows × 80 bytes), not O(n_rows × nt × 4).

    Construct via :func:`build_seismic_plan` or load via :meth:`load`.
    """

    grouping: str
    files: list[Path]
    trace_size_per_file: np.ndarray  # (n_files,) int64
    sample_format: int
    samples_per_trace: int
    dt_s: float

    row_file_id: np.ndarray       # (n_rows,) int64
    row_trace_offset: np.ndarray  # (n_rows,) int64
    row_source_xyz: np.ndarray    # (n_rows, 3) float64
    row_receiver_xyz: np.ndarray  # (n_rows, 3) float64

    group_id: np.ndarray          # (n_groups,) int64
    group_xyz: np.ndarray         # (n_groups, 3) float64
    group_offsets: np.ndarray     # (n_groups + 1,) int64

    build_meta: dict = field(default_factory=dict)

    # ----------------------------------------------------------- validation
    def __post_init__(self) -> None:
        if self.grouping not in VALID_GROUPINGS:
            raise ValueError(
                f"SeismicPlan: grouping {self.grouping!r} must be one of "
                f"{VALID_GROUPINGS}"
            )
        n_rows = int(self.row_file_id.shape[0])
        for name in ("row_trace_offset",):
            if int(getattr(self, name).shape[0]) != n_rows:
                raise ValueError(
                    f"SeismicPlan: {name} length {getattr(self, name).shape[0]} "
                    f"!= row_file_id length {n_rows}"
                )
        for name, expected in (("row_source_xyz", (n_rows, 3)),
                               ("row_receiver_xyz", (n_rows, 3))):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(
                    f"SeismicPlan: {name} shape {getattr(self, name).shape} "
                    f"!= expected {expected}"
                )
        n_groups = int(self.group_id.shape[0])
        if tuple(self.group_xyz.shape) != (n_groups, 3):
            raise ValueError(
                f"SeismicPlan: group_xyz shape {self.group_xyz.shape} "
                f"!= ({n_groups}, 3)"
            )
        if int(self.group_offsets.shape[0]) != n_groups + 1:
            raise ValueError(
                f"SeismicPlan: group_offsets length {self.group_offsets.shape[0]} "
                f"!= n_groups + 1 = {n_groups + 1}"
            )
        if int(self.group_offsets[0]) != 0:
            raise ValueError(
                f"SeismicPlan: group_offsets[0] must be 0, got "
                f"{int(self.group_offsets[0])}"
            )
        if int(self.group_offsets[-1]) != n_rows:
            raise ValueError(
                f"SeismicPlan: group_offsets[-1] must equal n_rows "
                f"({n_rows}), got {int(self.group_offsets[-1])}"
            )
        if int(self.trace_size_per_file.shape[0]) != len(self.files):
            raise ValueError(
                f"SeismicPlan: trace_size_per_file length "
                f"{self.trace_size_per_file.shape[0]} != n_files "
                f"{len(self.files)}"
            )
        # Bounds-check row_file_id.
        if n_rows > 0:
            max_fid = int(self.row_file_id.max())
            if max_fid >= len(self.files):
                raise ValueError(
                    f"SeismicPlan: row_file_id max {max_fid} >= n_files "
                    f"{len(self.files)}"
                )

    # ------------------------------------------------------------ derived
    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def n_rows(self) -> int:
        return int(self.row_file_id.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.group_id.shape[0])

    def group_slice(self, g: int) -> slice:
        return slice(int(self.group_offsets[int(g)]),
                     int(self.group_offsets[int(g) + 1]))

    def group_row_count(self, g: int) -> int:
        return int(self.group_offsets[int(g) + 1]
                   - self.group_offsets[int(g)])

    def per_group_row_counts(self) -> np.ndarray:
        return np.diff(self.group_offsets).astype(np.int64)

    # ----------------------------------------------------- serialization
    def save(self, path: str | Path) -> Path:
        """Save as a single npz; loads back via :meth:`load`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps({
            "schema_version": 1,
            "build_meta": self.build_meta,
        })
        np.savez(
            path,
            format=np.asarray(SCHEMA_FORMAT, dtype="U32"),
            grouping=np.asarray(self.grouping, dtype="U16"),
            files=np.asarray([str(p) for p in self.files], dtype="U512"),
            trace_size_per_file=np.asarray(self.trace_size_per_file, dtype=np.int64),
            sample_format=np.asarray(self.sample_format, dtype=np.int32),
            samples_per_trace=np.asarray(self.samples_per_trace, dtype=np.int32),
            dt_s=np.asarray(self.dt_s, dtype=np.float64),
            row_file_id=self.row_file_id.astype(np.int64, copy=False),
            row_trace_offset=self.row_trace_offset.astype(np.int64, copy=False),
            row_source_xyz=self.row_source_xyz.astype(np.float64, copy=False),
            row_receiver_xyz=self.row_receiver_xyz.astype(np.float64, copy=False),
            group_id=self.group_id.astype(np.int64, copy=False),
            group_xyz=self.group_xyz.astype(np.float64, copy=False),
            group_offsets=self.group_offsets.astype(np.int64, copy=False),
            _meta=meta_json,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SeismicPlan":
        """Load a ``seismic_plan_v1`` npz."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            fmt = str(np.asarray(z["format"]).item())
            if fmt != SCHEMA_FORMAT:
                raise ValueError(
                    f"SeismicPlan.load: unexpected format {fmt!r} in {path}; "
                    f"expected {SCHEMA_FORMAT!r}"
                )
            grouping = str(np.asarray(z["grouping"]).item())
            files = [Path(_remap_segy_path(p)) for p in z["files"]]
            meta = json.loads(str(z["_meta"]))
            return cls(
                grouping=grouping,
                files=files,
                trace_size_per_file=np.asarray(z["trace_size_per_file"], dtype=np.int64),
                sample_format=int(np.asarray(z["sample_format"]).item()),
                samples_per_trace=int(np.asarray(z["samples_per_trace"]).item()),
                dt_s=float(np.asarray(z["dt_s"]).item()),
                row_file_id=np.asarray(z["row_file_id"], dtype=np.int64),
                row_trace_offset=np.asarray(z["row_trace_offset"], dtype=np.int64),
                row_source_xyz=np.asarray(z["row_source_xyz"], dtype=np.float64),
                row_receiver_xyz=np.asarray(z["row_receiver_xyz"], dtype=np.float64),
                group_id=np.asarray(z["group_id"], dtype=np.int64),
                group_xyz=np.asarray(z["group_xyz"], dtype=np.float64),
                group_offsets=np.asarray(z["group_offsets"], dtype=np.int64),
                build_meta=meta.get("build_meta", {}) or {},
            )

    # ----------------------------------------------------- filter helpers
    def filter_rows(self, row_mask: np.ndarray) -> "SeismicPlan":
        """Drop rows where ``row_mask`` is False; recompute group_offsets.

        Groups that lose every row survive with a zero-length slice;
        compose with :meth:`drop_empty_groups` to remove them.
        """
        m = np.asarray(row_mask, dtype=bool)
        if m.shape != (self.n_rows,):
            raise ValueError(
                f"filter_rows: mask shape {m.shape} != ({self.n_rows},)"
            )
        if m.all():
            return self
        new_counts = np.zeros(self.n_groups, dtype=np.int64)
        for g in range(self.n_groups):
            sl = self.group_slice(g)
            new_counts[g] = int(m[sl].sum())
        new_offsets = np.empty(self.n_groups + 1, dtype=np.int64)
        new_offsets[0] = 0
        new_offsets[1:] = np.cumsum(new_counts)
        return SeismicPlan(
            grouping=self.grouping,
            files=list(self.files),
            trace_size_per_file=self.trace_size_per_file.copy(),
            sample_format=int(self.sample_format),
            samples_per_trace=int(self.samples_per_trace),
            dt_s=float(self.dt_s),
            row_file_id=self.row_file_id[m].copy(),
            row_trace_offset=self.row_trace_offset[m].copy(),
            row_source_xyz=self.row_source_xyz[m].copy(),
            row_receiver_xyz=self.row_receiver_xyz[m].copy(),
            group_id=self.group_id.copy(),
            group_xyz=self.group_xyz.copy(),
            group_offsets=new_offsets,
            build_meta={**self.build_meta, "post_filter_rows": int(self.n_rows - m.sum())},
        )

    def filter_groups(self, group_mask: np.ndarray) -> "SeismicPlan":
        """Drop groups where ``group_mask`` is False; recompute offsets + row arrays."""
        m = np.asarray(group_mask, dtype=bool)
        if m.shape != (self.n_groups,):
            raise ValueError(
                f"filter_groups: mask shape {m.shape} != ({self.n_groups},)"
            )
        if m.all():
            return self
        keep_groups = np.flatnonzero(m)
        new_counts = self.per_group_row_counts()[keep_groups]
        new_offsets = np.empty(keep_groups.size + 1, dtype=np.int64)
        new_offsets[0] = 0
        new_offsets[1:] = np.cumsum(new_counts)
        if keep_groups.size == 0:
            rows_kept = np.empty(0, dtype=np.int64)
        else:
            rows_kept = np.concatenate(
                [np.arange(int(self.group_offsets[g]),
                           int(self.group_offsets[g + 1]),
                           dtype=np.int64) for g in keep_groups]
            )
        return SeismicPlan(
            grouping=self.grouping,
            files=list(self.files),
            trace_size_per_file=self.trace_size_per_file.copy(),
            sample_format=int(self.sample_format),
            samples_per_trace=int(self.samples_per_trace),
            dt_s=float(self.dt_s),
            row_file_id=self.row_file_id[rows_kept].copy(),
            row_trace_offset=self.row_trace_offset[rows_kept].copy(),
            row_source_xyz=self.row_source_xyz[rows_kept].copy(),
            row_receiver_xyz=self.row_receiver_xyz[rows_kept].copy(),
            group_id=self.group_id[keep_groups].copy(),
            group_xyz=self.group_xyz[keep_groups].copy(),
            group_offsets=new_offsets,
            build_meta={**self.build_meta, "post_filter_groups": int(self.n_groups - m.sum())},
        )

    def drop_empty_groups(self) -> "SeismicPlan":
        """Drop groups with zero rows; preserves order of surviving groups."""
        counts = self.per_group_row_counts()
        if counts.all():
            return self
        return self.filter_groups(counts > 0)


# ============================================================================
# Builder: SEGYIndex -> SeismicPlan
# ============================================================================
def build_seismic_plan(
    index,
    *,
    grouping: str = "csg",
    shot_ids: Sequence[int] | None = None,
    receiver_quantize_m: float | None = None,
    offset_min_m: float | None = None,
    offset_max_m: float | None = None,
    max_traces_per_group: int | None = None,
    seed: int = 0,
    build_label: str | None = None,
) -> SeismicPlan:
    """Build a :class:`SeismicPlan` from a :class:`SEGYIndex`.

    Parameters
    ----------
    index
        A :class:`sweep_io.segy_index.SEGYIndex` — the full per-trace
        catalog. Not mutated.
    grouping
        ``"csg"`` (default) → one group per unique ``shot_id`` (FFID).
        ``"crg"`` → one group per receiver cell (use
        ``receiver_quantize_m`` to define the cell size).
    shot_ids
        Optional whitelist of shot IDs to keep (intersected with what
        the index has). ``None`` = all shots in the index.
    receiver_quantize_m
        For ``grouping="crg"``: quantization tolerance for collapsing
        per-trace receiver positions to a finite set of cells. Required
        for CRG; ignored for CSG.
    offset_min_m, offset_max_m
        Per-trace source-receiver horizontal-offset filter (meters).
        Traces falling outside ``[offset_min_m, offset_max_m]`` are
        dropped pre-grouping.
    max_traces_per_group
        Hard cap on rows per group after grouping. Excess rows are
        randomly sub-sampled (seeded via ``seed``).
    seed
        RNG seed for the per-group cap sub-sampling.
    build_label
        Optional human-readable tag stored in ``build_meta`` for
        provenance.
    """
    if grouping not in VALID_GROUPINGS:
        raise ValueError(f"build_seismic_plan: grouping {grouping!r} invalid")
    if grouping == "supershot":
        raise NotImplementedError(
            "build_seismic_plan: supershot grouping is reserved for a future "
            "release; the runner can still source-encode at iter time."
        )

    # ----------- 1) per-trace selection (shot whitelist + offset filter)
    n_total = int(index.n_traces)
    keep_mask = np.ones(n_total, dtype=bool)

    if shot_ids is not None:
        wl = set(int(s) for s in shot_ids)
        keep_mask &= np.array([int(s) in wl for s in index.shot_id], dtype=bool)

    if offset_min_m is not None or offset_max_m is not None:
        dx = index.rx_m - index.sx_m
        dy = index.ry_m - index.sy_m
        off = np.hypot(dx, dy)
        if offset_min_m is not None:
            keep_mask &= off >= float(offset_min_m)
        if offset_max_m is not None:
            keep_mask &= off <= float(offset_max_m)

    if not keep_mask.any():
        raise ValueError(
            "build_seismic_plan: shot_ids + offset filter dropped every trace"
        )

    rows_idx = np.flatnonzero(keep_mask)
    row_file_id = np.asarray(index.file_id[rows_idx], dtype=np.int64)
    row_trace_offset = np.asarray(index.byte_offset[rows_idx], dtype=np.int64)
    row_sx = np.asarray(index.sx_m[rows_idx], dtype=np.float64)
    row_sy = np.asarray(index.sy_m[rows_idx], dtype=np.float64)
    row_sz = np.asarray(index.sz_m[rows_idx], dtype=np.float64)
    row_rx = np.asarray(index.rx_m[rows_idx], dtype=np.float64)
    row_ry = np.asarray(index.ry_m[rows_idx], dtype=np.float64)
    row_rz = np.asarray(index.rz_m[rows_idx], dtype=np.float64)
    row_shot_id = np.asarray(index.shot_id[rows_idx], dtype=np.int64)

    # ----------- 2) compute group key per row
    if grouping == "csg":
        # One group per unique shot_id (FFID); group_xyz = source position.
        group_keys = row_shot_id
    else:  # crg
        if receiver_quantize_m is None or float(receiver_quantize_m) <= 0:
            raise ValueError(
                "build_seismic_plan: grouping='crg' requires "
                "receiver_quantize_m > 0"
            )
        q = float(receiver_quantize_m)
        # Quantize per-trace receiver xyz to an integer cell key, CENTERED
        # by per-axis min so the resulting indices fit a 21-bit packed
        # field (max 2^21 = 2,097,152 cells per axis, ≈ 1000 km at q=0.5
        # m — enough for any real survey). Storing raw UTM cell indices
        # would overflow on UTM northings (e.g. 5e6 m / 0.5 m =
        # 1e7 cells, needs 24 bits per axis) and silently lose the
        # upper bits on unpacking.
        rx_min = float(row_rx.min())
        ry_min = float(row_ry.min())
        rz_min = float(row_rz.min())
        kx = np.round((row_rx - rx_min) / q).astype(np.int64)
        ky = np.round((row_ry - ry_min) / q).astype(np.int64)
        kz = np.round((row_rz - rz_min) / q).astype(np.int64)
        # Guard against the next overflow (huge survey extents at very
        # fine quantization). 21 bits per axis = 2,097,152 cells.
        _crg_bits = 21
        _crg_limit = 1 << _crg_bits
        if (int(kx.max()) >= _crg_limit
                or int(ky.max()) >= _crg_limit
                or int(kz.max()) >= _crg_limit):
            raise ValueError(
                f"build_seismic_plan: CRG cell index exceeds the "
                f"{_crg_bits}-bit packed field "
                f"(kx_max={int(kx.max())}, ky_max={int(ky.max())}, "
                f"kz_max={int(kz.max())}, limit={_crg_limit - 1}). "
                f"Increase receiver_quantize_m (currently {q}) or widen "
                "the per-axis bit allocation in build_seismic_plan."
            )
        # Pack (kx, ky, kz) into a single int64 group key. OR-of-shifted
        # is safe because the 21-bit shifts make the per-axis fields
        # bit-disjoint.
        group_keys = (kx << (_crg_bits * 2)) | (ky << _crg_bits) | kz

    # ----------- 3) group + sort rows so each group is contiguous
    sort_idx = np.argsort(group_keys, kind="stable")
    row_file_id = row_file_id[sort_idx]
    row_trace_offset = row_trace_offset[sort_idx]
    row_sx = row_sx[sort_idx]; row_sy = row_sy[sort_idx]; row_sz = row_sz[sort_idx]
    row_rx = row_rx[sort_idx]; row_ry = row_ry[sort_idx]; row_rz = row_rz[sort_idx]
    group_keys_sorted = group_keys[sort_idx]

    unique_keys, starts, counts = np.unique(
        group_keys_sorted, return_index=True, return_counts=True
    )
    n_groups = unique_keys.size
    group_offsets = np.empty(n_groups + 1, dtype=np.int64)
    group_offsets[:-1] = starts
    group_offsets[-1] = group_keys_sorted.size

    # ----------- 4) group_id + group_xyz (representative per group)
    if grouping == "csg":
        # Re-derive shot_id per group from first row of the group.
        group_id = np.asarray(
            [int(unique_keys[g]) for g in range(n_groups)], dtype=np.int64
        )
        # Source position = first row of each group (constant within a shot).
        group_xyz = np.stack(
            [row_sx[starts], row_sy[starts], row_sz[starts]], axis=-1
        )
    else:  # crg
        # group_id = sequential cell index (the hashed key is implementation detail).
        group_id = np.arange(n_groups, dtype=np.int64)
        # Use the QUANTIZED cell-center as the group position (deterministic).
        # Unpack the bit-packed key, then reverse the centering shift
        # applied during the pack step to recover the UTM coordinate.
        q = float(receiver_quantize_m)
        _crg_bits = 21
        _crg_mask = (1 << _crg_bits) - 1
        kx_recovered = (unique_keys >> (_crg_bits * 2)) & _crg_mask
        ky_recovered = (unique_keys >> _crg_bits) & _crg_mask
        kz_recovered = unique_keys & _crg_mask
        # All values are non-negative since we centered by min before
        # packing — no signed-recovery branch needed.
        group_xyz = np.stack(
            [kx_recovered.astype(np.float64) * q + rx_min,
             ky_recovered.astype(np.float64) * q + ry_min,
             kz_recovered.astype(np.float64) * q + rz_min],
            axis=-1,
        ).astype(np.float64)

    # ----------- 5) optional per-group sub-sample cap
    if max_traces_per_group is not None and int(max_traces_per_group) > 0:
        cap = int(max_traces_per_group)
        rng = np.random.default_rng(int(seed))
        keep_rows = np.ones(group_offsets[-1], dtype=bool)
        for g in range(n_groups):
            a, b = int(group_offsets[g]), int(group_offsets[g + 1])
            n = b - a
            if n > cap:
                pick = rng.choice(n, size=cap, replace=False)
                drop = np.ones(n, dtype=bool)
                drop[pick] = False
                keep_rows[a:b] = ~drop
        if not keep_rows.all():
            # Reapply mask + rebuild offsets.
            row_file_id = row_file_id[keep_rows]
            row_trace_offset = row_trace_offset[keep_rows]
            row_sx = row_sx[keep_rows]; row_sy = row_sy[keep_rows]; row_sz = row_sz[keep_rows]
            row_rx = row_rx[keep_rows]; row_ry = row_ry[keep_rows]; row_rz = row_rz[keep_rows]
            new_counts = np.zeros(n_groups, dtype=np.int64)
            for g in range(n_groups):
                a, b = int(group_offsets[g]), int(group_offsets[g + 1])
                new_counts[g] = int(keep_rows[a:b].sum())
            group_offsets = np.empty(n_groups + 1, dtype=np.int64)
            group_offsets[0] = 0
            group_offsets[1:] = np.cumsum(new_counts)

    # ----------- 6) assemble + provenance
    files = [Path(p) for p in index.file_paths]
    if hasattr(index, "meta"):
        # build_segy_index doesn't always populate trace_size_per_file;
        # derive from SEG-Y trace header + sample_format if missing.
        ts = index.meta.get("trace_size_per_file") if isinstance(index.meta, dict) else None
    else:
        ts = None
    if ts is None:
        from .segy import SEGY_TRACE_HEADER_SIZE
        bytes_per_sample = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(int(index.sample_format), 4)
        uniform = SEGY_TRACE_HEADER_SIZE + int(index.n_samples) * bytes_per_sample
        ts = np.full(len(files), uniform, dtype=np.int64)
    else:
        ts = np.asarray(ts, dtype=np.int64)

    build_meta = {
        "schema": SCHEMA_FORMAT,
        "grouping": grouping,
        "n_traces_total": int(n_total),
        "n_rows_kept": int(row_file_id.shape[0]),
        "n_groups": int(n_groups),
        "shot_ids_filter": (None if shot_ids is None
                            else int(len(list(shot_ids)))),
        "receiver_quantize_m": (None if receiver_quantize_m is None
                                else float(receiver_quantize_m)),
        "offset_min_m": offset_min_m,
        "offset_max_m": offset_max_m,
        "max_traces_per_group": max_traces_per_group,
        "seed": int(seed),
        "label": build_label,
        # Index identity hash (so reader can sanity-check it's reading
        # the SEG-Y files the plan was built from).
        "index_hash": _hash_index(index),
    }

    row_source_xyz = np.stack([row_sx, row_sy, row_sz], axis=-1)
    row_receiver_xyz = np.stack([row_rx, row_ry, row_rz], axis=-1)

    return SeismicPlan(
        grouping=grouping,
        files=files,
        trace_size_per_file=ts,
        sample_format=int(index.sample_format),
        samples_per_trace=int(index.n_samples),
        dt_s=float(index.dt_s),
        row_file_id=row_file_id,
        row_trace_offset=row_trace_offset,
        row_source_xyz=row_source_xyz,
        row_receiver_xyz=row_receiver_xyz,
        group_id=group_id,
        group_xyz=group_xyz,
        group_offsets=group_offsets,
        build_meta=build_meta,
    )


def _hash_index(index) -> str:
    """Stable short hash of an SEGYIndex's identity (files + n_traces + dt)."""
    h = hashlib.sha256()
    for p in index.file_paths:
        h.update(os.path.basename(str(p)).encode())
    h.update(str(int(index.n_traces)).encode())
    h.update(f"{float(index.dt_s):.9e}".encode())
    h.update(str(int(index.n_samples)).encode())
    return h.hexdigest()[:16]


# ============================================================================
# PlanReader: lazy SEG-Y trace reader keyed by group id
# ============================================================================
class PlanReader:
    """Lazy SEG-Y trace reader bound to a :class:`SeismicPlan`.

    The plan tells *what* to read (file_ids + byte offsets, grouped);
    this reader does the actual IBM-float decoding via
    :class:`MultiFileSEGYReader`. Trace order within a group is
    preserved (matches ``plan.row_*[group_slice(g)]``).

    Parameters
    ----------
    plan
        A :class:`SeismicPlan` (typically loaded from disk).
    mmap
        Whether the underlying SEG-Y reader should mmap files. Default
        ``True`` — the OS page cache then shares pages across processes
        in distributed runs.
    cache_all
        When ``True``, read every plan row into a single ``(n_rows, nt)``
        float32 tensor at construction and serve subsequent
        ``read_group(g)`` calls from the cache. Useful for small
        datasets (e.g. 2-D Viking — ~700 MB fits in RAM trivially). For
        large 3-D surveys leave it ``False`` so memory stays
        bounded.
    """

    def __init__(
        self,
        plan: SeismicPlan,
        *,
        mmap: bool = True,
        cache_all: bool = False,
        trace_cache_bytes: int = 0,
        coalesce_gap: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        plan
            A :class:`SeismicPlan` (typically loaded from disk).
        mmap
            Whether the underlying SEG-Y reader should mmap files.
        cache_all
            When True, eagerly read every plan row into a ``(n_rows, nt)``
            float32 tensor at construction.
        trace_cache_bytes
            Optional in-RAM LRU cache for decoded traces, keyed by
            ``(file_id, byte_offset)``. Used by :meth:`read_rows` to
            absorb cold-Lustre re-reads when the same physical traces are
            sampled across multiple iterations (the OBN multisource
            supershot pattern). ``0`` disables. ``-1`` means unbounded
            (matches the legacy ``--trace-cache-bytes -1`` default for
            large surveys). Any positive value is a byte budget.
            Ignored when ``cache_all=True`` (the full cache is already
            materialised).
        coalesce_gap
            Bytes-within-the-same-file gap below which two miss-traces
            get merged into a single ``pread`` call. ``0`` disables.
            Sensible value: 4× the trace stride (~``240 + nt*4`` bytes)
            so any two sequentially-indexed traces in the same file get
            coalesced. Critical for Lustre where per-call latency
            dominates small random reads. Only used by :meth:`read_rows`
            (the cached path); ``read_group`` always reads the full group
            slice which is already contiguous.
        """
        self.plan = plan
        self._reader = MultiFileSEGYReader(
            [str(p) for p in plan.files], mmap_mode=mmap,
        )
        self._cache: np.ndarray | None = None
        if cache_all:
            self._cache = self._reader.read_traces(
                plan.row_file_id, plan.row_trace_offset,
            )
        # Per-trace LRU cache + read-coalescing for the sampler-driven
        # path. Disabled when cache_all is on (the whole plan is cached
        # contiguously, no per-trace caching needed).
        self._trace_cache = None
        self._coalesce_gap = int(max(0, coalesce_gap))
        if not cache_all and int(trace_cache_bytes) != 0:
            from .crg_dataset import TraceCache
            # Legacy convention: -1 means unbounded → TraceCache(0).
            cap = 0 if int(trace_cache_bytes) < 0 else int(trace_cache_bytes)
            self._trace_cache = TraceCache(max_bytes=cap)

    # ----------------------------------------------------------- read API
    def read_group(self, g: int) -> np.ndarray:
        """Return ``(n_in_group, nt)`` float32 for group ``g``."""
        sl = self.plan.group_slice(int(g))
        if self._cache is not None:
            return self._cache[sl]
        return self._reader.read_traces(
            self.plan.row_file_id[sl],
            self.plan.row_trace_offset[sl],
        )

    def read_groups(self, gs: Iterable[int]) -> list[np.ndarray]:
        """Return one ``(n_in_g, nt)`` array per group id in ``gs``."""
        return [self.read_group(int(g)) for g in gs]

    def read_rows(self, row_idx: np.ndarray) -> np.ndarray:
        """Return ``(n_picked, nt)`` float32 for an arbitrary set of plan rows.

        Used by sampler-driven loaders (e.g. CRG shared-shot batches) that
        pick row indices across multiple groups rather than reading whole
        groups. Row order in the output matches ``row_idx``.

        Three code paths, in priority order:

        1. ``cache_all=True`` at construction → fancy-index slice on the
           in-RAM ``(n_rows, nt)`` tensor (single numpy op, instant).
        2. ``trace_cache_bytes`` set (per-trace LRU + ``coalesce_gap``)
           → :func:`sweep_io.crg_dataset.read_traces_cached`. This is the
           legacy OBN supershot IO path: hot traces (re-sampled across
           iters) come from RAM, cold traces are coalesced into bigger
           ``pread`` calls. Cuts wait_io about
           five-fold on a large survey.
        3. Otherwise → raw :meth:`MultiFileSEGYReader.read_traces` (one
           ``pread`` per miss, no caching). Slow on Lustre with random
           access patterns; only useful for one-shot reads.
        """
        idx = np.asarray(row_idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return np.empty((0, int(self.plan.samples_per_trace)),
                            dtype=np.float32)
        if self._cache is not None:
            return self._cache[idx]
        fids = self.plan.row_file_id[idx]
        offs = self.plan.row_trace_offset[idx]
        if self._trace_cache is not None:
            from .crg_dataset import read_traces_cached
            return read_traces_cached(
                self._reader, self._trace_cache,
                fids, offs,
                coalesce_gap=self._coalesce_gap,
            )
        return self._reader.read_traces(fids, offs)

    @property
    def trace_cache_stats(self) -> dict | None:
        """Cache hit/miss counters + current byte usage, or None when
        ``trace_cache_bytes=0`` was set at construction."""
        if self._trace_cache is None:
            return None
        return {
            "hits": int(self._trace_cache.hits),
            "misses": int(self._trace_cache.misses),
            "entries": len(self._trace_cache),
            "bytes": int(self._trace_cache.cur_bytes),
        }

    def read_all(self) -> np.ndarray:
        """Return ``(n_rows, nt)`` float32 of every plan row, in plan order."""
        if self._cache is not None:
            return self._cache
        return self._reader.read_traces(
            self.plan.row_file_id, self.plan.row_trace_offset,
        )

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> "PlanReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----------------------------------------------------------- shortcuts
    @property
    def grouping(self) -> str:
        return self.plan.grouping

    @property
    def n_groups(self) -> int:
        return self.plan.n_groups

    @property
    def n_rows(self) -> int:
        return self.plan.n_rows

    @property
    def n_samples(self) -> int:
        return int(self.plan.samples_per_trace)

    @property
    def dt_s(self) -> float:
        return float(self.plan.dt_s)


# ============================================================================
# Shared-shot sampler (CRG-grouped plans → source-encoded supershot batches)
# ============================================================================
@dataclass(frozen=True)
class SharedShotBatch:
    """One iteration's shared-shot sample for source-encoded FWI.

    Every group in ``group_indices`` (each a virtual source / OBN node when
    the underlying plan is CRG-grouped) records the SAME set of physical
    shots, identified by mm-quantised ``(sx, sy)`` keys. That invariant is
    what source-encoded supershot FWI needs: a single encoded wavelet plays
    against a single encoded obs trace per virtual receiver.

    Attributes
    ----------
    group_indices
        ``(B,)`` int64 — plan groups in this batch (= virtual sources).
    rows_per_group
        Length-``B`` list of ``(n_shared,)`` int64 arrays, each holding
        flat plan-row indices for the i-th group. All arrays have the
        same length; row ``j`` across groups refers to the SAME physical
        shot.
    shared_shot_xyz_m
        ``(n_shared, 3)`` float64 — physical shot positions of the
        intersection shots.
    source_xyz_m
        ``(B, 3)`` float64 — virtual-source (group / OBN node) positions
        (copied from ``plan.group_xyz``).
    n_shared
        Convenience, ``= rows_per_group[0].size``.
    """

    group_indices: np.ndarray
    rows_per_group: list
    shared_shot_xyz_m: np.ndarray
    source_xyz_m: np.ndarray
    n_shared: int


def _shot_xy_keys(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Pack mm-quantised ``(sx, sy)`` into one int64 each.

    Identical to ``sweep_io.crg_plan._shot_xy_keys`` (kept here so the
    unified plan path has no dependency on the legacy CRG module).
    """
    sxi = np.rint(np.asarray(sx, dtype=np.float64) * 1000.0).astype(np.int64)
    syi = np.rint(np.asarray(sy, dtype=np.float64) * 1000.0).astype(np.int64)
    return ((sxi & 0xFFFFFFFF) << 32) | (syi & 0xFFFFFFFF)


def precompute_group_unique_keys(plan: SeismicPlan) -> list[np.ndarray]:
    """Precompute one sorted-unique ``(sx, sy)`` int64-key array per CRG group.

    The shared-shot sampler's per-iter cost is dominated by computing
    these key arrays from scratch each call (24× ``_shot_xy_keys`` +
    ``np.unique`` on every group's rows, seconds per call). Since the plan is frozen,
    these arrays only depend on ``plan.row_source_xyz`` + ``group_offsets``
    — pre-compute once at setup, pass into the sampler via
    ``precomputed_group_unique_keys=``.

    Returns a length-``plan.n_groups`` list; each entry is a sorted
    1-D int64 array of unique mm-quantised ``(sx, sy)`` shot keys for
    that group's rows.
    """
    if plan.grouping != "crg":
        raise ValueError(
            f"precompute_group_unique_keys: expects grouping='crg' "
            f"(got {plan.grouping!r})."
        )
    out: list[np.ndarray] = []
    for g in range(int(plan.n_groups)):
        sl = plan.group_slice(g)
        sx = plan.row_source_xyz[sl, 0]
        sy = plan.row_source_xyz[sl, 1]
        out.append(np.unique(_shot_xy_keys(sx, sy)))
    return out


def sample_shared_shots_from_plan(
    plan: SeismicPlan,
    rng: np.random.Generator,
    *,
    batch_size: int,
    source_lines_per_group: int = 0,
    max_traces_per_sourceline: int = 0,
    min_coverage: int = 0,
    eligible_groups: np.ndarray | None = None,
    max_retries: int = 10,
    precomputed_group_unique_keys: list[np.ndarray] | None = None,
) -> SharedShotBatch:
    """Pick ``batch_size`` groups whose physical-shot sets intersect.

    Operates on a ``grouping='crg'`` :class:`SeismicPlan`: each group is a
    virtual source (e.g. one OBN node, or a quantised receiver cell), and
    its rows are the physical shots that group recorded. The intersection
    across the chosen groups gives the shots EVERY picked group recorded —
    the invariant needed for source-encoded supershot FWI.

    This is the unified-plan port of
    :func:`sweep_io.crg_plan.sample_crg_shared_shots`; behaviour is
    deliberately byte-equivalent so existing runner code can swap one for
    the other with no FP drift (assuming the same RNG state).

    Optional sub-sampling within the intersection:

    * ``source_lines_per_group > 0`` — keep that many random source-line
      ``file_id``s (= source-line SEG-Y files).
    * ``max_traces_per_sourceline > 0`` — keep that many random shots per
      kept source line.
    * ``0`` for either disables that level.

    Parameters
    ----------
    plan
        A ``grouping='crg'`` :class:`SeismicPlan` (CSG plans cannot be
        sampled this way — each CSG group has a single source, so the
        intersection is trivially that one source).
    rng
        Caller-owned ``np.random.Generator`` (advanced in-place).
    batch_size
        Number of groups (= virtual sources) per batch.
    min_coverage
        Drop groups whose row count is below this. ``0`` disables.
    eligible_groups
        Optional pre-curated set of group indices to sample from. When
        ``None`` and ``min_coverage > 0``, computed from
        :meth:`SeismicPlan.per_group_row_counts`.
    max_retries
        Re-roll group selection up to this many times when the picked
        groups' shot intersection is empty (rare with ``min_coverage``
        on but possible for partial-coverage edge nodes).

    Raises
    ------
    ValueError
        When ``plan.grouping != 'crg'``, or when the shot intersection is
        empty after ``max_retries`` attempts, or when sub-sampling
        empties the intersection.
    """
    if plan.grouping != "crg":
        raise ValueError(
            f"sample_shared_shots_from_plan: requires a grouping='crg' "
            f"SeismicPlan, got {plan.grouping!r}. CSG groups have a "
            "single source each, so the shared-shot intersection is "
            "trivially that one source — use SeismicPlan.group_slice "
            "directly instead."
        )
    n_groups = int(plan.n_groups)
    if eligible_groups is None:
        if min_coverage > 0:
            counts = plan.per_group_row_counts()
            eligible = np.flatnonzero(counts >= int(min_coverage))
            if eligible.size == 0:
                eligible = np.arange(n_groups, dtype=np.int64)
        else:
            eligible = np.arange(n_groups, dtype=np.int64)
    else:
        eligible = np.asarray(eligible_groups, dtype=np.int64).reshape(-1)
    bs = max(1, min(int(batch_size), int(eligible.size)))

    K = int(source_lines_per_group) if source_lines_per_group and int(source_lines_per_group) > 0 else 0
    M = int(max_traces_per_sourceline) if max_traces_per_sourceline and int(max_traces_per_sourceline) > 0 else 0

    # Retry group sampling when the picked groups' shot-(sx,sy) intersection
    # turns out to be empty (rare with min_coverage on, but can happen for
    # partial-coverage edge nodes). Mirrors the legacy soft fallback.
    group_indices: np.ndarray | None = None
    per_group_keys_full: list[np.ndarray] = []
    per_group_keys_uniq: list[np.ndarray] = []
    inter = np.empty(0, dtype=np.int64)
    last_attempt: np.ndarray = np.empty(0, dtype=np.int64)
    for _attempt in range(max(1, int(max_retries))):
        cand = np.sort(rng.choice(eligible, size=bs, replace=False)).astype(np.int64)
        last_attempt = cand
        cand_keys_full: list[np.ndarray] = []
        cand_keys_uniq: list[np.ndarray] = []
        for g in cand:
            sl = plan.group_slice(int(g))
            sx = plan.row_source_xyz[sl, 0]
            sy = plan.row_source_xyz[sl, 1]
            keys_full = _shot_xy_keys(sx, sy)
            cand_keys_full.append(keys_full)
            # (C) Reuse precomputed sorted-unique keys when caller has
            # already paid the np.unique cost once per group. Saves
            # batch_size × O(rows_per_group log rows_per_group) per iter.
            if precomputed_group_unique_keys is not None:
                cand_keys_uniq.append(precomputed_group_unique_keys[int(g)])
            else:
                cand_keys_uniq.append(np.unique(keys_full))
        # (B) Counting-based intersect: one np.unique on the concatenation
        # vs (batch_size - 1) sequential np.intersect1d calls. Each
        # intersect1d itself does a concat+sort under the hood, so the
        # sequential reduce ends up O(B^2 N log N); this is O(B N log(B N))
        # — for a 24-node batch on a large survey, seconds → under half a second.
        if len(cand_keys_uniq) == 0:
            cand_inter = np.empty(0, dtype=np.int64)
        elif len(cand_keys_uniq) == 1:
            cand_inter = cand_keys_uniq[0]
        else:
            concat = np.concatenate(cand_keys_uniq)
            uniq, counts = np.unique(concat, return_counts=True)
            cand_inter = uniq[counts == len(cand_keys_uniq)]
        if cand_inter.size > 0:
            group_indices = cand
            per_group_keys_full = cand_keys_full
            per_group_keys_uniq = cand_keys_uniq
            inter = cand_inter
            break
    if group_indices is None:
        raise ValueError(
            f"sample_shared_shots_from_plan: shot-(sx,sy) intersection is "
            f"empty for batch_size={bs} after {max_retries} retries (last "
            f"attempt groups={last_attempt.tolist()}). Raise min_coverage "
            "or lower batch_size."
        )

    # Look up (file_id, sx, sy, sz) of intersection shots via group 0's rows.
    master_sl = plan.group_slice(int(group_indices[0]))
    master_fids = plan.row_file_id[master_sl].astype(np.int64)
    master_keys = per_group_keys_full[0]
    master_sort = np.argsort(master_keys)
    pos_in_master = master_sort[
        np.searchsorted(master_keys[master_sort], inter)
    ]
    inter_fids = master_fids[pos_in_master]
    master_xyz = plan.row_source_xyz[master_sl][pos_in_master]
    master_sx = master_xyz[:, 0]
    master_sy = master_xyz[:, 1]
    master_sz = master_xyz[:, 2]

    # Hierarchical sub-sample (line → trace).
    unique_fids = np.unique(inter_fids)
    if K > 0 and unique_fids.size > K:
        keep_fids = rng.choice(unique_fids, size=K, replace=False)
        line_mask = np.isin(inter_fids, keep_fids)
        inter = inter[line_mask]
        inter_fids = inter_fids[line_mask]
        master_sx = master_sx[line_mask]
        master_sy = master_sy[line_mask]
        master_sz = master_sz[line_mask]
    if M > 0:
        kept = []
        for f in np.unique(inter_fids):
            f_mask = inter_fids == f
            f_idx = np.flatnonzero(f_mask)
            if f_idx.size > M:
                f_idx = rng.choice(f_idx, size=M, replace=False)
            kept.append(f_idx)
        kept = np.sort(np.concatenate(kept)) if kept else np.empty(0, dtype=np.int64)
        inter = inter[kept]
        inter_fids = inter_fids[kept]
        master_sx = master_sx[kept]
        master_sy = master_sy[kept]
        master_sz = master_sz[kept]

    if inter.size == 0:
        raise ValueError(
            "sample_shared_shots_from_plan: empty after sub-sampling — "
            "relax source_lines_per_group / max_traces_per_sourceline."
        )

    # For each picked group, look up plan-row indices of the shared shots.
    rows_per_group: list[np.ndarray] = []
    for gi, g in enumerate(group_indices):
        sl = plan.group_slice(int(g))
        group_keys = per_group_keys_full[gi]
        sort_idx = np.argsort(group_keys)
        sorted_keys = group_keys[sort_idx]
        ins = np.searchsorted(sorted_keys, inter)
        local_pos = sort_idx[ins]
        rows_per_group.append((local_pos + int(sl.start)).astype(np.int64))

    shared_xyz = np.stack([master_sx, master_sy, master_sz], axis=-1).astype(np.float64)
    source_xyz = plan.group_xyz[group_indices].astype(np.float64)
    return SharedShotBatch(
        group_indices=group_indices,
        rows_per_group=rows_per_group,
        shared_shot_xyz_m=shared_xyz,
        source_xyz_m=source_xyz,
        n_shared=int(inter.size),
    )


@dataclass
class PerCRGBatch:
    """One iteration's per-CRG independent-coverage sample (NO intersection).

    Companion to :class:`SharedShotBatch`. For the per-shot (non-encoded)
    FWI path every CRG node is a SEPARATE forward solve, so the shared-shot
    intersection that source encoding needs is unnecessary — and, on
    partial-coverage OBN data, actively harmful: the intersection keeps only
    shots recorded by EVERY node in the batch, discarding exactly the
    wide-offset diving-wave shots that just a few nearby nodes recorded.
    This batch instead gives each node its OWN sub-sampled rows, so
    ``rows_per_group`` is RAGGED (per-node length in ``per_group_counts``).

    ``valid_mask`` / ``recv_rows_padded`` are filled by the runner after it
    reads + grid-dedups + zero-pads the ragged traces into a dense
    ``(B, max_nrec, nt)`` batch (the padded tail is masked out of the loss).

    Attributes
    ----------
    group_indices : (B,) int64 — plan groups (virtual sources / OBN nodes).
    rows_per_group : length-B list of int64 arrays (RAGGED) — each group's
        own sub-sampled plan-row indices.
    source_xyz_m : (B, 3) float64 — virtual-source (node) positions.
    per_group_counts : (B,) int64 — ``rows_per_group[i].size``.
    """

    group_indices: np.ndarray
    rows_per_group: list
    source_xyz_m: np.ndarray
    per_group_counts: np.ndarray
    valid_mask: np.ndarray | None = None        # (B, max_nrec) bool, runner-filled
    recv_rows_padded: np.ndarray | None = None  # (B, max_nrec) int64, runner-filled
    n_shared: int = 0                           # = max_nrec (padded width), runner-filled


def sample_percrg_independent(
    plan: SeismicPlan,
    rng: np.random.Generator,
    *,
    batch_size: int,
    source_lines_per_group: int = 0,
    max_traces_per_sourceline: int = 0,
    min_coverage: int = 0,
    eligible_groups: np.ndarray | None = None,
) -> PerCRGBatch:
    """Pick ``batch_size`` CRG groups; each keeps its OWN sub-sampled rows.

    Unlike :func:`sample_shared_shots_from_plan`, this does NOT intersect the
    groups' shot sets — there is no shared receiver geometry. The sub-sampling
    knobs (``source_lines_per_group`` → keep that many random source-line
    ``file_id``s; ``max_traces_per_sourceline`` → that many random shots per
    kept line) are applied PER GROUP on that group's own recorded shots. Use
    for the per-shot (non-encoded) FWI path so each CRG node inverts its full
    aperture; over iterations the stochastic per-node draw sweeps each node's
    entire coverage.

    Grid-cell dedup of the receivers is deliberately left to the caller — it
    needs the model frame / grid origin this plan-space sampler is agnostic to.
    """
    if plan.grouping != "crg":
        raise ValueError(
            f"sample_percrg_independent: requires grouping='crg', "
            f"got {plan.grouping!r}."
        )
    n_groups = int(plan.n_groups)
    if eligible_groups is None:
        if min_coverage > 0:
            counts = plan.per_group_row_counts()
            eligible = np.flatnonzero(counts >= int(min_coverage))
            if eligible.size == 0:
                eligible = np.arange(n_groups, dtype=np.int64)
        else:
            eligible = np.arange(n_groups, dtype=np.int64)
    else:
        eligible = np.asarray(eligible_groups, dtype=np.int64).reshape(-1)
    bs = max(1, min(int(batch_size), int(eligible.size)))
    K = int(source_lines_per_group) if source_lines_per_group and int(source_lines_per_group) > 0 else 0
    M = int(max_traces_per_sourceline) if max_traces_per_sourceline and int(max_traces_per_sourceline) > 0 else 0

    group_indices = np.sort(
        rng.choice(eligible, size=bs, replace=False)
    ).astype(np.int64)

    rows_per_group: list[np.ndarray] = []
    for g in group_indices:
        sl = plan.group_slice(int(g))
        start = int(sl.start)
        fids = plan.row_file_id[sl].astype(np.int64)
        local_idx = np.arange(fids.size, dtype=np.int64)
        # (line level) keep K random source-line file_ids
        if K > 0:
            uf = np.unique(fids)
            if uf.size > K:
                keep_fids = rng.choice(uf, size=K, replace=False)
                m = np.isin(fids, keep_fids)
                local_idx = local_idx[m]
                fids = fids[m]
        # (trace level) keep M random shots per kept line
        if M > 0:
            kept: list[np.ndarray] = []
            for f in np.unique(fids):
                f_idx = local_idx[fids == f]
                if f_idx.size > M:
                    f_idx = rng.choice(f_idx, size=M, replace=False)
                kept.append(f_idx)
            local_idx = (np.sort(np.concatenate(kept))
                         if kept else np.empty(0, dtype=np.int64))
        rows_per_group.append((local_idx + start).astype(np.int64))

    per_group_counts = np.array([r.size for r in rows_per_group], dtype=np.int64)
    source_xyz = plan.group_xyz[group_indices].astype(np.float64)
    return PerCRGBatch(
        group_indices=group_indices,
        rows_per_group=rows_per_group,
        source_xyz_m=source_xyz,
        per_group_counts=per_group_counts,
    )


def build_shotkey_to_nodes_index(
    plan: SeismicPlan,
    *,
    min_coverage: int = 0,
    eligible_groups: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Build the receiver-first reverse index ``shot_key -> covering groups``.

    For each eligible CRG group (one whose recorded-shot row count is
    ``>= min_coverage``; all groups when ``min_coverage <= 0``) take its
    UNIQUE mm-quantised ``(sx, sy)`` shot keys and accumulate, per key, the
    group (node) indices that record it. Lets
    :func:`sample_shared_shots_receiver_first` answer "which nodes recorded
    shot K?" in O(1) instead of an O(n_groups) per-iter scan. Built ONCE and
    reused for the whole run (pass via ``shotkey_to_nodes=``).

    ``eligible_groups`` (optional) restricts the index to a caller-supplied
    set of group indices — e.g. the in-grid (``slot_in``) nodes the encoded
    FWI path will accept. WITHOUT it the index can surface a node whose
    source position is OUTSIDE the grid, which the random-group sampler
    excludes via its own ``eligible_groups`` arg, so receiver-first would
    inject a source out of bounds. The final eligible set is the
    intersection of the ``min_coverage`` filter and this set.

    Returns ``dict[int shot_key -> np.ndarray(group_idx, int64)]`` (each
    array sorted, unique). Unified-plan port of the fwi_workflow
    receiver-first reverse index.
    """
    if plan.grouping != "crg":
        raise ValueError(
            f"build_shotkey_to_nodes_index: requires grouping='crg', "
            f"got {plan.grouping!r}."
        )
    n_groups = int(plan.n_groups)
    if min_coverage and int(min_coverage) > 0:
        counts = plan.per_group_row_counts()
        eligible = np.flatnonzero(counts >= int(min_coverage))
        if eligible.size == 0:
            eligible = np.arange(n_groups, dtype=np.int64)
    else:
        eligible = np.arange(n_groups, dtype=np.int64)
    # Intersect with a caller-supplied eligible set (the in-grid nodes). Keep
    # the in-grid invariant even if the min_coverage filter would empty it
    # (out-of-grid sources are a hard error; low coverage is a soft loss).
    if eligible_groups is not None:
        eg = np.unique(np.asarray(eligible_groups, dtype=np.int64).reshape(-1))
        eligible = np.intersect1d(eligible, eg, assume_unique=False)
        if eligible.size == 0:
            eligible = eg
    acc: dict = {}
    for g in eligible:
        g = int(g)
        sl = plan.group_slice(g)
        if sl.stop == sl.start:
            continue
        keys = np.unique(_shot_xy_keys(plan.row_source_xyz[sl, 0],
                                       plan.row_source_xyz[sl, 1]))
        for k in keys.tolist():
            lst = acc.get(k)
            if lst is None:
                acc[k] = [g]
            else:
                lst.append(g)
    index = {k: np.asarray(sorted(set(v)), dtype=np.int64) for k, v in acc.items()}
    if verbose:
        print(
            f"[receiver-first] built shot_key->nodes reverse index: "
            f"{int(eligible.size)} eligible groups "
            f"(min_coverage={int(min_coverage)}), "
            f"{len(index)} distinct shot keys",
            flush=True,
        )
    return index


def sample_shared_shots_receiver_first(
    plan: SeismicPlan,
    rng: np.random.Generator,
    *,
    batch_size: int,
    source_lines_per_group: int = 0,
    max_traces_per_sourceline: int = 0,
    min_coverage: int = 0,
    max_retries: int = 20,
    eligible_groups: np.ndarray | None = None,
    shotkey_to_nodes: dict | None = None,
    shotkey_keys_arr: np.ndarray | None = None,
    precomputed_group_unique_keys: list | None = None,
) -> SharedShotBatch:
    """Receiver-first shared-shot sampler (anti-bias for survey periphery).

    :func:`sample_shared_shots_from_plan` picks ``batch_size`` groups at
    random then intersects their shots; on a partial-coverage OBN survey that
    intersection collapses toward the survey CENTRE, leaving the periphery
    chronically under-sampled by the encoded gradient. This variant inverts
    the order so the per-iter target sweeps the WHOLE survey:

      1) Pick ONE target shot uniformly from all distinct shot keys.
      2) Look up (via the reverse index) every eligible group that recorded
         it.
      3) If ``>= batch_size`` such groups exist, draw ``batch_size`` of them;
         else retry with a fresh target (up to ``max_retries``). On exhausted
         retries fall back to :func:`sample_shared_shots_from_plan`.
      4) Delegate the intersection + hierarchical sub-sample to
         :func:`sample_shared_shots_from_plan` with ``eligible_groups`` = the
         chosen groups (whose intersection is guaranteed non-empty — it
         contains the target).

    Port of fwi_workflow ``sample_crg_iter_receiver_first`` onto the unified
    SeismicPlan. Build ``shotkey_to_nodes`` ONCE via
    :func:`build_shotkey_to_nodes_index` and pass it in (building per call is
    O(n_groups × rows)). Uses ONLY the passed ``rng``.
    """
    bs_req = int(batch_size)
    if shotkey_to_nodes is None:
        shotkey_to_nodes = build_shotkey_to_nodes_index(
            plan, min_coverage=int(min_coverage),
            eligible_groups=eligible_groups, verbose=True,
        )
    if shotkey_keys_arr is None:
        shotkey_keys_arr = np.fromiter(
            shotkey_to_nodes.keys(), dtype=np.int64, count=len(shotkey_to_nodes)
        )

    chosen: np.ndarray | None = None
    if shotkey_keys_arr.size > 0 and bs_req >= 1:
        for _ in range(max(1, int(max_retries))):
            cand_key = int(shotkey_keys_arr[rng.integers(shotkey_keys_arr.size)])
            nodes = shotkey_to_nodes[cand_key]
            if nodes.size >= bs_req:
                chosen = np.sort(
                    rng.choice(nodes, size=bs_req, replace=False)
                ).astype(np.int64)
                break

    if chosen is None:
        # No target reached batch_size within retries — plain shared-shots.
        # Forward the in-grid eligible set so the fallback also refuses
        # out-of-grid nodes (matches the random-sampler path).
        return sample_shared_shots_from_plan(
            plan, rng,
            batch_size=bs_req,
            source_lines_per_group=int(source_lines_per_group),
            max_traces_per_sourceline=int(max_traces_per_sourceline),
            min_coverage=int(min_coverage),
            eligible_groups=eligible_groups,
            precomputed_group_unique_keys=precomputed_group_unique_keys,
        )

    # The chosen groups all record the target shot, so their intersection is
    # non-empty; reuse the canonical intersection + sub-sampling path.
    return sample_shared_shots_from_plan(
        plan, rng,
        batch_size=bs_req,
        source_lines_per_group=int(source_lines_per_group),
        max_traces_per_sourceline=int(max_traces_per_sourceline),
        min_coverage=0,
        eligible_groups=chosen,
        precomputed_group_unique_keys=precomputed_group_unique_keys,
    )


__all__ = [
    "SCHEMA_FORMAT",
    "VALID_GROUPINGS",
    "SeismicPlan",
    "SharedShotBatch",
    "build_seismic_plan",
    "PlanReader",
    "precompute_group_unique_keys",
    "sample_shared_shots_from_plan",
    "build_shotkey_to_nodes_index",
    "sample_shared_shots_receiver_first",
]
