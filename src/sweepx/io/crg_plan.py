"""CRG (common-receiver-gather) FWI plan loader.

For OBN / OBC surveys the physical receivers (ocean-bottom nodes) act as
**virtual sources** in CRG-mode FWI via source-receiver reciprocity. A
"plan cache" npz pre-computes, for each virtual source (one per OBN
node), the list of (file_id, byte_offset) tuples pointing into the raw
source-line SEG-Y files that hold the physical shots recorded by that
node — together with the per-shot source coordinates.

Loading the slim plan cache is O(MB) and instant; the alternative is
lazy-loading a multi-GB CRG index npz that is slow to decompress.

Plan-cache schema (``format = "crg_fwi_plan_v1"``)::

    files                  : (n_files,)             U256   SEG-Y paths
    trace_size_per_file    : (n_files,)             int64  bytes per trace inc. header
    sample_format          : scalar                 int32  SEG-Y sample format code
    samples_per_trace      : scalar                 int32  nt of every trace
    dt_s                   : scalar                 float64
    receiver_indices       : (n_recv_used,)         int64  legacy index into original CRG
    receiver_xyz_used      : (n_recv_used, 3)       float64 OBN node positions (UTM, m)
    plan_file_ids          : (n_plan_rows,)         int64  per-row file index
    plan_trace_offsets     : (n_plan_rows,)         int64  per-row byte offset
    plan_sx, plan_sy, plan_sz : (n_plan_rows,)      float64 physical shot positions (UTM, m)
    plan_offsets           : (n_recv_used + 1,)     int64  CSR-style row pointer

For each virtual-source slot ``i``, the rows
``plan_*[plan_offsets[i] : plan_offsets[i+1]]`` are its complete gather:
the shots fall along multiple source lines (file_ids), and the
physical-shot coordinates ``(plan_sx, plan_sy, plan_sz)`` are the
**virtual receiver** coordinates fed to the FWI solver.

See :func:`load_crg_shot_plan_cache` for the loader entry point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


def _remap_segy_path(p: str | os.PathLike) -> str:
    """Apply ``FWI_SEGY_ROOT`` / ``FWI_SEGY_REMAP`` env-var remaps to a path.

    The CRG plan cache stores absolute SEG-Y paths captured on the host
    where the cache was built. When running on a different host where
    the SEG-Y archive lives elsewhere, set:

    * ``FWI_SEGY_ROOT=/path/to/segy``    — replace the directory of every
      recorded path with this prefix; the basename is preserved.
      Recommended when filenames are unique and stable across hosts.
    * ``FWI_SEGY_REMAP=OLD:NEW;OLD2:NEW2`` — prefix substitution. Multiple
      rules separated by ``;``. Use when the on-disk directory tree under
      a common root is preserved.

    ``FWI_SEGY_ROOT`` wins when both are set. Idempotent when neither is.
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


@dataclass(frozen=True)
class CRGPlan:
    """Slim per-virtual-source CRG plan loaded from a ``crg_fwi_plan_v1`` cache.

    Each ``i ∈ [0, n_receivers_used)`` is one virtual source. Its
    physical-shot rows are ``plan_*[slot_slice(i)]``.

    Attributes
    ----------
    files
        SEG-Y source-line files (per-file ``read_traces`` dispatch keys
        are ``plan_file_ids``).
    trace_size_per_file
        ``(n_files,)`` byte stride per trace (header + samples).
    sample_format, samples_per_trace, dt_s
        SEG-Y trace parameters; identical across files in a plan.
    n_receivers_used
        Number of virtual sources (OBN nodes) in the plan.
    receiver_xyz_used
        ``(n_receivers_used, 3)`` virtual-source positions in UTM (m).
        These become the FWI **sources**.
    plan_file_ids, plan_trace_offsets
        Flat per-row arrays of (file index, byte offset) tuples handed
        to :meth:`sweep_io.segy.MultiFileSEGYReader.read_traces`.
    plan_sx, plan_sy, plan_sz
        Flat per-row physical shot positions in UTM (m). These become
        the FWI **receivers** (one per recorded trace).
    plan_offsets
        ``(n_receivers_used + 1,)`` CSR-style row pointer:
        ``plan_offsets[i] : plan_offsets[i+1]`` are the row indices of
        slot ``i``'s gather.
    """

    files: list[Path]
    trace_size_per_file: np.ndarray
    sample_format: int
    samples_per_trace: int
    dt_s: float
    receiver_indices: np.ndarray
    receiver_xyz_used: np.ndarray
    plan_file_ids: np.ndarray
    plan_trace_offsets: np.ndarray
    plan_sx: np.ndarray
    plan_sy: np.ndarray
    plan_sz: np.ndarray
    plan_offsets: np.ndarray

    # --------------------------------------------------------- derived
    @property
    def n_receivers_used(self) -> int:
        return int(self.receiver_xyz_used.shape[0])

    @property
    def n_plan_rows(self) -> int:
        return int(self.plan_file_ids.shape[0])

    @property
    def virtual_source_xy(self) -> np.ndarray:
        """``(n_receivers_used, 2)`` virtual-source horizontal coordinates (m)."""
        return self.receiver_xyz_used[:, :2]

    def slot_slice(self, slot: int) -> slice:
        """Flat-row slice for the ``slot``-th virtual source."""
        return slice(int(self.plan_offsets[int(slot)]),
                     int(self.plan_offsets[int(slot) + 1]))

    def slot_row_count(self, slot: int) -> int:
        """Number of physical shots that hit virtual source ``slot``."""
        return int(self.plan_offsets[int(slot) + 1] - self.plan_offsets[int(slot)])

    def per_slot_row_counts(self) -> np.ndarray:
        """``(n_receivers_used,)`` shot count per virtual source."""
        return np.diff(self.plan_offsets).astype(np.int64)

    def slot_receiver_xyz(self, slot: int) -> np.ndarray:
        """Position of the ``slot``-th OBN node (= virtual source position)."""
        return np.asarray(self.receiver_xyz_used[int(slot)], dtype=np.float64)

    def slot_shot_xyz(self, slot: int) -> np.ndarray:
        """``(n_shots_for_slot, 3)`` physical shot positions (= virtual recv)."""
        sl = self.slot_slice(slot)
        return np.stack(
            [self.plan_sx[sl], self.plan_sy[sl], self.plan_sz[sl]],
            axis=-1,
        ).astype(np.float64)

    def filter_by_row_mask(self, row_mask: np.ndarray) -> "CRGPlan":
        """Return a new plan keeping only flat rows where ``row_mask`` is True.

        The per-slot ``plan_offsets`` are recomputed so each slot still
        addresses a contiguous run of its surviving rows. Slots that lose
        ALL their rows survive with a zero-length slice — caller should
        compose with :meth:`filter_by_coverage` to drop them entirely.

        Used by the OBN CRG path to drop physical shots whose model-frame
        position falls outside the (optionally cropped) inversion grid.
        """
        m = np.asarray(row_mask, dtype=bool)
        if m.shape != (self.n_plan_rows,):
            raise ValueError(
                f"row_mask shape {m.shape} must equal (n_plan_rows,) = "
                f"({self.n_plan_rows},)"
            )
        if m.all():
            return self
        # Per-slot surviving row count drives the new offsets.
        new_counts = np.zeros(self.n_receivers_used, dtype=np.int64)
        for i in range(self.n_receivers_used):
            sl = self.slot_slice(i)
            new_counts[i] = int(m[sl].sum())
        new_offsets = np.empty(self.n_receivers_used + 1, dtype=np.int64)
        new_offsets[0] = 0
        new_offsets[1:] = np.cumsum(new_counts)
        return CRGPlan(
            files=list(self.files),
            trace_size_per_file=self.trace_size_per_file.copy(),
            sample_format=int(self.sample_format),
            samples_per_trace=int(self.samples_per_trace),
            dt_s=float(self.dt_s),
            receiver_indices=self.receiver_indices.copy(),
            receiver_xyz_used=self.receiver_xyz_used.copy(),
            plan_file_ids=self.plan_file_ids[m].copy(),
            plan_trace_offsets=self.plan_trace_offsets[m].copy(),
            plan_sx=self.plan_sx[m].copy(),
            plan_sy=self.plan_sy[m].copy(),
            plan_sz=self.plan_sz[m].copy(),
            plan_offsets=new_offsets,
        )

    def filter_by_coverage(self, min_shots: int) -> "CRGPlan":
        """Return a new plan keeping only slots with ``>= min_shots`` shots.

        Mirrors the legacy ``min_coverage`` filter — drops edge / partial-
        coverage OBN nodes that record so few shots they would hurt
        source-encoded supershot inversions (per ``crg_data._sample_crg_iter_shared_shots``).
        """
        if int(min_shots) <= 0:
            return self
        counts = self.per_slot_row_counts()
        keep = np.flatnonzero(counts >= int(min_shots))
        if keep.size == self.n_receivers_used:
            return self
        if keep.size == 0:
            raise ValueError(
                f"filter_by_coverage(min_shots={min_shots}) dropped every "
                f"virtual source (max count was {int(counts.max())})."
            )
        new_offsets = np.empty(keep.size + 1, dtype=np.int64)
        new_offsets[0] = 0
        new_offsets[1:] = np.cumsum(counts[keep])
        rows_kept = np.concatenate(
            [np.arange(int(self.plan_offsets[k]), int(self.plan_offsets[k + 1]),
                       dtype=np.int64)
             for k in keep]
        ) if keep.size else np.empty(0, dtype=np.int64)
        return CRGPlan(
            files=list(self.files),
            trace_size_per_file=self.trace_size_per_file.copy(),
            sample_format=int(self.sample_format),
            samples_per_trace=int(self.samples_per_trace),
            dt_s=float(self.dt_s),
            receiver_indices=self.receiver_indices[keep].copy(),
            receiver_xyz_used=self.receiver_xyz_used[keep].copy(),
            plan_file_ids=self.plan_file_ids[rows_kept].copy(),
            plan_trace_offsets=self.plan_trace_offsets[rows_kept].copy(),
            plan_sx=self.plan_sx[rows_kept].copy(),
            plan_sy=self.plan_sy[rows_kept].copy(),
            plan_sz=self.plan_sz[rows_kept].copy(),
            plan_offsets=new_offsets,
        )


def load_crg_shot_plan_cache(cache_path: str | Path) -> CRGPlan:
    """Load a ``crg_fwi_plan_v1`` npz cache into a :class:`CRGPlan`.

    SEG-Y paths recorded in the cache are remapped via
    :func:`_remap_segy_path` so the same cache works across hosts when
    ``FWI_SEGY_ROOT`` or ``FWI_SEGY_REMAP`` is exported.
    """
    cache_path = Path(cache_path)
    npz = np.load(cache_path, allow_pickle=False)
    try:
        fmt = str(np.asarray(npz["format"]).item())
        if fmt != "crg_fwi_plan_v1":
            raise ValueError(
                f"Unexpected plan cache format {fmt!r} in {cache_path}; "
                "expected 'crg_fwi_plan_v1'."
            )
        plan = CRGPlan(
            files=[Path(_remap_segy_path(p)) for p in npz["files"]],
            trace_size_per_file=np.asarray(npz["trace_size_per_file"], dtype=np.int64),
            sample_format=int(np.asarray(npz["sample_format"]).item()),
            samples_per_trace=int(np.asarray(npz["samples_per_trace"]).item()),
            dt_s=float(np.asarray(npz["dt_s"]).item()),
            receiver_indices=np.asarray(npz["receiver_indices"], dtype=np.int64),
            receiver_xyz_used=np.asarray(npz["receiver_xyz_used"], dtype=np.float64),
            plan_file_ids=np.asarray(npz["plan_file_ids"], dtype=np.int64),
            plan_trace_offsets=np.asarray(npz["plan_trace_offsets"], dtype=np.int64),
            plan_sx=np.asarray(npz["plan_sx"], dtype=np.float64),
            plan_sy=np.asarray(npz["plan_sy"], dtype=np.float64),
            plan_sz=np.asarray(npz["plan_sz"], dtype=np.float64),
            plan_offsets=np.asarray(npz["plan_offsets"], dtype=np.int64),
        )
    finally:
        npz.close()
    return plan


@dataclass(frozen=True)
class CRGSharedShotBatch:
    """One iteration's shared-shots CRG sample (for source encoding).

    Every slot in ``slot_indices`` records the SAME set of physical shots
    (identified by mm-quantized ``(sx, sy)`` UTM coordinates) — the
    invariant source-encoded supershot FWI needs. The shared shots get
    different byte offsets within each source-line SEG-Y because the
    file organises traces by (shot, receiver) and each slot is a
    different receiver — :attr:`rows_per_slot` resolves those per-slot
    byte addresses for downstream SEG-Y reads.

    Attributes
    ----------
    slot_indices
        ``(B,)`` int64 — virtual sources (OBN nodes) in this batch.
    rows_per_slot
        Length ``B`` list of ``(n_shared,)`` int64 arrays, each holding
        flat plan-row indices (= rows into ``plan.plan_file_ids`` /
        ``plan_trace_offsets`` / ``plan_sx`` / ...) for the i-th slot.
        All arrays have the same length; row ``j`` across slots refers
        to the SAME physical shot.
    shared_shot_xyz_m
        ``(n_shared, 3)`` float64 — physical shot positions of the
        intersection shots (= virtual receivers in CRG geometry).
    source_xyz_m
        ``(B, 3)`` float64 — virtual-source (OBN node) positions.
    n_shared
        Convenience, ``= rows_per_slot[0].size``.
    """

    slot_indices: np.ndarray
    rows_per_slot: list
    shared_shot_xyz_m: np.ndarray
    source_xyz_m: np.ndarray
    n_shared: int


def _shot_xy_keys(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Pack mm-quantised ``(sx, sy)`` into one int64 each (legacy convention).

    mm precision is far below any survey positioning accuracy (~meters),
    so two SEG-Y traces with the same physical shot point produce the
    same key. The packed form makes set operations O(N).
    """
    sxi = np.rint(np.asarray(sx, dtype=np.float64) * 1000.0).astype(np.int64)
    syi = np.rint(np.asarray(sy, dtype=np.float64) * 1000.0).astype(np.int64)
    return ((sxi & 0xFFFFFFFF) << 32) | (syi & 0xFFFFFFFF)


def sample_crg_shared_shots(
    plan: CRGPlan,
    rng: np.random.Generator,
    *,
    batch_size: int,
    source_lines_per_crg: int = 0,
    max_traces_per_sourceline: int = 0,
    min_coverage: int = 0,
    eligible_slots: np.ndarray | None = None,
    max_retries: int = 10,
) -> CRGSharedShotBatch:
    """Pick ``batch_size`` virtual sources whose shot positions intersect.

    Mirrors ``fwi_workflow-dev``'s ``_sample_crg_iter_shared_shots``:
    physical shots are identified by their mm-quantised ``(sx, sy)``
    keys (not byte offsets — each slot has a different byte offset for
    the same physical shot because the SEG-Y file groups traces by
    receiver). The intersection across slots gives the shots that EVERY
    chosen OBN node recorded.

    Optional sub-sampling within the intersection:

    * ``source_lines_per_crg > 0`` — keep that many random source-line
      ``file_id``s (= source lines).
    * ``max_traces_per_sourceline > 0`` — keep that many random shots
      per kept source line.
    * ``0`` for either disables that level.

    Raises ``ValueError`` when the intersection is empty after filtering
    (typical when ``batch_size`` is too large for partial-coverage nodes
    — raise ``min_coverage`` or lower ``batch_size``).
    """
    n_recv = int(plan.n_receivers_used)
    if eligible_slots is None:
        if min_coverage > 0:
            counts = plan.per_slot_row_counts()
            eligible = np.flatnonzero(counts >= int(min_coverage))
            if eligible.size == 0:
                eligible = np.arange(n_recv, dtype=np.int64)
        else:
            eligible = np.arange(n_recv, dtype=np.int64)
    else:
        eligible = np.asarray(eligible_slots, dtype=np.int64).reshape(-1)
    bs = max(1, min(int(batch_size), int(eligible.size)))

    K = int(source_lines_per_crg) if source_lines_per_crg and int(source_lines_per_crg) > 0 else 0
    M = int(max_traces_per_sourceline) if max_traces_per_sourceline and int(max_traces_per_sourceline) > 0 else 0

    # Retry slot sampling when the picked slots' shot (sx,sy) intersection
    # turns out to be empty (rare with min_coverage on, but can happen for
    # partial-coverage edge nodes). Mirrors the legacy soft fallback.
    slot_indices: np.ndarray | None = None
    per_slot_keys_full: list[np.ndarray] = []
    per_slot_keys_uniq: list[np.ndarray] = []
    inter = np.empty(0, dtype=np.int64)
    last_attempt: np.ndarray = np.empty(0, dtype=np.int64)
    for _attempt in range(max(1, int(max_retries))):
        cand = np.sort(rng.choice(eligible, size=bs, replace=False)).astype(np.int64)
        last_attempt = cand
        cand_keys_full: list[np.ndarray] = []
        cand_keys_uniq: list[np.ndarray] = []
        for slot in cand:
            sl = plan.slot_slice(int(slot))
            keys_full = _shot_xy_keys(plan.plan_sx[sl], plan.plan_sy[sl])
            cand_keys_full.append(keys_full)
            cand_keys_uniq.append(np.unique(keys_full))
        cand_inter = cand_keys_uniq[0]
        for ks in cand_keys_uniq[1:]:
            if cand_inter.size == 0 or ks.size == 0:
                cand_inter = np.empty(0, dtype=np.int64)
                break
            cand_inter = np.intersect1d(cand_inter, ks, assume_unique=True)
        if cand_inter.size > 0:
            slot_indices = cand
            per_slot_keys_full = cand_keys_full
            per_slot_keys_uniq = cand_keys_uniq
            inter = cand_inter
            break
    if slot_indices is None:
        raise ValueError(
            f"sample_crg_shared_shots: shot-(sx,sy) intersection is empty "
            f"for batch_size={bs} after {max_retries} retries (last attempt "
            f"slots={last_attempt.tolist()}). Raise min_coverage or lower "
            "batch_size."
        )

    # --- Look up the (file_id, sx, sy) of intersection shots via slot 0's rows.
    master_sl = plan.slot_slice(int(slot_indices[0]))
    master_fids = plan.plan_file_ids[master_sl].astype(np.int64)
    master_keys = per_slot_keys_full[0]
    master_sort = np.argsort(master_keys)
    pos_in_master = master_sort[
        np.searchsorted(master_keys[master_sort], inter)
    ]
    inter_fids = master_fids[pos_in_master]
    master_sx = plan.plan_sx[master_sl][pos_in_master]
    master_sy = plan.plan_sy[master_sl][pos_in_master]
    master_sz = plan.plan_sz[master_sl][pos_in_master]

    # --- Hierarchical sub-sample (line → trace).
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
            "sample_crg_shared_shots: empty after sub-sampling — relax "
            "source_lines_per_crg / max_traces_per_sourceline."
        )

    # --- For each slot, look up plan-row indices of the shared shots.
    rows_per_slot: list[np.ndarray] = []
    for si, slot in enumerate(slot_indices):
        sl = plan.slot_slice(int(slot))
        slot_keys = per_slot_keys_full[si]
        sort_idx = np.argsort(slot_keys)
        sorted_keys = slot_keys[sort_idx]
        ins = np.searchsorted(sorted_keys, inter)
        local_pos = sort_idx[ins]
        rows_per_slot.append((local_pos + int(sl.start)).astype(np.int64))

    shared_xyz = np.stack([master_sx, master_sy, master_sz], axis=-1).astype(np.float64)
    source_xyz = plan.receiver_xyz_used[slot_indices].astype(np.float64)
    return CRGSharedShotBatch(
        slot_indices=slot_indices,
        rows_per_slot=rows_per_slot,
        shared_shot_xyz_m=shared_xyz,
        source_xyz_m=source_xyz,
        n_shared=int(inter.size),
    )


__all__ = [
    "CRGPlan",
    "CRGSharedShotBatch",
    "load_crg_shot_plan_cache",
    "sample_crg_shared_shots",
]
