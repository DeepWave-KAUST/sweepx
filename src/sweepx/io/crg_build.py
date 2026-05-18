"""Build a :class:`CRGPlan` from raw source-line SEG-Y files.

Pipeline (sweep-tasks self-contained, no dependency on the legacy
``fwi_workflow-dev`` MPI scripts):

    raw SEG-Y (N files, headers only)
        │  scan_segy_headers via build_segy_index (thread pool)
        ▼
    SEGYIndex (per-trace (file_id, byte_offset, sx, sy, sz, rx, ry, rz))
        │  build_crg_plan_from_index:
        │    * quantise (rx, ry, rz) to ``receiver_quantize_m``
        │    * group traces by quantised receiver tuple → slot indices
        │    * (optional) receiver_stride / receiver_max / receiver_ids
        │    * (optional) shots_per_source_line / source_line_stride
        ▼
    CRGPlan (slim per-virtual-source plan)
        │  save_crg_plan → ``crg_fwi_plan_v1`` npz
        ▼
    Disk artifact (~120 MB; symmetric with load_crg_shot_plan_cache)

Output schema matches the legacy ``crg_fwi_plan_v1``, so files produced
by this builder are drop-in for any consumer of
:func:`sweep_io.crg_plan.load_crg_shot_plan_cache`.

The CLI front-end lives in ``sweep-tasks/cli.py`` as the
``sweep-tasks build-crg-plan`` subcommand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .crg_plan import CRGPlan
from .segy import SEGY_TRACE_HEADER_SIZE
from .segy_index import SEGYIndex, build_segy_index


def build_crg_plan_from_index(
    index: SEGYIndex,
    *,
    receiver_quantize_m: float = 0.5,
    receiver_stride: int = 1,
    receiver_max: int | None = None,
    receiver_ids: Sequence[int] | None = None,
    shots_per_source_line: int | None = None,
    source_line_stride: int = 1,
    rng_seed: int = 20260509,
) -> CRGPlan:
    """Group an :class:`SEGYIndex` by quantised receiver position, then
    apply optional receiver / shot sub-sampling, into a :class:`CRGPlan`.

    Quantisation: traces whose ``(rx, ry, rz)`` collapse to the same cell
    at ``receiver_quantize_m`` tolerance are considered the same OBN node.
    A regular node grid can use e.g. ``0.5 m``.

    Parameters
    ----------
    index
        Loaded :class:`SEGYIndex` over all source-line SEG-Y files. Built
        by :func:`sweep_io.segy_index.build_segy_index`.
    receiver_quantize_m
        Cell size (m) for receiver position quantisation.
    receiver_stride
        Keep every N-th unique receiver (after quantisation). ``1`` = all.
    receiver_max
        Cap the kept-receiver count after striding.
    receiver_ids
        Explicit receiver indices to keep (overrides stride / max).
        Indices refer to the receiver-id ordering AFTER quantisation +
        sort (same order ``CRGPlan.receiver_xyz_used`` is written in).
    shots_per_source_line
        Per-receiver cap on shots picked from each source-line file.
        ``None`` keeps all; e.g. ``4`` keeps at most four shots per line
        when building the canonical plan from CRG.
    source_line_stride
        Drop every Nth source line per receiver before the per-line cap.
        ``1`` keeps all lines.
    rng_seed
        Deterministic RNG seed for the per-line shot sub-sampler.

    Returns
    -------
    CRGPlan
        Ready to feed ``obs.crg_plan.plan_path`` in a sweep-tasks FWI
        YAML, or to ``save_crg_plan`` for the on-disk cache.
    """
    rng = np.random.default_rng(int(rng_seed))

    # --- 1) Per-trace receiver-cell key.
    rx, ry, rz = index.rx_m, index.ry_m, index.rz_m
    q = float(receiver_quantize_m)
    if q <= 0.0:
        raise ValueError(f"receiver_quantize_m must be > 0; got {q}")
    rcell = np.stack(
        [np.rint(rx / q).astype(np.int64),
         np.rint(ry / q).astype(np.int64),
         np.rint(rz / q).astype(np.int64)],
        axis=-1,
    )

    # Unique receiver cells + per-trace receiver-id.
    rcell_packed = (
        (rcell[:, 0] & 0x1FFFFF) << 42
        | (rcell[:, 1] & 0x1FFFFF) << 21
        | (rcell[:, 2] & 0x1FFFFF)
    )
    uniq_keys, inverse = np.unique(rcell_packed, return_inverse=True)
    n_recv_total = uniq_keys.size
    # Representative xyz per receiver: mean of all traces hitting that cell.
    receiver_xyz_all = np.zeros((n_recv_total, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_recv_total).astype(np.int64)
    np.add.at(receiver_xyz_all, inverse, np.stack([rx, ry, rz], axis=-1))
    receiver_xyz_all /= counts[:, None].astype(np.float64)

    # --- 2) Receiver subsetting.
    if receiver_ids is not None:
        kept_idx = np.asarray(list(receiver_ids), dtype=np.int64)
        if (kept_idx < 0).any() or (kept_idx >= n_recv_total).any():
            raise ValueError(
                f"receiver_ids out of [0, {n_recv_total}); "
                f"got min={kept_idx.min()} max={kept_idx.max()}"
            )
    else:
        kept_idx = np.arange(n_recv_total, dtype=np.int64)
        if int(receiver_stride) > 1:
            kept_idx = kept_idx[:: int(receiver_stride)]
        if receiver_max is not None:
            kept_idx = kept_idx[: int(receiver_max)]

    # --- 3) Sort traces by (receiver_id, file_id, byte_offset) for CSR groups.
    # We want per-receiver contiguous runs, but only for the kept receivers.
    kept_set = set(int(k) for k in kept_idx.tolist())
    trace_mask = np.fromiter(
        (rid in kept_set for rid in inverse.tolist()),
        count=inverse.size, dtype=bool,
    )
    sel_inverse = inverse[trace_mask]
    sel_file_id = index.file_id[trace_mask].astype(np.int64)
    sel_byte_off = index.byte_offset[trace_mask].astype(np.int64)
    sel_sx = index.sx_m[trace_mask]
    sel_sy = index.sy_m[trace_mask]
    sel_sz = index.sz_m[trace_mask]

    # Lexsort: primary by (kept-receiver-order-position), secondary by file,
    # tertiary by byte offset.
    # Map global receiver-id → kept-receiver-position (0..n_kept-1).
    pos_in_kept = np.full(n_recv_total, -1, dtype=np.int64)
    for k, gid in enumerate(kept_idx.tolist()):
        pos_in_kept[int(gid)] = k
    sel_pos = pos_in_kept[sel_inverse]
    order = np.lexsort((sel_byte_off, sel_file_id, sel_pos))
    sel_pos = sel_pos[order]
    sel_file_id = sel_file_id[order]
    sel_byte_off = sel_byte_off[order]
    sel_sx = sel_sx[order]
    sel_sy = sel_sy[order]
    sel_sz = sel_sz[order]

    # CSR offsets per kept receiver.
    n_kept = int(kept_idx.size)
    plan_offsets = np.zeros(n_kept + 1, dtype=np.int64)
    counts_kept = np.bincount(sel_pos, minlength=n_kept).astype(np.int64)
    plan_offsets[1:] = np.cumsum(counts_kept)

    # --- 4) Optional per-line / per-shot sub-sampling within each receiver.
    K = int(source_line_stride) if source_line_stride and source_line_stride > 0 else 1
    M = int(shots_per_source_line) if shots_per_source_line and shots_per_source_line > 0 else 0
    if K > 1 or M > 0:
        new_file_ids = []
        new_byte_offs = []
        new_sx = []
        new_sy = []
        new_sz = []
        new_offsets = [0]
        for slot in range(n_kept):
            s = int(plan_offsets[slot])
            e = int(plan_offsets[slot + 1])
            if s == e:
                new_offsets.append(int(new_offsets[-1]))
                continue
            slot_fids = sel_file_id[s:e]
            slot_offs = sel_byte_off[s:e]
            slot_sx = sel_sx[s:e]
            slot_sy = sel_sy[s:e]
            slot_sz = sel_sz[s:e]
            keep_in_slot: list[np.ndarray] = []
            unique_fids = np.unique(slot_fids)
            if K > 1:
                unique_fids = unique_fids[::K]
                line_mask = np.isin(slot_fids, unique_fids)
                local_idx = np.flatnonzero(line_mask)
            else:
                local_idx = np.arange(e - s, dtype=np.int64)
            if M > 0:
                # Per-line cap on shots.
                for fid in unique_fids:
                    line_local = local_idx[slot_fids[local_idx] == fid]
                    if line_local.size > M:
                        line_local = rng.choice(line_local, size=M, replace=False)
                    keep_in_slot.append(line_local)
                kept = np.sort(np.concatenate(keep_in_slot)) if keep_in_slot else np.empty(0, dtype=np.int64)
            else:
                kept = np.sort(local_idx)
            new_file_ids.append(slot_fids[kept])
            new_byte_offs.append(slot_offs[kept])
            new_sx.append(slot_sx[kept])
            new_sy.append(slot_sy[kept])
            new_sz.append(slot_sz[kept])
            new_offsets.append(int(new_offsets[-1]) + int(kept.size))
        sel_file_id = np.concatenate(new_file_ids) if new_file_ids else np.empty(0, dtype=np.int64)
        sel_byte_off = np.concatenate(new_byte_offs) if new_byte_offs else np.empty(0, dtype=np.int64)
        sel_sx = np.concatenate(new_sx) if new_sx else np.empty(0, dtype=np.float64)
        sel_sy = np.concatenate(new_sy) if new_sy else np.empty(0, dtype=np.float64)
        sel_sz = np.concatenate(new_sz) if new_sz else np.empty(0, dtype=np.float64)
        plan_offsets = np.asarray(new_offsets, dtype=np.int64)

    # --- 5) Trace stride per file (legacy schema needs it).
    sample_bytes = {1: 4, 3: 2, 5: 4, 6: 8, 8: 1}.get(int(index.sample_format), 4)
    trace_size_per_file = np.full(
        len(index.file_paths),
        SEGY_TRACE_HEADER_SIZE + int(index.n_samples) * sample_bytes,
        dtype=np.int64,
    )

    return CRGPlan(
        files=[Path(p) for p in index.file_paths],
        trace_size_per_file=trace_size_per_file,
        sample_format=int(index.sample_format),
        samples_per_trace=int(index.n_samples),
        dt_s=float(index.dt_s),
        receiver_indices=kept_idx.copy(),
        receiver_xyz_used=receiver_xyz_all[kept_idx].copy(),
        plan_file_ids=sel_file_id.astype(np.int64),
        plan_trace_offsets=sel_byte_off.astype(np.int64),
        plan_sx=sel_sx.astype(np.float64),
        plan_sy=sel_sy.astype(np.float64),
        plan_sz=sel_sz.astype(np.float64),
        plan_offsets=plan_offsets,
    )


def save_crg_plan(plan: CRGPlan, path: str | Path) -> Path:
    """Write a :class:`CRGPlan` to disk in the ``crg_fwi_plan_v1`` schema.

    Symmetric with :func:`sweep_io.crg_plan.load_crg_shot_plan_cache`;
    files written here load back identically (modulo path remapping via
    ``FWI_SEGY_ROOT`` / ``FWI_SEGY_REMAP``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        format=np.asarray("crg_fwi_plan_v1", dtype="U32"),
        files=np.asarray([str(p) for p in plan.files], dtype="U256"),
        trace_size_per_file=plan.trace_size_per_file,
        sample_format=np.asarray(plan.sample_format, dtype=np.int32),
        samples_per_trace=np.asarray(plan.samples_per_trace, dtype=np.int32),
        dt_s=np.asarray(plan.dt_s, dtype=np.float64),
        receiver_indices=plan.receiver_indices,
        receiver_xyz_used=plan.receiver_xyz_used,
        plan_file_ids=plan.plan_file_ids,
        plan_trace_offsets=plan.plan_trace_offsets,
        plan_sx=plan.plan_sx,
        plan_sy=plan.plan_sy,
        plan_sz=plan.plan_sz,
        plan_offsets=plan.plan_offsets,
    )
    return path


def _partition_files_contiguous(n: int, rank: int, size: int) -> tuple[int, int]:
    """Return ``(start, end)`` for rank ``rank``'s contiguous slice of
    ``range(n)``. The remainder is spread over the first ``n % size``
    ranks so partition sizes differ by at most one."""
    base, extra = divmod(int(n), int(size))
    if rank < extra:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rank * base + extra
        end = start + base
    return int(start), int(end)


def _merge_segy_indices(parts: list[SEGYIndex]) -> SEGYIndex:
    """Concatenate per-rank :class:`SEGYIndex` slices into one global index.

    Each part holds the file_id values local to its own ``file_paths``
    slice (0..len(slice)-1). On merge we shift those by the rank's file
    offset so the resulting index addresses the global path list.
    """
    if not parts:
        raise ValueError("_merge_segy_indices: empty parts")
    file_paths_all: list[str] = []
    file_id_chunks: list[np.ndarray] = []
    keys = ("byte_offset", "shot_id", "receiver_id",
            "sx_m", "sy_m", "sz_m", "rx_m", "ry_m", "rz_m")
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    n_samples = int(parts[0].n_samples)
    dt_s = float(parts[0].dt_s)
    sample_format = int(parts[0].sample_format)
    for p in parts:
        if int(p.n_samples) != n_samples or float(p.dt_s) != dt_s:
            raise ValueError(
                "_merge_segy_indices: inconsistent SEG-Y scalars across ranks; "
                "every file must share the same nt + dt."
            )
        offset = len(file_paths_all)
        file_paths_all.extend(p.file_paths)
        file_id_chunks.append(p.file_id.astype(np.int32, copy=False) + np.int32(offset))
        for k in keys:
            arrays[k].append(getattr(p, k))
    return SEGYIndex(
        file_paths=file_paths_all,
        file_id=np.concatenate(file_id_chunks),
        byte_offset=np.concatenate(arrays["byte_offset"]),
        shot_id=np.concatenate(arrays["shot_id"]),
        receiver_id=np.concatenate(arrays["receiver_id"]),
        sx_m=np.concatenate(arrays["sx_m"]),
        sy_m=np.concatenate(arrays["sy_m"]),
        sz_m=np.concatenate(arrays["sz_m"]),
        rx_m=np.concatenate(arrays["rx_m"]),
        ry_m=np.concatenate(arrays["ry_m"]),
        rz_m=np.concatenate(arrays["rz_m"]),
        dt_s=dt_s,
        n_samples=n_samples,
        sample_format=sample_format,
        meta={"byte_map": parts[0].meta.get("byte_map", {}),
              "n_files": len(file_paths_all),
              "merged_from_ranks": len(parts)},
    )


def _scan_segy_mpi(
    paths: list[Path],
    *,
    mpi_comm,
    byte_map: dict | None,
    source_depth_m_override: float | None,
    receiver_depth_m_override: float | None,
    coord_scalar_override: float | None,
    show_progress: bool,
    stage_dir: str | Path | None = None,
) -> "SEGYIndex | None":
    """MPI header-scan path: rank R scans ``paths[start_R:end_R]`` then
    rank 0 collects per-rank indices and returns the merged global index.

    Non-root ranks return ``None`` and should exit the pipeline early.

    Each rank scans single-threaded inside its slice (``num_workers=1``);
    rely on MPI ranks for the parallelism instead of nested threads.
    Hybrid threads + MPI doesn't help on header-scan because the GIL
    isn't a bottleneck (mmap reads release it) but multiple Python
    interpreters compete for file-descriptor cache.

    Collection uses file-based staging — every rank dumps its local
    :class:`SEGYIndex` to ``stage_dir/rank_<R>.npz`` and root reloads
    them sequentially. ``mpi_comm.gather`` of pickled SEGYIndex objects
    aborts with ``MPI.Exception: Invalid argument`` once the aggregate
    pickle stream on root exceeds mpi4py's pickled-gather limits (seen
    with tens of ranks on a large survey). The
    file path is the only object that crosses MPI. ``stage_dir`` must
    be visible from rank 0 — on multi-node jobs that means a shared
    filesystem; on single-node MPI any local dir works.
    """
    import shutil
    import tempfile

    from .segy_index import SEGYIndex

    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    n = len(paths)
    start, end = _partition_files_contiguous(n, rank, size)
    local_paths = list(paths[start:end])

    # Rank 0 picks the staging dir and broadcasts it so every rank writes
    # into the same place.
    if rank == 0:
        base = Path(stage_dir) if stage_dir is not None else Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        stage_path = Path(tempfile.mkdtemp(prefix="segy_mpi_scan_", dir=str(base)))
        stage_str = str(stage_path)
    else:
        stage_str = None
    stage_str = mpi_comm.bcast(stage_str, root=0)
    stage = Path(stage_str)

    if rank == 0 and show_progress:
        print(f"[build-crg-plan] MPI: {size} ranks; rank 0 scans "
              f"{len(local_paths)}/{n} files; stage_dir={stage}")

    if local_paths:
        local_index = build_segy_index(
            local_paths,
            byte_map=byte_map,
            source_depth_m_override=source_depth_m_override,
            receiver_depth_m_override=receiver_depth_m_override,
            coord_scalar_override=coord_scalar_override,
            num_workers=1,
            show_progress=False,
        )
        local_index.save(stage / f"rank_{rank:05d}.npz")

    mpi_comm.Barrier()
    if rank != 0:
        return None

    # Streaming merge: load one part, fold into chunk accumulators, drop it.
    # Holding every rank's SEGYIndex + the final merged copy peaked at
    # several times the merged size during the merge sort;
    # streaming caps peak at ~one part + accumulator.
    keys = ("byte_offset", "shot_id", "receiver_id",
            "sx_m", "sy_m", "sz_m", "rx_m", "ry_m", "rz_m")
    file_paths_all: list[str] = []
    file_id_chunks: list[np.ndarray] = []
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    n_samples: int | None = None
    dt_s: float | None = None
    sample_format: int = 5
    byte_map_meta: dict = {}
    n_merged = 0
    for r in range(size):
        p = stage / f"rank_{r:05d}.npz"
        if not p.exists():
            continue
        part = SEGYIndex.load(p)
        if n_samples is None:
            n_samples = int(part.n_samples)
            dt_s = float(part.dt_s)
            sample_format = int(part.sample_format)
            byte_map_meta = part.meta.get("byte_map", {}) or {}
        elif int(part.n_samples) != n_samples or float(part.dt_s) != dt_s:
            raise ValueError(
                "_scan_segy_mpi: inconsistent SEG-Y scalars across ranks; "
                "every file must share the same nt + dt."
            )
        offset = len(file_paths_all)
        file_paths_all.extend(part.file_paths)
        file_id_chunks.append(
            part.file_id.astype(np.int32, copy=False) + np.int32(offset)
        )
        for k in keys:
            chunks[k].append(getattr(part, k))
        n_merged += 1
        del part
    try:
        shutil.rmtree(stage)
    except OSError:
        pass

    if n_merged == 0:
        raise RuntimeError("No SEG-Y files were scanned by any rank.")

    return SEGYIndex(
        file_paths=file_paths_all,
        file_id=np.concatenate(file_id_chunks),
        byte_offset=np.concatenate(chunks["byte_offset"]),
        shot_id=np.concatenate(chunks["shot_id"]),
        receiver_id=np.concatenate(chunks["receiver_id"]),
        sx_m=np.concatenate(chunks["sx_m"]),
        sy_m=np.concatenate(chunks["sy_m"]),
        sz_m=np.concatenate(chunks["sz_m"]),
        rx_m=np.concatenate(chunks["rx_m"]),
        ry_m=np.concatenate(chunks["ry_m"]),
        rz_m=np.concatenate(chunks["rz_m"]),
        dt_s=dt_s,
        n_samples=n_samples,
        sample_format=sample_format,
        meta={
            "byte_map": byte_map_meta,
            "n_files": len(file_paths_all),
            "merged_from_ranks": n_merged,
        },
    )


def build_crg_plan_from_segy(
    paths: Sequence[str | Path],
    *,
    byte_map: dict | None = None,
    source_depth_m_override: float | None = None,
    receiver_depth_m_override: float | None = None,
    coord_scalar_override: float | None = None,
    num_workers: int = 16,
    use_mpi: bool | None = None,
    mpi_stage_dir: str | Path | None = None,
    receiver_quantize_m: float = 0.5,
    receiver_stride: int = 1,
    receiver_max: int | None = None,
    receiver_ids: Sequence[int] | None = None,
    shots_per_source_line: int | None = None,
    source_line_stride: int = 1,
    rng_seed: int = 20260509,
    show_progress: bool = False,
) -> "CRGPlan | None":
    """End-to-end: scan raw SEG-Y headers, then group into a :class:`CRGPlan`.

    Convenience wrapper around :func:`sweep_io.segy_index.build_segy_index`
    (parallel header scan) followed by :func:`build_crg_plan_from_index`
    (receiver grouping + sub-sampling).

    Parameters mirror both helpers. See
    :func:`sweep_io.segy_index.build_segy_index` for the SEG-Y header
    knobs (``byte_map``, ``source_depth_m_override`` etc.) and the
    arguments of :func:`build_crg_plan_from_index` for the receiver-side
    knobs.

    Parallelism modes (mutually exclusive):

    * **MPI** (set ``use_mpi=True`` *or* run under ``mpiexec`` with
      ``use_mpi=None`` auto-detect): rank 0 partitions the file list,
      every rank scans a contiguous slice single-threaded, rank 0
      gathers + merges. **Non-root ranks return** ``None``.
    * **ThreadPool** (default, ``use_mpi=False``): one process,
      ``num_workers`` background threads for the per-file scan.

    The ``num_workers`` knob is ignored under MPI (MPI ranks ARE the
    parallelism; nesting threads inside MPI workers competes for
    file-descriptor cache without helping).
    """
    paths_list = [Path(p) for p in paths]

    # Auto-detect: if mpi4py is importable AND size > 1, default to MPI.
    mpi_comm = None
    if use_mpi is not False:
        try:
            from mpi4py import MPI  # type: ignore

            comm = MPI.COMM_WORLD
            if use_mpi is True or comm.Get_size() > 1:
                mpi_comm = comm
        except ImportError:
            if use_mpi is True:
                raise RuntimeError(
                    "build_crg_plan_from_segy: use_mpi=True but mpi4py is not "
                    "installed. ``pip install mpi4py`` or run without --mpi."
                )

    if mpi_comm is not None:
        idx = _scan_segy_mpi(
            paths_list,
            mpi_comm=mpi_comm,
            byte_map=byte_map,
            source_depth_m_override=source_depth_m_override,
            receiver_depth_m_override=receiver_depth_m_override,
            coord_scalar_override=coord_scalar_override,
            show_progress=show_progress,
            stage_dir=mpi_stage_dir,
        )
        if idx is None:
            # Non-root rank: nothing to do.
            return None
    else:
        idx = build_segy_index(
            paths_list,
            byte_map=byte_map,
            source_depth_m_override=source_depth_m_override,
            receiver_depth_m_override=receiver_depth_m_override,
            coord_scalar_override=coord_scalar_override,
            num_workers=int(num_workers),
            show_progress=show_progress,
        )
    return build_crg_plan_from_index(
        idx,
        receiver_quantize_m=receiver_quantize_m,
        receiver_stride=receiver_stride,
        receiver_max=receiver_max,
        receiver_ids=receiver_ids,
        shots_per_source_line=shots_per_source_line,
        source_line_stride=source_line_stride,
        rng_seed=rng_seed,
    )


__all__ = [
    "build_crg_plan_from_index",
    "build_crg_plan_from_segy",
    "save_crg_plan",
]
