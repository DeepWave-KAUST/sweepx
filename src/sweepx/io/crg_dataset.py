"""``torch.utils.data.Dataset`` adapter over a :class:`CRGPlan`.

Each ``__getitem__(i)`` returns one virtual-source gather (the i-th OBN
node's traces + per-shot xyz). When wrapped in a ``DataLoader`` with
``num_workers > 0``, the SEG-Y reads run on the worker side and the
main process only sees pinned-host batches ready for H2D transfer to the
solver.

Worker setup
------------
``MultiFileSEGYReader`` keeps mmap handles open; those don't survive
``fork`` cleanly. :func:`worker_init_fn` re-opens the reader per worker
process the first time the worker draws an item.

Padding
-------
Virtual-source gathers have variable lengths (each OBN node sees a
different number of shots). :func:`pad_collate_fn` builds a rectangular
``(B, n_recv_max, nt)`` batch by zero-padding the receiver axis and
returns a matching ``(B, n_recv_max)`` ``valid_receiver_mask`` so the
loss / forward solver can ignore the pad rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .crg_plan import CRGPlan
from .segy import MultiFileSEGYReader, SEGY_TRACE_HEADER_SIZE


@dataclass
class _PerSlotSample:
    """Concrete numpy payload returned by :meth:`CRGBatchDataset.__getitem__`."""

    traces: np.ndarray         # (n_shots, nt) float32 — physical shots at this OBN node
    shot_xyz_m: np.ndarray     # (n_shots, 3) float64 — physical shot coords (= virt recv)
    source_xyz_m: np.ndarray   # (3,) float64 — virtual-source coord (= OBN node)
    slot: int                  # plan slot index
    n_shots: int               # convenience, = traces.shape[0]


class CRGBatchDataset:
    """``Dataset`` yielding one virtual-source gather per ``__getitem__`` call.

    Designed for ``torch.utils.data.DataLoader`` with ``num_workers > 0``
    + :func:`pad_collate_fn` + :func:`worker_init_fn`.

    Parameters
    ----------
    plan
        Loaded :class:`CRGPlan`.
    slot_indices
        Subset of slot indices to expose (default: all). Use this to
        shard slots across distributed ranks **before** the DataLoader.
    max_shots_per_slot
        Truncate each slot's shot list to this length (or ``None`` to
        keep them all). Truncation is deterministic (first ``max_shots``);
        per-iter random sub-sampling belongs upstream in the runner.
    """

    def __init__(
        self,
        plan: CRGPlan,
        *,
        slot_indices: np.ndarray | None = None,
        max_shots_per_slot: int | None = None,
    ) -> None:
        self._plan = plan
        if slot_indices is None:
            self._slot_indices = np.arange(plan.n_receivers_used, dtype=np.int64)
        else:
            self._slot_indices = np.asarray(slot_indices, dtype=np.int64)
            if self._slot_indices.ndim != 1:
                raise ValueError(
                    f"slot_indices must be 1-D; got shape {self._slot_indices.shape}"
                )
        self._max_shots_per_slot = (
            int(max_shots_per_slot)
            if max_shots_per_slot is not None and int(max_shots_per_slot) > 0
            else None
        )
        # The reader is reopened per process (workers fork before draw).
        self._reader: MultiFileSEGYReader | None = None

    # --------------------------------------------------------- accessors
    def __len__(self) -> int:
        return int(self._slot_indices.size)

    @property
    def plan(self) -> CRGPlan:
        return self._plan

    @property
    def slot_indices(self) -> np.ndarray:
        return self._slot_indices

    @property
    def samples_per_trace(self) -> int:
        return int(self._plan.samples_per_trace)

    @property
    def dt_s(self) -> float:
        return float(self._plan.dt_s)

    @property
    def n_shots_per_slot(self) -> np.ndarray:
        """``(len(self),)`` shot count after ``max_shots_per_slot`` cap."""
        counts = np.diff(self._plan.plan_offsets).astype(np.int64)
        sub = counts[self._slot_indices]
        if self._max_shots_per_slot is not None:
            sub = np.minimum(sub, self._max_shots_per_slot)
        return sub

    @property
    def n_shots_max(self) -> int:
        """Max gather length across exposed slots — for ``n_recv_max`` sizing."""
        return int(self.n_shots_per_slot.max(initial=0))

    # --------------------------------------------------------- reader mgmt
    def _ensure_reader(self) -> MultiFileSEGYReader:
        if self._reader is None:
            self._reader = MultiFileSEGYReader(
                [str(p) for p in self._plan.files], mmap_mode=True,
            )
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __getstate__(self) -> dict[str, Any]:
        # ``mmap``-backed reader doesn't pickle across fork. Drop it; the
        # worker reopens on first __getitem__.
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # --------------------------------------------------------- main API
    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        slot = int(self._slot_indices[int(i)])
        sl = self._plan.slot_slice(slot)
        file_ids = self._plan.plan_file_ids[sl]
        # SEG-Y "byte offset" semantics in the plan cache: ``plan_trace_offsets``
        # holds the trace-header byte offset within each file (i.e. the start
        # of the (header + samples) record). ``MultiFileSEGYReader.read_traces``
        # expects exactly that.
        byte_offs = self._plan.plan_trace_offsets[sl]
        sx = self._plan.plan_sx[sl]
        sy = self._plan.plan_sy[sl]
        sz = self._plan.plan_sz[sl]
        if self._max_shots_per_slot is not None and file_ids.size > self._max_shots_per_slot:
            keep = slice(0, self._max_shots_per_slot)
            file_ids = file_ids[keep]
            byte_offs = byte_offs[keep]
            sx = sx[keep]; sy = sy[keep]; sz = sz[keep]
        reader = self._ensure_reader()
        traces = reader.read_traces(file_ids, byte_offs).astype(np.float32, copy=False)
        shot_xyz = np.stack([sx, sy, sz], axis=-1).astype(np.float64)
        source_xyz = np.asarray(self._plan.receiver_xyz_used[slot], dtype=np.float64)
        return {
            "traces": traces,                       # (n_shots, nt)
            "shot_xyz_m": shot_xyz,                 # (n_shots, 3)
            "source_xyz_m": source_xyz,             # (3,)
            "slot": np.int64(slot),
            "n_shots": np.int64(traces.shape[0]),
        }


def worker_init_fn(worker_id: int) -> None:
    """``DataLoader.worker_init_fn`` for :class:`CRGBatchDataset`.

    Closes any pre-fork :class:`MultiFileSEGYReader` so each worker
    re-opens its own mmap handles. Without this, multiple workers may
    share a single inherited handle, race on its file descriptor, and
    return scrambled traces.
    """
    import torch.utils.data

    info = torch.utils.data.get_worker_info()
    if info is None or not hasattr(info, "dataset"):
        return
    ds = info.dataset
    if isinstance(ds, CRGBatchDataset):
        ds.close()  # next __getitem__ will reopen in this worker process


def pad_collate_fn(
    batch: list[dict[str, np.ndarray]],
    *,
    n_recv_max: int | None = None,
    nt: int | None = None,
) -> dict[str, Any]:
    """Stack variable-length virtual-source gathers into a padded batch.

    Output dict keys:

    * ``traces``               ``(B, n_recv_max, nt)`` float32
    * ``valid_receiver_mask``  ``(B, n_recv_max)``    bool
    * ``shot_xyz_m``           ``(B, n_recv_max, 3)`` float64
    * ``source_xyz_m``         ``(B, 3)``             float64
    * ``slot``                 ``(B,)``               int64
    * ``n_shots``              ``(B,)``               int64
    """
    if not batch:
        raise ValueError("pad_collate_fn called on empty batch")
    counts = np.asarray([int(b["n_shots"]) for b in batch], dtype=np.int64)
    if n_recv_max is None:
        n_recv_max = int(counts.max())
    if nt is None:
        nt = int(batch[0]["traces"].shape[-1])
    B = len(batch)
    traces = np.zeros((B, n_recv_max, nt), dtype=np.float32)
    shot_xyz = np.zeros((B, n_recv_max, 3), dtype=np.float64)
    valid = np.zeros((B, n_recv_max), dtype=bool)
    source_xyz = np.zeros((B, 3), dtype=np.float64)
    slots = np.zeros(B, dtype=np.int64)
    for i, item in enumerate(batch):
        n = int(item["n_shots"])
        if n > n_recv_max:
            # truncate — caller chose n_recv_max smaller than the natural max
            n = n_recv_max
        if n > 0:
            traces[i, :n, :] = item["traces"][:n]
            shot_xyz[i, :n, :] = item["shot_xyz_m"][:n]
            valid[i, :n] = True
        source_xyz[i] = item["source_xyz_m"]
        slots[i] = int(item["slot"])
    return {
        "traces": traces,
        "valid_receiver_mask": valid,
        "shot_xyz_m": shot_xyz,
        "source_xyz_m": source_xyz,
        "slot": slots,
        "n_shots": counts,
    }


class TraceCache:
    """Bounded LRU cache for SEG-Y traces keyed by ``(file_id, byte_offset)``.

    Mirrors the in-RAM trace cache the legacy ``CRGBatchPrefetcher`` uses
    to absorb cold-Lustre re-reads in source-encoded supershot FWI: each
    iter touches ~24×100 traces, ~16 KB each → ~40 MB/iter. With a 5 GB
    cap (~300k traces) the cache covers ~12 random batches before LRU
    eviction kicks in. The legacy ``--trace-cache-bytes -1`` (unlimited)
    corresponds to ``max_bytes=0``.

    Thread-safe enough for the CRG path's use: the cache is consulted
    only from the prefetch thread + its inner io pool, which are the
    same process; insertion is dict-atomic in CPython. For multi-process
    DataLoader use, instantiate one cache per worker.
    """

    def __init__(self, max_bytes: int = 0) -> None:
        from collections import OrderedDict

        self.max_bytes = int(max_bytes)  # 0 = unbounded
        self._store: "OrderedDict[tuple[int, int], np.ndarray]" = OrderedDict()
        self._cur_bytes = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    @property
    def cur_bytes(self) -> int:
        return self._cur_bytes

    def get(self, file_id: int, byte_offset: int) -> "np.ndarray | None":
        key = (int(file_id), int(byte_offset))
        v = self._store.get(key)
        if v is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return v

    def put(self, file_id: int, byte_offset: int, trace: "np.ndarray") -> None:
        key = (int(file_id), int(byte_offset))
        existing = self._store.get(key)
        if existing is not None:
            self._store.move_to_end(key)
            return
        # Store a contiguous copy so the cache is decoupled from any
        # mmap-backed source array that might be released by the reader.
        arr = np.ascontiguousarray(trace, dtype=np.float32)
        self._store[key] = arr
        self._cur_bytes += int(arr.nbytes)
        if self.max_bytes > 0:
            while self._cur_bytes > self.max_bytes and self._store:
                _, evicted = self._store.popitem(last=False)
                self._cur_bytes -= int(evicted.nbytes)


def read_traces_cached(
    reader,
    cache: TraceCache,
    file_ids,
    byte_offsets,
    *,
    coalesce_gap: int = 0,
) -> np.ndarray:
    """Look up traces in ``cache`` first, only call ``reader.read_traces``
    for misses, then insert the freshly read traces back into the cache.

    ``coalesce_gap`` is forwarded to :meth:`MultiFileSEGYReader.read_traces`
    so adjacent miss traces within the same file get merged into a single
    ``pread`` — dramatic speedup on Lustre where per-call latency
    dominates small random reads. Sensible values: the per-trace stride
    (≈ ``240 + samples_per_trace * 4`` bytes) to coalesce any two
    sequentially-indexed traces in the same file, or up through one
    filesystem block (~1 MB) for more aggressive merging.

    Returns a ``(n, n_samples)`` ``float32`` ndarray in caller-order
    (matches :meth:`sweep_io.segy.MultiFileSEGYReader.read_traces`).
    """
    fids = np.asarray(file_ids, dtype=np.int64)
    offs = np.asarray(byte_offsets, dtype=np.int64)
    n = int(fids.size)
    out = np.empty((n, reader.n_samples), dtype=np.float32)
    miss_idx_list: list[int] = []
    for i in range(n):
        cached = cache.get(int(fids[i]), int(offs[i]))
        if cached is None:
            miss_idx_list.append(i)
        else:
            out[i] = cached
    if miss_idx_list:
        miss_idx = np.asarray(miss_idx_list, dtype=np.int64)
        sub = reader.read_traces(
            fids[miss_idx], offs[miss_idx], coalesce_gap=coalesce_gap,
        )
        for j, i in enumerate(miss_idx_list):
            out[i] = sub[j]
            cache.put(int(fids[i]), int(offs[i]), sub[j])
    return out


__all__ = [
    "CRGBatchDataset",
    "TraceCache",
    "pad_collate_fn",
    "read_traces_cached",
    "worker_init_fn",
]
