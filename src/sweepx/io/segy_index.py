"""Persistent SEG-Y header catalog + lazy shot-gather dataset.

For projects where the data is spread across many SEG-Y files,
the canonical workflow is:

    1. Run :func:`build_segy_index` **once** to scan all files in parallel
       and produce a small (~few tens of MB) ``SEGYIndex`` with one row
       per trace: ``(file_id, byte_offset, shot_id, receiver_id, sx, sy,
       sz, rx, ry, rz)`` + a per-shot lookup table.

    2. For every subsequent inversion / imaging run, load the index
       (:meth:`SEGYIndex.load`) and iterate shots via
       :class:`IndexedShotGatherDataset`, which fetches each shot's
       traces with one coalesced byte-offset read against the underlying
       SEG-Y files. Pairs naturally with :class:`sweep_io.prefetch.Prefetcher`
       and :class:`sweep_io.cuda_prefetch.CUDAPrefetcher` for I/O / compute
       overlap.

For single-file mid-sized datasets (Viking line-12: 750 MB), the
heavier index workflow is overkill — use :class:`sweep_io.segy.SEGYReader`
directly. The two approaches share the same byte-offset reader; the
index just persists the catalog so you don't pay header-scan cost on
every run.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .prefetch import ThreadPoolPrefetcher
from .segy import (
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    MultiFileSEGYReader,
    SEGYReader,
)


# SEG-Y rev1 standard trace-header byte offsets (0-indexed in the 240-byte header)
SEGY_REV1_BYTES = {
    "shot":              8,    # 9-12   field record number / FFID
    "trace_in_shot":     12,   # 13-16  trace number within FFID
    "coord_scalar":      70,   # 71-72  int16 multiplier (negative = divisor)
    "sx":                72,   # 73-76
    "sy":                76,   # 77-80
    "rx":                80,   # 81-84
    "ry":                84,   # 85-88
    "source_elev":       40,   # 41-44  (raw, scaled by coord_scalar)
    "receiver_elev":     104,  # 105-108
    "source_depth":      48,   # 49-52  (default; OBN-3D overrides to 44)
    "receiver_depth":    52,   # 53-56  (typical)
}


# ============================================================================
# SEGYIndex
# ============================================================================
@dataclass
class SEGYIndex:
    """Per-trace catalog backed by 1-D numpy columns.

    All trace-level arrays have the same length = total trace count across
    all files. ``file_id[i]`` indexes :attr:`file_paths`. ``shot_id[i]`` is
    the group key (usually FFID); the same value across all traces of one
    shot. ``receiver_id[i]`` is the trace's index within its shot (0-based
    after sort).

    Per-shot fast lookup is built on construction via
    :meth:`_build_shot_lookup`. ``lookup_shot(shot_id)`` returns the slice
    in O(log n).
    """

    file_paths: list[str]
    file_id:     np.ndarray   # int32
    byte_offset: np.ndarray   # int64
    shot_id:     np.ndarray   # int64
    receiver_id: np.ndarray   # int32
    sx_m: np.ndarray          # float64
    sy_m: np.ndarray
    sz_m: np.ndarray
    rx_m: np.ndarray
    ry_m: np.ndarray
    rz_m: np.ndarray
    dt_s: float
    n_samples: int
    sample_format: int = 5
    meta: dict = field(default_factory=dict)

    # Built post-init: shot lookup
    _shot_starts: np.ndarray = field(init=False, repr=False)
    _shot_ids_unique: np.ndarray = field(init=False, repr=False)

    SCHEMA_VERSION: int = 1

    def __post_init__(self) -> None:
        n = self.file_id.size
        for name in ("byte_offset", "shot_id", "receiver_id",
                     "sx_m", "sy_m", "sz_m", "rx_m", "ry_m", "rz_m"):
            arr = getattr(self, name)
            if arr.size != n:
                raise ValueError(
                    f"SEGYIndex column {name!r} has length {arr.size}, "
                    f"expected {n} to match file_id."
                )
        self._build_shot_lookup()

    def _build_shot_lookup(self) -> None:
        """Sort traces by shot_id (stable) and compute per-shot start offsets."""
        order = np.argsort(self.shot_id, kind="stable")
        for name in ("file_id", "byte_offset", "shot_id", "receiver_id",
                     "sx_m", "sy_m", "sz_m", "rx_m", "ry_m", "rz_m"):
            setattr(self, name, getattr(self, name)[order])
        # Unique shot ids and the start index of each shot's contiguous run.
        self._shot_ids_unique, starts = np.unique(self.shot_id, return_index=True)
        # Append the tail so we can slice via [start:end] without special-case.
        self._shot_starts = np.concatenate([starts, [self.shot_id.size]])

    # ------------------------------------------------------------------ derived
    @property
    def n_traces(self) -> int:
        return int(self.file_id.size)

    @property
    def n_shots(self) -> int:
        return int(self._shot_ids_unique.size)

    @property
    def shot_ids(self) -> np.ndarray:
        return self._shot_ids_unique

    # ------------------------------------------------------------- shot lookup
    def lookup_shot(self, shot_id: int) -> dict:
        """Return all per-trace + source info for one shot. O(log n_shots)."""
        idx = int(np.searchsorted(self._shot_ids_unique, shot_id))
        if idx >= self._shot_ids_unique.size or int(self._shot_ids_unique[idx]) != int(shot_id):
            raise KeyError(f"shot_id {shot_id} not in index")
        a, b = int(self._shot_starts[idx]), int(self._shot_starts[idx + 1])
        return {
            "shot_id": int(shot_id),
            "file_id": self.file_id[a:b],
            "byte_offset": self.byte_offset[a:b],
            "receiver_id": self.receiver_id[a:b],
            "rx_m": self.rx_m[a:b],
            "ry_m": self.ry_m[a:b],
            "rz_m": self.rz_m[a:b],
            "sx_m": float(self.sx_m[a]),  # constant within a shot
            "sy_m": float(self.sy_m[a]),
            "sz_m": float(self.sz_m[a]),
        }

    # -------------------------------------------------------- serialization
    def save(self, path: str | Path) -> Path:
        """Save as `.npz` (compressed). Loads back via :meth:`load`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            file_id=self.file_id,
            byte_offset=self.byte_offset,
            shot_id=self.shot_id,
            receiver_id=self.receiver_id,
            sx_m=self.sx_m, sy_m=self.sy_m, sz_m=self.sz_m,
            rx_m=self.rx_m, ry_m=self.ry_m, rz_m=self.rz_m,
            file_paths=np.array(self.file_paths, dtype=object),
            _meta=json.dumps({
                "dt_s": self.dt_s,
                "n_samples": self.n_samples,
                "sample_format": self.sample_format,
                "schema_version": self.SCHEMA_VERSION,
                "meta": self.meta,
            }),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SEGYIndex":
        with np.load(path, allow_pickle=True) as z:
            meta = json.loads(str(z["_meta"]))
            return cls(
                file_paths=list(z["file_paths"]),
                file_id=np.asarray(z["file_id"]),
                byte_offset=np.asarray(z["byte_offset"]),
                shot_id=np.asarray(z["shot_id"]),
                receiver_id=np.asarray(z["receiver_id"]),
                sx_m=np.asarray(z["sx_m"]), sy_m=np.asarray(z["sy_m"]), sz_m=np.asarray(z["sz_m"]),
                rx_m=np.asarray(z["rx_m"]), ry_m=np.asarray(z["ry_m"]), rz_m=np.asarray(z["rz_m"]),
                dt_s=float(meta["dt_s"]),
                n_samples=int(meta["n_samples"]),
                sample_format=int(meta.get("sample_format", 5)),
                meta=meta.get("meta", {}) or {},
            )

    # ------------------------------------------------------ geometry adapter
    def to_physical_geometry(self, *, shot_ids: Sequence[int] | None = None):
        """Materialise a :class:`sweep_io.geometry.PhysicalGeometry`.

        Expects every shot to have the same receiver count (typical for
        streamers). For OBN / variable-receiver-count layouts, slice the
        index by shot_id first.
        """
        from .geometry import PhysicalGeometry

        sids = np.asarray(shot_ids if shot_ids is not None else self.shot_ids)
        # nrec per shot (must be uniform)
        per_shot = []
        for sid in sids:
            slc = self.lookup_shot(int(sid))
            per_shot.append(slc)
        n_recs = {len(s["rx_m"]) for s in per_shot}
        if len(n_recs) != 1:
            raise ValueError(
                f"to_physical_geometry needs a uniform receiver count per shot; "
                f"got {sorted(n_recs)}. Slice the index first if intentional."
            )
        nrec = n_recs.pop()
        sources = np.stack(
            [np.array([s["sx_m"], s["sz_m"]]) for s in per_shot]
        )
        receivers = np.stack(
            [np.stack([s["rx_m"], s["rz_m"]], axis=1) for s in per_shot]
        )
        return PhysicalGeometry(
            sources_xyz_m=sources,
            receivers_xyz_m=receivers,
            dt=self.dt_s, nt=self.n_samples,
            meta={**self.meta, "source": "SEGYIndex"},
        )


# ============================================================================
# Index builder
# ============================================================================
def _scan_one_file(
    path: str | Path,
    file_id: int,
    *,
    byte_map: dict,
    source_depth_m_override: float | None,
    receiver_depth_m_override: float | None,
    coord_scalar_override: float | None,
) -> dict:
    """Worker: scan all trace headers in one SEG-Y; return per-trace columns."""
    path = Path(path)
    reader = SEGYReader(path, mmap_mode=True)
    try:
        nt = reader.n_traces
        ts = SEGY_TRACE_HEADER_SIZE
        per_trace_bytes = ts + reader.n_samples * 4    # IBM/IEEE 32-bit
        # Defensive: handle non-float formats too
        if reader.sample_format in (3,):
            per_trace_bytes = ts + reader.n_samples * 2
        elif reader.sample_format in (8,):
            per_trace_bytes = ts + reader.n_samples
        elif reader.sample_format in (6,):
            per_trace_bytes = ts + reader.n_samples * 8

        base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
        shot_id      = np.empty(nt, dtype=np.int64)
        trace_in_shot = np.empty(nt, dtype=np.int32)
        sx_raw       = np.empty(nt, dtype=np.int64)
        sy_raw       = np.empty(nt, dtype=np.int64)
        rx_raw       = np.empty(nt, dtype=np.int64)
        ry_raw       = np.empty(nt, dtype=np.int64)
        sz_raw       = np.empty(nt, dtype=np.int64)
        rz_raw       = np.empty(nt, dtype=np.int64)
        coord_scale  = np.empty(nt, dtype=np.int64)
        byte_offset  = np.empty(nt, dtype=np.int64)

        for i in range(nt):
            hdr_off = base + i * per_trace_bytes
            byte_offset[i] = hdr_off
            h = reader._pread(hdr_off, ts)
            shot_id[i]       = struct.unpack(">i", h[byte_map["shot"]    :byte_map["shot"]    + 4])[0]
            trace_in_shot[i] = struct.unpack(">i", h[byte_map["trace_in_shot"]:byte_map["trace_in_shot"]+4])[0]
            coord_scale[i]   = struct.unpack(">h", h[byte_map["coord_scalar"]:byte_map["coord_scalar"]+2])[0]
            sx_raw[i] = struct.unpack(">i", h[byte_map["sx"]:byte_map["sx"] + 4])[0]
            sy_raw[i] = struct.unpack(">i", h[byte_map["sy"]:byte_map["sy"] + 4])[0]
            rx_raw[i] = struct.unpack(">i", h[byte_map["rx"]:byte_map["rx"] + 4])[0]
            ry_raw[i] = struct.unpack(">i", h[byte_map["ry"]:byte_map["ry"] + 4])[0]
            # Depths are typically zero on marine sets — overrides applied below.
            sz_raw[i] = struct.unpack(">i", h[byte_map["source_depth"]:byte_map["source_depth"] + 4])[0]
            rz_raw[i] = struct.unpack(">i", h[byte_map["receiver_depth"]:byte_map["receiver_depth"] + 4])[0]

        # Apply coord scalar: negative = divisor, positive = multiplier, 0 → 1.
        scale_per_trace = coord_scale.astype(np.float64)
        factor = np.where(
            scale_per_trace == 0, 1.0,
            np.where(scale_per_trace > 0, scale_per_trace, 1.0 / np.abs(scale_per_trace)),
        )
        if coord_scalar_override is not None:
            factor = np.full_like(factor, float(coord_scalar_override))

        sx_m = sx_raw.astype(np.float64) * factor
        sy_m = sy_raw.astype(np.float64) * factor
        rx_m = rx_raw.astype(np.float64) * factor
        ry_m = ry_raw.astype(np.float64) * factor
        # Depth: use override if provided (Viking case — bytes are zero in the file).
        if source_depth_m_override is not None:
            sz_m = np.full(nt, float(source_depth_m_override), dtype=np.float64)
        else:
            sz_m = sz_raw.astype(np.float64) * factor
        if receiver_depth_m_override is not None:
            rz_m = np.full(nt, float(receiver_depth_m_override), dtype=np.float64)
        else:
            rz_m = rz_raw.astype(np.float64) * factor

        return {
            "n_traces": nt,
            "file_id": np.full(nt, file_id, dtype=np.int32),
            "byte_offset": byte_offset,
            "shot_id": shot_id,
            "trace_in_shot": trace_in_shot,
            "sx_m": sx_m, "sy_m": sy_m, "sz_m": sz_m,
            "rx_m": rx_m, "ry_m": ry_m, "rz_m": rz_m,
            "dt_s": reader.dt,
            "n_samples": reader.n_samples,
            "sample_format": reader.sample_format,
        }
    finally:
        reader.close()


def build_segy_index(
    paths: Sequence[str | Path],
    *,
    byte_map: dict | None = None,
    source_depth_m_override: float | None = None,
    receiver_depth_m_override: float | None = None,
    coord_scalar_override: float | None = None,
    num_workers: int = 4,
    show_progress: bool = False,
) -> SEGYIndex:
    """Scan headers of `paths` in parallel and build a unified :class:`SEGYIndex`.

    Parameters
    ----------
    paths
        SEG-Y file paths in the order you want them assigned ``file_id``.
    byte_map
        Trace-header byte-offset overrides. Defaults to SEG-Y rev1 standard
        (:data:`SEGY_REV1_BYTES`). Override per-key, e.g.
        ``{"source_depth": 44}`` for OBN-3D's airgun depth.
    source_depth_m_override, receiver_depth_m_override
        If the SEG-Y file's depth bytes are zero (or unreliable), use these
        instead. Common case for marine streamers.
    coord_scalar_override
        If set, ignore the per-trace ``coord_scalar`` field and apply this
        constant multiplier to (sx, sy, rx, ry). Use when you know the
        files have inconsistent or wrong scalar values.
    num_workers
        Parallel header-scan workers. ``find -name "*.sgy" | wc -l`` is a
        good upper bound; 4-8 is typical for SSD / Lustre.
    show_progress
        Print one line per finished file. Useful when scanning hundreds.

    Returns
    -------
    SEGYIndex
        With ``len(paths)`` distinct file_id values.
    """
    if not paths:
        raise ValueError("paths must be non-empty.")
    bm = dict(SEGY_REV1_BYTES)
    if byte_map:
        bm.update(byte_map)

    paths_list = [str(Path(p)) for p in paths]

    def _worker(file_id: int) -> dict:
        result = _scan_one_file(
            paths_list[file_id], file_id,
            byte_map=bm,
            source_depth_m_override=source_depth_m_override,
            receiver_depth_m_override=receiver_depth_m_override,
            coord_scalar_override=coord_scalar_override,
        )
        if show_progress:
            print(f"  scanned {Path(paths_list[file_id]).name}  "
                  f"(file_id={file_id}, traces={result['n_traces']})")
        return result

    results: list[dict] = []
    with ThreadPoolPrefetcher(_worker, range(len(paths_list)),
                              num_workers=num_workers) as pf:
        for r in pf:
            results.append(r)

    # All files must agree on dt / n_samples / format (typical for one survey)
    dt0 = results[0]["dt_s"]
    ns0 = results[0]["n_samples"]
    fmt0 = results[0]["sample_format"]
    for r in results[1:]:
        if (r["dt_s"], r["n_samples"], r["sample_format"]) != (dt0, ns0, fmt0):
            raise ValueError(
                "SEG-Y files disagree on dt / n_samples / sample_format. "
                "Cannot build a single index. Build per-format indexes "
                f"instead. File-0={ (dt0, ns0, fmt0) }  mismatch={ (r['dt_s'], r['n_samples'], r['sample_format']) }"
            )

    return SEGYIndex(
        file_paths=paths_list,
        file_id=np.concatenate([r["file_id"]      for r in results]),
        byte_offset=np.concatenate([r["byte_offset"] for r in results]),
        shot_id=np.concatenate([r["shot_id"]      for r in results]),
        receiver_id=np.concatenate([r["trace_in_shot"] for r in results]).astype(np.int32),
        sx_m=np.concatenate([r["sx_m"] for r in results]),
        sy_m=np.concatenate([r["sy_m"] for r in results]),
        sz_m=np.concatenate([r["sz_m"] for r in results]),
        rx_m=np.concatenate([r["rx_m"] for r in results]),
        ry_m=np.concatenate([r["ry_m"] for r in results]),
        rz_m=np.concatenate([r["rz_m"] for r in results]),
        dt_s=float(dt0), n_samples=int(ns0), sample_format=int(fmt0),
        meta={"byte_map": bm, "n_files": len(paths_list)},
    )


# ============================================================================
# Lazy shot-gather dataset
# ============================================================================
class IndexedShotGatherDataset:
    """Per-shot dataset backed by :class:`SEGYIndex` + :class:`MultiFileSEGYReader`.

    Each ``__getitem__`` issues **one coalesced byte-offset read** for the
    selected shot's traces (sorted ascending, runs merged). Output shape
    is ``(n_recv, n_samples)`` float32.

    Pair with :class:`sweep_io.prefetch.Prefetcher` or
    :class:`sweep_io.cuda_prefetch.CUDAPrefetcher` for I/O / compute overlap.

    Parameters
    ----------
    index
        Pre-built SEGYIndex.
    reader
        Optional :class:`MultiFileSEGYReader`. Built from ``index.file_paths``
        if omitted.
    shot_ids
        Subset (and order) of shot IDs to expose. Default: all shots, sorted.
    coalesce_gap
        Forwarded to ``reader.read_traces``; merges adjacent reads with at
        most this many bytes of gap.
    """

    def __init__(
        self,
        index: SEGYIndex,
        reader: MultiFileSEGYReader | None = None,
        *,
        shot_ids: Sequence[int] | None = None,
        coalesce_gap: int = 0,
    ) -> None:
        self.index = index
        self.reader = reader or MultiFileSEGYReader(index.file_paths)
        self._owned_reader = reader is None
        self.shot_ids = (
            np.asarray(shot_ids, dtype=np.int64)
            if shot_ids is not None
            else np.asarray(index.shot_ids, dtype=np.int64)
        )
        self.coalesce_gap = coalesce_gap

    def __len__(self) -> int:
        return int(self.shot_ids.size)

    def __getitem__(self, idx: int) -> dict:
        sid = int(self.shot_ids[idx])
        slc = self.index.lookup_shot(sid)
        obs = self.reader.read_traces(
            slc["file_id"], slc["byte_offset"], coalesce_gap=self.coalesce_gap,
        )
        return {
            "shot_id": sid,
            "obs": obs,                                # (nrec, nt) float32
            "source_xyz_m": np.array([slc["sx_m"], slc["sy_m"], slc["sz_m"]]),
            "receivers_xyz_m": np.stack(
                [slc["rx_m"], slc["ry_m"], slc["rz_m"]], axis=-1,
            ),
            "receiver_id": slc["receiver_id"],
        }

    def close(self) -> None:
        if self._owned_reader:
            self.reader.close()

    def __enter__(self) -> "IndexedShotGatherDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "SEGYIndex",
    "build_segy_index",
    "IndexedShotGatherDataset",
    "SEGY_REV1_BYTES",
]
