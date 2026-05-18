"""SEG-Y readers — two layers.

``read_segy`` / ``write_segy``
    High-level wrappers around the optional ``segyio`` dependency.
    Convenient for one-shot, small-to-medium reads; pulls the whole
    cube into memory.

``SEGYReader`` / ``MultiFileSEGYReader``
    Low-level, **no-segyio** byte-offset readers. Designed for hot-path
    FWI iteration where you have a pre-built ``(file_id, byte_offset)``
    catalog and want fast, coalesced trace reads without re-parsing
    headers each iteration. Pure stdlib + numpy — no extras required.

The codecs (:func:`ibm_to_ieee`, :func:`ieee_to_ibm`) are exposed
separately for users who do their own raw reads.
"""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ constants
SEGY_TEXT_HEADER_SIZE = 3200
SEGY_BIN_HEADER_SIZE = 400
SEGY_TRACE_HEADER_SIZE = 240

# Sample format codes from SEG-Y rev 1 binary header byte 25-26.
FORMAT_IBM_FLOAT32 = 1
FORMAT_INT32 = 2
FORMAT_INT16 = 3
FORMAT_IEEE_FLOAT32 = 5
FORMAT_IEEE_FLOAT64 = 6
FORMAT_INT8 = 8

_BYTES_PER_SAMPLE = {
    FORMAT_IBM_FLOAT32: 4,
    FORMAT_INT32: 4,
    FORMAT_INT16: 2,
    FORMAT_IEEE_FLOAT32: 4,
    FORMAT_IEEE_FLOAT64: 8,
    FORMAT_INT8: 1,
}


# =============================================================================
# Codecs
# =============================================================================
def ibm_to_ieee(arr_u32: np.ndarray) -> np.ndarray:
    """Decode IBM 32-bit floats (uint32 bit-pattern) to IEEE float32.

    Vectorized — handles tens of millions of samples per second per core.

    IBM float layout:
    - bit 31: sign (1 = negative)
    - bits 30-24: exponent, biased by 64, base 16
    - bits 23-0: mantissa (unsigned, with implicit radix-16 point at top)

    value = (-1)**sign × mantissa × 16**(exponent - 64) / 16**6
    """
    u32 = np.ascontiguousarray(arr_u32, dtype=np.uint32)
    sign = (u32 >> 31).astype(np.int8)        # 0 / 1
    expo = ((u32 >> 24) & 0x7F).astype(np.int16)
    mant = (u32 & 0x00FFFFFF).astype(np.float32)
    # mant × 16**(expo - 64) / 16**6  =  mant × 2**(4·(expo-64) - 24)
    out = np.ldexp(mant, (expo.astype(np.int32) - 64) * 4 - 24).astype(np.float32)
    return np.where(sign != 0, -out, out)


def ieee_to_ibm(arr_f32: np.ndarray) -> np.ndarray:
    """Encode IEEE float32 to IBM 32-bit floats (uint32 bit-pattern).

    Inverse of :func:`ibm_to_ieee`. Lossy by a couple of mantissa bits
    (IBM-FP32 stores 24 bits of mantissa under a radix-16 exponent;
    IEEE-FP32 has 23 explicit + 1 implicit). Acceptable for writing FWI
    synthetics back to SEG-Y; **not** bit-for-bit reversible.
    """
    f = np.ascontiguousarray(arr_f32, dtype=np.float64)
    out = np.zeros(f.shape, dtype=np.uint32)
    nz = f != 0
    if not np.any(nz):
        return out
    a = np.abs(f[nz])
    sign_nz = (f[nz] < 0).astype(np.uint32)

    # Pick expo16 so that mant_norm = a / 16**expo16 lies in [1/16, 1).
    # log_16(a) = log2(a) / 4.
    expo16 = np.floor(np.log2(a) / 4.0).astype(np.int32) + 1
    mant_norm = a / np.power(16.0, expo16.astype(np.float64))

    # Numerical safety net — float rounding can push mant_norm slightly
    # out of [1/16, 1) at radix-16 boundaries (e.g. a == 1.0). A pair
    # of one-shot corrections snaps it back in range.
    too_big = mant_norm >= 1.0
    expo16 = expo16 + too_big.astype(np.int32)
    mant_norm = np.where(too_big, mant_norm / 16.0, mant_norm)
    too_small = mant_norm < (1.0 / 16.0)
    expo16 = expo16 - too_small.astype(np.int32)
    mant_norm = np.where(too_small, mant_norm * 16.0, mant_norm)

    mant = (mant_norm * (1 << 24)).astype(np.uint32)
    mant = np.minimum(mant, np.uint32(0x00FFFFFF))
    biased = (expo16 + 64).astype(np.uint32) & 0x7F
    out[nz] = (sign_nz << 31) | (biased << 24) | mant
    return out


# =============================================================================
# Low-level reader
# =============================================================================
class SEGYReader:
    """Single-file SEG-Y reader keyed by absolute byte offsets.

    On construction, parses the 400-byte binary header to extract
    ``n_samples``, ``dt``, and ``sample_format``. Trace data is then
    read **on demand** by absolute byte offset — no header re-parsing
    per trace, no Python loop per sample.

    Designed to be the fast lane behind a pre-built shot index. Pair it
    with :class:`MultiFileSEGYReader` when shots live across many files.

    Parameters
    ----------
    path
        File path.
    mmap_mode
        If ``True`` (default), memory-map the file (zero-copy reads,
        kernel page cache reuse). If ``False``, use ``pread`` /
        ``seek+read``. mmap can misbehave on certain network filesystems
        — try ``False`` if you see ``SIGBUS`` or weird truncation.

    Concurrency
    -----------
    A single ``SEGYReader`` instance is **thread-safe for reads** when
    ``mmap_mode=True`` (each ``read_trace_data`` call takes a numpy view
    of the mmap; no shared file pointer). With ``mmap_mode=False`` reads
    use ``os.pread``, which is also thread-safe. So either way you can
    share one reader across all your prefetch workers.
    """

    def __init__(self, path: str | os.PathLike, *, mmap_mode: bool = True) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._fd = os.open(str(self.path), os.O_RDONLY)
        self._mmap = None
        if mmap_mode:
            try:
                size = os.fstat(self._fd).st_size
                self._mmap = mmap.mmap(self._fd, length=size, prot=mmap.PROT_READ)
            except (OSError, ValueError):
                # fall back to pread
                self._mmap = None

        bh = self._pread(SEGY_TEXT_HEADER_SIZE, SEGY_BIN_HEADER_SIZE)
        # bytes 17-18 (1-indexed) = sample interval in microseconds
        self.dt_us = struct.unpack(">H", bh[16:18])[0]
        self.dt = float(self.dt_us) * 1e-6
        # bytes 21-22 = number of samples per trace
        self.n_samples = struct.unpack(">H", bh[20:22])[0]
        # bytes 25-26 = sample format code
        self.sample_format = struct.unpack(">H", bh[24:26])[0]
        if self.sample_format not in _BYTES_PER_SAMPLE:
            raise NotImplementedError(
                f"{self.path}: SEG-Y format code {self.sample_format} not supported"
            )
        self._bytes_per_sample = _BYTES_PER_SAMPLE[self.sample_format]
        self._trace_data_bytes = self.n_samples * self._bytes_per_sample
        self._trace_total_bytes = SEGY_TRACE_HEADER_SIZE + self._trace_data_bytes

    # --------------------------------------------------------- low-level IO
    def _pread(self, offset: int, size: int) -> bytes:
        """Read ``size`` bytes starting at ``offset``. Thread-safe."""
        if self._mmap is not None:
            return bytes(self._mmap[offset : offset + size])
        return os.pread(self._fd, size, offset)

    def _pread_into(self, dst: bytearray, offset: int, size: int) -> None:
        """Read directly into a pre-allocated buffer."""
        if self._mmap is not None:
            dst[:] = self._mmap[offset : offset + size]
            return
        # os.pread doesn't take a buffer; read into bytes and copy.
        dst[:] = os.pread(self._fd, size, offset)

    # ------------------------------------------------------------ properties
    @property
    def n_traces(self) -> int:
        """Total number of traces in the file (computed from file size)."""
        size = os.fstat(self._fd).st_size
        body = size - SEGY_TEXT_HEADER_SIZE - SEGY_BIN_HEADER_SIZE
        if body % self._trace_total_bytes != 0:
            # variable-length traces not supported; report best estimate
            return body // self._trace_total_bytes
        return body // self._trace_total_bytes

    # ------------------------------------------------------------- decoding
    def _decode(self, raw: np.ndarray) -> np.ndarray:
        """Decode a uint8 view of trace payloads to float32."""
        nt = self.n_samples
        if self.sample_format == FORMAT_IEEE_FLOAT32:
            # SEG-Y stores big-endian; numpy default float32 is native.
            return raw.view(">f4").reshape(-1, nt).astype(np.float32, copy=False)
        if self.sample_format == FORMAT_IBM_FLOAT32:
            u32 = raw.view(">u4").reshape(-1, nt)
            return ibm_to_ieee(u32)
        if self.sample_format == FORMAT_INT32:
            return raw.view(">i4").reshape(-1, nt).astype(np.float32)
        if self.sample_format == FORMAT_INT16:
            return raw.view(">i2").reshape(-1, nt).astype(np.float32)
        if self.sample_format == FORMAT_IEEE_FLOAT64:
            return raw.view(">f8").reshape(-1, nt).astype(np.float32)
        if self.sample_format == FORMAT_INT8:
            return raw.view("i1").reshape(-1, nt).astype(np.float32)
        raise AssertionError(f"unhandled sample_format {self.sample_format}")

    # ------------------------------------------------------------- hot path
    def read_trace_data(
        self,
        byte_offsets: Sequence[int] | np.ndarray,
        *,
        coalesce_gap: int = 0,
    ) -> np.ndarray:
        """Read trace data at given absolute byte offsets.

        Each offset must point at a trace *header* (not at the data
        block — the 240-byte header is skipped internally).

        Parameters
        ----------
        byte_offsets
            1-D array of absolute byte offsets, one per trace to read.
            Order is **preserved** in the output, but the actual disk
            reads are issued in ascending order (and coalesced where
            possible) for sequential bandwidth.
        coalesce_gap
            Adjacent reads whose gap (between the end of one and the
            start of the next) is ≤ this many bytes are merged into a
            single ``pread``. ``0`` disables coalescing; sensible values
            on Lustre / NVMe are ``0`` (already sorted-sequential is
            fast enough) up through one filesystem block (~1 MB).

        Returns
        -------
        np.ndarray
            ``(n_traces, n_samples)`` float32 array, ordered the same as
            ``byte_offsets``.
        """
        offsets = np.asarray(byte_offsets, dtype=np.int64)
        if offsets.ndim != 1:
            raise ValueError(f"byte_offsets must be 1-D; got shape {offsets.shape}")
        n = offsets.size
        if n == 0:
            return np.empty((0, self.n_samples), dtype=np.float32)

        # Sort ascending; remember permutation to undo.
        order = np.argsort(offsets, kind="stable")
        sorted_offs = offsets[order]
        # data starts after the 240-byte trace header
        data_offs = sorted_offs + SEGY_TRACE_HEADER_SIZE
        size = self._trace_data_bytes

        # Coalesce runs of (near-)adjacent reads.
        if coalesce_gap < 0:
            raise ValueError(f"coalesce_gap must be >= 0; got {coalesce_gap}")
        starts: list[int] = [int(data_offs[0])]
        ends: list[int] = [int(data_offs[0] + size)]
        owners: list[list[int]] = [[0]]
        for i in range(1, n):
            s = int(data_offs[i])
            if s - ends[-1] <= coalesce_gap:
                ends[-1] = s + size
                owners[-1].append(i)
            else:
                starts.append(s)
                ends.append(s + size)
                owners.append([i])

        # Read each run, gathering bytes into a single stacked buffer so
        # we can decode once at the end. Per-trace _decode() in a Python
        # loop is dominated by numpy/Python dispatch overhead (~70 us per
        # trace × tens of thousands of traces = several seconds in the
        # shared-shot OBN supershot path). Batched decode amortises that.
        stacked = np.empty((n, size), dtype=np.uint8)
        _arange_size = np.arange(size, dtype=np.int64)
        for run_start, run_end, run_owners in zip(starts, ends, owners):
            buf = self._pread(run_start, run_end - run_start)
            raw = np.frombuffer(buf, dtype=np.uint8)
            owners_arr = np.asarray(run_owners, dtype=np.int64)
            rels = data_offs[owners_arr] - run_start
            K = int(owners_arr.size)
            if K == 1:
                # Single-trace run (common when sampler picks scattered traces).
                stacked[int(owners_arr[0])] = raw[int(rels[0]):int(rels[0]) + size]
            else:
                # Multi-trace run. Densely-packed (coalesce_gap=0) gives
                # rels = [0, size, 2*size, ...], so we can reshape without
                # building an index. Otherwise gather via add.outer; the
                # index buffer is K*size int64 (~K*16 KB for nt=4001) and
                # K is small per run for shared-shot sampling (≤ a few).
                if rels[0] == 0 and K * size == (run_end - run_start) and \
                        np.array_equal(rels, _arange_size[:K] * size):
                    stacked[owners_arr] = raw[:K * size].reshape(K, size)
                else:
                    col_idx = rels[:, None] + _arange_size[None, :]
                    stacked[owners_arr] = raw[col_idx]

        # ONE batched decode across all n traces — turns 24k × ~70 us
        # dispatch overhead into one big numpy view+astype.
        out_sorted = self._decode(stacked.reshape(-1)).reshape(n, self.n_samples)

        # Undo the sort.
        out = np.empty_like(out_sorted)
        out[order] = out_sorted
        return out

    def read_trace_header(self, byte_offset: int) -> bytes:
        """Return the 240-byte trace header at ``byte_offset``."""
        return self._pread(int(byte_offset), SEGY_TRACE_HEADER_SIZE)

    def read_text_header(self) -> bytes:
        """First 3200 bytes (the EBCDIC text header)."""
        return self._pread(0, SEGY_TEXT_HEADER_SIZE)

    def read_binary_header(self) -> bytes:
        """Bytes 3200–3600 (the binary header)."""
        return self._pread(SEGY_TEXT_HEADER_SIZE, SEGY_BIN_HEADER_SIZE)

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1

    def __enter__(self) -> "SEGYReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# Multi-file fan-out
# =============================================================================
class MultiFileSEGYReader:
    """Dispatch byte-offset reads across many SEG-Y files by ``file_id``.

    Holds one open :class:`SEGYReader` per path. All readers must agree
    on ``n_samples``, ``dt``, and ``sample_format`` (the usual case for a
    single survey split across many files). A mismatch raises at construction.

    Parameters
    ----------
    paths
        List of file paths. ``file_id = 0`` corresponds to ``paths[0]``,
        etc.
    mmap_mode
        Passed through to each underlying :class:`SEGYReader`.
    """

    def __init__(self, paths: Sequence[str | os.PathLike], *, mmap_mode: bool = True) -> None:
        if len(paths) == 0:
            raise ValueError("MultiFileSEGYReader needs at least one path.")
        self.readers: list[SEGYReader] = [
            SEGYReader(p, mmap_mode=mmap_mode) for p in paths
        ]
        ref = self.readers[0]
        for i, r in enumerate(self.readers[1:], start=1):
            if (r.n_samples, r.dt_us, r.sample_format) != (
                ref.n_samples,
                ref.dt_us,
                ref.sample_format,
            ):
                raise ValueError(
                    f"file {i} ({r.path}) disagrees with file 0 ({ref.path}) on "
                    f"(n_samples={r.n_samples} vs {ref.n_samples}, "
                    f"dt_us={r.dt_us} vs {ref.dt_us}, "
                    f"format={r.sample_format} vs {ref.sample_format})"
                )
        self.n_samples = ref.n_samples
        self.dt = ref.dt
        self.sample_format = ref.sample_format

    def read_traces(
        self,
        file_ids: Sequence[int] | np.ndarray,
        byte_offsets: Sequence[int] | np.ndarray,
        *,
        coalesce_gap: int = 0,
    ) -> np.ndarray:
        """Read traces from heterogeneous files, preserving caller order.

        Parameters
        ----------
        file_ids
            1-D array, same length as ``byte_offsets``. Index into
            ``self.readers``.
        byte_offsets
            1-D array of trace-header offsets within each file.
        coalesce_gap
            Forwarded to each per-file read.

        Returns
        -------
        np.ndarray
            ``(n, n_samples)`` float32, ordered by the caller.
        """
        fids = np.asarray(file_ids, dtype=np.int64)
        offs = np.asarray(byte_offsets, dtype=np.int64)
        if fids.shape != offs.shape:
            raise ValueError(
                f"file_ids {fids.shape} and byte_offsets {offs.shape} must match"
            )
        n = fids.size
        out = np.empty((n, self.n_samples), dtype=np.float32)
        # group by file_id
        for fid in np.unique(fids):
            mask = fids == fid
            sub = self.readers[int(fid)].read_trace_data(
                offs[mask], coalesce_gap=coalesce_gap
            )
            out[mask] = sub
        return out

    def close(self) -> None:
        for r in self.readers:
            r.close()
        self.readers = []

    def __enter__(self) -> "MultiFileSEGYReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# =============================================================================
# Optional convenience: full-file read/write via segyio
# =============================================================================
def read_segy(
    path: str | os.PathLike,
    *,
    iline: int = 189,
    xline: int = 193,
    strict: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Read an entire SEG-Y volume into memory (via the optional ``segyio`` dep)."""
    try:
        import segyio  # type: ignore
    except ImportError as e:
        raise ImportError(
            "read_segy requires `segyio`. Install with `pip install sweep-io[segy]`."
        ) from e
    with segyio.open(str(path), "r", iline=iline, xline=xline, strict=strict) as f:
        dt = segyio.dt(f) / 1e6
        n_samples = f.samples.size
        try:
            data = segyio.tools.cube(f)
            ilines = np.asarray(f.ilines)
            xlines = np.asarray(f.xlines)
        except Exception:
            data = np.stack([np.asarray(tr) for tr in f.trace[:]], axis=0)
            ilines = np.array([])
            xlines = np.array([])
    return data, {
        "dt": dt,
        "n_samples": int(n_samples),
        "ilines": ilines,
        "xlines": xlines,
    }


def write_segy(
    path: str | os.PathLike,
    data: np.ndarray,
    *,
    dt: float,
    delay_recording_time: float = 0.0,
) -> None:
    """Write a 2-D ``(n_traces, n_samples)`` SEG-Y via ``segyio``."""
    try:
        import segyio  # type: ignore
    except ImportError as e:
        raise ImportError(
            "write_segy requires `segyio`. Install with `pip install sweep-io[segy]`."
        ) from e
    if data.ndim != 2:
        raise ValueError(f"write_segy expects (n_traces, n_samples); got {data.shape}")
    n_traces, n_samples = data.shape
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(n_samples)
    spec.tracecount = n_traces
    spec.sorting = segyio.TraceSortingFormat.UNKNOWN_SORTING
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = int(dt * 1e6)
        for i in range(n_traces):
            f.trace[i] = np.asarray(data[i], dtype=np.float32)
            f.header[i] = {
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(dt * 1e6),
                segyio.TraceField.DelayRecordingTime: int(delay_recording_time * 1000),
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
            }


# =============================================================================
# Writing a minimal SEG-Y header (for tests / synthetics; no segyio needed)
# =============================================================================
def write_segy_minimal(
    path: str | os.PathLike,
    data: np.ndarray,
    *,
    dt: float,
    sample_format: int = FORMAT_IEEE_FLOAT32,
) -> None:
    """Write a SEG-Y file with empty text/trace headers — useful for fixtures.

    No segyio required. Produces a valid layout that :class:`SEGYReader`
    can parse:
    - 3200-byte zeroed text header
    - 400-byte binary header with dt / n_samples / sample_format set
    - one (zeroed trace header + sample data) per row

    Only formats 1 (IBM) and 5 (IEEE32) are supported here; we don't
    write integers from this helper.
    """
    if data.ndim != 2:
        raise ValueError(f"write_segy_minimal expects (n_traces, n_samples); got {data.shape}")
    if sample_format not in (FORMAT_IBM_FLOAT32, FORMAT_IEEE_FLOAT32):
        raise ValueError(f"write_segy_minimal supports formats 1 and 5; got {sample_format}")
    n_traces, n_samples = data.shape
    path = Path(path)
    with open(path, "wb") as f:
        f.write(b"\x00" * SEGY_TEXT_HEADER_SIZE)
        bh = bytearray(SEGY_BIN_HEADER_SIZE)
        struct.pack_into(">H", bh, 16, int(round(dt * 1e6)))  # dt in microseconds
        struct.pack_into(">H", bh, 20, int(n_samples))
        struct.pack_into(">H", bh, 24, int(sample_format))
        f.write(bh)
        if sample_format == FORMAT_IEEE_FLOAT32:
            encoded = np.ascontiguousarray(data, dtype=">f4")
        else:
            encoded = ieee_to_ibm(np.ascontiguousarray(data, dtype=np.float32))
            encoded = encoded.astype(">u4")
        trace_header = b"\x00" * SEGY_TRACE_HEADER_SIZE
        for i in range(n_traces):
            f.write(trace_header)
            f.write(encoded[i].tobytes())


__all__ = [
    "SEGY_TEXT_HEADER_SIZE",
    "SEGY_BIN_HEADER_SIZE",
    "SEGY_TRACE_HEADER_SIZE",
    "FORMAT_IBM_FLOAT32",
    "FORMAT_INT32",
    "FORMAT_INT16",
    "FORMAT_IEEE_FLOAT32",
    "FORMAT_IEEE_FLOAT64",
    "FORMAT_INT8",
    "ibm_to_ieee",
    "ieee_to_ibm",
    "SEGYReader",
    "MultiFileSEGYReader",
    "read_segy",
    "write_segy",
    "write_segy_minimal",
]
