"""End-to-end demo: prefetched SEG-Y shot reads vs. blocking reads.

Creates a few synthetic SEG-Y files in a temp dir, builds a trivial
shot index, then walks the index twice:
  1. blocking — read each shot, then do fake compute
  2. prefetched — same, but with a background thread reading ahead
Reports wall-clock and the time the consumer spent waiting on I/O.

Run:
    python examples/prefetch_demo.py
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from sweep_io.prefetch import Prefetcher, TimingPrefetcher
from sweep_io.segy import MultiFileSEGYReader, write_segy_minimal


def main() -> None:
    n_files = 4
    shots_per_file = 8
    n_samples, n_receivers = 1024, 240    # ~1 MB per shot (float32)
    compute_seconds = 0.02                 # fake forward-and-backward time

    tmp = Path(tempfile.mkdtemp(prefix="sweep_io_prefetch_demo_"))
    try:
        print(f"writing {n_files} synthetic SEG-Y files into {tmp} ...")
        paths: list[Path] = []
        for k in range(n_files):
            rng = np.random.default_rng(seed=k)
            d = rng.standard_normal((shots_per_file * n_receivers, n_samples)).astype(
                "float32"
            )
            p = tmp / f"shot_line_{k:03d}.segy"
            write_segy_minimal(p, d, dt=0.001)
            paths.append(p)

        # Build a trivial (file_id, byte_offset) index.
        # Each "shot" = a contiguous run of n_receivers traces in one file.
        # Real workflows store this as a numpy structured array on disk.
        from sweep_io.segy import (
            SEGY_BIN_HEADER_SIZE,
            SEGY_TEXT_HEADER_SIZE,
            SEGY_TRACE_HEADER_SIZE,
        )

        trace_total = SEGY_TRACE_HEADER_SIZE + n_samples * 4
        base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
        records: list[tuple[int, np.ndarray]] = []
        for k in range(n_files):
            for s in range(shots_per_file):
                offs = base + np.arange(
                    s * n_receivers, (s + 1) * n_receivers, dtype="int64"
                ) * trace_total
                records.append((k, offs))

        reader = MultiFileSEGYReader(paths)

        def load_shot(idx: int) -> np.ndarray:
            file_id, offs = records[idx]
            return reader.readers[file_id].read_trace_data(offs)

        n_shots = len(records)
        print(
            f"\n{n_shots} shots, ~{n_samples * n_receivers * 4 / 1e6:.1f} MB each, "
            f"fake compute = {compute_seconds * 1e3:.0f} ms/shot\n"
        )

        # --- 1: blocking reads ---------------------------------------------------
        t0 = time.perf_counter()
        io_wait = 0.0
        for i in range(n_shots):
            t_io = time.perf_counter()
            _ = load_shot(i)
            io_wait += time.perf_counter() - t_io
            time.sleep(compute_seconds)
        dt_blocking = time.perf_counter() - t0
        print(
            f"[blocking ] total {dt_blocking * 1e3:7.1f} ms   "
            f"io_wait {io_wait * 1e3:7.1f} ms  "
            f"compute {n_shots * compute_seconds * 1e3:.1f} ms"
        )

        # --- 2: prefetched reads -------------------------------------------------
        t0 = time.perf_counter()
        with TimingPrefetcher(
            (load_shot(i) for i in range(n_shots)), queue_depth=2
        ) as pf:
            for _ in pf:
                time.sleep(compute_seconds)
        dt_pf = time.perf_counter() - t0
        io_wait_pf = sum(pf.wait_times)
        print(
            f"[prefetch ] total {dt_pf * 1e3:7.1f} ms   "
            f"io_wait {io_wait_pf * 1e3:7.1f} ms  "
            f"compute {n_shots * compute_seconds * 1e3:.1f} ms"
        )

        speedup = dt_blocking / dt_pf
        print(
            f"\nspeedup ≈ {speedup:.2f}x; "
            f"io_wait dropped from {io_wait*1e3:.1f} ms to {io_wait_pf*1e3:.1f} ms"
        )

        reader.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
