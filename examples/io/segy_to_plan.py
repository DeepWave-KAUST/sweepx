"""SEG-Y to shot gathers: index the headers once, build a plan, read gathers.

The path sweep-tasks takes with field data, on a small synthetic file so it runs
anywhere in a second: write a 3-shot SEG-Y with source and receiver coordinates in
the trace headers, scan it into a SEGYIndex, organise it as shot gathers (a
SeismicPlan), then read the gathers through a PlanReader with a prefetcher.

Run:
    python examples/segy_to_plan.py
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np

from sweep_io.prefetch import Prefetcher
from sweep_io.segy import (
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    write_segy_minimal,
)
from sweep_io.segy_index import SEGY_REV1_BYTES, build_segy_index
from sweep_io.seismic_plan import PlanReader, SeismicPlan


def write_demo_segy(path: Path, nshots: int = 3, nrec: int = 48, nt: int = 500,
                    dt: float = 0.004) -> np.ndarray:
    """A streamer-like line: shots every 50 m, receivers every 25 m behind them."""
    data = np.random.default_rng(0).standard_normal((nshots * nrec, nt)).astype("float32")
    write_segy_minimal(path, data, dt=dt)
    # write_segy_minimal leaves the trace headers blank. Fill in what the index
    # reads: shot number, trace number, and coordinates (scalar 1 = metres).
    b = SEGY_REV1_BYTES
    trace_bytes = SEGY_TRACE_HEADER_SIZE + nt * 4
    with open(path, "r+b") as f:
        for s in range(nshots):
            for r in range(nrec):
                pos = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE + (s * nrec + r) * trace_bytes
                f.seek(pos)
                head = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
                struct.pack_into(">i", head, b["shot"], s + 1)
                struct.pack_into(">i", head, b["trace_in_shot"], r + 1)
                struct.pack_into(">h", head, b["coord_scalar"], 1)
                struct.pack_into(">i", head, b["sx"], 1000 + 50 * s)
                struct.pack_into(">i", head, b["rx"], 1000 + 50 * s + 100 + 25 * r)
                f.seek(pos)
                f.write(head)
    return data


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = write_demo_segy(tmp / "line.sgy")

        # 1. Scan the headers once. The index is small and reusable.
        index = build_segy_index([tmp / "line.sgy"])
        print(f"index: {index.n_traces} traces in {index.n_shots} shots")

        # 2. Organise the traces as shot gathers, drop offsets under 200 m, and
        #    save the plan; later runs load it instead of rescanning.
        plan = index.to_seismic_plan(grouping="csg", offset_min_m=200.0)
        plan.save(tmp / "plan.npz")
        plan = SeismicPlan.load(tmp / "plan.npz")
        print(f"plan: {plan.n_groups} gathers, {plan.n_rows} traces")

        # 3. Read gathers by number. The prefetcher reads the next gather while
        #    the loop body (your forward and backward) runs.
        reader = PlanReader(plan)
        gathers = (reader.read_group(g) for g in range(plan.n_groups))
        with Prefetcher(gathers, queue_depth=2) as pf:
            for g, gather in enumerate(pf):
                print(f"gather {g}: {gather.shape}")
        reader.close()

        # Offsets are 100 m + 25 m per receiver, so each gather keeps receivers 4-47.
        with PlanReader(plan) as check:
            assert np.array_equal(check.read_group(0), data[4:48])
        print("gather 0 holds exactly the traces written")


if __name__ == "__main__":
    main()
