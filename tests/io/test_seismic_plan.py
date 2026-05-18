"""Tests for the unified ``seismic_plan_v1`` schema + PlanReader.

Builds tiny synthetic SEG-Y fixtures (shared helper with test_segy_index)
so the suite runs without external data.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from sweep_io.segy import (
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    FORMAT_IEEE_FLOAT32,
    write_segy_minimal,
)
from sweep_io.segy_index import SEGY_REV1_BYTES, build_segy_index
from sweep_io.seismic_plan import (
    SCHEMA_FORMAT,
    PlanReader,
    SeismicPlan,
    build_seismic_plan,
)


# ============================================================================
# Helpers (mirror test_segy_index.py — kept local so the two test files
# stay independent).
# ============================================================================
def _patch_trace_header(
    raw: bytearray, *, shot_id, trace_in_shot, sx, sy, rx, ry,
    coord_scalar=1, source_depth=0, receiver_depth=0,
):
    bm = SEGY_REV1_BYTES
    struct.pack_into(">i", raw, bm["shot"], shot_id)
    struct.pack_into(">i", raw, bm["trace_in_shot"], trace_in_shot)
    struct.pack_into(">h", raw, bm["coord_scalar"], coord_scalar)
    struct.pack_into(">i", raw, bm["sx"], sx)
    struct.pack_into(">i", raw, bm["sy"], sy)
    struct.pack_into(">i", raw, bm["rx"], rx)
    struct.pack_into(">i", raw, bm["ry"], ry)
    struct.pack_into(">i", raw, bm["source_depth"], source_depth)
    struct.pack_into(">i", raw, bm["receiver_depth"], receiver_depth)


def _write_synthetic_segy(
    tmp_path: Path, *,
    nshots: int = 4, nrec: int = 6, nt: int = 32,
    sx_step: int = 50, rx_step: int = 25,
    fname: str = "syn.segy", shot_id_start: int = 1,
    rec_y_per_shot: list[int] | None = None,
) -> Path:
    """Synthetic SEG-Y with deterministic per-trace sample values.

    Each trace stores ``data[i, t] = shot_id * 1000 + r * 10 + t`` so we can
    bit-for-bit verify what PlanReader actually reads.
    """
    n_total = nshots * nrec
    data = np.zeros((n_total, nt), dtype="float32")
    for s in range(nshots):
        sid = shot_id_start + s
        for r in range(nrec):
            i = s * nrec + r
            for t in range(nt):
                data[i, t] = float(sid) * 1000 + r * 10 + t
    path = tmp_path / fname
    write_segy_minimal(path, data, dt=0.002, sample_format=FORMAT_IEEE_FLOAT32)

    trace_total = SEGY_TRACE_HEADER_SIZE + nt * 4
    base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
    with open(path, "r+b") as f:
        for s in range(nshots):
            sx = 1000 + s * sx_step
            sy = 0
            for r in range(nrec):
                rx = r * rx_step
                ry = (rec_y_per_shot[s] if rec_y_per_shot is not None else 0)
                i = s * nrec + r
                hdr_off = base + i * trace_total
                f.seek(hdr_off)
                buf = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
                _patch_trace_header(
                    buf,
                    shot_id=shot_id_start + s,
                    trace_in_shot=r,
                    sx=sx, sy=sy, rx=rx, ry=ry, coord_scalar=1,
                )
                f.seek(hdr_off)
                f.write(bytes(buf))
    return path


# ============================================================================
# build_seismic_plan — CSG grouping (default, 2-D streamer style)
# ============================================================================
def test_csg_plan_no_filter_keeps_everything(tmp_path):
    """CSG with no filter: n_groups = nshots, all rows kept."""
    p = _write_synthetic_segy(tmp_path, nshots=4, nrec=6, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")

    assert plan.grouping == "csg"
    assert plan.n_groups == 4
    assert plan.n_rows == 4 * 6
    assert plan.per_group_row_counts().tolist() == [6, 6, 6, 6]
    # Sources are at sx = 1000, 1050, 1100, 1150 (sx_step=50).
    assert plan.group_xyz[:, 0].tolist() == [1000.0, 1050.0, 1100.0, 1150.0]


def test_csg_plan_shot_whitelist(tmp_path):
    """shot_ids filter narrows the plan to the picked shots only."""
    p = _write_synthetic_segy(tmp_path, nshots=4, nrec=6, nt=16,
                              shot_id_start=10)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg", shot_ids=[11, 13])

    assert plan.n_groups == 2
    assert plan.group_id.tolist() == [11, 13]
    assert plan.n_rows == 2 * 6


def test_csg_plan_offset_filter(tmp_path):
    """Offset filter drops near-offset traces."""
    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=6, nt=16,
                              rx_step=100, sx_step=0)
    idx = build_segy_index([p])
    # rx = 0, 100, 200, 300, 400, 500; sx = 1000 → offsets 1000, 900, 800, 700, 600, 500.
    plan = build_seismic_plan(idx, grouping="csg", offset_min_m=700.0)

    # Per shot, only rows with offset >= 700 survive → rx in {0, 100, 200, 300}
    # for each shot, i.e. 4 rows per shot.
    assert plan.n_groups == 2
    assert plan.per_group_row_counts().tolist() == [4, 4]


# ============================================================================
# build_seismic_plan — CRG grouping (3-D OBN style)
# ============================================================================
def test_crg_plan_quantize_collapses_receivers(tmp_path):
    """CRG quantization groups nearby physical receivers into one cell."""
    # Shot 1's receivers at (rx=0..5*25=125, ry=0); shot 2 at (rx=0..125, ry=10).
    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=6, nt=16,
                              rx_step=25,
                              rec_y_per_shot=[0, 10])
    idx = build_segy_index([p])
    # With quantize=30 m, rx={0,25,50,75,100,125} bin to {0,1,2,2,3,4} ish.
    # ry differences of 10 m → with quantize=30, they collapse to the same y-cell.
    plan = build_seismic_plan(idx, grouping="crg", receiver_quantize_m=30.0)

    # Expect 5 unique receiver cells (rx bins of width 30 from 0..125 →
    # bins 0,1,2,3,4 = 5 cells; ry both round to 0).
    assert plan.grouping == "crg"
    assert plan.n_groups == 5
    # Each cell sees 2 shots (one row per shot per receiver-cell).
    # Note: rx_step=25 with quantize=30: cells are
    #   rx=0  → 0/30=0
    #   rx=25 → 25/30=1
    #   rx=50 → 50/30=2
    #   rx=75 → 75/30=2 or 3 (round-half-to-even: 75/30=2.5 → 2)
    #   rx=100→ 100/30=3
    #   rx=125→ 125/30=4
    # → bin sizes [1, 1, 2, 1, 1] per shot = 2 shots → [2,2,4,2,2] total
    counts = plan.per_group_row_counts()
    assert counts.sum() == 2 * 6
    # All groups have a multiple-of-2 count (each cell sees BOTH shots).
    assert (counts % 2 == 0).all()


def test_crg_plan_group_xyz_round_trip_at_utm_scale(tmp_path):
    """Regression: CRG group_xyz must equal the quantized receiver UTM
    coords, even when the coords are at UTM scale (millions of metres of
    northing). The original packed-key code
    used 21 bits per axis but failed to center by min before packing —
    UTM northings (~9e6 cells at q=0.5) overflowed the 21-bit
    field and silently lost the upper bits, yielding group_xyz off by a
    multiple of 2^21 * q ≈ 1,048,576 m."""
    import struct

    from sweep_io.segy import (
        SEGY_BIN_HEADER_SIZE,
        SEGY_TEXT_HEADER_SIZE,
        SEGY_TRACE_HEADER_SIZE,
        FORMAT_IEEE_FLOAT32,
        write_segy_minimal,
    )
    nshots, nrec, nt = 2, 4, 16
    data = np.zeros((nshots * nrec, nt), dtype="float32")
    path = tmp_path / "utm_scale.sgy"
    write_segy_minimal(path, data, dt=0.002, sample_format=FORMAT_IEEE_FLOAT32)
    # Patch UTM-scale headers. coord_scalar=1, sx ≈ 500000, sy ≈ 4500000
    # (a synthetic UTM-scale northing, 4.5e6 m).
    bm = SEGY_REV1_BYTES
    base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
    trace_total = SEGY_TRACE_HEADER_SIZE + nt * 4
    with open(path, "r+b") as f:
        for s in range(nshots):
            for r in range(nrec):
                i = s * nrec + r
                hdr_off = base + i * trace_total
                f.seek(hdr_off)
                buf = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
                struct.pack_into(">i", buf, bm["shot"], s + 1)
                struct.pack_into(">i", buf, bm["trace_in_shot"], r)
                struct.pack_into(">h", buf, bm["coord_scalar"], 1)
                struct.pack_into(">i", buf, bm["sx"], 500_000 + s * 25)
                struct.pack_into(">i", buf, bm["sy"], 4_500_000 + s * 25)
                # Two distinct receiver cells:
                #   r in {0,1} -> cell A (rx=499000, ry=4498000)
                #   r in {2,3} -> cell B (rx=511500, ry=4509500)
                rx = 499_000 if r < 2 else 511_500
                ry = 4_498_000 if r < 2 else 4_509_500
                struct.pack_into(">i", buf, bm["rx"], rx)
                struct.pack_into(">i", buf, bm["ry"], ry)
                f.seek(hdr_off)
                f.write(bytes(buf))
    idx = build_segy_index([path])
    plan = build_seismic_plan(idx, grouping="crg", receiver_quantize_m=0.5)
    assert plan.grouping == "crg"
    assert plan.n_groups == 2, f"expected 2 unique receiver cells, got {plan.n_groups}"
    # group_xyz must be at UTM scale (within ±1 m of the cell centers),
    # NOT modulo 2^21 * 0.5 ≈ 1,048,576 m as the original bug produced.
    xy_sorted = plan.group_xyz[np.argsort(plan.group_xyz[:, 0])]
    assert abs(xy_sorted[0, 0] - 499_000) < 1.0, (
        f"group_xyz[0, x] = {xy_sorted[0, 0]} (expected ~499000); "
        "likely the 21-bit packed-key overflow bug regressed."
    )
    assert abs(xy_sorted[0, 1] - 4_498_000) < 1.0, (
        f"group_xyz[0, y] = {xy_sorted[0, 1]} (expected ~4498000); "
        "likely the 21-bit packed-key overflow bug regressed."
    )
    assert abs(xy_sorted[1, 0] - 511_500) < 1.0
    assert abs(xy_sorted[1, 1] - 4_509_500) < 1.0


def test_crg_plan_raises_on_pack_overflow(tmp_path):
    """When the survey range × 1/q exceeds the 21-bit packing budget,
    raise a clear error instead of silently corrupting group_xyz."""
    import struct

    from sweep_io.segy import (
        SEGY_BIN_HEADER_SIZE,
        SEGY_TEXT_HEADER_SIZE,
        SEGY_TRACE_HEADER_SIZE,
        FORMAT_IEEE_FLOAT32,
        write_segy_minimal,
    )
    nshots, nrec, nt = 1, 2, 8
    data = np.zeros((nshots * nrec, nt), dtype="float32")
    path = tmp_path / "huge_extent.sgy"
    write_segy_minimal(path, data, dt=0.002, sample_format=FORMAT_IEEE_FLOAT32)
    bm = SEGY_REV1_BYTES
    base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
    trace_total = SEGY_TRACE_HEADER_SIZE + nt * 4
    # Force receiver-x range > 2^21 * 0.001 m at q=0.001 → overflow.
    rx_vals = [0, 2_500_000]  # 2.5M m range, /q=0.001 → 2.5e9 cells, > 2^21
    with open(path, "r+b") as f:
        for r, rx in enumerate(rx_vals):
            hdr_off = base + r * trace_total
            f.seek(hdr_off)
            buf = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
            struct.pack_into(">i", buf, bm["shot"], 1)
            struct.pack_into(">i", buf, bm["trace_in_shot"], r)
            struct.pack_into(">h", buf, bm["coord_scalar"], 1)
            struct.pack_into(">i", buf, bm["sx"], 100)
            struct.pack_into(">i", buf, bm["sy"], 100)
            struct.pack_into(">i", buf, bm["rx"], rx)
            struct.pack_into(">i", buf, bm["ry"], 0)
            f.seek(hdr_off)
            f.write(bytes(buf))
    idx = build_segy_index([path])
    with pytest.raises(ValueError, match="21-bit packed field"):
        build_seismic_plan(idx, grouping="crg", receiver_quantize_m=0.001)


def test_crg_plan_requires_quantize(tmp_path):
    """CRG without ``receiver_quantize_m`` is an error."""
    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=4, nt=16)
    idx = build_segy_index([p])
    with pytest.raises(ValueError, match="receiver_quantize_m"):
        build_seismic_plan(idx, grouping="crg")


def test_supershot_grouping_not_implemented(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=4, nt=16)
    idx = build_segy_index([p])
    with pytest.raises(NotImplementedError, match="supershot"):
        build_seismic_plan(idx, grouping="supershot")


def test_invalid_grouping(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=4, nt=16)
    idx = build_segy_index([p])
    with pytest.raises(ValueError, match="grouping"):
        build_seismic_plan(idx, grouping="weirdmode")


# ============================================================================
# max_traces_per_group cap
# ============================================================================
def test_max_traces_per_group_caps(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=10, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg", max_traces_per_group=4,
                              seed=42)

    assert plan.n_groups == 3
    assert plan.per_group_row_counts().tolist() == [4, 4, 4]


# ============================================================================
# Save / load round-trip
# ============================================================================
def test_save_load_roundtrip(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=5, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg", build_label="rt_test")

    out = tmp_path / "plan.npz"
    plan.save(out)
    plan2 = SeismicPlan.load(out)

    assert plan2.grouping == plan.grouping
    assert plan2.n_groups == plan.n_groups
    assert plan2.n_rows == plan.n_rows
    np.testing.assert_array_equal(plan2.row_file_id, plan.row_file_id)
    np.testing.assert_array_equal(plan2.row_trace_offset, plan.row_trace_offset)
    np.testing.assert_array_equal(plan2.group_offsets, plan.group_offsets)
    np.testing.assert_array_almost_equal(plan2.row_source_xyz, plan.row_source_xyz)
    np.testing.assert_array_almost_equal(plan2.row_receiver_xyz, plan.row_receiver_xyz)
    assert plan2.dt_s == plan.dt_s
    assert plan2.samples_per_trace == plan.samples_per_trace
    assert plan2.build_meta.get("label") == "rt_test"


def test_load_rejects_wrong_format(tmp_path):
    """Loading a file whose ``format`` field is not seismic_plan_v1 errors."""
    np.savez(tmp_path / "fake.npz",
             format=np.asarray("crg_fwi_plan_v1", dtype="U32"))
    with pytest.raises(ValueError, match=SCHEMA_FORMAT):
        SeismicPlan.load(tmp_path / "fake.npz")


# ============================================================================
# PlanReader: bit-for-bit SEG-Y read
# ============================================================================
def test_plan_reader_matches_raw_decode(tmp_path):
    """PlanReader.read_group(g) returns the same bytes as direct SEG-Y read."""
    p = _write_synthetic_segy(tmp_path, nshots=4, nrec=5, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")

    reader = PlanReader(plan)
    try:
        # For each shot (CSG group), the per-trace values should be
        # data[i, t] = shot_id * 1000 + r * 10 + t.
        for g in range(plan.n_groups):
            arr = reader.read_group(g)
            assert arr.shape == (5, 16)
            sid = int(plan.group_id[g])
            # Trace r should be exactly [sid*1000 + r*10 + 0, +1, +2, ..., +15]
            for r in range(5):
                expected = sid * 1000 + r * 10 + np.arange(16, dtype=np.float32)
                np.testing.assert_array_equal(arr[r], expected)
    finally:
        reader.close()


def test_plan_reader_cache_all_parity(tmp_path):
    """cache_all=True returns identical arrays to lazy reads."""
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=4, nt=8)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")

    lazy = PlanReader(plan, cache_all=False)
    eager = PlanReader(plan, cache_all=True)
    try:
        for g in range(plan.n_groups):
            np.testing.assert_array_equal(lazy.read_group(g), eager.read_group(g))
        np.testing.assert_array_equal(lazy.read_all(), eager.read_all())
    finally:
        lazy.close()
        eager.close()


def test_plan_reader_context_manager_closes(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=3, nt=8)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")
    with PlanReader(plan) as reader:
        arr = reader.read_group(0)
        assert arr.shape == (3, 8)


# ============================================================================
# Filters (compose-able, no SEG-Y re-read)
# ============================================================================
def test_filter_rows_recomputes_offsets(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=5, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")

    # Drop every other row globally; each group should lose half.
    mask = np.zeros(plan.n_rows, dtype=bool)
    mask[::2] = True
    plan2 = plan.filter_rows(mask)
    assert plan2.n_groups == plan.n_groups
    # 15 rows total, every other = 8 (rows 0,2,4,6,8,10,12,14).
    assert plan2.n_rows == 8


def test_filter_groups_drops_groups(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=4, nrec=3, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")

    mask = np.array([True, False, True, False])
    plan2 = plan.filter_groups(mask)
    assert plan2.n_groups == 2
    assert plan2.n_rows == 2 * 3
    assert plan2.group_id.tolist() == [plan.group_id[0], plan.group_id[2]]


def test_drop_empty_groups(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=4, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")

    # Drop all rows belonging to group 1 → that group becomes empty.
    mask = np.ones(plan.n_rows, dtype=bool)
    sl = plan.group_slice(1)
    mask[sl] = False
    plan2 = plan.filter_rows(mask)
    assert plan2.n_groups == 3  # still there
    assert plan2.per_group_row_counts().tolist() == [4, 0, 4]

    plan3 = plan2.drop_empty_groups()
    assert plan3.n_groups == 2
    assert plan3.per_group_row_counts().tolist() == [4, 4]


# ============================================================================
# SEGYIndex.to_seismic_plan() convenience hop
# ============================================================================
def test_segy_index_to_seismic_plan_hop(tmp_path):
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=4, nt=16)
    idx = build_segy_index([p])

    plan_via_method = idx.to_seismic_plan(grouping="csg")
    plan_direct = build_seismic_plan(idx, grouping="csg")

    np.testing.assert_array_equal(plan_via_method.row_file_id, plan_direct.row_file_id)
    np.testing.assert_array_equal(plan_via_method.group_offsets, plan_direct.group_offsets)
    assert plan_via_method.grouping == "csg"


# ============================================================================
# PlanReader.read_rows — sampler-driven row reads
# ============================================================================
def test_plan_reader_read_rows_lazy(tmp_path):
    """read_rows on the lazy path returns rows in the requested order."""
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=4, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")
    full = plan.row_source_xyz.shape[0]

    with PlanReader(plan, cache_all=False) as reader:
        # Pick a non-contiguous subset across two groups; deliberately mix
        # ascending/descending order so we can confirm the output preserves
        # the index order.
        picked = np.array([5, 0, 7, 3, 10], dtype=np.int64)
        rows = reader.read_rows(picked)
        assert rows.shape == (5, 16)
        assert rows.dtype == np.float32
        # Cross-check against read_all using the same indices.
        all_rows = reader.read_all()
        np.testing.assert_array_equal(rows, all_rows[picked])
        # Empty input must produce a shape-(0, nt) result, not raise.
        empty = reader.read_rows(np.empty(0, dtype=np.int64))
        assert empty.shape == (0, 16)


def test_plan_reader_read_rows_trace_cache_hits_on_repeat(tmp_path):
    """trace_cache_bytes>0 hands repeated row reads from in-RAM LRU.

    Reads the same row indices twice; second pass should be all hits.
    """
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=4, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")
    picked = np.array([1, 4, 9, 2, 7, 0, 5], dtype=np.int64)
    with PlanReader(plan, cache_all=False,
                    trace_cache_bytes=-1, coalesce_gap=0) as reader:
        first = reader.read_rows(picked)
        stats_after_first = reader.trace_cache_stats
        assert stats_after_first["misses"] == picked.size
        assert stats_after_first["hits"] == 0
        second = reader.read_rows(picked)
        stats_after_second = reader.trace_cache_stats
        # Second pass: 7 fresh hits, 0 new misses.
        assert stats_after_second["hits"] == picked.size
        assert stats_after_second["misses"] == picked.size  # unchanged
        # Bit-for-bit equal.
        np.testing.assert_array_equal(first, second)
        # Cross-check raw (cache disabled) gives the same bytes.
    with PlanReader(plan, cache_all=False, trace_cache_bytes=0) as raw:
        assert raw.trace_cache_stats is None
        baseline = raw.read_rows(picked)
    np.testing.assert_array_equal(first, baseline)


def test_plan_reader_read_rows_coalesce_gap_yields_same_bytes(tmp_path):
    """coalesce_gap is an IO optimisation — must not change byte output."""
    p = _write_synthetic_segy(tmp_path, nshots=4, nrec=6, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")
    picked = np.array([0, 1, 2, 5, 6, 18, 19, 20], dtype=np.int64)
    with PlanReader(plan, cache_all=False, trace_cache_bytes=0,
                    coalesce_gap=0) as a:
        no_coalesce = a.read_rows(picked)
    with PlanReader(plan, cache_all=False, trace_cache_bytes=-1,
                    coalesce_gap=4096) as b:
        coalesced = b.read_rows(picked)
    np.testing.assert_array_equal(no_coalesce, coalesced)


def test_plan_reader_trace_cache_disabled_by_cache_all(tmp_path):
    """cache_all=True suppresses trace_cache entirely (full plan already in RAM)."""
    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=4, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")
    with PlanReader(plan, cache_all=True, trace_cache_bytes=-1) as reader:
        assert reader.trace_cache_stats is None  # no separate cache
        reader.read_rows(np.array([0, 1, 2], dtype=np.int64))


def test_plan_reader_read_rows_cache_all_parity(tmp_path):
    """read_rows must yield identical bytes whether or not cache_all is on."""
    p = _write_synthetic_segy(tmp_path, nshots=3, nrec=4, nt=16)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="csg")
    picked = np.array([1, 4, 9, 2], dtype=np.int64)

    with PlanReader(plan, cache_all=False) as lazy:
        lazy_rows = lazy.read_rows(picked)
    with PlanReader(plan, cache_all=True) as cached:
        cached_rows = cached.read_rows(picked)
    np.testing.assert_array_equal(lazy_rows, cached_rows)


# ============================================================================
# sample_shared_shots_from_plan — CRG sampler on a SeismicPlan
# ============================================================================
def _build_synthetic_crg_plan_with_shared_shots(tmp_path: Path):
    """Build a CRG SeismicPlan whose receiver cells share many shots.

    Layout: 4 shots × 6 receivers, rx_step=25 with quantize=30 → 5 cells.
    Every cell ends up recording ALL 4 shots (different byte offsets per
    cell, same physical (sx, sy) keys). Perfect for shared-shot sampling.
    """
    p = _write_synthetic_segy(tmp_path, nshots=4, nrec=6, nt=16, rx_step=25)
    idx = build_segy_index([p])
    plan = build_seismic_plan(idx, grouping="crg", receiver_quantize_m=30.0)
    return plan


def test_sample_shared_shots_basic(tmp_path):
    """Happy path: sample B groups, get the intersection of shots they all saw."""
    from sweep_io.seismic_plan import sample_shared_shots_from_plan

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    rng = np.random.default_rng(0)

    batch = sample_shared_shots_from_plan(plan, rng, batch_size=3)

    assert batch.group_indices.shape == (3,)
    assert batch.group_indices.dtype == np.int64
    assert batch.n_shared >= 1
    # All groups share the same shots (each cell saw every shot in this fixture).
    assert batch.n_shared == 4
    # rows_per_group: length-B list, each (n_shared,) int64.
    assert len(batch.rows_per_group) == 3
    for rows in batch.rows_per_group:
        assert rows.shape == (4,)
        assert rows.dtype == np.int64
        # Indices must point inside that group's slice.
    # Geometry: source_xyz_m = group_xyz at the picked groups.
    np.testing.assert_array_equal(
        batch.source_xyz_m, plan.group_xyz[batch.group_indices],
    )
    # shared_shot_xyz_m shape.
    assert batch.shared_shot_xyz_m.shape == (4, 3)


def test_sample_shared_shots_rows_point_to_same_physical_shots(tmp_path):
    """rows[j] across groups must reference the SAME physical shot."""
    from sweep_io.seismic_plan import sample_shared_shots_from_plan

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    rng = np.random.default_rng(1)
    batch = sample_shared_shots_from_plan(plan, rng, batch_size=3)

    # For each shared-shot column j, the (sx, sy) lookup via every picked
    # group's row_source_xyz must agree (this is the invariant the sampler
    # promises and the supershot encoder relies on).
    for j in range(batch.n_shared):
        sxy_set = set()
        for gi, rows in enumerate(batch.rows_per_group):
            sxy = (
                int(round(plan.row_source_xyz[rows[j], 0] * 1000)),
                int(round(plan.row_source_xyz[rows[j], 1] * 1000)),
            )
            sxy_set.add(sxy)
        assert len(sxy_set) == 1, (
            f"row[{j}] across groups points to different physical shots: "
            f"{sxy_set}"
        )


def test_sample_shared_shots_rejects_csg_plan(tmp_path):
    """The sampler only makes sense for grouping='crg'."""
    from sweep_io.seismic_plan import sample_shared_shots_from_plan

    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=4, nt=16)
    idx = build_segy_index([p])
    csg = build_seismic_plan(idx, grouping="csg")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="grouping='crg'"):
        sample_shared_shots_from_plan(csg, rng, batch_size=1)


def test_sample_shared_shots_deterministic_under_seed(tmp_path):
    """Same RNG state -> identical batch."""
    from sweep_io.seismic_plan import sample_shared_shots_from_plan

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    a = sample_shared_shots_from_plan(plan, np.random.default_rng(42), batch_size=3)
    b = sample_shared_shots_from_plan(plan, np.random.default_rng(42), batch_size=3)

    np.testing.assert_array_equal(a.group_indices, b.group_indices)
    assert a.n_shared == b.n_shared
    for ra, rb in zip(a.rows_per_group, b.rows_per_group):
        np.testing.assert_array_equal(ra, rb)


def test_sample_shared_shots_min_coverage_filter(tmp_path):
    """min_coverage drops low-coverage groups before sampling."""
    from sweep_io.seismic_plan import sample_shared_shots_from_plan

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    counts = plan.per_group_row_counts()
    high_threshold = int(counts.max())  # only the busiest groups qualify
    eligible_high = np.flatnonzero(counts >= high_threshold)
    rng = np.random.default_rng(0)
    batch = sample_shared_shots_from_plan(
        plan, rng,
        batch_size=min(2, int(eligible_high.size)),
        min_coverage=high_threshold,
    )
    # All picked groups must satisfy the coverage threshold.
    for g in batch.group_indices:
        assert plan.group_row_count(int(g)) >= high_threshold


def test_sample_shared_shots_with_read_rows_round_trip(tmp_path):
    """Sampler output + PlanReader.read_rows yields the supershot obs slice."""
    from sweep_io.seismic_plan import sample_shared_shots_from_plan

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    rng = np.random.default_rng(7)
    batch = sample_shared_shots_from_plan(plan, rng, batch_size=3)

    with PlanReader(plan, cache_all=False) as reader:
        # Read each picked group's shared rows; shape (B, n_shared, nt).
        per_group = np.stack(
            [reader.read_rows(rows) for rows in batch.rows_per_group],
            axis=0,
        )
        assert per_group.shape == (3, batch.n_shared, plan.samples_per_trace)
        assert per_group.dtype == np.float32
        # Bit-for-bit parity with cache_all.
    with PlanReader(plan, cache_all=True) as reader_cached:
        per_group_cached = np.stack(
            [reader_cached.read_rows(rows) for rows in batch.rows_per_group],
            axis=0,
        )
    np.testing.assert_array_equal(per_group, per_group_cached)


# ============================================================================
# precompute_group_unique_keys + (B)+(C) sampler fast-path
# ============================================================================
def test_precompute_group_unique_keys_matches_per_call_unique(tmp_path):
    """The precomputed array per group must equal np.unique(_shot_xy_keys)."""
    from sweep_io.seismic_plan import (
        _shot_xy_keys,
        precompute_group_unique_keys,
    )

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    precomputed = precompute_group_unique_keys(plan)
    assert len(precomputed) == int(plan.n_groups)
    for g in range(int(plan.n_groups)):
        sl = plan.group_slice(g)
        sx = plan.row_source_xyz[sl, 0]
        sy = plan.row_source_xyz[sl, 1]
        expected = np.unique(_shot_xy_keys(sx, sy))
        np.testing.assert_array_equal(precomputed[g], expected)
        assert precomputed[g].dtype == np.int64
        # Sorted-unique invariant relied on by the (B) counting intersect.
        assert np.all(np.diff(precomputed[g]) > 0) or precomputed[g].size <= 1


def test_precompute_group_unique_keys_rejects_csg(tmp_path):
    """CRG-only helper — bail out cleanly on CSG plans."""
    from sweep_io.seismic_plan import precompute_group_unique_keys

    p = _write_synthetic_segy(tmp_path, nshots=2, nrec=4, nt=16)
    idx = build_segy_index([p])
    csg = build_seismic_plan(idx, grouping="csg")
    with pytest.raises(ValueError, match="grouping='crg'"):
        precompute_group_unique_keys(csg)


def test_sample_shared_shots_precomputed_keys_byte_equivalent(tmp_path):
    """(B)+(C) fast path must return the SAME batch as the slow path.

    Regression guard: the counting-intersect + reused-uniq optimisation
    only touches *how* the candidate intersection is computed, not the
    set itself or the row look-up. Given identical RNG state, the
    returned ``SharedShotBatch`` must be bit-for-bit identical.
    """
    from sweep_io.seismic_plan import (
        precompute_group_unique_keys,
        sample_shared_shots_from_plan,
    )

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    precomputed = precompute_group_unique_keys(plan)
    slow = sample_shared_shots_from_plan(
        plan, np.random.default_rng(13), batch_size=3,
    )
    fast = sample_shared_shots_from_plan(
        plan, np.random.default_rng(13), batch_size=3,
        precomputed_group_unique_keys=precomputed,
    )
    np.testing.assert_array_equal(slow.group_indices, fast.group_indices)
    assert slow.n_shared == fast.n_shared
    for ra, rb in zip(slow.rows_per_group, fast.rows_per_group):
        np.testing.assert_array_equal(ra, rb)
    np.testing.assert_array_equal(slow.source_xyz_m, fast.source_xyz_m)
    np.testing.assert_array_equal(
        slow.shared_shot_xyz_m, fast.shared_shot_xyz_m,
    )


def test_sample_shared_shots_precomputed_keys_with_subsample(tmp_path):
    """Fast path must stay byte-equivalent with the sub-sampling knobs on."""
    from sweep_io.seismic_plan import (
        precompute_group_unique_keys,
        sample_shared_shots_from_plan,
    )

    plan = _build_synthetic_crg_plan_with_shared_shots(tmp_path)
    precomputed = precompute_group_unique_keys(plan)
    common = dict(
        batch_size=3,
        source_lines_per_group=1,
        max_traces_per_sourceline=2,
        min_coverage=1,
    )
    slow = sample_shared_shots_from_plan(
        plan, np.random.default_rng(2026), **common,
    )
    fast = sample_shared_shots_from_plan(
        plan, np.random.default_rng(2026),
        precomputed_group_unique_keys=precomputed, **common,
    )
    np.testing.assert_array_equal(slow.group_indices, fast.group_indices)
    assert slow.n_shared == fast.n_shared
    for ra, rb in zip(slow.rows_per_group, fast.rows_per_group):
        np.testing.assert_array_equal(ra, rb)
