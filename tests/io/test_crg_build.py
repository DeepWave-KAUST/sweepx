"""Tests for :mod:`sweep_io.crg_build` — SEG-Y → CRGPlan pipeline.

Strategy: build a tiny synthetic survey on disk (4 SEG-Y files, 8 traces
each, simulating 4 OBN nodes each recorded by all 4 source-line files
twice), then exercise the full
``build_segy_index → build_crg_plan_from_index → save_crg_plan``
pipeline and verify against hand-computed expectations.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from sweep_io.crg_build import (
    build_crg_plan_from_index,
    build_crg_plan_from_segy,
    save_crg_plan,
)
from sweep_io.crg_plan import load_crg_shot_plan_cache
from sweep_io.segy import (
    FORMAT_IEEE_FLOAT32,
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
)
from sweep_io.segy_index import SEGYIndex, build_segy_index


SAMPLES = 32
DT_S = 0.002


def _trace_byte_offset(trace_idx: int) -> int:
    return (SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
            + trace_idx * (SEGY_TRACE_HEADER_SIZE + SAMPLES * 4))


def _write_segy_with_headers(
    path: Path,
    *,
    shot_ids: list[int],
    receiver_ids: list[int],
    sx_m: list[float], sy_m: list[float],
    rx_m: list[float], ry_m: list[float],
    rz_m: list[float],
    coord_scalar: int = 1,
) -> None:
    """Write a SEG-Y where every trace has full (sx, sy, sz=0, rx, ry, rz)
    headers populated. Sample data = zero (we only test the header path)."""
    n_traces = len(shot_ids)
    with open(path, "wb") as f:
        # 3200 B text header (zeros)
        f.write(b"\x00" * SEGY_TEXT_HEADER_SIZE)
        # 400 B binary header — set dt + n_samples + sample_format.
        bh = bytearray(SEGY_BIN_HEADER_SIZE)
        struct.pack_into(">H", bh, 16, int(round(DT_S * 1.0e6)))   # dt µs
        struct.pack_into(">H", bh, 20, SAMPLES)
        struct.pack_into(">H", bh, 24, FORMAT_IEEE_FLOAT32)
        f.write(bytes(bh))
        # Per-trace header + zero samples.
        zero_samples = (b"\x00" * SAMPLES * 4)
        for i in range(n_traces):
            th = bytearray(SEGY_TRACE_HEADER_SIZE)
            # Standard rev1 bytes (matches SEGY_REV1_BYTES):
            #   shot          @ byte 8  (4 B int32, FFID)
            #   trace_in_shot @ byte 12 (4 B int32)
            #   source_elev   @ byte 40 (4 B int32 — used as rz here)
            #   source_depth  @ byte 48 (4 B int32, ignored in this fixture)
            #   receiver_depth@ byte 52 (4 B int32, ignored)
            #   coord_scalar  @ byte 70 (2 B int16)
            #   sx, sy        @ bytes 72, 76 (4 B int32 each)
            #   rx, ry        @ bytes 80, 84 (4 B int32 each)
            struct.pack_into(">i", th, 8, int(shot_ids[i]))
            struct.pack_into(">i", th, 12, int(receiver_ids[i]))
            struct.pack_into(">h", th, 70, int(coord_scalar))
            struct.pack_into(">i", th, 72, int(round(sx_m[i] / max(1.0 / coord_scalar, 1.0))))
            struct.pack_into(">i", th, 76, int(round(sy_m[i] / max(1.0 / coord_scalar, 1.0))))
            struct.pack_into(">i", th, 80, int(round(rx_m[i] / max(1.0 / coord_scalar, 1.0))))
            struct.pack_into(">i", th, 84, int(round(ry_m[i] / max(1.0 / coord_scalar, 1.0))))
            # Use source_depth byte (48) for sz; receiver_depth byte (52)
            # for rz. The default byte_map in segy_index reads these.
            struct.pack_into(">i", th, 48, 0)  # sz = 0 (we override at scan)
            struct.pack_into(">i", th, 52, int(round(rz_m[i])))
            f.write(bytes(th))
            f.write(zero_samples)


def _build_tiny_survey(tmp_path: Path) -> tuple[list[Path], dict]:
    """Build 2 SEG-Y files with 6 traces each, simulating 3 OBN nodes
    seeing 2 shots from each of 2 source lines.

    Layout:
      file_0 (source-line 1): shots @ sx=100 and sx=200, recorded by 3 OBN nodes
      file_1 (source-line 2): shots @ sx=300 and sx=400, recorded by 3 OBN nodes
    Total: 12 traces, 3 unique OBN nodes (~(rx, ry, rz)), 4 unique shots.
    """
    # 3 OBN node positions (gx, gy, gz):
    obn = [
        (500.0, 1000.0, 50.0),
        (700.0, 1000.0, 60.0),
        (900.0, 1000.0, 70.0),
    ]
    files = []
    for fi, line_sx in enumerate(([100.0, 200.0], [300.0, 400.0])):
        shots, recs, sx, sy, rx, ry, rz = [], [], [], [], [], [], []
        for shot_idx, sxi in enumerate(line_sx):
            for ri, (gxi, gyi, gzi) in enumerate(obn):
                shots.append(shot_idx + 1)
                recs.append(ri)
                sx.append(sxi); sy.append(0.0)
                rx.append(gxi); ry.append(gyi); rz.append(gzi)
        p = tmp_path / f"sourceLine_{fi:03d}.sgy"
        _write_segy_with_headers(
            p, shot_ids=shots, receiver_ids=recs,
            sx_m=sx, sy_m=sy, rx_m=rx, ry_m=ry, rz_m=rz,
            coord_scalar=1,
        )
        files.append(p)
    expected = {
        "n_obn": 3,
        "obn_xyz": np.asarray(obn, dtype=np.float64),
        "n_traces_per_obn": 4,  # 2 lines × 2 shots each
        "n_files": 2,
    }
    return files, expected


def test_build_segy_index_returns_expected_per_trace_arrays(tmp_path):
    files, exp = _build_tiny_survey(tmp_path)
    idx = build_segy_index(files, num_workers=1)
    # 12 total traces.
    assert idx.file_id.size == 12
    assert int(idx.n_samples) == SAMPLES
    assert idx.dt_s == pytest.approx(DT_S)
    # 3 unique (rx, ry, rz) cells.
    rcell = np.stack([idx.rx_m, idx.ry_m, idx.rz_m], axis=-1)
    unique = np.unique(rcell.astype(np.int64).view([("", np.int64)] * 3))
    assert unique.size == 3


def test_build_crg_plan_groups_by_receiver(tmp_path):
    files, exp = _build_tiny_survey(tmp_path)
    idx = build_segy_index(files, num_workers=1)
    plan = build_crg_plan_from_index(idx, receiver_quantize_m=1.0)
    assert plan.n_receivers_used == exp["n_obn"]
    assert plan.n_plan_rows == 12
    # Each slot should have 4 traces (2 lines × 2 shots).
    np.testing.assert_array_equal(plan.per_slot_row_counts(), [4, 4, 4])
    # Receiver xyz round-trip approx (mean of contributing trace coords).
    for k, expected_xyz in enumerate(exp["obn_xyz"]):
        np.testing.assert_allclose(plan.receiver_xyz_used[k], expected_xyz, atol=1.0e-6)
    # Plan-row file_ids per slot: 2 from file 0, 2 from file 1.
    for k in range(3):
        sl = plan.slot_slice(k)
        fids = plan.plan_file_ids[sl]
        assert int((fids == 0).sum()) == 2
        assert int((fids == 1).sum()) == 2


def test_build_crg_plan_receiver_stride(tmp_path):
    files, _ = _build_tiny_survey(tmp_path)
    idx = build_segy_index(files, num_workers=1)
    plan = build_crg_plan_from_index(
        idx, receiver_quantize_m=1.0, receiver_stride=2,
    )
    # Stride 2 over 3 receivers → keep 2 (indices 0 and 2).
    assert plan.n_receivers_used == 2
    # Each kept slot still has 4 traces.
    np.testing.assert_array_equal(plan.per_slot_row_counts(), [4, 4])


def test_build_crg_plan_shots_per_source_line_caps(tmp_path):
    files, _ = _build_tiny_survey(tmp_path)
    idx = build_segy_index(files, num_workers=1)
    # Cap to 1 shot per source line per slot → each slot keeps 2 traces.
    plan = build_crg_plan_from_index(
        idx, receiver_quantize_m=1.0, shots_per_source_line=1,
    )
    assert plan.n_receivers_used == 3
    np.testing.assert_array_equal(plan.per_slot_row_counts(), [2, 2, 2])


def test_save_crg_plan_round_trip(tmp_path):
    files, _ = _build_tiny_survey(tmp_path)
    idx = build_segy_index(files, num_workers=1)
    plan = build_crg_plan_from_index(idx, receiver_quantize_m=1.0)
    out = tmp_path / "plan_v1.npz"
    save_crg_plan(plan, out)
    assert out.exists()
    loaded = load_crg_shot_plan_cache(out)
    # Schema fields match.
    assert loaded.n_receivers_used == plan.n_receivers_used
    assert loaded.n_plan_rows == plan.n_plan_rows
    np.testing.assert_array_equal(loaded.plan_file_ids, plan.plan_file_ids)
    np.testing.assert_array_equal(loaded.plan_trace_offsets, plan.plan_trace_offsets)
    np.testing.assert_allclose(loaded.plan_sx, plan.plan_sx)
    np.testing.assert_allclose(loaded.receiver_xyz_used, plan.receiver_xyz_used)


def test_build_crg_plan_from_segy_end_to_end(tmp_path):
    """The convenience wrapper does scan + group in one call."""
    files, exp = _build_tiny_survey(tmp_path)
    plan = build_crg_plan_from_segy(
        files, num_workers=1,
        receiver_quantize_m=1.0,
    )
    assert plan.n_receivers_used == exp["n_obn"]
    # And byte offsets point at valid SEG-Y trace records.
    for slot in range(plan.n_receivers_used):
        sl = plan.slot_slice(slot)
        for fid, off in zip(plan.plan_file_ids[sl].tolist(),
                            plan.plan_trace_offsets[sl].tolist()):
            assert fid in (0, 1)
            assert off >= SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE


def test_partition_files_contiguous_uneven():
    """Files split contiguously with remainder spread over the first ranks."""
    from sweep_io.crg_build import _partition_files_contiguous

    # n=10 across size=4 → splits of (3, 3, 2, 2)
    spans = [_partition_files_contiguous(10, r, 4) for r in range(4)]
    counts = [e - s for s, e in spans]
    assert counts == [3, 3, 2, 2]
    # Spans cover [0, 10) contiguously without gaps or overlaps.
    assert spans[0] == (0, 3)
    assert spans[1] == (3, 6)
    assert spans[2] == (6, 8)
    assert spans[3] == (8, 10)


def test_partition_files_contiguous_evenly_divisible():
    from sweep_io.crg_build import _partition_files_contiguous

    spans = [_partition_files_contiguous(16, r, 4) for r in range(4)]
    assert spans == [(0, 4), (4, 8), (8, 12), (12, 16)]


def test_merge_segy_indices_remaps_file_ids(tmp_path):
    """Build two disjoint single-rank indices, merge → global file_ids
    address the concatenated path list. SEGYIndex sorts by shot_id post-
    construction, so we count totals per file_id rather than positional
    expectations.
    """
    from sweep_io.crg_build import _merge_segy_indices

    files, _ = _build_tiny_survey(tmp_path)
    idx_a = build_segy_index([files[0]], num_workers=1)
    idx_b = build_segy_index([files[1]], num_workers=1)
    merged = _merge_segy_indices([idx_a, idx_b])
    # Concatenated file list.
    assert len(merged.file_paths) == 2
    assert merged.file_paths[0] == str(files[0])
    assert merged.file_paths[1] == str(files[1])
    # Trace count = sum of parts.
    assert merged.file_id.size == idx_a.file_id.size + idx_b.file_id.size
    # Per-file trace counts preserved after the file_id remap.
    assert int((merged.file_id == 0).sum()) == idx_a.file_id.size
    assert int((merged.file_id == 1).sum()) == idx_b.file_id.size


def test_merge_segy_indices_matches_single_process(tmp_path):
    """Merged-from-shards index must equal a single-process scan."""
    from sweep_io.crg_build import (
        _merge_segy_indices,
        build_crg_plan_from_index,
    )

    files, _ = _build_tiny_survey(tmp_path)
    ref = build_segy_index(files, num_workers=1)
    # Mimic 2-rank split: rank 0 gets files[0], rank 1 gets files[1].
    sub = _merge_segy_indices(
        [build_segy_index([files[0]], num_workers=1),
         build_segy_index([files[1]], num_workers=1)]
    )
    # Single-process and merged should produce the same CRG plan
    # (sorted by receiver_id which is invariant to scan order).
    plan_ref = build_crg_plan_from_index(ref, receiver_quantize_m=1.0)
    plan_sub = build_crg_plan_from_index(sub, receiver_quantize_m=1.0)
    assert plan_ref.n_receivers_used == plan_sub.n_receivers_used
    assert plan_ref.n_plan_rows == plan_sub.n_plan_rows
    np.testing.assert_array_equal(
        plan_ref.per_slot_row_counts(),
        plan_sub.per_slot_row_counts(),
    )
    np.testing.assert_allclose(plan_ref.receiver_xyz_used, plan_sub.receiver_xyz_used)


def test_save_load_plan_then_read_traces_with_reader(tmp_path):
    """Round-trip the plan through disk, then verify the byte offsets it
    holds are actually valid for the MultiFileSEGYReader."""
    from sweep_io.segy import MultiFileSEGYReader

    files, _ = _build_tiny_survey(tmp_path)
    plan = build_crg_plan_from_segy(files, num_workers=1, receiver_quantize_m=1.0)
    out = tmp_path / "plan_v1.npz"
    save_crg_plan(plan, out)
    loaded = load_crg_shot_plan_cache(out)
    reader = MultiFileSEGYReader([str(p) for p in loaded.files], mmap_mode=True)
    try:
        sl = loaded.slot_slice(0)
        traces = reader.read_traces(
            loaded.plan_file_ids[sl], loaded.plan_trace_offsets[sl],
        )
        # All-zero sample payload (we wrote zeros) at the right shape.
        assert traces.shape == (4, SAMPLES)
        np.testing.assert_allclose(traces, 0.0)
    finally:
        reader.close()
