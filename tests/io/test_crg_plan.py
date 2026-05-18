"""Tests for :func:`sweep_io.crg_plan.load_crg_shot_plan_cache` +
:class:`sweep_io.crg_dataset.CRGBatchDataset`.

The fixture builds a tiny synthetic survey:
- 2 SEG-Y source-line files via :func:`sweep_io.segy.write_segy_minimal`
- a 3-virtual-source plan cache pointing into those files
- exercises the end-to-end load → dataset → pad_collate path

This is the smallest exercise of the CRG ingest pipeline that doesn't
require any field data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sweep_io.crg_plan import CRGPlan, _remap_segy_path, load_crg_shot_plan_cache
from sweep_io.segy import (
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    FORMAT_IEEE_FLOAT32,
    write_segy_minimal,
)


SAMPLES_PER_TRACE = 64
DT_S = 0.002


def _write_tiny_segy(path: Path, n_traces: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_traces, SAMPLES_PER_TRACE)).astype(np.float32)
    # Encode the (file_seed, trace_idx) into the first two samples so the
    # round-trip assertion can verify trace identity unambiguously.
    for i in range(n_traces):
        data[i, 0] = float(seed)
        data[i, 1] = float(i)
    write_segy_minimal(path, data, dt=DT_S, sample_format=FORMAT_IEEE_FLOAT32)


def _trace_byte_offset(trace_idx: int) -> int:
    """Byte offset of the trace-header for ``trace_idx`` in a v1 file."""
    bytes_per_trace = SEGY_TRACE_HEADER_SIZE + SAMPLES_PER_TRACE * 4
    return SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE + trace_idx * bytes_per_trace


@pytest.fixture
def tiny_crg_cache(tmp_path: Path) -> Path:
    # Two source-line files with 8 / 6 traces respectively.
    f0 = tmp_path / "sourceLine_001.sgy"
    f1 = tmp_path / "sourceLine_002.sgy"
    _write_tiny_segy(f0, n_traces=8, seed=11)
    _write_tiny_segy(f1, n_traces=6, seed=22)

    # 3 virtual sources (OBN nodes).
    receiver_xyz_used = np.array(
        [[1000.0, 2000.0, 100.0],
         [1500.0, 2000.0, 100.0],
         [1000.0, 2500.0, 100.0]],
        dtype=np.float64,
    )
    # Per-slot plan rows:
    # slot 0 -> file 0 traces [0, 2, 4]  (3 shots)
    # slot 1 -> file 0 trace 1 + file 1 traces [0, 1, 2, 3] (5 shots)
    # slot 2 -> file 1 traces [4, 5]    (2 shots)
    file_id_blocks = [
        np.array([0, 0, 0], dtype=np.int64),
        np.array([0, 1, 1, 1, 1], dtype=np.int64),
        np.array([1, 1], dtype=np.int64),
    ]
    trace_idx_blocks = [
        np.array([0, 2, 4], dtype=np.int64),
        np.array([1, 0, 1, 2, 3], dtype=np.int64),
        np.array([4, 5], dtype=np.int64),
    ]
    plan_file_ids = np.concatenate(file_id_blocks)
    plan_trace_offsets = np.array(
        [_trace_byte_offset(int(t)) for t in np.concatenate(trace_idx_blocks)],
        dtype=np.int64,
    )
    plan_sx = np.arange(plan_file_ids.size, dtype=np.float64) * 10.0
    plan_sy = np.arange(plan_file_ids.size, dtype=np.float64) * 5.0
    plan_sz = np.full(plan_file_ids.size, 8.0, dtype=np.float64)
    plan_offsets = np.array([0, 3, 8, 10], dtype=np.int64)

    cache_path = tmp_path / "plan.npz"
    np.savez(
        cache_path,
        format=np.asarray("crg_fwi_plan_v1", dtype="U32"),
        files=np.asarray([str(f0), str(f1)], dtype="U256"),
        trace_size_per_file=np.asarray(
            [SEGY_TRACE_HEADER_SIZE + SAMPLES_PER_TRACE * 4] * 2, dtype=np.int64),
        sample_format=np.asarray(FORMAT_IEEE_FLOAT32, dtype=np.int32),
        samples_per_trace=np.asarray(SAMPLES_PER_TRACE, dtype=np.int32),
        dt_s=np.asarray(DT_S, dtype=np.float64),
        receiver_indices=np.asarray([10, 11, 12], dtype=np.int64),
        receiver_xyz_used=receiver_xyz_used,
        plan_file_ids=plan_file_ids,
        plan_trace_offsets=plan_trace_offsets,
        plan_sx=plan_sx, plan_sy=plan_sy, plan_sz=plan_sz,
        plan_offsets=plan_offsets,
    )
    return cache_path


def test_load_crg_shot_plan_cache_shapes(tiny_crg_cache):
    plan = load_crg_shot_plan_cache(tiny_crg_cache)
    assert isinstance(plan, CRGPlan)
    assert plan.n_receivers_used == 3
    assert plan.n_plan_rows == 10
    assert plan.samples_per_trace == SAMPLES_PER_TRACE
    assert plan.dt_s == pytest.approx(DT_S)
    np.testing.assert_array_equal(plan.per_slot_row_counts(), [3, 5, 2])
    np.testing.assert_allclose(plan.virtual_source_xy[0], [1000.0, 2000.0])


def test_load_crg_shot_plan_cache_segy_root_remap(tiny_crg_cache, tmp_path, monkeypatch):
    # Move the SEG-Y files to a sibling dir and remap via FWI_SEGY_ROOT.
    remap_dir = tmp_path / "remapped"
    remap_dir.mkdir()
    for f in tmp_path.glob("*.sgy"):
        f.rename(remap_dir / f.name)
    monkeypatch.setenv("FWI_SEGY_ROOT", str(remap_dir))
    plan = load_crg_shot_plan_cache(tiny_crg_cache)
    for p in plan.files:
        assert p.parent == remap_dir
        assert p.exists()


def test_remap_segy_path_via_remap_rules(monkeypatch, tmp_path):
    monkeypatch.delenv("FWI_SEGY_ROOT", raising=False)
    monkeypatch.setenv("FWI_SEGY_REMAP", "/old/root:/new/root")
    assert _remap_segy_path("/old/root/sub/x.sgy") == "/new/root/sub/x.sgy"
    assert _remap_segy_path("/unrelated/x.sgy") == "/unrelated/x.sgy"


def test_filter_by_coverage_drops_below_min(tiny_crg_cache):
    plan = load_crg_shot_plan_cache(tiny_crg_cache)
    filtered = plan.filter_by_coverage(min_shots=3)
    # slot 2 (count 2) is dropped; counts becomes [3, 5].
    assert filtered.n_receivers_used == 2
    np.testing.assert_array_equal(filtered.per_slot_row_counts(), [3, 5])
    # Plan-row data was contracted.
    assert filtered.n_plan_rows == 8


def test_crg_batch_dataset_returns_correct_traces(tiny_crg_cache):
    plan = load_crg_shot_plan_cache(tiny_crg_cache)
    from sweep_io.crg_dataset import CRGBatchDataset

    ds = CRGBatchDataset(plan)
    assert len(ds) == 3
    item0 = ds[0]
    assert item0["traces"].shape == (3, SAMPLES_PER_TRACE)
    # Identity check: trace ``i`` from file ``seed`` carries (seed, i) in
    # samples [0, 1]. Slot 0 picks file 0 traces [0, 2, 4].
    expected_file_seeds = np.array([11, 11, 11], dtype=np.float32)
    expected_trace_idx = np.array([0, 2, 4], dtype=np.float32)
    np.testing.assert_allclose(item0["traces"][:, 0], expected_file_seeds)
    np.testing.assert_allclose(item0["traces"][:, 1], expected_trace_idx)
    np.testing.assert_allclose(item0["source_xyz_m"], [1000.0, 2000.0, 100.0])
    ds.close()


def test_pad_collate_fn_padded_layout(tiny_crg_cache):
    plan = load_crg_shot_plan_cache(tiny_crg_cache)
    from sweep_io.crg_dataset import CRGBatchDataset, pad_collate_fn

    ds = CRGBatchDataset(plan)
    batch = [ds[0], ds[1], ds[2]]
    out = pad_collate_fn(batch)
    assert out["traces"].shape == (3, 5, SAMPLES_PER_TRACE)
    assert out["valid_receiver_mask"].shape == (3, 5)
    # Slot 0 has 3 valid, slot 1 has 5, slot 2 has 2.
    np.testing.assert_array_equal(
        out["valid_receiver_mask"],
        np.array([
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]),
    )
    np.testing.assert_array_equal(out["n_shots"], [3, 5, 2])
    np.testing.assert_array_equal(out["slot"], [0, 1, 2])
    # Padded rows are exactly zero.
    assert np.all(out["traces"][0, 3:, :] == 0.0)
    ds.close()


def test_shared_shots_sampler_empty_intersection_raises(tiny_crg_cache):
    """tiny_crg_cache slot 0 = sx [0, 20, 40]; slot 1 = sx [10, 30, ...].
    No (sx, sy) overlap → intersection is empty → sampler raises.
    """
    from sweep_io.crg_plan import sample_crg_shared_shots

    plan = load_crg_shot_plan_cache(tiny_crg_cache)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="intersection is empty"):
        sample_crg_shared_shots(plan, rng, batch_size=2,
                                eligible_slots=np.array([0, 1]))


def test_shared_shots_sampler_finds_intersection_when_present(tmp_path):
    """Build a fixture where two slots share two physical shots (same sx/sy
    but different byte offsets because each slot is a different OBN node).
    """
    from sweep_io.crg_plan import sample_crg_shared_shots

    f0 = tmp_path / "sourceLine_001.sgy"
    _write_tiny_segy(f0, n_traces=8, seed=11)
    receiver_xyz_used = np.array(
        [[1000.0, 2000.0, 100.0], [1500.0, 2000.0, 100.0]], dtype=np.float64,
    )
    # Per legacy: a shot is identified by (sx, sy). Build two slots where
    # the same physical shots (sx, sy) appear at DIFFERENT byte offsets
    # (mimicking a SEG-Y layout where each (shot, receiver) is a separate
    # trace and the receivers come in different file positions).
    #
    # Slot 0 (OBN node 0) records 3 shots at sx ∈ {100, 200, 300}
    # at SEG-Y byte offsets for trace_idx 0, 1, 2.
    # Slot 1 (OBN node 1) records 3 shots at sx ∈ {200, 300, 400}
    # at SEG-Y byte offsets for trace_idx 3, 4, 5.
    # Shared: sx ∈ {200, 300}.
    plan_file_ids = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    plan_trace_offsets = np.array(
        [_trace_byte_offset(i) for i in [0, 1, 2, 3, 4, 5]],
        dtype=np.int64,
    )
    plan_sx = np.array([100.0, 200.0, 300.0,
                        200.0, 300.0, 400.0], dtype=np.float64)
    plan_sy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    plan_sz = np.full(6, 8.0, dtype=np.float64)
    plan_offsets = np.array([0, 3, 6], dtype=np.int64)
    cache_path = tmp_path / "plan.npz"
    np.savez(
        cache_path,
        format=np.asarray("crg_fwi_plan_v1", dtype="U32"),
        files=np.asarray([str(f0)], dtype="U256"),
        trace_size_per_file=np.asarray(
            [SEGY_TRACE_HEADER_SIZE + SAMPLES_PER_TRACE * 4], dtype=np.int64),
        sample_format=np.asarray(FORMAT_IEEE_FLOAT32, dtype=np.int32),
        samples_per_trace=np.asarray(SAMPLES_PER_TRACE, dtype=np.int32),
        dt_s=np.asarray(DT_S, dtype=np.float64),
        receiver_indices=np.asarray([10, 11], dtype=np.int64),
        receiver_xyz_used=receiver_xyz_used,
        plan_file_ids=plan_file_ids,
        plan_trace_offsets=plan_trace_offsets,
        plan_sx=plan_sx, plan_sy=plan_sy, plan_sz=plan_sz,
        plan_offsets=plan_offsets,
    )
    plan = load_crg_shot_plan_cache(cache_path)
    rng = np.random.default_rng(0)
    batch = sample_crg_shared_shots(plan, rng, batch_size=2)
    # Shared shots are sx ∈ {200, 300}; shared_shot_xyz_m must contain those.
    assert batch.n_shared == 2
    np.testing.assert_array_equal(
        sorted(batch.shared_shot_xyz_m[:, 0].tolist()), [200.0, 300.0],
    )
    assert batch.source_xyz_m.shape == (2, 3)
    # rows_per_slot must be len==2 with same length entries:
    assert len(batch.rows_per_slot) == 2
    assert batch.rows_per_slot[0].size == 2 and batch.rows_per_slot[1].size == 2
    # Slot 0's rows point at flat indices 1, 2 (sx=200, 300 within slot 0's
    # plan range [0, 3)). Slot 1's rows point at flat indices 3, 4 (sx=200,
    # 300 within slot 1's plan range [3, 6)).
    s0 = set(batch.rows_per_slot[0].tolist())
    s1 = set(batch.rows_per_slot[1].tolist())
    assert s0 == {1, 2}, s0
    assert s1 == {3, 4}, s1


def test_max_shots_per_slot_truncates(tiny_crg_cache):
    plan = load_crg_shot_plan_cache(tiny_crg_cache)
    from sweep_io.crg_dataset import CRGBatchDataset

    ds = CRGBatchDataset(plan, max_shots_per_slot=2)
    np.testing.assert_array_equal(ds.n_shots_per_slot, [2, 2, 2])
    # Slot 1 normally has 5 shots; should now be truncated to 2.
    item1 = ds[1]
    assert item1["traces"].shape == (2, SAMPLES_PER_TRACE)
    ds.close()
