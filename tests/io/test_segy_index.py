"""Tests for SEGYIndex, build_segy_index, IndexedShotGatherDataset.

Builds tiny synthetic SEG-Y fixtures via `write_segy_minimal` so the suite
runs without external data.
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
from sweep_io.segy_index import (
    SEGY_REV1_BYTES,
    SEGYIndex,
    build_segy_index,
    IndexedShotGatherDataset,
)


# ============================================================================
# Helpers — write a minimal SEG-Y where the trace-header fields we care about
# (FFID, sx, sy, rx, ry, coord_scalar) actually have meaningful values.
# ============================================================================
def _patch_trace_header(
    raw_bytes: bytearray,
    *,
    shot_id: int,
    trace_in_shot: int,
    sx: int, sy: int, rx: int, ry: int,
    coord_scalar: int = 1,
    source_depth: int = 0,
    receiver_depth: int = 0,
) -> None:
    """Mutate a 240-byte header to carry the supplied fields."""
    bm = SEGY_REV1_BYTES
    struct.pack_into(">i", raw_bytes, bm["shot"], shot_id)
    struct.pack_into(">i", raw_bytes, bm["trace_in_shot"], trace_in_shot)
    struct.pack_into(">h", raw_bytes, bm["coord_scalar"], coord_scalar)
    struct.pack_into(">i", raw_bytes, bm["sx"], sx)
    struct.pack_into(">i", raw_bytes, bm["sy"], sy)
    struct.pack_into(">i", raw_bytes, bm["rx"], rx)
    struct.pack_into(">i", raw_bytes, bm["ry"], ry)
    struct.pack_into(">i", raw_bytes, bm["source_depth"], source_depth)
    struct.pack_into(">i", raw_bytes, bm["receiver_depth"], receiver_depth)


def _write_fixture(
    tmp_path: Path, *, nshots: int = 3, nrec: int = 8, nt: int = 64,
    rx_step: int = 25, sx_start: int = 100, sx_step: int = 50,
    fname: str = "tiny.segy", shot_id_start: int = 1,
) -> Path:
    """Write a synthetic SEG-Y where each (shot, rec) carries an identifiable obs value."""
    n_total = nshots * nrec
    # Data per trace = arange offset so we can verify lookups
    data = np.zeros((n_total, nt), dtype="float32")
    for s in range(nshots):
        for r in range(nrec):
            i = s * nrec + r
            data[i, 0] = float(shot_id_start + s) * 100 + r   # tag value at sample 0
    path = tmp_path / fname
    write_segy_minimal(path, data, dt=0.001, sample_format=FORMAT_IEEE_FLOAT32)

    # Now patch each trace header.
    trace_total = SEGY_TRACE_HEADER_SIZE + nt * 4
    base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
    with open(path, "r+b") as f:
        for s in range(nshots):
            sx = sx_start + s * sx_step
            sy = 0
            for r in range(nrec):
                i = s * nrec + r
                hdr_off = base + i * trace_total
                f.seek(hdr_off)
                buf = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
                _patch_trace_header(
                    buf,
                    shot_id=shot_id_start + s,
                    trace_in_shot=r,
                    sx=sx, sy=sy,
                    rx=r * rx_step, ry=0,
                    coord_scalar=1,
                )
                f.seek(hdr_off)
                f.write(bytes(buf))
    return path


# ============================================================================
# build_segy_index
# ============================================================================
def test_build_index_single_file(tmp_path):
    p = _write_fixture(tmp_path, nshots=3, nrec=5, nt=32)
    idx = build_segy_index([p], num_workers=1)
    assert idx.n_traces == 15
    assert idx.n_shots == 3
    assert idx.file_paths == [str(p)]
    np.testing.assert_array_equal(idx.shot_ids, [1, 2, 3])
    # All same file
    assert int(idx.file_id.min()) == 0
    assert int(idx.file_id.max()) == 0


def test_build_index_multi_file(tmp_path):
    p1 = _write_fixture(tmp_path, nshots=2, nrec=4, nt=32, fname="a.segy", shot_id_start=10)
    p2 = _write_fixture(tmp_path, nshots=3, nrec=4, nt=32, fname="b.segy", shot_id_start=20)
    idx = build_segy_index([p1, p2], num_workers=2)
    assert idx.n_traces == 8 + 12
    assert idx.n_shots == 5
    np.testing.assert_array_equal(idx.shot_ids, [10, 11, 20, 21, 22])
    # file_id matches the input order
    sids_per_file = {int(f): set(int(s) for s in idx.shot_id[idx.file_id == f])
                     for f in np.unique(idx.file_id)}
    assert sids_per_file == {0: {10, 11}, 1: {20, 21, 22}}


def test_build_index_rejects_mismatched_files(tmp_path):
    # Same dt but different nt
    p1 = _write_fixture(tmp_path, nshots=1, nrec=2, nt=32, fname="x.segy", shot_id_start=1)
    p2 = _write_fixture(tmp_path, nshots=1, nrec=2, nt=64, fname="y.segy", shot_id_start=2)
    with pytest.raises(ValueError, match="dt / n_samples / sample_format"):
        build_segy_index([p1, p2], num_workers=1)


def test_index_depth_overrides(tmp_path):
    p = _write_fixture(tmp_path, nshots=2, nrec=3, nt=16)
    idx = build_segy_index(
        [p],
        source_depth_m_override=6.0,
        receiver_depth_m_override=10.0,
        num_workers=1,
    )
    np.testing.assert_array_equal(idx.sz_m, np.full(6, 6.0))
    np.testing.assert_array_equal(idx.rz_m, np.full(6, 10.0))


# ============================================================================
# Lookups + serialization
# ============================================================================
def test_lookup_shot_returns_correct_traces(tmp_path):
    p = _write_fixture(tmp_path, nshots=4, nrec=6, nt=16, sx_start=200, sx_step=25, rx_step=10)
    idx = build_segy_index([p], num_workers=1)
    slc = idx.lookup_shot(3)   # third shot has sx = 200 + 2*25 = 250
    assert slc["sx_m"] == pytest.approx(250.0)
    assert slc["sy_m"] == pytest.approx(0.0)
    np.testing.assert_array_equal(slc["receiver_id"], np.arange(6))
    np.testing.assert_array_equal(slc["rx_m"], np.arange(6) * 10.0)


def test_lookup_unknown_shot_raises(tmp_path):
    p = _write_fixture(tmp_path, nshots=2, nrec=2, nt=8)
    idx = build_segy_index([p], num_workers=1)
    with pytest.raises(KeyError):
        idx.lookup_shot(999)


def test_index_save_load_roundtrip(tmp_path):
    p = _write_fixture(tmp_path, nshots=3, nrec=4, nt=16)
    idx = build_segy_index([p], num_workers=1)
    save_path = tmp_path / "idx.npz"
    idx.save(save_path)
    assert save_path.exists()
    idx2 = SEGYIndex.load(save_path)
    np.testing.assert_array_equal(idx.shot_ids, idx2.shot_ids)
    np.testing.assert_array_equal(idx.byte_offset, idx2.byte_offset)
    assert idx.dt_s == idx2.dt_s
    assert idx.n_samples == idx2.n_samples


# ============================================================================
# IndexedShotGatherDataset
# ============================================================================
def test_indexed_dataset_reads_correct_shot(tmp_path):
    p = _write_fixture(tmp_path, nshots=3, nrec=5, nt=32, shot_id_start=10)
    idx = build_segy_index([p], num_workers=1)
    with IndexedShotGatherDataset(idx) as ds:
        assert len(ds) == 3
        sample = ds[1]                  # second shot, shot_id=11
        assert sample["shot_id"] == 11
        assert sample["obs"].shape == (5, 32)
        # Tagged values at sample 0: (11*100 + r) for r in 0..4
        np.testing.assert_array_equal(
            sample["obs"][:, 0], [1100, 1101, 1102, 1103, 1104],
        )


def test_indexed_dataset_subset_shot_ids(tmp_path):
    p = _write_fixture(tmp_path, nshots=4, nrec=3, nt=16, shot_id_start=1)
    idx = build_segy_index([p], num_workers=1)
    with IndexedShotGatherDataset(idx, shot_ids=[3, 1]) as ds:
        assert len(ds) == 2
        assert ds[0]["shot_id"] == 3
        assert ds[1]["shot_id"] == 1


def test_indexed_dataset_to_physical_geometry(tmp_path):
    p = _write_fixture(tmp_path, nshots=3, nrec=5, nt=16, shot_id_start=1,
                       sx_start=100, sx_step=25, rx_step=10)
    idx = build_segy_index(
        [p], source_depth_m_override=6.0, receiver_depth_m_override=10.0,
        num_workers=1,
    )
    pg = idx.to_physical_geometry()
    assert pg.nshots == 3
    assert pg.nreceivers == 5
    assert pg.ndim == 2
    # Source x: 100, 125, 150; Source z: 6 (override)
    np.testing.assert_array_equal(pg.sources_xyz_m, [[100, 6], [125, 6], [150, 6]])
    # Receiver x for shot 0: 0, 10, 20, 30, 40
    np.testing.assert_array_equal(pg.receivers_xyz_m[0, :, 0], [0, 10, 20, 30, 40])


def test_indexed_dataset_with_prefetch(tmp_path):
    """Glue test: IndexedShotGatherDataset wraps cleanly with the Prefetcher."""
    from sweep_io.prefetch import Prefetcher
    p = _write_fixture(tmp_path, nshots=4, nrec=3, nt=16, shot_id_start=100)
    idx = build_segy_index([p], num_workers=1)
    ds = IndexedShotGatherDataset(idx)
    seen = []
    with Prefetcher((ds[i] for i in range(len(ds))), queue_depth=2) as pf:
        for sample in pf:
            seen.append(int(sample["shot_id"]))
    assert seen == [100, 101, 102, 103]
    ds.close()
