"""Tests for SEGYReader, MultiFileSEGYReader, and the IBM<->IEEE codec."""

import numpy as np
import pytest

from sweep_io.segy import (
    FORMAT_IBM_FLOAT32,
    FORMAT_IEEE_FLOAT32,
    MultiFileSEGYReader,
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    SEGYReader,
    ibm_to_ieee,
    ieee_to_ibm,
    write_segy_minimal,
)


def _expected_offsets(n_traces: int, n_samples: int, bytes_per_sample: int = 4) -> np.ndarray:
    """Trace-header offsets for a file written by write_segy_minimal."""
    trace_total = SEGY_TRACE_HEADER_SIZE + n_samples * bytes_per_sample
    base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
    return base + np.arange(n_traces, dtype=np.int64) * trace_total


# ---------------------------------------------------------------- IBM codec
def test_ibm_zero_roundtrip():
    z = np.zeros(8, dtype=np.float32)
    enc = ieee_to_ibm(z)
    dec = ibm_to_ieee(enc)
    np.testing.assert_array_equal(dec, z)


def test_ibm_roundtrip_random():
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal(2048) * 1e-3).astype(np.float32)
    enc = ieee_to_ibm(arr)
    dec = ibm_to_ieee(enc)
    # IBM->IEEE round-trip is lossy in low bits (radix-16 normalization).
    np.testing.assert_allclose(dec, arr, rtol=2e-6, atol=1e-12)


def test_ibm_decoder_handles_negatives():
    arr = np.array([-1.0, -1e-3, -100.0, 0.5, 0.0, 1e6], dtype=np.float32)
    enc = ieee_to_ibm(arr)
    dec = ibm_to_ieee(enc)
    np.testing.assert_allclose(dec, arr, rtol=2e-6)


# ------------------------------------------------------------- SEGYReader
def test_segy_reader_roundtrip_ieee(tmp_path):
    nt, nrec = 500, 16
    rng = np.random.default_rng(0)
    data = rng.standard_normal((nrec, nt)).astype("float32")
    path = tmp_path / "test.segy"
    write_segy_minimal(path, data, dt=0.001, sample_format=FORMAT_IEEE_FLOAT32)

    with SEGYReader(path) as r:
        assert r.n_samples == nt
        assert r.dt == pytest.approx(0.001)
        assert r.sample_format == FORMAT_IEEE_FLOAT32
        assert r.n_traces == nrec

        offsets = _expected_offsets(nrec, nt)
        out = r.read_trace_data(offsets)
        np.testing.assert_allclose(out, data, rtol=1e-6)


def test_segy_reader_roundtrip_ibm(tmp_path):
    nt, nrec = 200, 10
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((nrec, nt)) * 1e-3).astype("float32")
    path = tmp_path / "test.segy"
    write_segy_minimal(path, data, dt=0.002, sample_format=FORMAT_IBM_FLOAT32)

    with SEGYReader(path) as r:
        assert r.sample_format == FORMAT_IBM_FLOAT32
        offsets = _expected_offsets(nrec, nt)
        out = r.read_trace_data(offsets)
        np.testing.assert_allclose(out, data, rtol=2e-6, atol=1e-12)


def test_segy_reader_preserves_caller_order(tmp_path):
    """Even when reads are sorted internally, output must be in caller order."""
    nt, nrec = 128, 8
    data = np.arange(nrec * nt, dtype="float32").reshape(nrec, nt)
    path = tmp_path / "test.segy"
    write_segy_minimal(path, data, dt=0.001)

    with SEGYReader(path) as r:
        offsets = _expected_offsets(nrec, nt)
        # Read in reverse order
        out = r.read_trace_data(offsets[::-1])
        np.testing.assert_allclose(out, data[::-1], rtol=1e-6)
        # Random permutation
        perm = np.array([3, 0, 7, 1, 5, 2, 6, 4])
        out2 = r.read_trace_data(offsets[perm])
        np.testing.assert_allclose(out2, data[perm], rtol=1e-6)


def test_segy_reader_coalesce_gives_same_result(tmp_path):
    nt, nrec = 64, 32
    rng = np.random.default_rng(0)
    data = rng.standard_normal((nrec, nt)).astype("float32")
    path = tmp_path / "test.segy"
    write_segy_minimal(path, data, dt=0.001)

    with SEGYReader(path) as r:
        offsets = _expected_offsets(nrec, nt)
        baseline = r.read_trace_data(offsets, coalesce_gap=0)
        coalesced = r.read_trace_data(offsets, coalesce_gap=4096)
        np.testing.assert_array_equal(baseline, coalesced)


def test_segy_reader_empty_read(tmp_path):
    nt, nrec = 32, 4
    data = np.zeros((nrec, nt), dtype="float32")
    path = tmp_path / "test.segy"
    write_segy_minimal(path, data, dt=0.001)
    with SEGYReader(path) as r:
        out = r.read_trace_data(np.array([], dtype=np.int64))
        assert out.shape == (0, nt)


def test_segy_reader_with_and_without_mmap(tmp_path):
    nt, nrec = 64, 8
    rng = np.random.default_rng(0)
    data = rng.standard_normal((nrec, nt)).astype("float32")
    path = tmp_path / "test.segy"
    write_segy_minimal(path, data, dt=0.001)

    offsets = _expected_offsets(nrec, nt)
    with SEGYReader(path, mmap_mode=True) as r:
        a = r.read_trace_data(offsets)
    with SEGYReader(path, mmap_mode=False) as r:
        b = r.read_trace_data(offsets)
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------- MultiFileSEGYReader
def test_multifile_reader_dispatches_correctly(tmp_path):
    nt = 32
    files: list = []
    arrays: list = []
    for k in range(3):
        nrec = 4 + k
        rng = np.random.default_rng(k)
        d = (rng.standard_normal((nrec, nt)) * (k + 1)).astype("float32")
        p = tmp_path / f"f{k}.segy"
        write_segy_minimal(p, d, dt=0.001)
        files.append(p)
        arrays.append(d)

    with MultiFileSEGYReader(files) as mr:
        # Read all traces across all 3 files, interleaved.
        file_ids, byte_offsets, expected_rows = [], [], []
        for k, arr in enumerate(arrays):
            offs = _expected_offsets(arr.shape[0], nt)
            for i in range(arr.shape[0]):
                file_ids.append(k)
                byte_offsets.append(int(offs[i]))
                expected_rows.append(arr[i])
        # Shuffle
        rng = np.random.default_rng(0)
        order = rng.permutation(len(file_ids))
        file_ids = np.asarray(file_ids)[order]
        byte_offsets = np.asarray(byte_offsets)[order]
        expected = np.stack([expected_rows[i] for i in order])

        got = mr.read_traces(file_ids, byte_offsets)
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_multifile_reader_rejects_mismatched_files(tmp_path):
    a = np.zeros((4, 16), dtype="float32")
    b = np.zeros((4, 32), dtype="float32")  # different n_samples
    pa, pb = tmp_path / "a.segy", tmp_path / "b.segy"
    write_segy_minimal(pa, a, dt=0.001)
    write_segy_minimal(pb, b, dt=0.001)
    with pytest.raises(ValueError, match="n_samples"):
        MultiFileSEGYReader([pa, pb])


def test_multifile_reader_needs_at_least_one_path():
    with pytest.raises(ValueError):
        MultiFileSEGYReader([])
