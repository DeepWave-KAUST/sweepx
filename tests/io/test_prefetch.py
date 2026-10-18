"""Tests for sweep_io.prefetch — order, errors, lifecycle, overlap."""

import time

import pytest

from sweep_io.prefetch import (
    Prefetcher,
    ThreadPoolPrefetcher,
    TimingPrefetcher,
    prefetch,
)


def test_prefetch_preserves_order():
    with Prefetcher(range(10), queue_depth=3) as pf:
        out = list(pf)
    assert out == list(range(10))


def test_prefetch_handles_empty_source():
    with Prefetcher([], queue_depth=2) as pf:
        out = list(pf)
    assert out == []


def test_prefetch_propagates_exceptions():
    def bad_gen():
        yield 1
        yield 2
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        with Prefetcher(bad_gen(), queue_depth=1) as pf:
            list(pf)


def test_prefetch_rejects_bad_queue_depth():
    with pytest.raises(ValueError):
        Prefetcher(range(3), queue_depth=0)


def test_prefetch_close_is_idempotent():
    pf = Prefetcher(range(100), queue_depth=2)
    next(pf)
    pf.close()
    pf.close()  # should not raise
    # After close, no more iteration
    with pytest.raises(StopIteration):
        next(pf)


def test_prefetch_overlaps_with_consumer():
    """If load takes Lt and consume takes Lc, total ≈ max(Lt, Lc) per item
    (plus startup overhead), not Lt + Lc."""
    Lt = 0.02  # seconds per item, "I/O" side
    Lc = 0.02  # seconds per item, "compute" side
    N = 8

    def slow_iter():
        for i in range(N):
            time.sleep(Lt)
            yield i

    t0 = time.perf_counter()
    with Prefetcher(slow_iter(), queue_depth=2) as pf:
        for _ in pf:
            time.sleep(Lc)
    dt = time.perf_counter() - t0
    # Without prefetch: N * (Lt + Lc) = 8 * 0.04 = 0.32 s
    # With prefetch:    Lt + N * max(Lt, Lc) ≈ 0.02 + 8 * 0.02 = 0.18 s
    # Generous threshold (CI jitter):
    assert dt < N * (Lt + Lc) * 0.8, f"prefetch overlap looks ineffective: {dt:.3f}s"


def test_threadpool_prefetcher_preserves_order():
    def load(i): return i * i
    with ThreadPoolPrefetcher(load, range(20), num_workers=4, queue_depth=4) as pf:
        out = list(pf)
    assert out == [i * i for i in range(20)]


def test_threadpool_prefetcher_uses_multiple_workers_for_parallel_loads():
    """4 workers + 0.05s per load → 8 loads should take ~0.1s, not ~0.4s."""
    N = 8
    Lt = 0.05

    def slow_load(i):
        time.sleep(Lt)
        return i

    t0 = time.perf_counter()
    with ThreadPoolPrefetcher(slow_load, range(N), num_workers=4, queue_depth=4) as pf:
        out = list(pf)
    dt = time.perf_counter() - t0
    assert out == list(range(N))
    # Serial would be N*Lt = 0.4s; 4 workers should hit ~N/4 * Lt = 0.1s.
    # Loose threshold for CI jitter.
    assert dt < N * Lt * 0.6, f"4 workers didn't parallelize: {dt:.3f}s"


def test_threadpool_prefetcher_invalid_args():
    with pytest.raises(ValueError):
        ThreadPoolPrefetcher(lambda i: i, range(3), num_workers=0)
    with pytest.raises(ValueError):
        ThreadPoolPrefetcher(lambda i: i, range(3), queue_depth=0)


def test_timing_prefetcher_records_waits():
    with TimingPrefetcher(range(10), queue_depth=2) as pf:
        for _ in pf:
            pass
    assert len(pf.wait_times) == 10
    assert all(w >= 0 for w in pf.wait_times)


def test_prefetch_helper():
    with prefetch(range(5)) as pf:
        out = list(pf)
    assert out == [0, 1, 2, 3, 4]
