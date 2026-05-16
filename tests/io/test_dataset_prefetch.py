"""Tests for ShotGatherDataset's prefetch hooks (skipped when torch missing)."""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep_io.datasets import (  # noqa: E402
    PrefetchingShotDataset,
    ShotGatherDataset,
)
from sweep_io.geometry import Geometry  # noqa: E402


def _make_dataset(nshots: int = 6, nrec: int = 8, nt: int = 32):
    sources = np.stack(
        [np.linspace(10, 100, nshots, dtype="int64"), np.zeros(nshots, dtype="int64")],
        axis=1,
    )
    receivers = np.broadcast_to(
        np.stack(
            [np.arange(nrec, dtype="int64"), np.zeros(nrec, dtype="int64")], axis=1
        ),
        (nshots, nrec, 2),
    ).copy()
    geom = Geometry(sources=sources, receivers=receivers, dt=0.001, nt=nt)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((nshots, nt, nrec)).astype("float32")
    return ShotGatherDataset(geom, obs), obs


def test_iter_prefetched_preserves_order():
    ds, obs = _make_dataset()
    seen = []
    for sample in ds.iter_prefetched(queue_depth=2):
        seen.append(sample["shot_index"])
        np.testing.assert_array_equal(sample["obs"].numpy(), obs[sample["shot_index"]])
    assert seen == list(range(len(ds)))


def test_iter_prefetched_thread_pool():
    ds, _ = _make_dataset(nshots=8)
    seen = [s["shot_index"] for s in ds.iter_prefetched(num_workers=2, queue_depth=2)]
    assert seen == list(range(8))


def test_prefetching_dataset_iterable_view():
    ds, obs = _make_dataset()
    view = PrefetchingShotDataset(ds, queue_depth=2)
    assert len(view) == len(ds)
    seen = list(view)
    assert [s["shot_index"] for s in seen] == list(range(len(ds)))


def test_iter_prefetched_with_slow_obs_overlaps():
    """If obs callable sleeps 20ms and compute sleeps 20ms, total < serial."""
    nshots, nt, nrec = 6, 16, 4
    sources = np.zeros((nshots, 2), dtype="int64")
    receivers = np.zeros((nshots, nrec, 2), dtype="int64")
    geom = Geometry(sources=sources, receivers=receivers, dt=0.001, nt=nt)

    def slow_obs(i: int) -> np.ndarray:
        time.sleep(0.02)
        return np.zeros((nt, nrec), dtype="float32") + i

    ds = ShotGatherDataset(geom, slow_obs)

    t0 = time.perf_counter()
    for _ in ds.iter_prefetched(queue_depth=2):
        time.sleep(0.02)  # fake compute
    dt = time.perf_counter() - t0
    # Serial: nshots * (0.02 + 0.02) = 0.24 s
    # Overlapped: ~0.02 + 0.02 * nshots ≈ 0.14 s
    serial = nshots * 0.04
    assert dt < serial * 0.8, f"prefetch overlap looks ineffective: dt={dt:.3f}s"


def test_iter_prefetched_with_subset_indices():
    ds, _ = _make_dataset(nshots=10)
    seen = [s["shot_index"] for s in ds.iter_prefetched([7, 3, 0])]
    assert seen == [7, 3, 0]
