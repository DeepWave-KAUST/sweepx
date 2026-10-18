"""Background-thread prefetching primitives.

Lets you overlap I/O (or any slow per-item work) with downstream compute.
Pure stdlib — no numpy, no torch — so this module is import-safe even
when the rest of the package's optional deps aren't installed.

Two flavors:

``Prefetcher``
    Single background thread consumes an iterable and pushes items into
    a bounded queue. Best when each item's load is itself fast or
    serial (e.g. one big SEG-Y read).

``ThreadPoolPrefetcher``
    A pool of N workers, each fetching items in parallel by index.
    Best when individual loads are slow but independent and the
    bottleneck isn't a single file's bandwidth (e.g. many SEG-Y files,
    or many random byte-offset reads in one file with `pread`).

Both:
- yield items **in order**;
- propagate exceptions from the worker side to the consumer side cleanly;
- shut down their threads on ``close()`` / context-manager exit.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generic, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

_STOP = object()  # sentinel pushed by worker on clean EOS


class _Err:
    """Wrapper for a worker-thread exception ferried over the queue."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class Prefetcher(Generic[T]):
    """Yield items from ``source`` while a background thread reads ahead.

    Parameters
    ----------
    source
        Any iterable. Must be safe to consume from a single background
        thread (most generators are; only iterators with hard
        thread-affinity requirements are not).
    queue_depth
        Maximum number of items the worker may hold ahead of the
        consumer. ``2`` is the canonical double-buffered default.
        Higher values trade peak RAM for more slack against bursty
        compute / I/O.
    name
        Optional thread name (helps in py-spy / gdb).
    daemon
        Whether the worker is a daemon thread. Default ``True`` so
        Python doesn't wait for it on interpreter exit.

    Lifecycle
    ---------
    The worker starts in ``__init__`` and runs until the source is
    exhausted, an exception is raised, or ``close()`` is called.
    Reusing a consumed instance is **not** supported; build a new one.

    Examples
    --------
    >>> def load(i):
    ...     # imagine slow disk I/O here
    ...     return i * 2
    >>> with Prefetcher((load(i) for i in range(5))) as pf:
    ...     out = list(pf)
    >>> out
    [0, 2, 4, 6, 8]
    """

    def __init__(
        self,
        source: Iterable[T],
        *,
        queue_depth: int = 2,
        name: str | None = None,
        daemon: bool = True,
    ) -> None:
        if queue_depth < 1:
            raise ValueError(f"queue_depth must be >= 1; got {queue_depth}")
        self._source_iter = iter(source)
        self._q: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name=name or "Prefetcher", daemon=daemon
        )
        self._thread.start()

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        try:
            for item in self._source_iter:
                if self._stop.is_set():
                    return
                # Block until consumer makes room — backpressure.
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # noqa: BLE001 — ferry everything
            try:
                self._q.put(_Err(e), timeout=1.0)
            except queue.Full:
                pass
            return
        # Clean EOS
        try:
            self._q.put(_STOP, timeout=1.0)
        except queue.Full:
            pass

    # ------------------------------------------------------------- iter / ctx
    def __iter__(self) -> "Prefetcher[T]":
        return self

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        item = self._q.get()
        if item is _STOP:
            self._closed = True
            raise StopIteration
        if isinstance(item, _Err):
            self._closed = True
            raise item.exc
        return item  # type: ignore[return-value]

    def __enter__(self) -> "Prefetcher[T]":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self, *, timeout: float = 1.0) -> None:
        """Signal the worker to stop, drain the queue, join the thread."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # Wake the worker if it's blocked on a full queue
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)


class ThreadPoolPrefetcher(Generic[T]):
    """Order-preserving prefetcher with ``num_workers`` parallel loaders.

    Use this when each item is loaded by an independent call to a
    function ``load(idx)``, the calls are I/O-bound (so threading
    helps), and there's enough parallel bandwidth to benefit from
    multiple in-flight reads.

    Parameters
    ----------
    load_fn
        Callable ``(idx) -> item``. Must be thread-safe; if it mutates
        shared state, you must serialize access yourself.
    indices
        Sequence of indices to feed ``load_fn``. Items are yielded in
        the order of this sequence (not the order workers finish).
    num_workers
        Worker thread count. ``2`` is a sensible default for OS-cached
        spinning disk; ``4-8`` for NVMe or Lustre.
    queue_depth
        How far ahead of the consumer the worker pool may run.
        Roughly: peak RAM ≈ ``queue_depth × item_size``.

    Examples
    --------
    >>> def load(i): return i * i
    >>> with ThreadPoolPrefetcher(load, range(5), num_workers=2) as pf:
    ...     out = list(pf)
    >>> out
    [0, 1, 4, 9, 16]
    """

    def __init__(
        self,
        load_fn: Callable[[int], T],
        indices: Sequence[int] | Iterable[int],
        *,
        num_workers: int = 2,
        queue_depth: int = 2,
        name: str | None = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1; got {num_workers}")
        if queue_depth < 1:
            raise ValueError(f"queue_depth must be >= 1; got {queue_depth}")
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix=name or "PrefetcherWorker"
        )
        self._load_fn = load_fn
        # Materialize indices so we can iterate in order.
        self._indices = list(indices)
        self._queue_depth = max(queue_depth, num_workers)
        self._closed = False

    def __iter__(self) -> Iterator[T]:
        pending: dict[int, object] = {}  # future_idx -> Future
        next_to_submit = 0
        next_to_yield = 0
        n = len(self._indices)
        try:
            # Prime
            while next_to_submit < n and len(pending) < self._queue_depth:
                idx = self._indices[next_to_submit]
                pending[next_to_submit] = self._executor.submit(self._load_fn, idx)
                next_to_submit += 1

            while next_to_yield < n:
                fut = pending.pop(next_to_yield)
                result = fut.result()  # type: ignore[union-attr]
                next_to_yield += 1
                # Replenish the pipeline before yielding so the caller's
                # compute can overlap with the *next* load.
                if next_to_submit < n:
                    idx = self._indices[next_to_submit]
                    pending[next_to_submit] = self._executor.submit(
                        self._load_fn, idx
                    )
                    next_to_submit += 1
                yield result  # type: ignore[misc]
        finally:
            self.close()

    def __enter__(self) -> "ThreadPoolPrefetcher[T]":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the executor, cancelling any pending futures."""
        if self._closed:
            return
        self._closed = True
        # cancel_futures requires Python 3.9+, which we already require.
        self._executor.shutdown(wait=False, cancel_futures=True)


def prefetch(
    source: Iterable[T],
    *,
    queue_depth: int = 2,
) -> Prefetcher[T]:
    """Tiny convenience wrapper: ``for x in prefetch(my_iter): ...``."""
    return Prefetcher(source, queue_depth=queue_depth)


# ---------------------------------------------------------------- diagnostics
class TimingPrefetcher(Prefetcher[T]):
    """Variant that also records per-item wait times.

    Records, on the consumer side, how long ``__next__`` blocked waiting
    for the worker. ``wait_times`` accumulates one entry per consumed
    item. Useful for benchmarking the prefetch / compute overlap:

    - all-zero wait times  → consumer is the bottleneck (compute-bound)
    - large wait times     → worker can't keep up (I/O-bound)
    """

    def __init__(self, source: Iterable[T], *, queue_depth: int = 2) -> None:
        super().__init__(source, queue_depth=queue_depth, name="TimingPrefetcher")
        self.wait_times: list[float] = []

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        t0 = time.perf_counter()
        item = self._q.get()
        dt = time.perf_counter() - t0
        if item is _STOP:
            self._closed = True
            raise StopIteration
        if isinstance(item, _Err):
            self._closed = True
            raise item.exc
        self.wait_times.append(dt)
        return item  # type: ignore[return-value]


__all__ = [
    "Prefetcher",
    "ThreadPoolPrefetcher",
    "TimingPrefetcher",
    "prefetch",
]


# When run as a script: tiny smoke test you can pipe stderr.
if __name__ == "__main__":  # pragma: no cover
    def slow(i: int) -> int:
        time.sleep(0.01)
        return i

    with TimingPrefetcher((slow(i) for i in range(20))) as pf:
        for x in pf:
            time.sleep(0.01)
    sys.stderr.write(
        f"mean wait = {sum(pf.wait_times) / len(pf.wait_times) * 1e3:.2f} ms\n"
    )
