"""CUDA-aware prefetching with pinned host buffers and a side stream.

Sits on top of :mod:`sweep_io.prefetch` and adds two GPU-specific tricks:

1. **Pinned host buffers.** Page-locked memory is required for true
   async H2D copies. Without it, ``tensor.to(device, non_blocking=True)``
   silently degrades to a synchronous copy.

2. **Side-stream H2D.** The copy is issued on a dedicated CUDA stream
   so it overlaps with whatever the compute (main) stream is doing.
   The consumer waits on a CUDA event the producer recorded — the wait
   happens GPU-side, not host-side, so the host stays free.

For shot-gather-sized payloads (10s of MB) the round-trip can drop from
~10 ms per shot to ~0 ms (perfectly overlapped). For multi-GB batches the
PCIe link itself becomes the bottleneck — the side stream still runs at
~50% of compute time, but compute and copy continue to overlap.

Optional dep: ``torch``.
"""

from __future__ import annotations

from typing import Any, Iterable

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sweep_io.cuda_prefetch requires `torch`. "
        "Install with `pip install sweep-io[torch]`."
    ) from e

import numpy as np

from .prefetch import Prefetcher


def _to_pinned_tensor(x: Any) -> torch.Tensor:
    """Cast ``x`` (ndarray / tensor / scalar) to a CPU tensor in pinned memory."""
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if not t.is_pinned() and t.device.type == "cpu":
            t = t.pin_memory()
        return t
    return torch.from_numpy(np.ascontiguousarray(x)).pin_memory()


def _stage_to_device(item: Any, device: torch.device) -> Any:
    """Walk a nested item, moving tensors / arrays to ``device`` non-blockingly.

    Supports plain tensors / arrays, dicts, lists, tuples. Anything else
    is passed through.
    """
    if isinstance(item, (torch.Tensor, np.ndarray)) or np.isscalar(item):
        try:
            t = _to_pinned_tensor(item) if not isinstance(item, torch.Tensor) or item.device.type == "cpu" else item
        except TypeError:
            # numpy scalar or python scalar — small enough to skip pinning
            return torch.as_tensor(item, device=device)
        return t.to(device, non_blocking=True)
    if isinstance(item, dict):
        return {k: _stage_to_device(v, device) for k, v in item.items()}
    if isinstance(item, tuple):
        return tuple(_stage_to_device(v, device) for v in item)
    if isinstance(item, list):
        return [_stage_to_device(v, device) for v in item]
    return item


class CUDAPrefetcher:
    """Wrap an iterable so each yielded item lands on GPU before compute uses it.

    Parameters
    ----------
    source
        Any iterable. Items can be numpy arrays, torch tensors, or
        nested dict/list/tuple of those. Non-tensor leaves pass through.
    device
        Target CUDA device (``"cuda"``, ``"cuda:0"``, ``torch.device``).
    queue_depth
        How many items the background thread keeps ready. ``2`` for
        double buffering; ``3+`` if compute is very fast / bursty.
    stream
        CUDA stream to issue H2D on. Defaults to a fresh stream on
        ``device``.

    Notes
    -----
    - The consumer must use the returned tensors on the **current**
      stream. We insert a GPU-side wait so the current stream sees the
      copy as completed; the host returns immediately.
    - For maximum throughput, pair this with
      ``torch.cuda.set_per_process_memory_fraction`` or a persistent
      pool to avoid pinned-allocator pressure under high queue depth.

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>> def load(i): return np.random.randn(2048, 256).astype("float32")
    >>> pf = CUDAPrefetcher((load(i) for i in range(N)),
    ...                     device="cuda:0", queue_depth=2)   # doctest: +SKIP
    >>> for shot in pf:                                       # doctest: +SKIP
    ...     pred = solver(shot)                               # doctest: +SKIP
    """

    def __init__(
        self,
        source: Iterable[Any],
        *,
        device: str | torch.device = "cuda",
        queue_depth: int = 2,
        stream: "torch.cuda.Stream | None" = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"CUDAPrefetcher needs a CUDA device; got {self.device}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAPrefetcher needs CUDA, but torch.cuda is unavailable.")
        self._stream = stream or torch.cuda.Stream(device=self.device)
        # Inner prefetcher receives (item, event) tuples.
        self._inner = Prefetcher(
            self._stage_iter(source),
            queue_depth=queue_depth,
            name="CUDAPrefetcher",
        )

    def _stage_iter(self, source: Iterable[Any]):
        """Yield ``(staged_item, copy_done_event)`` per source item.

        The H2D copy is issued on the side stream and we record an event
        immediately after. The event is what the consumer waits on.
        """
        for item in source:
            with torch.cuda.stream(self._stream):
                staged = _stage_to_device(item, self.device)
                event = torch.cuda.Event(blocking=False)
                event.record(self._stream)
            yield (staged, event)

    def __iter__(self) -> "CUDAPrefetcher":
        return self

    def __next__(self) -> Any:
        staged, event = next(self._inner)
        # GPU-side wait: enqueue a wait on the *current* stream so the
        # next kernel sees the copied data as ready, without blocking
        # the host.
        torch.cuda.current_stream(self.device).wait_event(event)
        return staged

    def __enter__(self) -> "CUDAPrefetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._inner.close()


__all__ = ["CUDAPrefetcher"]
