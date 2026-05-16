"""Torch dataset wrappers — one sample per shot, with optional prefetching.

Optional dependency. Importing this module without `torch` installed
raises a clear error.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sweep_io.datasets requires `torch`. Install with `pip install sweep-io[torch]`."
    ) from e

from .geometry import Geometry
from .prefetch import Prefetcher, ThreadPoolPrefetcher


class ShotGatherDataset(Dataset):
    """Per-shot dataset: each item is ``(sources_i, receivers_i, obs_i)``.

    Parameters
    ----------
    geometry
        Acquisition geometry. Provides ``sources`` and ``receivers`` slices.
    obs
        Observed data, either:
        - a numpy array / torch tensor of shape ``(nshots, ...)``, or
        - a callable ``obs(shot_index) -> array_like`` for out-of-core access
          (e.g. one SEG-Y per shot).
    transform
        Optional ``transform(sample) -> sample`` applied lazily.
    """

    def __init__(
        self,
        geometry: Geometry,
        obs: np.ndarray | "torch.Tensor" | Callable[[int], Any],
        *,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.geometry = geometry
        self.obs = obs
        self.transform = transform

    def __len__(self) -> int:
        return self.geometry.nshots

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if callable(self.obs):
            obs_i = self.obs(idx)
        else:
            obs_i = self.obs[idx]
        sample = {
            "sources": torch.as_tensor(self.geometry.sources[idx]),
            "receivers": torch.as_tensor(self.geometry.receivers[idx]),
            "obs": torch.as_tensor(np.asarray(obs_i)),
            "shot_index": idx,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    # ------------------------------------------------------------------ helpers
    def iter_prefetched(
        self,
        indices: Sequence[int] | Iterable[int] | None = None,
        *,
        queue_depth: int = 2,
        num_workers: int = 0,
    ) -> Iterator[dict[str, Any]]:
        """Iterate samples with a background prefetch thread.

        Parameters
        ----------
        indices
            Sequence of shot indices to iterate. Defaults to
            ``range(len(self))``.
        queue_depth
            How many samples to keep ready ahead of the consumer.
        num_workers
            ``0`` → single background thread (good when the underlying
            ``__getitem__`` does one big I/O call).
            ``>0`` → a thread pool of that size (good when ``__getitem__``
            can run in parallel, e.g. distinct SEG-Y files per shot).

        Notes
        -----
        This is **not** ``torch.utils.data.DataLoader``. There's no batch
        collation here — one item per ``next``. Wrap with your own
        batching if you want it. For the typical FWI loop you want one
        shot at a time anyway, so the loop is:

            for sample in ds.iter_prefetched():
                pred = solver(sample["sources"], sample["receivers"], ...)
                ...
        """
        if indices is None:
            indices = range(len(self))
        if num_workers <= 0:
            return iter(
                Prefetcher(
                    (self[i] for i in indices),
                    queue_depth=queue_depth,
                )
            )
        return iter(
            ThreadPoolPrefetcher(
                load_fn=lambda i: self[i],
                indices=indices,
                num_workers=num_workers,
                queue_depth=queue_depth,
            )
        )


class PrefetchingShotDataset:
    """Lightweight iterable view over a ``ShotGatherDataset`` with prefetch.

    Useful when your training loop is purely iterative (no random
    access, no shuffling) — e.g. a multi-scale FWI epoch over all shots.
    Contrast with :class:`ShotGatherDataset` which is a
    ``torch.utils.data.Dataset`` (random-access) you'd feed to a
    ``DataLoader``.

    Example
    -------
    >>> ds = ShotGatherDataset(geom, obs)                       # doctest: +SKIP
    >>> for shot in PrefetchingShotDataset(ds, queue_depth=2):  # doctest: +SKIP
    ...     train_step(shot)                                    # doctest: +SKIP
    """

    def __init__(
        self,
        dataset: ShotGatherDataset,
        *,
        indices: Sequence[int] | Iterable[int] | None = None,
        queue_depth: int = 2,
        num_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.indices = (
            list(range(len(dataset))) if indices is None else list(indices)
        )
        self.queue_depth = queue_depth
        self.num_workers = num_workers

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.dataset.iter_prefetched(
            self.indices,
            queue_depth=self.queue_depth,
            num_workers=self.num_workers,
        )

    def __len__(self) -> int:
        return len(self.indices)


def collate_shots(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Default collate: stack sources/receivers/obs into a batch dim.

    Pass this as ``collate_fn=collate_shots`` when constructing a ``DataLoader``.
    """
    return {
        "sources": torch.stack([b["sources"] for b in batch]),
        "receivers": torch.stack([b["receivers"] for b in batch]),
        "obs": torch.stack([b["obs"] for b in batch]),
        "shot_index": [b["shot_index"] for b in batch],
    }


__all__ = [
    "ShotGatherDataset",
    "PrefetchingShotDataset",
    "collate_shots",
]
