"""torch.distributed helpers for the sweep task layer.

Phase-1 distributed support is *shot parallelism* only: every rank keeps the
full inverted model, runs forward / backward on its slice of the current
shot batch, then all-reduces gradients before stepping the optimiser. This
mirrors `examples/multi-gpu/torch/fwi_marmousi_dist.py` but lifted into the
task layer so any YAML can be launched via ``torchrun``:

    torchrun --nproc_per_node=4 python -m sweep.cli run task.yaml

Spatial decomposition (each rank holding only a chunk of the model with
halo exchange every timestep) is *not* in scope — sweep's compiled
propagator is single-GPU.

When ``WORLD_SIZE`` is unset or equals 1, every helper here degrades to a
no-op, so the same runner code path works in both single-process and
distributed launches.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    is_root: bool
    initialised: bool  # True only when torch.distributed was initialised by us

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def _world_size_from_env() -> int:
    raw = os.environ.get("WORLD_SIZE")
    if not raw:
        return 1
    try:
        return max(int(raw), 1)
    except ValueError:
        return 1


def is_distributed_launch() -> bool:
    """True when launched via torchrun (or any compatible multi-process launcher)."""

    return _world_size_from_env() > 1


def init_distributed_if_needed() -> DistInfo:
    """Initialise the process group if launched under torchrun. Idempotent."""

    world_size = _world_size_from_env()
    if world_size <= 1:
        return DistInfo(rank=0, world_size=1, local_rank=0, is_root=True, initialised=False)

    import torch
    import torch.distributed as dist

    backend = os.environ.get("SWEEP_DIST_BACKEND")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = int(os.environ.get("RANK", "0"))
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif torch.cuda.is_available():
        local_rank = rank % max(torch.cuda.device_count(), 1)
    else:
        local_rank = 0
    # Only set the CUDA device when we have at least that many devices; running
    # multi-rank on a single GPU (or pure CPU with gloo) is also valid.
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
    return DistInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_root=(rank == 0),
        initialised=True,
    )


def cleanup_distributed(info: DistInfo) -> None:
    if not info.initialised:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def resolve_dist_device(device_str: str, local_rank: int) -> "torch.device":
    """Pick the right `torch.device` for this rank.

    `device: auto` becomes `cuda:LOCAL_RANK` when CUDA is available, otherwise
    `cpu`. An explicit `cuda` (no index) gets pinned to `cuda:LOCAL_RANK`;
    explicit `cuda:N` and `cpu` are honoured as-is.
    """

    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    dev = torch.device(device_str)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return dev


def broadcast_object(obj: Any, info: DistInfo, src: int = 0) -> Any:
    """Broadcast a picklable Python object from `src` rank to all ranks."""

    if not info.is_distributed:
        return obj
    import torch.distributed as dist

    holder = [obj] if info.rank == src else [None]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def broadcast_shot_indices(shot_idx, batchsize: int, info: DistInfo, src: int = 0):
    """Broadcast a numpy int64 array of shot indices from `src` to all ranks."""

    if not info.is_distributed:
        return shot_idx

    import numpy as np
    import torch
    import torch.distributed as dist

    if info.rank == src:
        arr = np.asarray(shot_idx, dtype=np.int64)
        if arr.shape[0] != batchsize:
            raise ValueError(
                f"shot_idx length {arr.shape[0]} != expected batchsize {batchsize}."
            )
        buf = torch.from_numpy(arr).clone()
    else:
        buf = torch.zeros(batchsize, dtype=torch.int64)
    if dist.get_backend() == "nccl":
        buf = buf.cuda()
    dist.broadcast(buf, src=src)
    return buf.detach().cpu().numpy().astype(np.int64, copy=False)


def all_reduce_grad_sum(tensors, info: DistInfo) -> None:
    """In-place SUM-reduce each tensor's gradient across ranks.

    For shot-parallel FWI the caller has already divided the local loss by the
    *global* normalisation count (e.g. `batchsize * nt * nrec`), so summing the
    per-rank gradients yields the gradient of the global per-element mean — no
    further division by world_size needed.
    """

    if not info.is_distributed:
        return
    import torch.distributed as dist

    for t in tensors:
        if t.grad is None:
            continue
        dist.all_reduce(t.grad.data, op=dist.ReduceOp.SUM)


def all_reduce_scalar_sum(value: float, info: DistInfo) -> float:
    """SUM-reduce a Python scalar across ranks."""

    if not info.is_distributed:
        return float(value)
    import torch
    import torch.distributed as dist

    buf = torch.tensor([float(value)], dtype=torch.float64)
    if dist.get_backend() == "nccl":
        buf = buf.cuda()
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return float(buf.cpu().item())


def barrier(info: DistInfo) -> None:
    if not info.is_distributed:
        return
    import torch.distributed as dist

    dist.barrier()


def split_for_rank(shot_idx, info: DistInfo):
    """Return the round-robin slice of `shot_idx` assigned to this rank."""

    if not info.is_distributed:
        return shot_idx
    return shot_idx[info.rank::info.world_size]
