"""``torch.distributed`` helpers — both stateless and ``DistInfo``-stateful APIs.

Two API styles coexist; pick the one that fits your code:

**Stateless** (the original sweep-runner surface):
    Each function consults ``torch.distributed`` directly. Best for short
    scripts and ad-hoc helpers::

        if is_torchrun():
            init_process_group()
        with only_main() as ok:
            if ok:
                print("rank 0 only")
        avg_loss = reduce_mean(loss.item())

**``DistInfo``-stateful**:
    A small ``DistInfo`` dataclass passed around explicitly. Better for
    long-lived FWI training loops that need richer primitives (broadcast,
    grad all-reduce, shot-index splitting). Brought in from the original
    ``sweep-tasks/_distributed.py`` so that all sister packages converge
    on a single distributed surface::

        info = init_distributed_if_needed()
        try:
            shots = broadcast_shot_indices(shots, batchsize, info)
            mine = split_for_rank(shots, info)
            ...
            all_reduce_grad_sum(model_tensors, info)
        finally:
            cleanup_distributed(info)

When ``WORLD_SIZE`` is unset / ``==1``, every helper degrades to a no-op
that preserves single-process semantics.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.distributed as dist


# =============================================================================
# Stateless API (original sweep-runner surface — unchanged)
# =============================================================================
def is_torchrun() -> bool:
    """``True`` when launched under ``torchrun`` (env vars present)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_process_group(backend: str = "nccl") -> None:
    """Initialize the default process group if running under torchrun.

    Idempotent — safe to call from every script regardless of launch mode.
    """
    if not is_torchrun() or dist.is_initialized():
        return
    # NCCL's watchdog default is 10 minutes, which is shorter than a legitimate
    # stage start: building the observed-coefficient table streams hundreds of
    # GB and the per-shard scale reduction sits inside that window. A band that
    # takes 16 min to stage is not hung, but the watchdog aborts every rank as
    # if it were. Honour the knob fwi_freqsel already reads for its own
    # re-init, so both paths agree.
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=int(
            os.environ.get("SWEEP_DD_NCCL_TIMEOUT_S", "1800"))))


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return rank() == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def pick_device(preferred: str = "cuda") -> torch.device:
    """Pick a device respecting ``LOCAL_RANK`` when running under torchrun."""
    if preferred == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        return torch.device(f"cuda:{local_rank()}")
    return torch.device(preferred)


def reduce_mean(value: float | torch.Tensor) -> float:
    """All-reduce a scalar across ranks and return the mean as Python float."""
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.as_tensor(value, dtype=torch.float64, device=pick_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size())


@contextmanager
def only_main() -> Iterator[bool]:
    """Context manager that yields ``True`` on the main process, ``False`` elsewhere."""
    yield is_main_process()


def barrier(info: "DistInfo | None" = None) -> None:
    """Barrier across ranks. No-op outside torchrun.

    Accepts an optional :class:`DistInfo` for callers using the stateful
    API; when omitted, consults the global ``torch.distributed`` state.
    """
    if info is not None:
        if not info.is_distributed:
            return
        dist.barrier()
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# =============================================================================
# DistInfo-stateful API (imported from the former sweep-tasks/_distributed.py)
# =============================================================================
@dataclass
class DistInfo:
    """Snapshot of the current rank's distributed state."""

    rank: int
    world_size: int
    local_rank: int
    is_root: bool
    initialised: bool  # True only when this process called init_process_group

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
    """``True`` when launched via torchrun (or any compatible multi-process launcher).

    Synonym for :func:`is_torchrun` kept for the ``DistInfo`` API's vocabulary.
    """
    return _world_size_from_env() > 1


def init_distributed_if_needed(backend_env: str = "SWEEP_DIST_BACKEND") -> DistInfo:
    """Initialise the default process group if under torchrun. Idempotent.

    Returns a populated :class:`DistInfo`. Backend selection: looks up
    ``$backend_env`` first, then falls back to ``"nccl"`` when CUDA is
    available else ``"gloo"``.
    """
    world = _world_size_from_env()
    if world <= 1:
        return DistInfo(rank=0, world_size=1, local_rank=0, is_root=True, initialised=False)

    backend = os.environ.get(backend_env)
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rk = int(os.environ.get("RANK", "0"))
    if "LOCAL_RANK" in os.environ:
        lrk = int(os.environ["LOCAL_RANK"])
    elif torch.cuda.is_available():
        lrk = rk % max(torch.cuda.device_count(), 1)
    else:
        lrk = 0
    # Pin to the local GPU only when there's a matching device — allows multi-rank
    # CPU-only / single-GPU runs (gloo) for testing.
    if torch.cuda.is_available() and lrk < torch.cuda.device_count():
        torch.cuda.set_device(lrk)
    return DistInfo(
        rank=rk, world_size=world, local_rank=lrk,
        is_root=(rk == 0), initialised=True,
    )


def cleanup_distributed(info: DistInfo) -> None:
    """Tear down the process group we created (no-op if we didn't)."""
    if not info.initialised:
        return
    if dist.is_initialized():
        dist.destroy_process_group()


def resolve_dist_device(device_str: str, local_rank_: int) -> "torch.device":
    """Map a config string to a ``torch.device`` honouring this rank's local index.

    - ``"auto"``: cuda:LOCAL_RANK if CUDA is available, else cpu.
    - ``"cuda"`` (no index): cuda:LOCAL_RANK.
    - ``"cuda:N"`` / ``"cpu"``: honoured verbatim.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank_}")
        return torch.device("cpu")
    dev = torch.device(device_str)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank_}")
    return dev


def broadcast_object(obj: Any, info: DistInfo, src: int = 0) -> Any:
    """Broadcast a picklable Python object from ``src`` rank to all ranks."""
    if not info.is_distributed:
        return obj
    holder = [obj] if info.rank == src else [None]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def broadcast_shot_indices(shot_idx, batchsize: int, info: DistInfo, src: int = 0):
    """Broadcast a numpy int64 array of shot indices from ``src`` to all ranks.

    Length validation is enforced on the source rank to catch index/batchsize
    drift early.
    """
    if not info.is_distributed:
        return shot_idx

    import numpy as np

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

    For shot-parallel FWI the caller is expected to have already divided
    the local loss by the **global** normalisation count (e.g.
    ``batchsize * nt * nrec``), so summing the per-rank gradients yields
    the gradient of the global per-element mean — no extra division by
    ``world_size`` needed.
    """
    if not info.is_distributed:
        return
    for t in tensors:
        if t.grad is None:
            continue
        dist.all_reduce(t.grad.data, op=dist.ReduceOp.SUM)


def all_reduce_sum_inplace(tensor, info: DistInfo) -> None:
    """In-place SUM-reduce a tensor across ranks (mirrors gradient reduction).

    Unlike :func:`all_reduce_grad_sum` (which iterates a list of param
    tensors and reduces their ``.grad`` attribute) this operates on the
    tensor data itself. Use it for per-iter auxiliary buffers such as
    pseudo-Hessian illumination accumulators that get accumulated chunk-by-
    chunk and then need to be summed across ranks before the gradient
    division step.

    No-op when running single-process (``info.is_distributed`` is false).
    """
    if not info.is_distributed:
        return
    if tensor is None:
        return
    dist.all_reduce(tensor.data, op=dist.ReduceOp.SUM)


def all_reduce_scalar_sum(value: float, info: DistInfo) -> float:
    """SUM-reduce a Python scalar across ranks.

    Differs from :func:`reduce_mean` (stateless API) in two ways:
    (1) returns the *sum* not the mean, so the caller can divide by whatever
    normalisation makes sense; (2) takes an explicit ``DistInfo`` so the
    no-op path doesn't need to consult global ``dist`` state.
    """
    if not info.is_distributed:
        return float(value)
    buf = torch.tensor([float(value)], dtype=torch.float64)
    if dist.get_backend() == "nccl":
        buf = buf.cuda()
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return float(buf.cpu().item())


def split_for_rank(shot_idx, info: DistInfo):
    """Return the round-robin slice of ``shot_idx`` assigned to this rank."""
    if not info.is_distributed:
        return shot_idx
    return shot_idx[info.rank::info.world_size]


__all__ = [
    # stateless
    "is_torchrun",
    "init_process_group",
    "world_size",
    "rank",
    "is_main_process",
    "local_rank",
    "pick_device",
    "reduce_mean",
    "only_main",
    "barrier",
    # DistInfo-stateful
    "DistInfo",
    "is_distributed_launch",
    "init_distributed_if_needed",
    "cleanup_distributed",
    "resolve_dist_device",
    "broadcast_object",
    "broadcast_shot_indices",
    "all_reduce_grad_sum",
    "all_reduce_sum_inplace",
    "all_reduce_scalar_sum",
    "split_for_rank",
]
