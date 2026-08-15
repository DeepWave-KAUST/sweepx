"""Checkpoint save/load + top-row freeze. Verbatim from runner.py."""
from pathlib import Path


def _zero_top_rows(inv_tensors_in_order, n_rows: int) -> None:
    if n_rows <= 0:
        return
    for t in inv_tensors_in_order:
        if t.grad is None:
            continue
        if t.grad.dim() >= 2:
            t.grad[:n_rows].zero_()
        else:
            t.grad[:n_rows] = 0


def _save_checkpoint(task_dir: Path, payload: dict) -> Path:
    """Atomic checkpoint write via `sweep_tasks.runtime.checkpoint.save_payload`."""
    from sweep_tasks.runtime.checkpoint import save_payload
    return save_payload(task_dir / "checkpoint.pt", payload)


def _load_checkpoint(prev_task_dir: Path) -> dict:
    """Counterpart to :func:`_save_checkpoint`. Raises FileNotFoundError if missing."""
    from sweep_tasks.runtime.checkpoint import load_payload
    return load_payload(prev_task_dir / "checkpoint.pt", weights_only=False)
