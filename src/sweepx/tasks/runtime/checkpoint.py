"""Atomic checkpoint save/load for ``(model, optimizer, epoch, ...)``.

`model` is treated generically — anything with ``state_dict()`` /
``load_state_dict()`` works. The same goes for `optimizer`. Plain tensors
also work via a tiny adapter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _state_dict(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise TypeError(
        f"Don't know how to serialize {type(obj).__name__}; needs state_dict() or be a Tensor."
    )


def _load_state(obj: Any, sd: Any) -> None:
    if isinstance(obj, torch.Tensor):
        with torch.no_grad():
            obj.copy_(sd)
        return
    if hasattr(obj, "load_state_dict"):
        obj.load_state_dict(sd)
        return
    raise TypeError(
        f"Don't know how to restore {type(obj).__name__}; needs load_state_dict() or be a Tensor."
    )


def save_state(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    epoch: int | None = None,
    **extras: Any,
) -> None:
    """Save a checkpoint atomically (write to ``path.tmp`` then rename).

    `extras` are stored verbatim and returned by ``load_state``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": _state_dict(model)}
    if optimizer is not None:
        payload["optimizer"] = _state_dict(optimizer)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    payload.update(extras)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_state(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint. Returns the full payload (less ``model`` / ``optimizer``)."""
    payload = torch.load(str(path), map_location=map_location)
    _load_state(model, payload["model"])
    if optimizer is not None and "optimizer" in payload:
        _load_state(optimizer, payload["optimizer"])
    return {k: v for k, v in payload.items() if k not in ("model", "optimizer")}


def save_payload(
    path: str | Path,
    payload: dict[str, Any],
    *,
    weights_only: bool | None = None,
) -> Path:
    """Atomically save an arbitrary dict via ``torch.save`` (write ``.tmp`` then rename).

    Lower-level cousin of :func:`save_state` for callers that build their own
    payload dict (e.g. a YAML-driven runner that wants `model_state`,
    `optimizer_state`, `scheduler_state`, `losses`, RNG state, etc., all keyed
    however they like). ``save_state``'s ``model=``/``optimizer=`` kwargs are
    convenient when those are the only two things you serialize, but force a
    schema (`payload["model"]`, `payload["optimizer"]`); this helper imposes
    no schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if weights_only is None:
        torch.save(payload, tmp)
    else:
        torch.save(payload, tmp, _use_new_zipfile_serialization=True)
    os.replace(tmp, path)
    return path


def load_payload(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    """Load a dict written by :func:`save_payload` (or any ``torch.save`` dict).

    ``weights_only`` defaults to ``False`` because most FWI payloads contain
    arbitrary Python objects (e.g. metadata dicts) that the safe loader
    refuses. Set ``True`` if you control the source and want the
    safer torch>=2.4 pickle-restricted path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path} to resume from.")
    return torch.load(str(path), map_location=map_location, weights_only=weights_only)


__all__ = ["save_state", "load_state", "save_payload", "load_payload"]
