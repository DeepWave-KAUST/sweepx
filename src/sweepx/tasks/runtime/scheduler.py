"""Learning-rate scheduler primitives — thin discriminated-union dispatch.

The four shipped variants cover the FWI-common cases:

- ``constant``: no scheduling (returns ``None``).
- ``step``: multiplicative step at fixed intervals.
- ``exp``: per-epoch exponential decay.
- ``cosine``: cosine annealing between ``initial_lr`` and ``eta_min``.

Each variant maps to the corresponding ``torch.optim.lr_scheduler.*`` class.
The wrapper provides a single ``build(spec, optimizer, total_epochs)`` entry
so YAML / Pydantic-driven runners can dispatch without growing a switch
statement in every callsite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ConstantSpec:
    """No scheduling. Returns ``None`` from :func:`build` — the optimizer's
    initial learning rates are used unchanged."""

    kind: Literal["constant"] = "constant"


@dataclass
class StepSpec:
    """Multiply every ``param_group['lr']`` by ``gamma`` every ``step_size`` epochs."""

    step_size: int
    gamma: float = 0.5
    kind: Literal["step"] = "step"


@dataclass
class ExpSpec:
    """Per-epoch exponential decay: ``lr <- lr * gamma`` after each step."""

    gamma: float
    kind: Literal["exp"] = "exp"


@dataclass
class CosineSpec:
    """Cosine annealing from initial lr to ``eta_min`` over ``t_max`` epochs."""

    eta_min: float = 0.0
    t_max: int | None = None  # default: derived from `total_epochs`
    kind: Literal["cosine"] = "cosine"


def build(spec: Any, optimizer: Any, total_epochs: int):
    """Construct a torch LR scheduler from ``spec``.

    ``spec`` is any object exposing a ``kind`` attribute matching one of
    ``"constant" | "step" | "exp" | "cosine"`` plus the relevant fields
    (so a Pydantic discriminated union from a higher-level package
    plugs in without conversion).

    Returns ``None`` for ``kind="constant"``, else a
    ``torch.optim.lr_scheduler._LRScheduler`` instance.
    """
    import torch  # local; sweep_runner avoids touching torch at import time

    kind = getattr(spec, "kind")
    if kind == "constant":
        return None
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(spec.step_size), gamma=float(spec.gamma),
        )
    if kind == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(spec.gamma),
        )
    if kind == "cosine":
        t_max = spec.t_max if spec.t_max is not None else max(int(total_epochs), 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(t_max), eta_min=float(spec.eta_min),
        )
    raise ValueError(f"Unknown scheduler kind {kind!r}")


__all__ = [
    "ConstantSpec",
    "StepSpec",
    "ExpSpec",
    "CosineSpec",
    "build",
]
