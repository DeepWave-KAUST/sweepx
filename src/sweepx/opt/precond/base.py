"""Preconditioner abstract base class.

A preconditioner is anything that pre-multiplies a gradient before an
optimizer's ``step()`` consumes it. Subclasses must implement two methods:

- :meth:`update` is called once per outer iteration and refreshes any
  internal state (illumination tensors, running estimates, ...).
- :meth:`__call__` is called once per parameter on every step and returns the
  preconditioned gradient.

Subclasses are expected to be **stateless across instances** - they receive
the new state through ``update`` rather than reaching back into the optimizer.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor


class Preconditioner:
    """Abstract preconditioner.

    The default implementation acts as the identity. Subclasses override
    :meth:`update` and :meth:`__call__`.
    """

    def update(
        self,
        params: Iterable[Tensor],
        *,
        source_illum: Optional[object] = None,
        receiver_illum: Optional[object] = None,
        **kwargs,
    ) -> None:
        """Refresh internal state. Default no-op."""
        return

    def __call__(self, grad: Tensor, param: Tensor) -> Tensor:  # noqa: D401
        """Return the preconditioned gradient. Default identity."""
        return grad


class IdentityPrecond(Preconditioner):
    """Explicit identity preconditioner (useful for unit tests / clarity)."""
