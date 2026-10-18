"""Parameterization abstract base class.

A parameterization decouples the optimization variable (the *latent*) from
the physical variable that the forward solver consumes. The contract::

    physical = forward(latent)
    latent   = inverse(physical)

Any optimizer can be wrapped to optimize over a latent: the closure (or
loss_fn) calls ``solver(forward(latent))`` and the optimizer iterates
``latent``. Bounds, log-spacing, deep image priors, etc. all fit this
contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Parameterization(ABC):
    """Abstract reparameterization between latent and physical model."""

    @abstractmethod
    def forward(self, latent: Tensor) -> Tensor:
        """Map the latent variable to the physical model."""

    @abstractmethod
    def inverse(self, physical: Tensor) -> Tensor:
        """Recover the latent that maps to the given physical model."""


class IdentityParameterization(Parameterization):
    """Pass-through. Useful as a default and as a sanity baseline."""

    def forward(self, latent: Tensor) -> Tensor:
        return latent

    def inverse(self, physical: Tensor) -> Tensor:
        return physical
