"""Coarse-to-fine level masking for the multi-resolution hash encoder.

Wraps :class:`~sweep_nn.hash_encoding.MultiResHashGrid` with a per-level soft
mask so the fine (high-resolution) levels can be frozen early in FWI and
progressively unfrozen — recover the low-wavenumber background first, add
high-wavenumber detail later. This mirrors classical multiscale FWI on the
*model-parameterization* side and follows coarse-to-fine positional-encoding
schemes (BARF, Lin et al. 2021, https://doi.org/10.48550/arXiv.2104.06405;
Neuralangelo, Li et al. 2023, https://doi.org/10.48550/arXiv.2306.03092).

Mechanism: the encoder output ``(..., L*F)`` is multiplied by a per-level mask
in ``[0, 1]``. A masked level (weight 0) contributes nothing to the forward
pass, so its latent grid receives **exactly zero gradient** and stays frozen
at initialization until the schedule unfreezes it. The frontier level gets a
soft (cosine/linear) weight for a smooth unfreeze.

Levels are ordered coarse -> fine (level 0 = ``base_resolution``,
level L-1 = ``finest_resolution``), so opening levels low -> high index is
coarse -> fine.
"""

from __future__ import annotations

import math

import torch

from .hash_encoding import MultiResHashGrid


def coarse_to_fine_weights(
    alpha: float,
    n_levels: int,
    ramp: str = "cosine",
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Per-level weights in ``[0, 1]`` for a continuous active-level count.

    Parameters
    ----------
    alpha
        Number of active levels, a float in ``[0, n_levels]``. The integer part
        indexes fully-open levels (weight 1); the fractional part gives the
        single frontier level a soft weight for a smooth unfreeze.
    n_levels
        Total number of hash levels ``L``.
    ramp
        Frontier shape: ``"cosine"`` (default), ``"linear"``, or ``"hard"``
        (no soft frontier — level is either fully on or fully off).
    """
    L = int(n_levels)
    w = torch.zeros(L, dtype=dtype, device=device)
    if ramp == "hard":
        w[: min(int(math.floor(alpha)), L)] = 1.0
        return w
    for i in range(L):
        t = float(alpha) - i
        if t <= 0.0:
            wi = 0.0
        elif t >= 1.0:
            wi = 1.0
        elif ramp == "linear":
            wi = t
        elif ramp == "cosine":
            wi = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            raise ValueError(f"ramp must be 'cosine', 'linear', or 'hard'; got {ramp!r}")
        w[i] = wi
    return w


class CoarseToFineHashGrid(MultiResHashGrid):
    """``MultiResHashGrid`` whose levels can be masked coarse-to-fine.

    Drop-in replacement for :class:`MultiResHashGrid` — same constructor args,
    plus ``base_levels`` (how many coarsest levels are open at init) and
    ``ramp`` (soft-unfreeze shape). Drive the schedule during training with
    :meth:`set_progress` (epoch fraction), :meth:`set_active_levels` (explicit
    active-level count, e.g. tied to a multiscale frequency stage), or
    :meth:`set_level_weights` (a full mask you build yourself).

    Examples
    --------
    >>> enc = CoarseToFineHashGrid(dim=2, n_levels=8, n_features_per_level=2,
    ...                            base_resolution=64, finest_resolution=512,
    ...                            base_levels=2, ramp="cosine")
    >>> for epoch in range(n_epochs):                      # doctest: +SKIP
    ...     enc.set_progress(epoch / n_epochs, ramp_end=0.6)   # open all by 60%
    ...     features = enc(coords)                          # (..., L*F)
    ...     ...  # SIREN head + FWI forward + backward
    """

    def __init__(
        self,
        dim: int,
        *,
        base_levels: int = 2,
        ramp: str = "cosine",
        **hash_kwargs,
    ) -> None:
        super().__init__(dim, **hash_kwargs)
        self.ramp = str(ramp)
        self.base_levels = max(1, min(int(base_levels), self.L))
        w = torch.zeros(self.L)
        w[: self.base_levels] = 1.0
        # Non-persistent: the schedule state is derived, not a learned param.
        self.register_buffer("level_weights", w, persistent=False)

    # ------------------------------------------------------------------ #
    # mask                                                               #
    # ------------------------------------------------------------------ #
    @property
    def level_mask(self) -> torch.Tensor:
        """``(L*F,)`` mask aligned with the encoder's level-major output."""
        return self.level_weights.repeat_interleave(self.F)

    @property
    def n_active_levels(self) -> float:
        """Effective number of active levels (sum of the soft weights)."""
        return float(self.level_weights.sum().item())

    @torch.no_grad()
    def set_level_weights(self, weights) -> None:
        """Set the per-level mask directly (any tensor/sequence of length L)."""
        self.level_weights.copy_(
            torch.as_tensor(
                weights, dtype=self.level_weights.dtype, device=self.level_weights.device
            )
        )

    def set_active_levels(self, alpha: float, ramp: str | None = None) -> None:
        """Open ``alpha`` levels (float in ``[0, L]``), frontier soft-ramped."""
        self.set_level_weights(
            coarse_to_fine_weights(
                alpha, self.L, ramp or self.ramp,
                dtype=self.level_weights.dtype, device=self.level_weights.device,
            )
        )

    def set_progress(
        self,
        progress: float,
        *,
        base_levels: int | None = None,
        warmup: float = 0.0,
        ramp_end: float = 1.0,
        final_levels: int | None = None,
    ) -> None:
        """Set the mask from training ``progress`` in ``[0, 1]``.

        ``alpha`` ramps ``base_levels`` -> ``final_levels`` (default ``L``)
        linearly over ``[warmup, ramp_end]`` (fractions of training): before
        ``warmup`` only the base levels are open; after ``ramp_end`` all
        ``final_levels`` are open. Capping ``final_levels`` below ``L`` keeps
        the finest levels frozen for the whole stage — use in a multiscale
        chain to reserve them for higher-frequency stages. Call once per
        epoch with ``progress = epoch / n_epochs``.
        """
        base = self.base_levels if base_levels is None else max(1, int(base_levels))
        final = self.L if final_levels is None else min(int(final_levels), self.L)
        final = max(final, base)
        lo = float(warmup)
        hi = max(lo + 1e-9, float(ramp_end))
        frac = min(1.0, max(0.0, (float(progress) - lo) / (hi - lo)))
        self.set_active_levels(base + frac * (final - base))

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """Encode coords ``(..., dim)`` -> masked features ``(..., L*F)``."""
        return super().forward(pos) * self.level_mask


__all__ = ["CoarseToFineHashGrid", "coarse_to_fine_weights"]
