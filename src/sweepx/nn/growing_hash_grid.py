"""On-demand *growing* multi-resolution hash-grid encoder.

Same encoding as :class:`~sweep_nn.hash_encoding.MultiResHashGrid` (Instant-NGP,
Müller et al. 2022, https://doi.org/10.1145/3528223.3530127) but the fine levels
are **allocated lazily**. Where :class:`~sweep_nn.coarse_to_fine.CoarseToFineHashGrid`
allocates every level up front and freezes the fine ones behind a zero mask,
``GrowingHashGrid`` starts with only ``initial_levels`` and instantiates each
finer level's latent table on the first :meth:`grow` call.

Why: in 3-D the finest levels dominate the latent budget (their dense grids hit
the ``2**log2_hashmap_size`` cap), so a masked-but-allocated fine level costs the
same memory as an active one. Growing on demand pays for a level only once the
coarse-to-fine schedule actually needs it — the key lever when the hash table is
the thing that OOMs.

The output width is **fixed** at ``max_levels * n_features_per_level`` for the
lifetime of the module (levels not yet grown emit zeros), so the downstream
SIREN/MLP head never has to be resized as the grid grows.

Coords come in normalized to ``[0, 1)`` with shape ``(..., dim)`` (``dim`` 2/3);
output is ``(..., max_levels * F)``. Levels are ordered coarse -> fine (level 0 =
``base_resolution``, level ``max_levels-1`` = ``finest_resolution``).

Usage::

    enc = GrowingHashGrid(dim=3, max_levels=16, initial_levels=6, ...)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    ...                                   # train with 6 levels
    new = enc.grow(1)                     # allocate level 6
    opt.add_param_group({"params": new})  # register the new latents with Adam
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def _next_multiple(value: int, step: int) -> int:
    return (int(value) + int(step) - 1) // int(step) * int(step)


class GrowingHashGrid(nn.Module):
    """Multi-resolution hash grid whose fine levels are allocated on demand.

    Numerically identical to :class:`MultiResHashGrid` once fully grown; the
    per-level latent table for level ``i`` corresponds to the flat table slice
    ``MultiResHashGrid.latents[offsets[i]:offsets[i+1]]``.
    """

    def __init__(
        self,
        dim: int,
        *,
        max_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int | List[int] = 16,
        finest_resolution: int | List[int] = 512,
        initial_levels: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if int(dim) not in (2, 3):
            raise ValueError(f"dim must be 2 or 3; got {dim}")

        self.dim = int(dim)
        self.L = int(max_levels)
        self.F = int(n_features_per_level)
        self.T = 2 ** int(log2_hashmap_size)
        self.output_dim = self.L * self.F
        self.n_output_dims = self.output_dim  # legacy alias
        self._dtype = dtype

        if isinstance(base_resolution, int):
            base_resolution = [int(base_resolution)] * self.dim
        if isinstance(finest_resolution, int):
            finest_resolution = [int(finest_resolution)] * self.dim
        base_resolution = [int(v) for v in base_resolution]
        finest_resolution = [int(v) for v in finest_resolution]
        if len(base_resolution) != self.dim or len(finest_resolution) != self.dim:
            raise ValueError("base_resolution / finest_resolution must match dim")

        # Per-axis geometric growth factor across the L levels.
        b = [
            math.exp(
                (math.log(float(finest_resolution[d])) - math.log(float(base_resolution[d])))
                / max(1, self.L - 1)
            )
            for d in range(self.dim)
        ]

        # Precompute the geometry of ALL L levels up front (cheap: shape (L, dim));
        # only the latent tables are allocated lazily.
        scales: List[List[float]] = []
        resolutions: List[List[int]] = []
        strides: List[List[int]] = []
        n_entries: List[int] = []
        is_dense: List[bool] = []
        for level in range(self.L):
            scale_i = [float(base_resolution[d]) * (b[d] ** level) - 1.0 for d in range(self.dim)]
            res_i = [int(math.ceil(s) + 1) for s in scale_i]
            prod = 1
            for r in res_i:
                prod *= r
            dense = prod <= self.T
            stride_i = [1]
            for r in res_i[:-1]:
                stride_i.append(stride_i[-1] * int(r))
            scales.append(scale_i)
            resolutions.append(res_i)
            strides.append(stride_i)
            n_entries.append(_next_multiple(prod, 8) if dense else self.T)
            is_dense.append(dense)

        self.n_entries = n_entries          # python list — per-level table size
        self.is_dense = is_dense
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32), persistent=False)
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.int32), persistent=False)
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.long), persistent=False)
        # Knuth / MurmurHash primes — constants.
        self.register_buffer(
            "primes", torch.tensor([1, 2_654_435_761, 805_459_861], dtype=torch.long), persistent=False
        )
        if self.dim == 2:
            cell_offsets = [[0, 0], [0, 1], [1, 0], [1, 1]]
        else:
            cell_offsets = [
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
            ]
        self.register_buffer("cell_offsets", torch.tensor(cell_offsets, dtype=torch.float32), persistent=False)

        # Latent tables, one nn.Parameter per grown level; grows via .grow().
        self.initial_levels = max(1, min(int(initial_levels), self.L))
        self.latents = nn.ParameterList()
        self._allocated = 0
        for _ in range(self.initial_levels):
            self._add_one_level()

    # ---- growth -----------------------------------------------------------

    def _add_one_level(self) -> nn.Parameter:
        i = self._allocated
        p = nn.Parameter(
            torch.empty(int(self.n_entries[i]), self.F, dtype=self._dtype).uniform_(-1.0e-4, 1.0e-4)
        )
        p.data = p.data.to(self.scales.device)  # match the module's device
        self.latents.append(p)
        self._allocated += 1
        return p

    def grow(self, n: int = 1) -> List[nn.Parameter]:
        """Instantiate the next ``n`` finer levels (capped at ``max_levels``).

        Returns the newly created :class:`~torch.nn.Parameter` tables so the
        caller can register them with a live optimizer via
        ``optimizer.add_param_group({"params": new})``.
        """
        new: List[nn.Parameter] = []
        for _ in range(int(n)):
            if self._allocated < self.L:
                new.append(self._add_one_level())
        return new

    def grow_to_progress(
        self,
        progress: float,
        *,
        base_levels: int | None = None,
        final_levels: int | None = None,
        warmup: float = 0.0,
        ramp_end: float = 1.0,
    ) -> List[nn.Parameter]:
        """Grow to the level count a coarse-to-fine schedule wants at ``progress``.

        Mirrors :meth:`~sweep_nn.coarse_to_fine.CoarseToFineHashGrid.set_progress`:
        an ``alpha`` ramps ``base_levels`` -> ``final_levels`` linearly over
        ``[warmup, ramp_end]`` (fractions of training). A level is *allocated*
        once ``alpha`` reaches it (``floor(alpha)``), so memory is spent lazily
        as the schedule advances. Never ungrows. Returns any newly created
        :class:`~torch.nn.Parameter` tables — register them with a live optimizer
        via ``add_param_group``. Call once per epoch with
        ``progress = epoch / (n_epochs - 1)``.
        """
        base = self.initial_levels if base_levels is None else max(1, int(base_levels))
        final = self.L if final_levels is None else min(int(final_levels), self.L)
        final = max(final, base)
        lo = float(warmup)
        hi = max(lo + 1e-9, float(ramp_end))
        frac = min(1.0, max(0.0, (float(progress) - lo) / (hi - lo)))
        alpha = base + frac * (final - base)
        target = min(self.L, int(math.floor(alpha + 1e-9)))
        if target > self._allocated:
            return self.grow(target - self._allocated)
        return []

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Resume-aware load: grow the ParameterList to the checkpoint's level
        count *before* the standard load, so a run that grew to K levels can be
        loaded into a freshly-constructed grid (e.g. warm-starting a
        higher-frequency band from a saved lower-frequency run — two separate
        processes). Never shrinks, never exceeds ``max_levels``.

        Runs for both ``encoder.load_state_dict(...)`` and the recursive
        ``parent.load_state_dict(...)`` (which invokes this hook per submodule).
        """
        key = prefix + "latents."
        idxs = [int(k[len(key):].split(".", 1)[0])
                for k in state_dict if k.startswith(key)]
        n_ckpt = (max(idxs) + 1) if idxs else 0
        while self._allocated < min(n_ckpt, self.L):
            self._add_one_level()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def n_active(self) -> int:
        """Number of levels currently allocated (coarse levels 0 .. n_active-1)."""
        return self._allocated

    def n_params(self) -> int:
        """Total latent entries allocated so far (across grown levels)."""
        return sum(p.numel() for p in self.latents)

    # ---- encoding ---------------------------------------------------------

    def _level_features(self, pos: torch.Tensor, i: int) -> torch.Tensor:
        scale = self.scales[i]                                   # (dim,)
        pos_scaled = pos * scale
        pos_floored = torch.floor(pos_scaled)
        vert_pos = (pos_floored.unsqueeze(-2) + self.cell_offsets).to(torch.long)  # (N, 2^dim, dim)
        if self.is_dense[i]:
            idx = (vert_pos * self.strides[i]).sum(dim=-1)       # (N, 2^dim)
        else:
            mask = 0xFFFF_FFFF
            x = vert_pos[..., 0] & mask
            y = vert_pos[..., 1] & mask
            if self.dim == 2:
                idx = (x ^ ((y * self.primes[1]) & mask)) & mask
            else:
                z = vert_pos[..., 2] & mask
                idx = (x ^ ((y * self.primes[1]) & mask) ^ ((z * self.primes[2]) & mask)) & mask
            idx = idx % self.T
        idx = idx.clamp(0, int(self.n_entries[i]) - 1)
        vert_latents = self.latents[i][idx]                      # (N, 2^dim, F)
        pos_offset = pos_scaled - pos_floored
        widths = (1.0 - self.cell_offsets) + (2.0 * self.cell_offsets - 1.0) * pos_offset.unsqueeze(-2)
        weights = widths.clamp(0.0, 1.0).prod(dim=-1)            # (N, 2^dim)
        return (vert_latents * weights.unsqueeze(-1)).sum(dim=-2)  # (N, F)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """Encode normalized coords ``(..., dim)`` -> features ``(..., L*F)``.

        Levels that have not been grown yet contribute a zero block, so the
        output width is constant at ``max_levels * F``.
        """
        shape = pos.shape[:-1]
        pos = pos.reshape(-1, self.dim)
        n_samples = int(pos.shape[0])
        feats: List[torch.Tensor] = []
        for i in range(self.L):
            if i < self._allocated:
                feats.append(self._level_features(pos, i))
            else:
                feats.append(pos.new_zeros(n_samples, self.F))
        return torch.cat(feats, dim=-1).reshape(*shape, -1)


__all__ = ["GrowingHashGrid"]
