"""Multi-resolution hash-grid encoding (Instant-NGP style), pure PyTorch.

Müller, Evans, Schied, Keller (2022),
    "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding",
    ACM ToG (SIGGRAPH).
DOI: https://doi.org/10.1145/3528223.3530127

Coords come in normalized to ``[0, 1)`` with shape ``(..., dim)`` (``dim`` is
2 or 3). The output is ``(..., L * F)`` — ``L`` resolution levels concatenated,
each with ``F`` features per cell. Levels whose dense grid fits in the hash
table use tiled indexing; higher levels use spatial hashing with Knuth-style
primes (``[1, 2_654_435_761, 805_459_861]``).

Device-agnostic and TorchScript-friendly. Suitable for FWI velocity models
where the encoder pairs with a small SIREN/MLP head to map coords -> scalar
velocity field.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def _next_multiple(value: int, step: int) -> int:
    return (int(value) + int(step) - 1) // int(step) * int(step)


class MultiResHashGrid(nn.Module):
    """Anisotropic multi-resolution hash-grid encoder."""

    def __init__(
        self,
        dim: int,
        *,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int | List[int] = 2,
        finest_resolution: int | List[int] = 16,
        dtype: torch.dtype = torch.float32,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        if int(dim) not in (2, 3):
            raise ValueError(f"dim must be 2 or 3; got {dim}")
        if str(backend) not in ("pytorch", "triton"):
            raise ValueError(f"backend must be 'pytorch' or 'triton'; got {backend!r}")
        self.backend = str(backend)

        self.dim = int(dim)
        self.L = int(n_levels)
        self.F = int(n_features_per_level)
        self.T = 2 ** int(log2_hashmap_size)
        self.output_dim = self.L * self.F
        self.n_output_dims = self.output_dim  # legacy alias

        if isinstance(base_resolution, int):
            base_resolution = [int(base_resolution)] * self.dim
        if isinstance(finest_resolution, int):
            finest_resolution = [int(finest_resolution)] * self.dim
        if len(base_resolution) != self.dim or len(finest_resolution) != self.dim:
            raise ValueError("base_resolution / finest_resolution must match dim")

        # Per-axis geometric growth factor across L levels.
        b = [
            math.exp(
                (math.log(float(finest_resolution[d])) - math.log(float(base_resolution[d])))
                / max(1, self.L - 1)
            )
            for d in range(self.dim)
        ]

        scales: List[List[float]] = []
        resolutions: List[List[int]] = []
        strides: List[List[int]] = []
        offsets = [0]
        first_hash_level = 0

        for level in range(self.L):
            scale_i = [float(base_resolution[d]) * (b[d] ** level) - 1.0 for d in range(self.dim)]
            scales.append(scale_i)
            res_i = [int(math.ceil(s) + 1) for s in scale_i]
            resolutions.append(res_i)

            prod = 1
            for r in res_i:
                prod *= r
            if prod <= self.T:
                n_entries = _next_multiple(prod, 8)
                first_hash_level += 1
            else:
                n_entries = self.T
            offsets.append(offsets[-1] + n_entries)

            stride_i = [1]
            for r in res_i[:-1]:
                stride_i.append(stride_i[-1] * int(r))
            strides.append(stride_i)

        self.first_hash_level = int(first_hash_level)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.int32))
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.int64))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64))
        # Knuth / MurmurHash primes. Persistent=False — these are constants.
        self.register_buffer(
            "primes",
            torch.tensor([1, 2_654_435_761, 805_459_861], dtype=torch.long),
            persistent=False,
        )
        self.latents = nn.Parameter(
            torch.empty(offsets[-1], self.F, dtype=dtype).uniform_(-1.0e-4, 1.0e-4)
        )

        if self.dim == 2:
            cell_offsets = [[0, 0], [0, 1], [1, 0], [1, 1]]
        else:
            cell_offsets = [
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
            ]
        self.register_buffer("cell_offsets", torch.tensor(cell_offsets, dtype=torch.float32))

        # Buffers the fused Triton backend reads (cheap; harmless for the PyTorch path).
        # is_dense[i] = 1 for tiled (dense) levels, 0 for hashed; offsets_head = offsets[:-1].
        is_dense = torch.zeros(self.L, dtype=torch.int32)
        is_dense[: self.first_hash_level] = 1
        self.register_buffer("is_dense", is_dense, persistent=False)
        self.register_buffer("offsets_head", torch.tensor(offsets[:-1], dtype=torch.int64), persistent=False)

    def _make_vert_pos(self, pos_scaled: torch.Tensor) -> torch.Tensor:
        pos_floored = torch.floor(pos_scaled)
        vert_pos = pos_floored.unsqueeze(-2) + self.cell_offsets.view(1, 1, -1, self.dim)
        return vert_pos.to(torch.long)

    def _make_tiled_indices(self, strides: torch.Tensor, vert_pos: torch.Tensor) -> torch.Tensor:
        return (vert_pos * strides.view(-1, 1, 1, self.dim)).sum(dim=-1)

    def _make_hash_indices(self, vert_pos: torch.Tensor) -> torch.Tensor:
        mask = 0xFFFF_FFFF
        x = vert_pos[..., 0] & mask
        y = vert_pos[..., 1] & mask
        if self.dim == 2:
            idx = (x ^ ((y * self.primes[1]) & mask)) & mask
        else:
            z = vert_pos[..., 2] & mask
            idx = (
                x
                ^ ((y * self.primes[1]) & mask)
                ^ ((z * self.primes[2]) & mask)
            ) & mask
        return idx.to(torch.long)

    def _make_indices(self, vert_pos: torch.Tensor) -> torch.Tensor:
        parts = []
        dense_levels = int(self.first_hash_level)
        if dense_levels > 0:
            parts.append(self._make_tiled_indices(self.strides[:dense_levels], vert_pos[:dense_levels]))
        if dense_levels < self.L:
            parts.append(self._make_hash_indices(vert_pos[dense_levels:]))
        idx = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        idx = idx % self.T
        idx = idx + self.offsets[:-1].view(self.L, 1, 1)
        return idx.clamp(0, self.latents.shape[0] - 1)

    def _lerp_weights(self, pos_scaled: torch.Tensor) -> torch.Tensor:
        pos_offset = pos_scaled - torch.floor(pos_scaled)
        widths = (
            (1.0 - self.cell_offsets)
            + (2.0 * self.cell_offsets - 1.0) * pos_offset.unsqueeze(-2)
        )
        return widths.clamp(0.0, 1.0).prod(dim=-1)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """Encode normalized coords ``(..., dim)`` -> features ``(..., L*F)``."""
        shape = pos.shape[:-1]
        pos = pos.reshape(-1, self.dim)
        if self.backend == "triton":
            # Fused GPU path: numerically matches the PyTorch branch (cos=1.0) with
            # ~7-20x less encoder memory. See sweep_nn.triton_hash_encoding.
            from .triton_hash_encoding import triton_encode

            out = triton_encode(
                pos, self.latents, self.scales, self.strides,
                self.offsets_head, self.is_dense, self.T, self.dim, self.F,
            )
            return out.reshape(*shape, -1)
        n_samples = int(pos.shape[0])
        # Broadcast: (L, n_samples, dim)
        pos_scaled = pos[None, :, :] * self.scales[:, None, :]
        vert_pos = self._make_vert_pos(pos_scaled)
        indices = self._make_indices(vert_pos)
        vert_latents = self.latents[indices]
        vert_weights = self._lerp_weights(pos_scaled)
        enc = (vert_latents * vert_weights.unsqueeze(-1)).sum(dim=-2)
        # Re-order so each sample concatenates its L levels.
        enc = enc.permute(1, 0, 2).reshape(n_samples, self.L * self.F)
        return enc.reshape(*shape, -1)


__all__ = ["MultiResHashGrid"]
