"""Hash-encoded SIREN velocity reparameterization for FWI.

Combines a multi-resolution hash encoder (Instant-NGP, Müller et al. 2022;
DOI: https://doi.org/10.1145/3528223.3530127) with a small SIREN MLP head
(Sitzmann et al. 2020; DOI: https://doi.org/10.48550/arXiv.2006.09661)
to map normalized spatial coordinates to a velocity field.

By default the rendered velocity is ``base_velocity + mlp_out * vp_std + vp_mean``
(delta mode — start from a given initial model and learn the perturbation).
With ``direct_velocity=True`` the velocity is the un-shifted output and
``base_velocity`` is used only for grid shape / coordinates.

The multi-scale FWI advantage of this parameterization: when the grid
resolution changes between stages, you call :meth:`update_base_velocity`
with the resampled base, which re-creates the coordinate grid for the new
shape. The hash encoder and the SIREN MLP keep all their parameters, so
the geological knowledge accumulated at coarse scales is carried over
without re-fitting.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coarse_to_fine import CoarseToFineHashGrid
from .growing_hash_grid import GrowingHashGrid
from .hash_encoding import MultiResHashGrid
from .siren import SirenMLP


def _axis_coords(n: int, *, coord_min: float, coord_max: float, device, dtype) -> torch.Tensor:
    """One coordinate axis spanning ``[coord_min, coord_max)`` with N samples."""
    n = max(1, int(n))
    step = (coord_max - coord_min) / float(n)
    return torch.linspace(coord_min, coord_max - step, n, device=device, dtype=dtype)


def _make_coord_grid(
    shape: Tuple[int, ...],
    *,
    coord_min: float,
    coord_max: float,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return coords with shape ``(prod(shape), dim)`` in C-order over ``shape``."""
    axes = [
        _axis_coords(int(s), coord_min=coord_min, coord_max=coord_max,
                     device=device, dtype=dtype)
        for s in shape
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, len(shape))


class FourierFeatures(nn.Module):
    """NeRF-style positional encoding: log-spaced sin/cos of the coordinates.

    ``coords (N, dim)`` -> ``[coords?, sin(2^k * pi * c), cos(2^k * pi * c)]``
    for ``k = 0..n_levels-1``, flattened to ``(N, n_output_dims)``. With
    coords normalized to ``[0, 1)`` the finest level resolves ``2^(n_levels-1)``
    half-cycles across the model extent. Deterministic (no random projection),
    so the encoding itself is seed-independent.
    """

    def __init__(self, dim: int, n_levels: int = 6, include_input: bool = True) -> None:
        super().__init__()
        import math
        self.dim = int(dim)
        self.n_levels = int(n_levels)
        self.include_input = bool(include_input)
        freqs = (2.0 ** torch.arange(self.n_levels, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.n_output_dims = self.dim * (2 * self.n_levels + (1 if self.include_input else 0))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        ang = coords.unsqueeze(-1) * self.freqs                    # (N, dim, L)
        enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (N, dim, 2L)
        enc = enc.reshape(coords.shape[0], -1)
        if self.include_input:
            enc = torch.cat([coords, enc], dim=-1)
        return enc


class VelocityINR(nn.Module):
    """Hash-encoded SIREN representation of a 2-D or 3-D velocity field.

    Parameters
    ----------
    base_velocity
        Tensor ``(nz, nx)`` (2-D) or ``(nz, ny, nx)`` (3-D). Treated as a
        non-trainable buffer — the network learns the perturbation on top
        (delta mode) or replaces it entirely (``direct_velocity=True``).
    vp_mean, vp_std
        Output scaling: ``velocity = base + (mlp_out * vp_std + vp_mean)``.
        ``vp_std`` controls the typical perturbation magnitude in m/s; the
        network's natural output range is ~[-1, 1] so ``vp_std=50`` gives
        ±50 m/s typical updates. The legacy default in fwi_workflow-dev
        was 50 for vp.
    use_hash_encoding
        Wrap coordinates with :class:`MultiResHashGrid` before the SIREN
        head. Required for high-frequency velocity detail; without it the
        SIREN must do all the spatial-frequency work alone (slower, less
        expressive).
    hash_levels, hash_features_per_level, hash_log2_size,
    hash_base_resolution, hash_finest_resolution
        Hash encoder hyperparameters. See :class:`MultiResHashGrid`.
        Defaults match the fwi_workflow-dev production config.
    hash_c2f, hash_c2f_base_levels, hash_c2f_ramp
        If ``hash_c2f=True`` build a :class:`CoarseToFineHashGrid` instead:
        only the ``hash_c2f_base_levels`` coarsest levels are open at init
        and the training loop drives the unfreeze schedule via
        ``self.encoder.set_progress(...)``. Same hash hyperparameters.
    hidden_features, hidden_layers, first_omega0, hidden_omega0
        SIREN MLP hyperparameters. See :class:`SirenMLP`.
    direct_velocity
        If True, ignore the base and return ``mlp_out * vp_std + vp_mean``.
        Useful for from-scratch reconstruction.
    coord_min, coord_max
        Normalization range for the spatial coords (default ``[0, 1)``).
        Must match the convention expected by the hash encoder.
    bounds
        Optional ``(vp_min, vp_max)`` clamp applied at render time. Set
        to ``None`` (default) to let the caller clamp.
    water_mask
        Optional boolean tensor with the SAME shape as ``base_velocity``.
        Voxels where ``True`` are pinned to ``water_vp`` at render time
        — the SIREN's output for those cells is ignored entirely. This
        is the right tool for "I know the water column is 1500 m/s, do
        NOT let SIREN init noise contaminate it" cases: the SIREN at
        init has ~std=0.08 raw output, so with ``vp_std=500`` the water
        layer would otherwise sit at 1500±40 m/s of garbage from epoch 0.
        Because the rendered output doesn't depend on SIREN params at
        masked voxels, gradients there are exactly zero — equivalent to
        freezing those cells AND giving SIREN free model capacity to
        spend on the rest of the model.
    water_vp
        Velocity in m/s used at water-mask voxels. Default ``1500.0``.

    Notes
    -----
    Multi-stage FWI usage:
        net = VelocityINR(init_vp_75m, vp_std=50.0, ...)
        # ... train at stage 0 ...
        net.update_base_velocity(init_vp_37p5m)  # resample base, keep params
        # ... train at stage 1 ...
    """

    def __init__(
        self,
        base_velocity: torch.Tensor,
        *,
        vp_mean: float = 0.0,
        vp_std: float = 50.0,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        first_omega0: float = 30.0,
        hidden_omega0: float = 30.0,
        use_bias: bool = False,
        use_hash_encoding: bool = True,
        hash_levels: int = 16,
        hash_features_per_level: int = 2,
        hash_log2_size: int = 15,
        hash_base_resolution: int | list[int] = 4,
        hash_finest_resolution: int | list[int] = 512,
        hash_c2f: bool = False,
        hash_c2f_base_levels: int = 2,
        hash_c2f_ramp: str = "cosine",
        hash_growing: bool = False,
        hash_backend: str = "pytorch",
        use_fourier_encoding: bool = False,
        fourier_levels: int = 6,
        fourier_include_input: bool = True,
        direct_velocity: bool = False,
        coord_min: float = 0.0,
        coord_max: float = 1.0,
        bounds: Tuple[float, float] | None = None,
        water_mask: torch.Tensor | None = None,
        water_vp: float = 1500.0,
        lateral_downsample: Tuple[int, int] | int = 1,
        compile_render: bool = False,
    ) -> None:
        super().__init__()
        if base_velocity.ndim not in (2, 3):
            raise ValueError(
                f"base_velocity must be 2-D (nz, nx) or 3-D (nz, ny, nx); "
                f"got shape {tuple(base_velocity.shape)}"
            )
        self.dim = int(base_velocity.ndim)
        self.vp_mean = float(vp_mean)
        self.vp_std = float(vp_std)
        self.direct_velocity = bool(direct_velocity)
        self.coord_min = float(coord_min)
        self.coord_max = float(coord_max)
        self.use_hash_encoding = bool(use_hash_encoding)
        self.bounds = tuple(bounds) if bounds is not None else None

        # Base velocity is a buffer, not a parameter — it's the constant
        # background. `register_buffer` keeps it on the right device when
        # the module is `.to(device)`'d.
        self.register_buffer(
            "base_velocity",
            base_velocity.detach().to(dtype=torch.float32).clone(),
            persistent=False,
        )
        # Coord grid is also a buffer, refreshed by update_base_velocity().
        self.register_buffer(
            "coords",
            _make_coord_grid(
                tuple(int(s) for s in base_velocity.shape),
                coord_min=self.coord_min,
                coord_max=self.coord_max,
                device=base_velocity.device,
            ),
            persistent=False,
        )
        # Optional water-layer mask: if provided, render replaces those
        # voxels with ``water_vp`` (Pinned. Not learned. Gradient → 0).
        self.water_vp = float(water_vp)
        if water_mask is not None:
            if tuple(water_mask.shape) != tuple(base_velocity.shape):
                raise ValueError(
                    f"water_mask.shape {tuple(water_mask.shape)} != "
                    f"base_velocity.shape {tuple(base_velocity.shape)}"
                )
            self.register_buffer(
                "water_mask",
                water_mask.to(dtype=torch.bool).clone(),
                persistent=False,
            )
        else:
            self.water_mask = None

        self.use_fourier_encoding = bool(use_fourier_encoding)
        if self.use_hash_encoding and self.use_fourier_encoding:
            raise ValueError(
                "use_hash_encoding and use_fourier_encoding are mutually "
                "exclusive — pick one input encoding."
            )
        if self.use_hash_encoding:
            hash_kwargs = dict(
                n_levels=int(hash_levels),
                n_features_per_level=int(hash_features_per_level),
                log2_hashmap_size=int(hash_log2_size),
                base_resolution=hash_base_resolution,
                finest_resolution=hash_finest_resolution,
                backend=str(hash_backend),
            )
            if bool(hash_growing):
                if str(hash_backend) == "triton":
                    raise ValueError(
                        "hash_backend='triton' is not supported with hash_growing=True "
                        "(GrowingHashGrid uses a lazy per-level ParameterList, which the "
                        "fused kernel cannot index). Use hash_growing=False."
                    )
                # On-demand growth: fine levels allocated lazily (saves latent
                # memory). Grow schedule driven from the training loop via
                # encoder.grow_to_progress(...); takes precedence over c2f mask.
                self.encoder = GrowingHashGrid(
                    dim=self.dim,
                    max_levels=int(hash_levels),
                    n_features_per_level=int(hash_features_per_level),
                    log2_hashmap_size=int(hash_log2_size),
                    base_resolution=hash_base_resolution,
                    finest_resolution=hash_finest_resolution,
                    initial_levels=int(hash_c2f_base_levels),
                )
            elif bool(hash_c2f):
                self.encoder = CoarseToFineHashGrid(
                    dim=self.dim,
                    base_levels=int(hash_c2f_base_levels),
                    ramp=str(hash_c2f_ramp),
                    **hash_kwargs,
                )
            else:
                self.encoder = MultiResHashGrid(dim=self.dim, **hash_kwargs)
            self.hash_growing = bool(hash_growing)
            in_features = int(self.encoder.n_output_dims)
        elif self.use_fourier_encoding:
            self.encoder = FourierFeatures(
                dim=self.dim,
                n_levels=int(fourier_levels),
                include_input=bool(fourier_include_input),
            )
            in_features = int(self.encoder.n_output_dims)
        else:
            self.encoder = None
            in_features = self.dim

        self.mlp = SirenMLP(
            in_features=in_features,
            out_features=1,
            hidden_features=int(hidden_features),
            hidden_layers=int(hidden_layers),
            first_omega0=float(first_omega0),
            hidden_omega0=float(hidden_omega0),
            bias=bool(use_bias),
        )

        # --- Anisotropic lateral downsampling ---------------------------- #
        # Render the INR delta on a grid coarsened in the LATERAL axes only
        # (full resolution kept in z, where seismic structure is sharp), then
        # trilinearly upsample the delta back to full resolution and add the
        # full-res base. The hash+MLP is evaluated at far fewer coords
        # (lateral_ds**2 fewer in 3-D) → proportionally faster render AND
        # reparam backward, with negligible error where the model is laterally
        # smooth (the FWI-resolvable case). Param count is UNCHANGED.
        if isinstance(lateral_downsample, int):
            lat = (lateral_downsample,) * (self.dim - 1)
        else:
            lat = tuple(int(s) for s in lateral_downsample)
        if len(lat) != self.dim - 1:
            raise ValueError(
                f"lateral_downsample must have {self.dim - 1} entries for "
                f"{self.dim}-D; got {lateral_downsample!r}"
            )
        self.lateral_ds = tuple(max(1, int(s)) for s in lat)
        self.compile_render = bool(compile_render)
        self._zslab_fn = None  # lazily-built (optionally compiled) renderer
        if any(ds > 1 for ds in self.lateral_ds) or self.compile_render:
            # TF32 accelerates the hash+MLP matmuls ~2-3× (Ampere+) at ~6e-4
            # relative gradient error (cosine 1.000000) — essentially lossless.
            # Only affects torch matmuls (the INR), never the C/CUDA solver.
            # Enabled whenever the perf path is active (anisotropic OR compile),
            # independent of compile so the reliable eager+TF32+aniso config
            # (no torch.compile — which jitters on some GPUs) gets it too.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._build_coarse_coords()

    def _build_coarse_coords(self) -> None:
        """(Re)build the coarse-lateral coord grid used by anisotropic render."""
        import math
        full = tuple(int(s) for s in self.base_velocity.shape)
        nz = full[0]
        lateral_full = full[1:]
        coarse_lat = tuple(
            max(2, math.ceil(n / ds)) if ds > 1 else n
            for n, ds in zip(lateral_full, self.lateral_ds)
        )
        self._coarse_lateral = coarse_lat
        self._aniso = any(ds > 1 for ds in self.lateral_ds)
        if self._aniso:
            self._coarse_coords = _make_coord_grid(
                (nz,) + coarse_lat,
                coord_min=self.coord_min,
                coord_max=self.coord_max,
                device=self.base_velocity.device,
            ).reshape((nz,) + coarse_lat + (self.dim,))
        else:
            self._coarse_coords = None
        self._zslab_fn = None  # invalidate compiled fn on shape change
        self._cg_graph = None  # invalidate any captured CUDA graph on shape change

    # ------------------------------------------------------------------ #
    # Stage transitions: replace the base velocity (+ shape) but keep    #
    # all learnable parameters intact — this is the multi-scale benefit. #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update_base_velocity(
        self,
        new_base: torch.Tensor,
        water_mask: torch.Tensor | None = None,
    ) -> None:
        """Swap in a new base (possibly different shape); rebuild coords.

        When the new base has a different shape than the old, any
        previously installed ``water_mask`` becomes stale. Pass a
        ``water_mask`` matched to ``new_base.shape`` to refresh; leave
        ``None`` to clear the existing mask (and warn loudly that a
        stale mask was dropped).
        """
        if new_base.ndim != self.dim:
            raise ValueError(
                f"new_base.ndim {new_base.ndim} != self.dim {self.dim}"
            )
        device = self.base_velocity.device
        new_base = new_base.detach().to(device=device, dtype=torch.float32).clone()
        shape_changed = (tuple(self.base_velocity.shape) != tuple(new_base.shape))
        # Replace the registered buffer so .to(device) keeps working.
        self.base_velocity = new_base
        self.coords = _make_coord_grid(
            tuple(int(s) for s in new_base.shape),
            coord_min=self.coord_min,
            coord_max=self.coord_max,
            device=device,
        )
        self._build_coarse_coords()  # refresh anisotropic coarse grid for new shape
        if water_mask is not None:
            if tuple(water_mask.shape) != tuple(new_base.shape):
                raise ValueError(
                    f"water_mask.shape {tuple(water_mask.shape)} != "
                    f"new_base.shape {tuple(new_base.shape)}"
                )
            self.water_mask = water_mask.to(
                device=device, dtype=torch.bool,
            ).clone()
        elif shape_changed and self.water_mask is not None:
            # Old mask is stale; drop it rather than crash at render.
            import warnings
            warnings.warn(
                "VelocityINR.update_base_velocity: new_base.shape "
                f"{tuple(new_base.shape)} != old water_mask.shape "
                f"{tuple(self.water_mask.shape)}; dropping stale "
                "water_mask. Pass water_mask=... to refresh.",
                stacklevel=2,
            )
            self.water_mask = None

    # ------------------------------------------------------------------ #
    # Rendering                                                          #
    # ------------------------------------------------------------------ #

    def _render_at_coords(
        self,
        coords: torch.Tensor,
        shape: Tuple[int, ...],
        base: torch.Tensor,
        *,
        water_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encoder(coords) if self.encoder is not None else coords
        raw = self.mlp(features).reshape(*shape)
        delta = raw * self.vp_std + self.vp_mean
        if self.direct_velocity:
            vp = delta
        else:
            vp = base + delta
        if self.bounds is not None:
            vp = vp.clamp(self.bounds[0], self.bounds[1])
        # Water-layer pin: override masked voxels with the fixed water
        # velocity. ``torch.where`` keeps the autograd graph correct —
        # gradient at masked voxels is zero (the output doesn't depend
        # on SIREN params there), so SIREN can't waste capacity on the
        # water column and the water layer never drifts from water_vp.
        #
        # Caller may pass an explicitly-shaped ``water_mask`` (e.g.
        # ``render_window`` slices the full-grid mask to match the
        # rendered chunk); fall back to ``self.water_mask`` only when
        # the render covers the full base-velocity grid.
        if water_mask is None and self.water_mask is not None:
            if tuple(self.water_mask.shape) == tuple(shape):
                water_mask = self.water_mask
            # else: caller is rendering a different shape (chunk /
            # arbitrary render_shape); they must pass the matching mask.
        if water_mask is not None:
            if tuple(water_mask.shape) != tuple(shape):
                raise ValueError(
                    f"water_mask.shape {tuple(water_mask.shape)} != "
                    f"render shape {tuple(shape)}"
                )
            water_val = vp.new_full((), self.water_vp)
            vp = torch.where(water_mask, water_val, vp)
        return vp

    def forward(self) -> torch.Tensor:
        """Render the velocity model on the current base grid."""
        return self.render()

    def _render_coords_chunk(self, cc, base_chunk, mask_chunk):
        """Compile-friendly anisotropic render of one z-slab. Takes TENSORS
        only (no python ints) so torch.compile guards on tensor shapes — which
        are constant when chunk_rows divides nz — and compiles exactly ONCE.
        (Passing z0/z1 ints made dynamo guard on their *values*, recompiling
        per slab → recompile_limit(8) → eager fallback → reparam_bwd spikes.)
        """
        feats = self.encoder(cc) if self.encoder is not None else cc
        rows = base_chunk.shape[0]
        raw = self.mlp(feats).reshape(rows, *self._coarse_lateral)
        delta = raw * self.vp_std + self.vp_mean
        mode = "bilinear" if self.dim == 2 else "trilinear"
        delta = F.interpolate(
            delta[None, None], size=tuple(base_chunk.shape),
            mode=mode, align_corners=True)[0, 0]
        vp = delta if self.direct_velocity else base_chunk + delta
        if self.bounds is not None:
            vp = vp.clamp(self.bounds[0], self.bounds[1])
        if mask_chunk is not None:
            vp = torch.where(mask_chunk, vp.new_full((), self.water_vp), vp)
        return vp

    def _zslab(self, z0: int, z1: int) -> torch.Tensor:
        """Render z-rows [z0:z1] at FULL lateral resolution. Anisotropic:
        coarse-lateral render + upsample (z stays exact); isotropic: defer to
        render_window. Slicing is done eagerly here; the heavy compute goes
        through the (optionally compiled) tensor-only ``_render_coords_chunk``.
        """
        full = tuple(int(s) for s in self.base_velocity.shape)
        if not self._aniso:
            if self.dim == 2:
                return self.render_window(z0, z1, 0, full[1])
            return self.render_window(z0, z1, 0, full[1], 0, full[2])
        cc = self._coarse_coords[z0:z1].reshape(-1, self.dim)
        base_chunk = self.base_velocity[z0:z1]
        mask_chunk = self.water_mask[z0:z1] if self.water_mask is not None else None
        if self.compile_render:
            if self._zslab_fn is None:
                self._zslab_fn = torch.compile(self._render_coords_chunk, dynamic=False)
            return self._zslab_fn(cc, base_chunk, mask_chunk)
        return self._render_coords_chunk(cc, base_chunk, mask_chunk)

    def render(self, chunk_rows: int | None = None) -> torch.Tensor:
        full = tuple(int(s) for s in self.base_velocity.shape)
        nz = int(full[0])
        lateral = 1
        for s in full[1:]:
            lateral *= int(s)
        coarse_lat = 1
        for s in self._coarse_lateral:
            coarse_lat *= int(s)
        # Auto-chunk along z. The hash encoder materializes O(n_coords) vertex
        # positions, so a full fine-grid render can need tens of GB
        # on a large 3-D grid. Chunking z-slabs is bit-identical (pointwise) and bounds
        # peak memory under no_grad. With anisotropic lateral downsampling the
        # effective per-row coord count is coarse_lat, so chunks can be larger.
        eff_lat = coarse_lat if self._aniso else lateral
        if chunk_rows is None:
            _CHUNK_POINTS = 2_000_000
            chunk_rows = (
                max(1, _CHUNK_POINTS // eff_lat)
                if eff_lat > 0 and nz * eff_lat > _CHUNK_POINTS
                else nz
            )
        if chunk_rows >= nz and not self._aniso:
            return self._render_at_coords(self.coords, full, self.base_velocity)
        slabs = []
        for z0 in range(0, nz, int(chunk_rows)):
            z1 = min(nz, z0 + int(chunk_rows))
            slabs.append(self._zslab(z0, z1))
        return torch.cat(slabs, dim=0)

    def render_shape(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Render at an arbitrary grid shape (bilinear/trilinear-resample the base)."""
        if len(shape) != self.dim:
            raise ValueError(f"shape {shape} dim mismatch with self.dim {self.dim}")
        shape = tuple(int(s) for s in shape)
        coords = _make_coord_grid(
            shape,
            coord_min=self.coord_min,
            coord_max=self.coord_max,
            device=self.coords.device,
        )
        same_shape = shape == tuple(int(s) for s in self.base_velocity.shape)
        if same_shape:
            base = self.base_velocity
            water_mask = self.water_mask
        else:
            mode = "bilinear" if self.dim == 2 else "trilinear"
            base = F.interpolate(
                self.base_velocity.reshape(1, 1, *self.base_velocity.shape),
                size=shape,
                mode=mode,
                align_corners=True,
            ).reshape(*shape)
            # Resample the water_mask via nearest-neighbor (bool stays
            # crisp; trilinear on bool produces garbage).
            if self.water_mask is not None:
                water_mask = F.interpolate(
                    self.water_mask.to(torch.float32).reshape(
                        1, 1, *self.water_mask.shape
                    ),
                    size=shape, mode="nearest",
                ).reshape(*shape).to(torch.bool)
            else:
                water_mask = None
        return self._render_at_coords(
            coords, shape, base, water_mask=water_mask,
        )

    def render_window(self, *bounds: int) -> torch.Tensor:
        """Render a rectangular window on the base grid.

        Bounds layout:
            2-D: ``(z0, z1, x0, x1)``
            3-D: ``(z0, z1, y0, y1, x0, x1)``
        """
        if len(bounds) != 2 * self.dim:
            raise ValueError(
                f"need {2 * self.dim} bounds for {self.dim}-D; got {len(bounds)}"
            )
        full_shape = tuple(int(s) for s in self.base_velocity.shape)
        slices = []
        win_shape = []
        for d in range(self.dim):
            lo = max(0, int(bounds[2 * d]))
            hi = min(full_shape[d], int(bounds[2 * d + 1]))
            slices.append(slice(lo, hi))
            win_shape.append(hi - lo)
        win_shape = tuple(win_shape)
        # Slice the cached coords by reshaping back to full grid, indexing,
        # then flattening — avoids re-creating the grid.
        coords_full = self.coords.reshape(*full_shape, self.dim)
        coords_win = coords_full[tuple(slices)].reshape(-1, self.dim)
        base_win = self.base_velocity[tuple(slices)]
        # Slice the water_mask in lock-step so the chunked-backward
        # path (which renders one z-slab at a time) gets a mask matching
        # its chunk shape — otherwise torch.where broadcasts incorrectly.
        if self.water_mask is not None:
            water_mask_win = self.water_mask[tuple(slices)]
        else:
            water_mask_win = None
        # coords/base_velocity may live on CPU (to save GPU memory when the
        # global grid is huge and only per-tile windows are rendered on GPU);
        # move just this window's slices to the network's compute device.
        # No-op when they already share the encoder/MLP device.
        dev = next(self.parameters()).device
        coords_win = coords_win.to(dev, non_blocking=True)
        base_win = base_win.to(dev, non_blocking=True)
        if water_mask_win is not None:
            water_mask_win = water_mask_win.to(dev, non_blocking=True)
        return self._render_at_coords(
            coords_win, win_shape, base_win, water_mask=water_mask_win,
        )

    # ------------------------------------------------------------------ #
    # Memory-conscious backward for large grids                          #
    # ------------------------------------------------------------------ #

    def backward_velocity_gradient(
        self, velocity_grad: torch.Tensor, *, chunk_rows: int = 64
    ) -> None:
        """Back-propagate a full-grid velocity gradient row-by-row.

        Use this when the full-grid ``render() + .backward()`` would OOM:
        e.g. a 3-D model with millions of voxels and a deep SIREN. The
        method renders one slab of rows at a time, calls ``backward()``
        on that slab with the corresponding slice of ``velocity_grad``,
        and accumulates gradients onto the trainable parameters without
        ever holding the full graph in memory.

        Parameters
        ----------
        velocity_grad
            Tensor with the same shape as :attr:`base_velocity`.
        chunk_rows
            Rows along the slow axis (axis 0) per chunk. Smaller = less
            peak memory, more overhead. 64 is a sensible default for 2-D;
            for 3-D models start with 4-8.
        """
        grad = torch.as_tensor(
            velocity_grad, dtype=torch.float32, device=self.base_velocity.device
        )
        if tuple(grad.shape) != tuple(int(s) for s in self.base_velocity.shape):
            raise ValueError(
                f"velocity_grad shape {tuple(grad.shape)} != "
                f"base_velocity shape {tuple(self.base_velocity.shape)}"
            )
        full_shape = tuple(int(s) for s in self.base_velocity.shape)
        rows = max(1, int(chunk_rows))
        nz = full_shape[0]
        import os as _os
        _dbg = _os.environ.get("SWEEP_REPARAM_TPROF") == "1"
        _cuda = self.base_velocity.device.type == "cuda"
        if _dbg:
            import time as _time
            _tr = _tb = 0.0; _n = 0
            _dev = self.base_velocity.device
            def _sy():
                if _cuda: torch.cuda.synchronize(_dev)
            if _cuda:
                _ma0 = torch.cuda.memory_allocated(_dev)
                _mr0 = torch.cuda.memory_reserved(_dev)
                _tot = torch.cuda.get_device_properties(_dev).total_memory
                _s0 = torch.cuda.memory_stats(_dev)
                _retr0 = _s0.get("num_alloc_retries", 0)
                _ooms0 = _s0.get("num_ooms", 0)
                _peak0 = torch.cuda.max_memory_allocated(_dev)
        for z0 in range(0, nz, rows):
            z1 = min(nz, z0 + rows)
            # ``_zslab`` is anisotropic-aware (coarse lateral + upsample) and
            # optionally torch.compiled; per-slab .backward keeps peak memory
            # O(rows) regardless of grid size.
            if _dbg:
                _sy(); _t0 = _time.perf_counter()
                vp_chunk = self._zslab(z0, z1)
                _sy(); _tr += _time.perf_counter() - _t0; _t0 = _time.perf_counter()
                vp_chunk.backward(grad[z0:z1], retain_graph=False)
                _sy(); _tb += _time.perf_counter() - _t0; _n += 1
            else:
                vp_chunk = self._zslab(z0, z1)
                vp_chunk.backward(grad[z0:z1], retain_graph=False)
        if _dbg:
            _extra = ""
            if _cuda:
                _s1 = torch.cuda.memory_stats(_dev)
                _dretr = _s1.get("num_alloc_retries", 0) - _retr0
                _dooms = _s1.get("num_ooms", 0) - _ooms0
                _peak1 = torch.cuda.max_memory_allocated(_dev)
                _extra = (f" | gpu alloc={_ma0/1e9:.1f}G reserved={_mr0/1e9:.1f}G "
                          f"total={_tot/1e9:.1f}G free={(_tot-_mr0)/1e9:.1f}G "
                          f"peak_in_call={(_peak1-_ma0)/1e9:.2f}G "
                          f"alloc_retries+={_dretr} ooms+={_dooms}")
            print(f"[reparam_tprof] chunks={_n} render={_tr:.3f}s backward={_tb:.3f}s "
                  f"total={_tr+_tb:.3f}s{_extra}", flush=True)

    # ------------------------------------------------------------------ #
    # CUDA-graphed reparam backward — root-cure for the prefetch-GIL      #
    # stall. The eager render+backward fires ~500 tiny kernels; under     #
    # async SEG-Y prefetch (ThreadPoolExecutor in the main process) those #
    # launches get starved of the GIL and the NN backward inflates ~80x   #
    # (0.08 s -> 5-9 s). Capturing render(chunk_rows=nz)+backward into a   #
    # CUDA graph turns the whole thing into ONE host-side replay launch,   #
    # immune to GIL contention. Numerically equivalent to                 #
    # backward_velocity_gradient (same math, 1 chunk). Opt-in.            #
    # ------------------------------------------------------------------ #
    def backward_velocity_gradient_graphed(self, velocity_grad: torch.Tensor) -> None:
        dev = self.base_velocity.device
        if dev.type != "cuda":
            return self.backward_velocity_gradient(
                velocity_grad, chunk_rows=int(self.base_velocity.shape[0]))
        g = torch.as_tensor(velocity_grad, dtype=torch.float32, device=dev)
        shape = tuple(int(s) for s in g.shape)
        if getattr(self, "_cg_graph", None) is None or getattr(self, "_cg_shape", None) != shape:
            self._cg_capture(shape)
        self._cg_static_vgrad.copy_(g)
        for sb in self._cg_static_grads:
            sb.zero_()
        self._cg_graph.replay()
        # Expose the captured static grad buffers as the params' .grad so the
        # optimizer reads them (the runner's zero_grad(set_to_none) may have
        # nulled .grad between iters; the graph always writes the same buffers).
        for p, sb in zip(self._cg_params, self._cg_static_grads):
            p.grad = sb

    def _cg_capture(self, shape) -> None:
        dev = self.base_velocity.device
        nz = int(shape[0])
        self._cg_static_vgrad = torch.zeros(shape, dtype=torch.float32, device=dev)
        self._cg_params = [p for p in self.parameters() if p.requires_grad]
        # Warmup in a side stream so the caching allocator sizes the graph pool.
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                for p in self._cg_params:
                    p.grad = None
                vp = self.render(chunk_rows=nz)
                vp.backward(self._cg_static_vgrad)
        torch.cuda.current_stream(dev).wait_stream(s)
        for p in self._cg_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        self._cg_static_grads = [p.grad for p in self._cg_params]
        for sb in self._cg_static_grads:
            sb.zero_()
        self._cg_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cg_graph):
            vp = self.render(chunk_rows=nz)
            vp.backward(self._cg_static_vgrad)
        self._cg_shape = shape

    def _cg_invalidate(self) -> None:
        """Drop any captured graph (call on base-velocity/shape change)."""
        self._cg_graph = None


__all__ = ["VelocityINR"]
