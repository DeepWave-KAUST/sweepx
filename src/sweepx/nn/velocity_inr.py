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
        direct_velocity: bool = False,
        coord_min: float = 0.0,
        coord_max: float = 1.0,
        bounds: Tuple[float, float] | None = None,
        water_mask: torch.Tensor | None = None,
        water_vp: float = 1500.0,
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

        if self.use_hash_encoding:
            self.encoder = MultiResHashGrid(
                dim=self.dim,
                n_levels=int(hash_levels),
                n_features_per_level=int(hash_features_per_level),
                log2_hashmap_size=int(hash_log2_size),
                base_resolution=hash_base_resolution,
                finest_resolution=hash_finest_resolution,
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

    def render(self) -> torch.Tensor:
        shape = tuple(int(s) for s in self.base_velocity.shape)
        return self._render_at_coords(self.coords, shape, self.base_velocity)

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
        for z0 in range(0, nz, rows):
            z1 = min(nz, z0 + rows)
            if self.dim == 2:
                vp_chunk = self.render_window(z0, z1, 0, full_shape[1])
                vp_chunk.backward(grad[z0:z1], retain_graph=False)
            else:
                vp_chunk = self.render_window(
                    z0, z1, 0, full_shape[1], 0, full_shape[2]
                )
                vp_chunk.backward(grad[z0:z1], retain_graph=False)


__all__ = ["VelocityINR"]
