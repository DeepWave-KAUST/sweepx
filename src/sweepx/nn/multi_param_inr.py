"""Multi-parameter INR: ONE shared encoder + SIREN trunk with an N-channel head,
one output channel per solver-model parameter (vp, z, and later vs, rho, eps …).

Generalizes :class:`VelocityINR` (single-channel vp) to JOINT multi-parameter
reparameterization for variable-density (vp, z), elastic (vp, vs, rho), or
anisotropic FWI. All channels share the coordinate encoder and every SIREN
hidden layer — they diverge only at the final ``Linear(hidden, n_params)`` — so
the low-wavenumber structure is learned ONCE and every parameter reads it (this
coupling is what stops the strongly-constrained parameter, e.g. impedance z,
from starving the weakly-constrained one, e.g. vp; validated on Overthrust VRZ).

Each channel ``i`` maps its raw SIREN output to a physical field by its OWN
affine and clamp::

    field_i = base_i + raw_i * std_i + mean_i          (clamped to bounds_i)

with an optional per-channel water pin. Channel 0 is the PRIMARY parameter
(vp): :meth:`render`, :meth:`forward`, and the :attr:`base_velocity` alias
return it, so any caller written against the VelocityINR interface (snapshots,
final ``inverted_vp.npy``, DD tile bounds) keeps working unchanged. The other
channels are reached via :meth:`render_param` / :meth:`render_all`, and the
whole parameter set is trained jointly through :meth:`backward_gradients`.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .velocity_inr import _make_coord_grid
from .siren import SirenMLP
from .hash_encoding import MultiResHashGrid
from .coarse_to_fine import CoarseToFineHashGrid
from .growing_hash_grid import GrowingHashGrid


class MultiParamINR(nn.Module):
    """Shared-trunk INR with one output channel per FWI model parameter.

    Parameters
    ----------
    base_models
        List of ``n_params`` initial-model tensors, all the same shape
        (``(nz, nx)`` or ``(nz, ny, nx)``). ``base_models[0]`` is the primary
        parameter (vp). Each is kept as a (non-persistent) buffer; the channel
        renders ``base_i + delta_i``.
    means, stds
        Per-channel affine ``delta_i = raw_i * stds[i] + means[i]``. Length
        ``n_params``. ``stds[i]`` is the per-parameter "scale" (its effective
        learning rate is ``lr * stds[i]``); pick it per parameter's magnitude
        (vp ~ 500, impedance z ~ 2).
    bounds
        Per-channel ``(min, max)`` clamp, or ``None`` to leave a channel
        unbounded. Length ``n_params``.
    water_mask, water_values
        Optional shared boolean water mask (shape = model shape). Masked voxels
        of channel ``i`` are pinned to ``water_values[i]`` (gradient there is 0,
        so the SIREN never spends capacity on the water column). ``water_values``
        length ``n_params`` (e.g. ``[1500.0, 1.5]`` for vp, Gardner-water z).
    The remaining arguments mirror :class:`VelocityINR` (shared SIREN + hash
    encoder hyperparameters).
    """

    def __init__(
        self,
        base_models: Sequence[torch.Tensor],
        *,
        means: Sequence[float],
        stds: Sequence[float],
        bounds: Sequence[Tuple[float, float] | None],
        water_mask: torch.Tensor | None = None,
        water_values: Sequence[float] | None = None,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        first_omega0: float = 30.0,
        hidden_omega0: float = 30.0,
        use_bias: bool = False,
        use_hash_encoding: bool = True,
        hash_levels: int = 16,
        hash_features_per_level: int = 2,
        hash_log2_size: int = 15,
        hash_base_resolution: int | List[int] = 4,
        hash_finest_resolution: int | List[int] = 512,
        hash_c2f: bool = False,
        hash_c2f_base_levels: int = 2,
        hash_c2f_ramp: str = "cosine",
        hash_growing: bool = False,
        hash_backend: str = "pytorch",
        direct_velocity: bool = False,
        coord_min: float = 0.0,
        coord_max: float = 1.0,
    ) -> None:
        super().__init__()
        base_models = list(base_models)
        n = len(base_models)
        if n < 1:
            raise ValueError("MultiParamINR needs at least one base model")
        shape = tuple(int(s) for s in base_models[0].shape)
        if len(shape) not in (2, 3):
            raise ValueError(
                f"base models must be 2-D (nz, nx) or 3-D (nz, ny, nx); got {shape}")
        for k, b in enumerate(base_models):
            if tuple(b.shape) != shape:
                raise ValueError(
                    f"base_models[{k}] shape {tuple(b.shape)} != base_models[0] {shape}")
        if not (len(means) == len(stds) == len(bounds) == n):
            raise ValueError("means/stds/bounds must each have n_params entries")
        self.n_params = int(n)
        self.dim = len(shape)
        self.shape = shape
        self.direct_velocity = bool(direct_velocity)
        self.coord_min = float(coord_min)
        self.coord_max = float(coord_max)
        self.use_hash_encoding = bool(use_hash_encoding)
        dev = base_models[0].device
        self._bshape = (self.n_params,) + (1,) * self.dim   # broadcast shape

        # Per-channel base stacked to (n, *shape); affine + bounds as (n,) vecs.
        self.register_buffer(
            "base_stack",
            torch.stack([b.detach().to(torch.float32) for b in base_models], 0).clone(),
            persistent=False)
        self.register_buffer("means", torch.tensor([float(m) for m in means],
                                                    dtype=torch.float32, device=dev),
                             persistent=False)
        self.register_buffer("stds", torch.tensor([float(s) for s in stds],
                                                   dtype=torch.float32, device=dev),
                             persistent=False)
        lo = [(-torch.inf if (b is None or b[0] is None) else float(b[0])) for b in bounds]
        hi = [(torch.inf if (b is None or b[1] is None) else float(b[1])) for b in bounds]
        self.register_buffer("bound_lo", torch.tensor(lo, dtype=torch.float32, device=dev),
                             persistent=False)
        self.register_buffer("bound_hi", torch.tensor(hi, dtype=torch.float32, device=dev),
                             persistent=False)

        # Shared coordinate grid (C-order over shape).
        self.register_buffer(
            "coords",
            _make_coord_grid(shape, coord_min=self.coord_min, coord_max=self.coord_max,
                             device=dev),
            persistent=False)

        # Optional water pin (shared spatial mask, per-channel value).
        if water_mask is not None:
            if tuple(water_mask.shape) != shape:
                raise ValueError(
                    f"water_mask shape {tuple(water_mask.shape)} != model shape {shape}")
            self.register_buffer("water_mask", water_mask.to(torch.bool).clone(),
                                 persistent=False)
            wv = ([0.0] * n if water_values is None else [float(v) for v in water_values])
            if len(wv) != n:
                raise ValueError("water_values must have n_params entries")
            self.register_buffer("water_values",
                                 torch.tensor(wv, dtype=torch.float32, device=dev),
                                 persistent=False)
        else:
            self.water_mask = None
            self.water_values = None

        # ---- shared encoder (hash) + SIREN trunk with N-channel head ----
        if self.use_hash_encoding:
            hash_kwargs = dict(
                n_levels=int(hash_levels),
                n_features_per_level=int(hash_features_per_level),
                log2_hashmap_size=int(hash_log2_size),
                base_resolution=hash_base_resolution,
                finest_resolution=hash_finest_resolution,
                backend=str(hash_backend))
            if bool(hash_growing):
                self.encoder = GrowingHashGrid(
                    dim=self.dim, max_levels=int(hash_levels),
                    n_features_per_level=int(hash_features_per_level),
                    log2_hashmap_size=int(hash_log2_size),
                    base_resolution=hash_base_resolution,
                    finest_resolution=hash_finest_resolution,
                    initial_levels=int(hash_c2f_base_levels))
            elif bool(hash_c2f):
                self.encoder = CoarseToFineHashGrid(
                    dim=self.dim, base_levels=int(hash_c2f_base_levels),
                    ramp=str(hash_c2f_ramp), **hash_kwargs)
            else:
                self.encoder = MultiResHashGrid(dim=self.dim, **hash_kwargs)
            in_features = int(self.encoder.n_output_dims)
        else:
            self.encoder = None
            in_features = self.dim

        self.mlp = SirenMLP(
            in_features=in_features,
            out_features=self.n_params,
            hidden_features=int(hidden_features),
            hidden_layers=int(hidden_layers),
            first_omega0=float(first_omega0),
            hidden_omega0=float(hidden_omega0),
            bias=bool(use_bias))

    # -- VelocityINR-compatible aliases (channel 0 = primary parameter) ----
    @property
    def base_velocity(self) -> torch.Tensor:
        """Primary-parameter (channel-0) base — mirrors VelocityINR.base_velocity."""
        return self.base_stack[0]

    def forward(self) -> torch.Tensor:
        return self.render()

    # ---- rendering -------------------------------------------------------
    def _render_all_at(self, coords: torch.Tensor, shape: Tuple[int, ...],
                       base_stack: torch.Tensor, water_mask: torch.Tensor | None
                       ) -> torch.Tensor:
        """Render all channels at ``coords`` -> ``(n_params, *shape)``."""
        feats = self.encoder(coords) if self.encoder is not None else coords
        raw = self.mlp(feats)                              # (Npts, n_params)
        raw = raw.reshape(*shape, self.n_params).movedim(-1, 0)   # (n, *shape)
        delta = raw * self.stds.view(self._bshape) + self.means.view(self._bshape)
        fields = delta if self.direct_velocity else base_stack + delta
        # per-channel clamp (inf bounds are no-ops)
        fields = torch.minimum(fields, self.bound_hi.view(self._bshape))
        fields = torch.maximum(fields, self.bound_lo.view(self._bshape))
        if water_mask is not None and self.water_values is not None:
            # Per-channel pin: a NaN water_value leaves that channel un-pinned,
            # so different parameters can opt in/out of the water mask.
            pin = ~torch.isnan(self.water_values)                     # (n,)
            if bool(pin.any()):
                wv = torch.nan_to_num(self.water_values, nan=0.0).view(self._bshape)
                chan_mask = (water_mask.unsqueeze(0) & pin.view(self._bshape)).expand_as(fields)
                fields = torch.where(chan_mask, wv.expand_as(fields), fields)
        return fields

    def render_all(self, chunk_rows: int | None = None) -> torch.Tensor:
        """Render every parameter -> ``(n_params, *shape)`` (channel 0 = vp).

        ``chunk_rows`` bounds peak memory by rendering z-slabs (bit-identical,
        pointwise). Under ``no_grad`` the caller controls the graph.
        """
        nz = self.shape[0]
        lat = 1
        for s in self.shape[1:]:
            lat *= int(s)
        if chunk_rows is None:
            _CHUNK_POINTS = 2_000_000
            chunk_rows = (max(1, _CHUNK_POINTS // lat)
                          if lat > 0 and nz * lat > _CHUNK_POINTS else nz)
        if chunk_rows >= nz:
            return self._render_all_at(self.coords, self.shape, self.base_stack,
                                       self.water_mask)
        slabs = []
        for z0 in range(0, nz, int(chunk_rows)):
            z1 = min(nz, z0 + int(chunk_rows))
            slabs.append(self._render_slab(z0, z1))
        return torch.cat(slabs, dim=1)                     # cat along z (axis 1)

    def _render_slab(self, z0: int, z1: int) -> torch.Tensor:
        """Render all channels for z-rows [z0, z1) -> (n_params, z1-z0, *lateral)."""
        rows = z1 - z0
        slab_shape = (rows,) + self.shape[1:]
        cc = self.coords.reshape(self.shape[0], -1, self.dim)[z0:z1].reshape(-1, self.dim)
        base = self.base_stack[:, z0:z1]
        wm = self.water_mask[z0:z1] if self.water_mask is not None else None
        return self._render_all_at(cc, slab_shape, base, wm)

    def render_param(self, i: int, chunk_rows: int | None = None) -> torch.Tensor:
        """Render parameter ``i`` alone -> ``*shape``."""
        return self.render_all(chunk_rows=chunk_rows)[i]

    def render(self, chunk_rows: int | None = None) -> torch.Tensor:
        """Render the PRIMARY parameter (vp, channel 0) — VelocityINR-compatible."""
        return self.render_all(chunk_rows=chunk_rows)[0]

    # ---- joint chunked backward -----------------------------------------
    def backward_gradients(self, grads: Sequence[torch.Tensor], *,
                           chunk_rows: int = 64) -> None:
        """Back-propagate a per-parameter gradient list onto the shared trunk.

        ``grads[i]`` is ``dL/d(field_i)`` (same shape as the model). Renders one
        z-slab of ALL channels at a time and backwards them together, so the
        shared trunk accumulates every parameter's contribution while peak
        memory stays O(chunk_rows) — the multi-parameter analogue of
        VelocityINR.backward_velocity_gradient.
        """
        if len(grads) != self.n_params:
            raise ValueError(f"expected {self.n_params} grads, got {len(grads)}")
        dev = self.base_stack.device
        gs = []
        for k, g in enumerate(grads):
            g = torch.as_tensor(g, dtype=torch.float32, device=dev)
            if tuple(g.shape) != self.shape:
                raise ValueError(
                    f"grads[{k}] shape {tuple(g.shape)} != model shape {self.shape}")
            gs.append(g)
        nz = self.shape[0]
        rows = max(1, int(chunk_rows))
        for z0 in range(0, nz, rows):
            z1 = min(nz, z0 + rows)
            fields = self._render_slab(z0, z1)             # (n, rows, *lat) w/ grad
            torch.autograd.backward(
                [fields[i] for i in range(self.n_params)],
                [gs[i][z0:z1] for i in range(self.n_params)],
                retain_graph=False)
