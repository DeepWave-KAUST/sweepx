"""The run's water layer: one definition (``water:``), sampled on every grid.

:class:`sweepx.tasks.schemas.WaterSpec` says WHERE the water is once, on the
native model grid (``grid.dh``, the starting model's shape): a seabed map or a
mask. Every inversion path asks :class:`WaterRegion` for the water on the grid
it is about to use -- a stage grid with its own ``dh_m``, a model_plan crop --
so the layer it holds fixed is the same physical layer on every grid. Padding
a grid for DD tiles stays with the caller: it pads the mask the way it pads
the model there (an edge replicate).

* seabed: the grid's columns sample the map at their physical positions
  (:func:`sweepx.tasks._helpers.stage_grid._seabed_to_grid`) and
  :func:`sweepx.tasks.bathymetry.water_mask_from_seabed_depth` marks every cell
  whose centre lies above ``seabed + buffer``. The buffer is a DEPTH,
  ``buffer_cells`` cells of the native grid, whatever the grid's own spacing.
* mask: each cell takes the value of the native cell nearest to its position
  (``i * dh`` from the native origin). A tie goes to the cell with the larger
  index; a position past the native extent takes the edge cell.

Nothing here reads a velocity.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from sweepx.tasks._helpers.stage_grid import (
    _SNAP_CELLS,
    _per_axis,
    _same_spacing,
    _seabed_to_grid,
)


def _nearest_native(n_dst: int, dh: float, dh0: float, n_src: int) -> np.ndarray:
    """Index of the native sample nearest to each sample of a grid at ``dh``.

    Sample ``j`` sits at ``j * dh``, i.e. ``j * dh / dh0`` native cells. A
    tie goes to the larger index; a coordinate within ``_SNAP_CELLS`` below a
    half-integer counts as the tie (``0.7 * 15`` is 10.499999999999998), so
    the rule does not depend on how ``dh / dh0`` rounds.
    """
    c = np.arange(n_dst, dtype=np.float64) * float(dh) / float(dh0)
    return np.clip(np.floor(c + 0.5 + _SNAP_CELLS).astype(np.int64), 0,
                   int(n_src) - 1)


def _load_mask(path: Path, native_shape: tuple[int, ...]) -> np.ndarray:
    m = np.load(path, allow_pickle=False)
    if not isinstance(m, np.ndarray):
        raise ValueError(f"water.mask {path}: expected a .npy array, got an "
                         f"archive with {getattr(m, 'files', '?')}")
    if m.dtype != np.bool_:
        if not (np.issubdtype(m.dtype, np.integer)
                or np.issubdtype(m.dtype, np.floating)):
            raise ValueError(f"water.mask {path}: dtype {m.dtype} is not boolean "
                             "or 0/1")
        if not np.all((m == 0) | (m == 1)):
            raise ValueError(f"water.mask {path}: values must be booleans or "
                             "0/1 (True / 1 = water)")
        m = m != 0
    if tuple(m.shape) != tuple(native_shape):
        raise ValueError(
            f"water.mask {path} has shape {tuple(m.shape)} but the native model "
            f"grid is {tuple(native_shape)}: the mask lives on the grid of the "
            "starting model (grid.dh), before any model_plan crop or stage "
            "resampling")
    return np.ascontiguousarray(m, dtype=np.bool_)


def _load_seabed(path: Path, native_shape: tuple[int, ...]) -> np.ndarray:
    from sweepx.tasks.bathymetry import load_seabed_depth_npz

    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.ndarray):
        raise ValueError(
            f"water.seabed {path} is a plain array; the seabed file is an .npz "
            "with key 'seabed_depth' (sweepx.tasks.bathymetry."
            "save_seabed_depth_npz)")
    obj.close()
    try:
        sd = load_seabed_depth_npz(path)
    except KeyError as err:
        raise ValueError(f"water.seabed: {err.args[0]}") from None
    lat = tuple(native_shape[1:])
    if sd.ndim != len(lat):
        raise ValueError(
            f"water.seabed {path} has shape {tuple(sd.shape)}; a "
            f"{len(native_shape)}-D model {tuple(native_shape)} needs "
            f"{'(nx,)' if len(lat) == 1 else '(ny, nx)'} seabed depths")
    if min(sd.shape) < 1 or not np.all(np.isfinite(sd)):
        raise ValueError(f"water.seabed {path}: depths must be finite and the "
                         "map non-empty")
    if len(lat) == 2 and tuple(sd.shape) != lat and min(sd.shape) > 1:
        # The map may come at any lateral resolution (resampled onto every
        # grid's columns), but it must span the model's lateral window. A
        # survey-frame map would be stretched into the model silently.
        ar_model = float(lat[0] - 1) / float(max(lat[1] - 1, 1))
        ar_sd = float(sd.shape[0] - 1) / float(sd.shape[1] - 1)
        if ar_model > 0 and abs(ar_sd - ar_model) / ar_model > 0.03:
            raise ValueError(
                f"water.seabed {path} shape {tuple(sd.shape)} does not match "
                f"the model's lateral aspect ratio ({lat[0]} x {lat[1]}); "
                "wrong-frame bathymetry? It must span the model's lateral "
                "extent, first and last samples on its first and last columns")
    return sd


class WaterRegion:
    """Where the water is, defined in one frame; :meth:`mask_on_grid` samples it.

    The frame is the native model grid (``frame_shape`` at ``frame_dh``, model
    axis order), or a crop of it (:meth:`cropped`). A grid passed to
    :meth:`mask_on_grid` shares the frame's origin: its sample ``i`` sits at
    ``i * dh`` -- the stage-grid convention of ``_helpers.stage_grid``.
    """

    def __init__(self, spec, native_shape, native_dh):
        self.spec = spec
        self.mode = str(spec.mode)
        self.pin = self.mode == "pin"
        self.velocity = float(spec.velocity)
        self.frame_shape = tuple(int(s) for s in native_shape)
        if len(self.frame_shape) not in (2, 3):
            raise ValueError(f"water: the model must be 2-D or 3-D, got "
                             f"shape {self.frame_shape}")
        self.frame_dh = _per_axis(native_dh, len(self.frame_shape), "native_dh")
        if spec.seabed is not None:
            self.kind = "seabed"
            if isinstance(spec.seabed, float):
                # A flat seabed: the same map, one depth on every column.
                self.path = None
                self._seabed = np.full(self.frame_shape[1:], spec.seabed,
                                       dtype=np.float32)
            else:
                self.path = Path(spec.seabed)
                self._seabed = _load_seabed(self.path, self.frame_shape)
            self._mask = None
            # A depth: buffer_cells cells of the NATIVE grid, on every grid.
            self.buffer_m = float(spec.buffer_cells) * float(self.frame_dh[0])
        else:
            self.kind = "mask"
            self.path = Path(spec.mask)
            self._mask = _load_mask(self.path, self.frame_shape)
            self._seabed = None
            self.buffer_m = 0.0

    @classmethod
    def from_task(cls, task_spec, native_shape, native_dh) -> "WaterRegion | None":
        """The task's water region, or ``None`` when ``water`` is unset/false."""
        from sweepx.tasks.schemas import water_of

        w = water_of(task_spec)
        return None if w is None else cls(w, native_shape, native_dh)

    # ------------------------------------------------------------------ #

    def cropped(self, starts, stops) -> "WaterRegion":
        """The same water on the sub-box ``[starts, stops)`` of this frame.

        ``starts`` / ``stops`` are cell indices per model axis -- the window a
        model_plan crop cut out of the starting model. Seabed depths are
        re-measured from the crop's top row.
        """
        starts = tuple(int(v) for v in starts)
        stops = tuple(int(v) for v in stops)
        if len(starts) != len(self.frame_shape) or len(stops) != len(starts):
            raise ValueError(f"crop {starts}..{stops} vs frame {self.frame_shape}")
        if starts == (0,) * len(starts) and stops == self.frame_shape:
            return self
        if any(a < 0 or b > n or b <= a
               for a, b, n in zip(starts, stops, self.frame_shape)):
            raise ValueError(f"crop {starts}..{stops} leaves the frame "
                             f"{self.frame_shape}")
        out = copy.copy(self)
        box = tuple(slice(a, b) for a, b in zip(starts, stops))
        out.frame_shape = tuple(b - a for a, b in zip(starts, stops))
        if self.kind == "mask":
            out._mask = np.ascontiguousarray(self._mask[box])
            return out
        # The map on this frame's own columns, then the crop's columns.
        sd = _seabed_to_grid(self._seabed, self.frame_shape[1:], self.frame_dh[1:],
                             self.frame_shape[1:], self.frame_dh[1:])
        sd = np.ascontiguousarray(sd[box[1:]], dtype=np.float32)
        if starts[0]:
            sd = (sd - np.float32(starts[0] * self.frame_dh[0])).astype(np.float32)
        out._seabed = sd
        return out

    def mask_on_grid(self, shape, dh) -> np.ndarray:
        """Boolean water mask (True = water) of a grid sharing the frame's origin.

        ``shape`` is the grid's shape (model axes), ``dh`` its spacing (scalar
        or per axis). On the frame itself this is the frame's own mask.
        """
        shape = tuple(int(s) for s in shape)
        if len(shape) != len(self.frame_shape):
            raise ValueError(f"water: grid {shape} vs frame {self.frame_shape}")
        d = _per_axis(dh, len(shape), "dh")
        if self.kind == "seabed":
            from sweepx.tasks.bathymetry import water_mask_from_seabed_depth

            sd = _seabed_to_grid(self._seabed, self.frame_shape[1:],
                                 self.frame_dh[1:], shape[1:], d[1:])
            return water_mask_from_seabed_depth(
                sd, nz=shape[0], dh_z_m=d[0], buffer_m=self.buffer_m)
        if shape == self.frame_shape and _same_spacing(d, self.frame_dh):
            return self._mask.copy()
        idx = [_nearest_native(n, a, a0, n0)
               for n, a, n0, a0 in zip(shape, d, self.frame_shape, self.frame_dh)]
        return np.ascontiguousarray(self._mask[np.ix_(*idx)])

    def describe(self) -> str:
        if self.kind == "mask":
            what = f"mask {self.path}"
        else:
            what = (f"seabed {self.path}" if self.path is not None
                    else f"flat seabed at {float(self.spec.seabed):g} m")
            if self.spec.buffer_cells:
                what += (f" + {self.spec.buffer_cells} native cells "
                         f"({self.buffer_m:g} m)")
        return f"{what}, mode={self.mode}, velocity={self.velocity:g} m/s"


# ---------------------------------------------------------------------- #
# What the paths do with a mask                                           #
# ---------------------------------------------------------------------- #

def zero_water_grad_(tensors, mask) -> None:
    """Zero every tensor's gradient in the water (``mask``: torch bool)."""
    if mask is None:
        return
    for t in tensors:
        g = getattr(t, "grad", None)
        if g is None:
            continue
        if tuple(g.shape) != tuple(mask.shape):
            raise ValueError(
                f"water mask {tuple(mask.shape)} does not fit the gradient "
                f"{tuple(g.shape)}")
        g[mask] = 0.0


def stamp_water_(t, mask, value: float) -> None:
    """Set ``t`` to ``value`` in the water, in place, outside autograd."""
    if mask is None:
        return
    import torch

    if tuple(t.shape) != tuple(mask.shape):
        raise ValueError(f"water mask {tuple(mask.shape)} does not fit the model "
                         f"{tuple(t.shape)}")
    with torch.no_grad():
        t[mask] = value


def rebake_water_(net, mask, values, render=None) -> int:
    """``reset_each_stage`` on a reparam network: drop the render-time pin and
    shift the base in the water so the network's next render there is
    ``values`` (``base[water] += value - render[water]``), after which the water
    inverts freely. ``values`` is the vp value, or one value per channel of a
    multi-parameter network (``None`` = leave that channel). ``render`` renders
    the full grid after the pin is dropped (default ``net()``, or
    ``net.render_all()`` for a multi-parameter network). Returns the number of
    water cells.
    """
    import torch

    multi = hasattr(net, "base_stack")
    with torch.no_grad():
        net.water_mask = None
        m = mask.to(device=(net.base_stack if multi else net.base_velocity).device)
        # The render may live elsewhere (a host-tiled render of a big grid):
        # pick the water cells there, move only those.
        if multi:
            fields = render() if render is not None else net.render_all()
            mr = m.to(fields.device)
            for i, v in enumerate(list(values)):
                if v is None:
                    continue
                net.base_stack[i][m] += (float(v) - fields[i][mr]).to(m.device)
        else:
            r = render() if render is not None else net()
            net.base_velocity[m] += (float(values) - r[m.to(r.device)]).to(m.device)
    return int(m.sum())


__all__ = [
    "WaterRegion",
    "rebake_water_",
    "stamp_water_",
    "zero_water_grad_",
]
