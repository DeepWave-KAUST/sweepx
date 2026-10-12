"""One physical grid per stage: shapes, resampling, geometry, INR frames.

A stage that sets ``dh_m`` runs on its own grid, but that grid samples the
SAME physical model as the native grid (``grid.dh``): stage sample ``i`` sits
at ``i * dh`` from the native origin (the position of native index 0), which
is exactly where geometry lands, ``round(x_m / dh)``. Every model, mask and
coordinate frame that moves between grids goes through this module, so a
stage can never again sample one frame while its sources and receivers live
in another.

The stage grid covers the whole native extent ``(n - 1) * dh_native``::

    n_stage = ceil((n - 1) * dh_native / dh) + 1

so every native sample (and every source or receiver the native grid holds)
maps inside it. When the extent is not a whole number of stage cells, the
last stage sample lies up to one cell beyond the native extent; it takes the
native edge value (linear interpolation clamped at the edge), like the
edge-replicated padding the solvers already use.

What this replaced: ``round(n * dh_native / dh)`` points with the native
extent stretched onto them end to end (``align_corners=True``), while the
solver stepped with the nominal ``dh``. Every coarse stage ran on a model
compressed against its own geometry -- Marmousi 1361 x 281 at 12.5 m lost
2.9 % in depth at 100 m, Overthrust 801 x 801 x 187 at 25 m lost 0.5 %
laterally and 1 % in depth at 100 m -- and receivers at the far edge fell
off the grid.
"""
from __future__ import annotations

import math

import numpy as np

# Relative tolerance for "these two spacings are the same grid". The runner
# compares stage spacings at 1e-9 / 1e-12 elsewhere; anything this close is
# the same grid to the solver.
_SAME_DH_RTOL = 1e-9
# A sample coordinate within this many CELLS of an integer is snapped onto it,
# so a coincident sample is copied exactly even when dh/dh_native is not
# representable (0.3 / 0.1 = 2.9999999999999996).
_SNAP_CELLS = 1e-6
# An extent (or depth) within this many CELLS of a whole number of cells counts
# as that whole number: float noise and a dh_m written to a few decimals
# (33.3333 for 100/3: 600.0006 cells across 20 km) must not add a cell.
_COVER_CELLS = 1e-3


def _per_axis(dh, ndim: int, what: str) -> tuple[float, ...]:
    """Broadcast a scalar spacing to ``ndim`` axes; validate a tuple."""
    if np.ndim(dh) == 0:
        vals = (float(dh),) * ndim
    else:
        vals = tuple(float(v) for v in dh)
        if len(vals) != ndim:
            raise ValueError(f"{what}: expected {ndim} spacings, got {len(vals)}")
    if any(not (v > 0.0) or not math.isfinite(v) for v in vals):
        raise ValueError(f"{what}: spacings must be finite and > 0, got {vals}")
    return vals


def _same_spacing(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(abs(x - y) <= _SAME_DH_RTOL * max(x, y) for x, y in zip(a, b))


def _stage_grid_shape(native_shape, native_dh, dh) -> tuple[int, ...]:
    """Shape of the stage grid that spans the native extent at spacing ``dh``.

    ``native_dh`` / ``dh`` are scalars or one spacing per axis (model axis
    order). Returns ``native_shape`` unchanged when the spacing is the
    native one, so a native-grid stage is exactly the native grid.
    """
    n0 = tuple(int(s) for s in native_shape)
    if any(s < 1 for s in n0):
        raise ValueError(f"native_shape must be positive, got {n0}")
    d0 = _per_axis(native_dh, len(n0), "native_dh")
    d = _per_axis(dh, len(n0), "dh")
    if _same_spacing(d0, d):
        return n0
    out = []
    for n, a, b in zip(n0, d0, d):
        cells = (n - 1) * a / b
        # ceil, with float noise forgiven: 800 * 25 / 100 must stay 200 cells
        k = math.ceil(cells - _COVER_CELLS)
        out.append(max(0, k) + 1)
    return tuple(out)


def _axis_coords(n_dst: int, ratio: float, n_src: int) -> np.ndarray:
    """Source-index coordinate of each destination sample on one axis."""
    c = np.arange(n_dst, dtype=np.float64) * ratio
    r = np.rint(c)
    snap = np.abs(c - r) <= _SNAP_CELLS
    c[snap] = r[snap]
    return np.clip(c, 0.0, float(n_src - 1))


def _resample_to_grid(t, src_dh, dst_shape, dst_dh):
    """Sample a field given on one grid at the points of another.

    Both grids share their origin (index 0 at the same physical point);
    sample ``j`` of the destination sits at ``j * dst_dh``. The value there
    is the separable linear interpolation of ``t`` at that physical point,
    computed from explicit coordinates (no ``align_corners`` convention),
    clamped to the edge value beyond the source extent. Where a destination
    point coincides with a source sample the value is that sample, exactly.

    ``t`` is a float torch tensor (returned on its device and dtype, detached)
    or a numpy array (returned as numpy). Boolean / integer fields are
    refused: a mask must be rebuilt from its definition on the new grid, not
    interpolated.
    """
    import torch

    is_np = isinstance(t, np.ndarray)
    x = torch.from_numpy(np.ascontiguousarray(t)) if is_np else t.detach()
    if not x.is_floating_point():
        raise TypeError(
            f"_resample_to_grid: refusing to interpolate a {x.dtype} field; "
            "rebuild masks from their definition on the target grid")
    dst_shape = tuple(int(s) for s in dst_shape)
    if len(dst_shape) != x.ndim:
        raise ValueError(
            f"_resample_to_grid: field is {x.ndim}-D {tuple(x.shape)} but "
            f"dst_shape={dst_shape}")
    sd = _per_axis(src_dh, x.ndim, "src_dh")
    dd = _per_axis(dst_dh, x.ndim, "dst_dh")
    out = x
    for ax in range(x.ndim):
        n_src, n_dst = int(x.shape[ax]), dst_shape[ax]
        if n_src == n_dst and _same_spacing((sd[ax],), (dd[ax],)):
            continue                                    # identical axis
        c = _axis_coords(n_dst, dd[ax] / sd[ax], n_src)
        i0 = np.minimum(np.floor(c).astype(np.int64), n_src - 1)
        w = c - i0
        lo = out.index_select(ax, torch.as_tensor(i0, device=out.device))
        if not np.any(w > 0.0):
            out = lo
            continue
        i1 = np.minimum(i0 + 1, n_src - 1)
        hi = out.index_select(ax, torch.as_tensor(i1, device=out.device))
        view = [1] * out.ndim
        view[ax] = n_dst
        wt = torch.as_tensor(w, dtype=out.dtype, device=out.device).reshape(view)
        out = lo + (hi - lo) * wt
    out = out.clone() if out is x else out.contiguous()
    return out.numpy() if is_np else out


def _seabed_to_grid(seabed_depth, model_lateral_shape, native_dh,
                    lateral_shape, dh):
    """Seabed-depth map (metres, ``(nx,)`` or ``(ny, nx)``) on a stage grid.

    The map covers the model's lateral extent at any resolution, first and
    last samples on the model's first and last columns (the documented
    convention). At the model's own resolution its spacing is ``native_dh``.
    On the native grid this reproduces the end-to-end bilinear map the
    runner has always used, bit for bit; on a stage grid it samples the map
    at the stage columns' physical positions.
    """
    import torch
    import torch.nn.functional as F

    sd = np.asarray(seabed_depth, dtype=np.float32)
    m_lat = tuple(int(s) for s in model_lateral_shape)
    lat = tuple(int(s) for s in lateral_shape)
    if sd.ndim != len(m_lat) or sd.ndim != len(lat):
        raise ValueError(
            f"seabed_depth {sd.shape} vs model lateral {m_lat} / stage "
            f"lateral {lat}: dimensionality differs")
    d0 = _per_axis(native_dh, sd.ndim, "native_dh")
    d = _per_axis(dh, sd.ndim, "dh")
    if lat == m_lat and _same_spacing(d0, d):
        if sd.shape == lat:
            return sd.copy()
        mode = "linear" if sd.ndim == 1 else "bilinear"
        return F.interpolate(torch.from_numpy(sd)[None, None], size=lat,
                             mode=mode, align_corners=True)[0, 0].numpy()
    spacing = tuple(
        a if n_sd == n_m else (a * (n_m - 1) / (n_sd - 1) if n_sd > 1 else a)
        for n_sd, n_m, a in zip(sd.shape, m_lat, d0))
    return _resample_to_grid(sd, spacing, lat, d)


def _stage_rows_covering(n_rows: int, native_dh: float, dh: float) -> int:
    """Stage rows whose depth lies inside the top ``n_rows`` native rows.

    ``freeze_top_n_rows`` names native rows ``0 .. n_rows-1``, i.e. depths
    ``0 .. (n_rows-1) * native_dh``. A stage row ``j`` (depth ``j * dh``) is
    frozen when it lies in that interval, so the frozen depth is the same
    physical layer on every grid instead of ``n_rows`` cells of whatever size
    the stage uses.
    """
    n_rows = int(n_rows)
    if n_rows <= 0:
        return 0
    if _same_spacing((float(native_dh),), (float(dh),)):
        return n_rows
    depth_cells = (n_rows - 1) * float(native_dh) / float(dh)
    return int(math.floor(depth_cells + _COVER_CELLS)) + 1


def _snap_covered(stage_idx, native_idx, stage_shape, native_shape):
    """Keep positions the native grid holds inside the stage grid.

    ``stage_idx`` / ``native_idx`` are integer grid indices of the same
    positions, coordinates last in sweep order ``(x, z)`` / ``(x, y, z)``
    (the reverse of the model axes). A position inside the native grid is
    inside a coarser stage grid by construction; on a FINER stage a position
    within half a native cell outside the native extent can round one cell
    past the stage edge. Those are moved onto the edge node -- a shift of at
    most half a native cell, the snapping error the native grid already
    accepted. Positions outside the native grid are returned untouched (the
    native run treats them the same way). Returns ``(indices, n_moved)``.
    """
    s = np.asarray(stage_idx, dtype=np.int64)
    nat = np.asarray(native_idx, dtype=np.int64)
    if s.shape != nat.shape:
        raise ValueError(f"index arrays differ: {s.shape} vs {nat.shape}")
    bounds_s = np.asarray(tuple(stage_shape)[::-1], dtype=np.int64)
    bounds_n = np.asarray(tuple(native_shape)[::-1], dtype=np.int64)
    covered = np.all((nat >= 0) & (nat < bounds_n), axis=-1, keepdims=True)
    clipped = np.clip(s, 0, bounds_s - 1)
    moved = covered & (clipped != s)
    if np.any(np.abs(clipped - s)[np.broadcast_to(covered, s.shape)] > 1):
        raise AssertionError(
            "a position inside the native grid fell more than one cell outside "
            "the stage grid; the stage grid does not span the native extent")
    out = np.where(covered, clipped, s)
    return out, int(np.any(moved, axis=-1).sum())


def _set_physical_inr_frame(net, *, shape, dh, frame_shape, frame_dh) -> None:
    """Give a reparam network the native coordinate frame on a stage grid.

    sweep-nn evaluates the network at ``coord_min + (coord_max - coord_min)
    * i / n`` along each axis of whatever grid it holds, so the same physical
    point gets a different coordinate on every grid and the learned field
    stretches at each stage change. On the native grid (``frame_shape`` at
    ``frame_dh``, the grid the network was defined on) that is the frame:
    position ``x`` -> ``x / (n_native * dh_native)``. This writes the same
    mapping for the stage grid, with positions past the native extent
    clamped to the last native sample. A no-op on the native grid itself, so
    native stages render exactly as before. Call it after every
    ``update_base_velocity`` / ``update_base_models``, which rebuild the
    coordinates from the shape alone.
    """
    import torch

    shape = tuple(int(s) for s in shape)
    frame_shape = tuple(int(s) for s in frame_shape)
    if len(shape) != len(frame_shape):
        raise ValueError(f"stage shape {shape} vs frame shape {frame_shape}")
    d = _per_axis(dh, len(shape), "dh")
    d0 = _per_axis(frame_dh, len(shape), "frame_dh")
    if shape == frame_shape and _same_spacing(d, d0):
        return
    if getattr(net, "_aniso", False):
        raise NotImplementedError(
            "reparam.lateral_downsample > 1 renders on its own coarse lateral "
            "grid, whose coordinates sweep-tasks cannot put in the native "
            "frame; with per-stage dh_m the learned field would be stretched "
            "at every stage change. Use lateral_downsample: 1 with stage grids.")
    coords = getattr(net, "coords", None)
    n_pts = int(np.prod(shape))
    if coords is None or tuple(coords.shape) != (n_pts, len(shape)):
        raise RuntimeError(
            f"reparam network coords {None if coords is None else tuple(coords.shape)} "
            f"do not match the stage grid {shape}; call this right after "
            "update_base_velocity / update_base_models (or the sweep-nn "
            "coordinate layout changed)")
    lo, hi = float(net.coord_min), float(net.coord_max)
    axes = []
    for n, a, n0, a0 in zip(shape, d, frame_shape, d0):
        x = np.minimum(np.arange(n, dtype=np.float64) * a, (n0 - 1) * a0)
        axes.append(torch.as_tensor(lo + (hi - lo) * x / (n0 * a0),
                                    dtype=coords.dtype))
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    net.coords = grid.reshape(n_pts, len(shape)).to(coords.device)
    _inv = getattr(net, "_cg_invalidate", None)        # captured CUDA graph
    if callable(_inv):
        _inv()


__all__ = [
    "_stage_grid_shape",
    "_resample_to_grid",
    "_seabed_to_grid",
    "_stage_rows_covering",
    "_snap_covered",
    "_set_physical_inr_frame",
]
