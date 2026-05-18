"""Bathymetry (seabed-depth) helpers for FWI.

Generates a 2-D seabed-depth map from a 3-D init vp model and saves it
in the npz format consumed by :class:`sweep_tasks.schemas.FreezeWaterLayerSpec`
and the velocity-INR water-pin path (see ``ReparamSpec.seabed_depth_path``).

The convention — shared across these two consumers and matched by
:class:`sweep_nn.SeabedFreezeMask`:

* ``seabed_depth`` array, shape ``(nx,)`` for 2-D models or ``(ny, nx)``
  for 3-D models.
* Values are depths in **meters** measured from grid index ``z=0``
  (typically the sea surface).
* Saved as an npz with a single key ``"seabed_depth"``.

Building the map alongside the init vp avoids two foot-guns:

1. **Float-exact equality**: inferring the water mask via
   ``vp == 1500.0`` only works when the init was stamped to exactly
   1500. A smoothing step or unit conversion can break that silently.
2. **Per-column bathymetry**: real surveys have varying water depth.
   A 2-D map captures it correctly; a flat ``z < N`` rule does not.

CLI:
    sweep-tasks derive-seabed-depth INIT_VP.npy \\
        --dh-z 75 --water-vp 1500 --output seabed_depth.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def derive_seabed_depth_from_vp(
    init_vp: np.ndarray,
    *,
    dh_z_m: float,
    water_vp: float = 1500.0,
    atol: float = 1e-3,
) -> np.ndarray:
    """Compute the seabed depth (m) per ``(y, x)`` column.

    For each column, finds the first ``z`` row whose vp differs from
    ``water_vp`` (within ``atol``) and returns that boundary in meters
    via ``z * dh_z_m``. Columns that are 1500 m/s all the way down
    (no rock at all) get ``nz * dh_z_m`` (= the model bottom).

    Parameters
    ----------
    init_vp
        ``(nz, nx)`` (2-D) or ``(nz, ny, nx)`` (3-D) initial velocity
        model in m/s.
    dh_z_m
        Vertical cell size in meters.
    water_vp
        Velocity of the water column (default 1500 m/s).
    atol
        Absolute tolerance for the water-vs-non-water comparison.
        Use a tight default; bump up if the init was smoothed.

    Returns
    -------
    np.ndarray
        Shape ``(nx,)`` (for 2-D input) or ``(ny, nx)`` (for 3-D input),
        ``float32`` depths in meters.
    """
    vp = np.asarray(init_vp)
    if vp.ndim not in (2, 3):
        raise ValueError(
            f"derive_seabed_depth_from_vp: expected 2-D or 3-D vp; "
            f"got shape {vp.shape}"
        )
    nz = int(vp.shape[0])
    # Per-voxel "is water" indicator (within tolerance).
    is_water = np.abs(vp - float(water_vp)) <= float(atol)
    # First non-water row per column: argmin over False == argmax over
    # True-then-False prefix. argmax returns the FIRST True; we want the
    # first False, so look at ~is_water.
    first_nonwater = np.argmax(~is_water, axis=0).astype(np.int64)
    # ``argmax`` returns 0 when all values are False — the SAME index
    # as "first row is non-water". Disambiguate: if column is entirely
    # water (all True), set depth = nz (model bottom).
    all_water = is_water.all(axis=0)
    first_nonwater = np.where(all_water, nz, first_nonwater)
    return (first_nonwater.astype(np.float32) * float(dh_z_m)).astype(np.float32)


def save_seabed_depth_npz(
    path: Union[str, Path], seabed_depth: np.ndarray,
) -> Path:
    """Save a seabed-depth array to npz under the canonical key."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, seabed_depth=np.asarray(seabed_depth, dtype=np.float32))
    return out


def load_seabed_depth_npz(path: Union[str, Path]) -> np.ndarray:
    """Inverse of :func:`save_seabed_depth_npz`."""
    with np.load(Path(path)) as data:
        if "seabed_depth" not in data.files:
            raise KeyError(
                f"{path}: expected key 'seabed_depth' (found {data.files})"
            )
        return np.asarray(data["seabed_depth"], dtype=np.float32)


def water_mask_from_seabed_depth(
    seabed_depth: np.ndarray,
    *,
    nz: int,
    dh_z_m: float,
    buffer_cells: int = 0,
    use_cell_center: bool = True,
) -> np.ndarray:
    """Broadcast a 2-D seabed map into a 3-D bool water mask.

    Parameters
    ----------
    seabed_depth
        ``(nx,)`` or ``(ny, nx)`` array, depths in meters from z=0.
    nz
        Number of z cells in the target grid.
    dh_z_m
        Vertical cell size in meters.
    buffer_cells
        Number of extra z-cells BELOW the picked seabed to also mark
        as "water" — useful to absorb the wavelet's rise time across
        the interface.
    use_cell_center
        When True (default), a cell at index ``z`` represents the
        depth interval ``[z*dh, (z+1)*dh)``, and the cell is "water"
        when its CENTER ``(z+0.5)*dh < seabed_depth + buffer*dh``.
        When False, uses the cell TOP — slightly more aggressive
        masking (one extra cell at the boundary).

    Returns
    -------
    np.ndarray
        Bool mask, shape ``(nz, nx)`` (2-D) or ``(nz, ny, nx)`` (3-D),
        True where the cell is in the water column.
    """
    sd = np.asarray(seabed_depth, dtype=np.float32)
    if sd.ndim not in (1, 2):
        raise ValueError(
            f"water_mask_from_seabed_depth: seabed_depth must be 1-D "
            f"(nx,) or 2-D (ny, nx); got shape {sd.shape}"
        )
    z_idx = np.arange(int(nz), dtype=np.float32)
    z_pos = (z_idx + (0.5 if use_cell_center else 0.0)) * float(dh_z_m)
    buf_m = float(buffer_cells) * float(dh_z_m)
    threshold = sd + buf_m  # shape matches sd
    # Broadcast: (nz, 1) [< (ny, nx)] -> (nz, ny, nx) ; or (nz, 1) [< (nx,)] -> (nz, nx)
    if sd.ndim == 1:
        mask = z_pos[:, None] < threshold[None, :]
    else:
        mask = z_pos[:, None, None] < threshold[None, :, :]
    return mask.astype(np.bool_)


__all__ = [
    "derive_seabed_depth_from_vp",
    "save_seabed_depth_npz",
    "load_seabed_depth_npz",
    "water_mask_from_seabed_depth",
]
