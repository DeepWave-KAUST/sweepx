"""Velocity / parameter model load and save.

Format is selected by file extension. The dispatch table `_FORMATS` is
public — register your own format with::

    sweep_io.models.register_format(".myext", load_fn, save_fn)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np

_LoadFn = Callable[[Path, dict], np.ndarray]
_SaveFn = Callable[[Path, np.ndarray, dict], None]

_FORMATS: dict[str, Tuple[_LoadFn, _SaveFn]] = {}


def register_format(ext: str, load_fn: _LoadFn, save_fn: _SaveFn) -> None:
    """Register a (load, save) pair for a file extension (lowercase, with dot)."""
    _FORMATS[ext.lower()] = (load_fn, save_fn)


def load_velocity(
    path: str | Path,
    *,
    shape: Tuple[int, ...] | None = None,
    dtype: str | np.dtype = "float32",
    order: str = "C",
) -> np.ndarray:
    """Load a velocity/parameter model.

    Parameters
    ----------
    path
        File path. Format is dispatched by extension.
    shape, dtype, order
        Used only for raw-binary loads (e.g. `.bin`, `.raw`, or unknown
        extensions). Ignored for self-describing formats.

    Returns
    -------
    np.ndarray
        Array as stored on disk. The caller is responsible for reshape /
        transpose — `sweep-io` does not impose `(nz, nx)` vs `(nx, nz)`.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _FORMATS:
        return _FORMATS[ext][0](path, dict(shape=shape, dtype=dtype, order=order))
    # fallback: raw binary
    return _load_raw(path, {"shape": shape, "dtype": dtype, "order": order})


def save_velocity(
    path: str | Path,
    arr: np.ndarray,
    *,
    header: dict[str, Any] | None = None,
) -> None:
    """Save a velocity/parameter model. Format dispatched by extension.

    `header` is an optional metadata dict; honored only by formats that
    can store it (HDF5). Otherwise silently ignored.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _FORMATS:
        _FORMATS[ext][1](path, np.asarray(arr), header or {})
        return
    _save_raw(path, np.asarray(arr), header or {})


# ---------------------------------------------------------------------- npy
def _load_npy(path: Path, _opts: dict) -> np.ndarray:
    return np.load(path)


def _save_npy(path: Path, arr: np.ndarray, _hdr: dict) -> None:
    np.save(path, arr)


register_format(".npy", _load_npy, _save_npy)


# ---------------------------------------------------------------------- npz
def _load_npz(path: Path, _opts: dict) -> np.ndarray:
    with np.load(path) as f:
        keys = list(f.keys())
        if len(keys) != 1:
            raise ValueError(
                f"{path}: .npz must contain exactly one array (got {keys}); "
                "use np.load directly for multi-array files."
            )
        return f[keys[0]]


def _save_npz(path: Path, arr: np.ndarray, header: dict) -> None:
    if header:
        np.savez(path, vp=arr, **header)
    else:
        np.savez(path, vp=arr)


register_format(".npz", _load_npz, _save_npz)


# ----------------------------------------------------------------- raw / bin
def _load_raw(path: Path, opts: dict) -> np.ndarray:
    shape = opts.get("shape")
    dtype = np.dtype(opts.get("dtype", "float32"))
    order = opts.get("order", "C")
    if shape is None:
        raise ValueError(
            f"{path}: raw binary load requires `shape=...` to be passed."
        )
    return np.fromfile(path, dtype=dtype).reshape(shape, order=order)


def _save_raw(path: Path, arr: np.ndarray, _hdr: dict) -> None:
    np.ascontiguousarray(arr).tofile(path)


register_format(".bin", _load_raw, _save_raw)
register_format(".raw", _load_raw, _save_raw)


# ---------------------------------------------------------------------- h5
def _load_h5(path: Path, _opts: dict) -> np.ndarray:
    try:
        import h5py
    except ImportError as e:
        raise ImportError("HDF5 load requires `pip install sweep-io[hdf5]`.") from e
    with h5py.File(path, "r") as f:
        if "vp" in f:
            return f["vp"][...]
        keys = list(f.keys())
        if len(keys) != 1:
            raise ValueError(
                f"{path}: HDF5 must contain a 'vp' dataset or exactly one "
                f"dataset (got {keys})."
            )
        return f[keys[0]][...]


def _save_h5(path: Path, arr: np.ndarray, header: dict) -> None:
    try:
        import h5py
    except ImportError as e:
        raise ImportError("HDF5 save requires `pip install sweep-io[hdf5]`.") from e
    with h5py.File(path, "w") as f:
        d = f.create_dataset("vp", data=arr)
        for k, v in header.items():
            d.attrs[k] = v


register_format(".h5", _load_h5, _save_h5)
register_format(".hdf5", _load_h5, _save_h5)
