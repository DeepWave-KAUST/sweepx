"""``inspect_file`` — let the LLM peek at a file before deciding what to do.

Supports the formats the sweep stack actually consumes:
  - ``.npy`` / ``.npz``  → numpy
  - ``.h5`` / ``.hdf5``  → h5py (optional)
  - ``.segy`` / ``.sgy`` → segyio (optional)
  - ``.bin`` / ``.raw``  → unknown shape, just file size

The returned ``hint`` is a *guess* about what the file likely represents
(velocity model, observed shots, source wavelet, ...), based on shape and dtype
heuristics. The LLM is told to verify with the user, not trust the hint blindly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sweep_agent.tools import register


class InspectFileParams(BaseModel):
    path: str = Field(..., description="Absolute or working-dir-relative file path to inspect.")


def _hint_for_array(shape: tuple[int, ...], dtype: str) -> str:
    """Rough heuristic — what does an ndarray of this shape/dtype usually mean in sweep?"""
    nd = len(shape)
    if nd == 2:
        a, b = shape
        if min(a, b) >= 64 and max(a, b) <= 4096:
            return "likely a 2-D velocity / density model (nz, nx) or (nx, nz)"
        if a > b * 5:
            return "likely a single shot gather (nt, nreceivers) — nt much larger than ntraces"
        return "2-D array — could be a model slice or a single shot"
    if nd == 3:
        return "3-D array — could be a 3-D model (nz, ny, nx) or a stack of shots (nshots, nt, nreceivers)"
    if nd == 4:
        return "4-D array — typical sweep observed-data layout (nshots, nt, nreceivers, nchannels)"
    if nd == 1:
        return f"1-D array of length {shape[0]} — likely a source wavelet or a 1-D profile"
    return f"{nd}-D array — uncommon for sweep workflows; ask the user what it represents"


def _inspect_npy(path: Path) -> dict[str, Any]:
    arr = np.load(path, mmap_mode="r")
    return {
        "format": "npy",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "hint": _hint_for_array(arr.shape, str(arr.dtype)),
    }


def _inspect_npz(path: Path) -> dict[str, Any]:
    with np.load(path, mmap_mode="r") as z:
        entries = {}
        for key in z.files:
            arr = z[key]
            entries[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "hint": _hint_for_array(arr.shape, str(arr.dtype)),
            }
    return {
        "format": "npz",
        "entries": entries,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "hint": f"npz archive with {len(entries)} entries: {sorted(entries)[:10]}",
    }


def _inspect_h5(path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError:
        return {"format": "h5", "error": "h5py not installed; install with `pip install h5py`"}
    datasets: dict[str, Any] = {}

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "hint": _hint_for_array(obj.shape, str(obj.dtype)),
            }

    with h5py.File(path, "r") as f:
        f.visititems(_visit)
    return {
        "format": "h5",
        "datasets": datasets,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "hint": f"HDF5 file with {len(datasets)} datasets",
    }


def _inspect_segy(path: Path) -> dict[str, Any]:
    try:
        import segyio  # type: ignore
    except ImportError:
        return {"format": "segy", "error": "segyio not installed; install with `pip install segyio`"}
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size
        dt_us = int(f.bin[segyio.BinField.Interval])
    return {
        "format": "segy",
        "ntraces": n_traces,
        "nsamples_per_trace": n_samples,
        "dt_seconds": dt_us / 1e6,
        "total_duration_seconds": (n_samples - 1) * dt_us / 1e6,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "hint": "SEG-Y shot data — typically observed seismic; needs --geometry info to map to shots",
    }


def _inspect_raw(path: Path) -> dict[str, Any]:
    return {
        "format": "raw",
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "hint": "Raw binary — shape/dtype unknown; ask the user before loading",
    }


@register(
    name="inspect_file",
    description=(
        "Inspect a data file (velocity model, observed shots, wavelet, ...) without loading it "
        "into memory. Returns format, shape, dtype, and a short hint about what the file likely "
        "represents. Supports .npy, .npz, .h5/.hdf5, .segy/.sgy, .bin/.raw. Call this first "
        "whenever the user mentions a file path — never guess shapes."
    ),
    params_model=InspectFileParams,
)
def inspect_file(args: InspectFileParams) -> dict[str, Any]:
    path = Path(args.path).expanduser()
    if not path.exists():
        return {"error": f"file not found: {path}"}
    if not path.is_file():
        return {"error": f"not a regular file: {path}"}
    ext = path.suffix.lower()
    base = {"path": str(path.resolve())}
    if ext == ".npy":
        return {**base, **_inspect_npy(path)}
    if ext == ".npz":
        return {**base, **_inspect_npz(path)}
    if ext in {".h5", ".hdf5"}:
        return {**base, **_inspect_h5(path)}
    if ext in {".segy", ".sgy"}:
        return {**base, **_inspect_segy(path)}
    return {**base, **_inspect_raw(path)}
