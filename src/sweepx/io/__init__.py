"""sweep-io — file and dataset I/O for seismic FWI.

Always-available submodules:
- :mod:`sweepx.io.models`       — velocity / parameter model I/O (numpy)
- :mod:`sweepx.io.geometry`     — acquisition geometry, UTM ↔ model-frame rotation
- :mod:`sweepx.io.prefetch`     — background prefetchers (stdlib only)
- :mod:`sweepx.io.segy`         — SEG-Y readers (low-level byte-offset readers
                                  are stdlib + numpy; the full-file
                                  :func:`sweepx.io.segy.read_segy` /
                                  :func:`sweepx.io.segy.write_segy` defer
                                  ``segyio`` import until called)
- :mod:`sweepx.io.segy_index`   — header catalog over many SEG-Y files
- :mod:`sweepx.io.seismic_plan` — CSG / CRG plans, lazy reader, samplers
- :mod:`sweepx.io.plan`         — shot / receiver / model-window selection
- :mod:`sweepx.io.crg_build`, :mod:`sweepx.io.crg_plan` — CRG plan cache
- :mod:`sweepx.io.wavelet`      — source wavelet from an ``.npz``

Lazy-imported (require optional extras):
- :mod:`sweepx.io.datasets`        — torch Dataset wrappers (extra: ``[torch]``)
- :mod:`sweepx.io.cuda_prefetch`   — CUDA-aware prefetcher (extra: ``[torch]``)
- :mod:`sweepx.io.crg_dataset`     — Dataset over a CRG plan (extra: ``[torch]``)
"""

from __future__ import annotations

__version__ = "0.1.1"

from . import (
    crg_build,
    crg_plan,
    geometry,
    models,
    plan,
    prefetch,
    segy,
    segy_index,
    seismic_plan,
    wavelet,
)

__all__ = [
    "crg_build", "crg_plan", "models", "geometry", "plan", "prefetch",
    "segy", "segy_index", "seismic_plan", "wavelet", "__version__",
]


def __getattr__(name: str):
    if name == "datasets":
        from . import datasets as _m
        return _m
    if name == "cuda_prefetch":
        from . import cuda_prefetch as _m
        return _m
    if name == "crg_dataset":
        # Lazy because it imports torch.utils.data; sweep-io's base install
        # is torch-free until the user asks for the torch dataset wrappers.
        from . import crg_dataset as _m
        return _m
    raise AttributeError(f"module 'sweepx.io' has no attribute {name!r}")
