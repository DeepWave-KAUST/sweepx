"""sweep-io — file and dataset I/O for seismic FWI.

Always-available submodules:
- :mod:`sweep_io.models`     — velocity / parameter model I/O (numpy)
- :mod:`sweep_io.geometry`   — acquisition geometry dataclass
- :mod:`sweep_io.prefetch`   — background prefetchers (stdlib only)
- :mod:`sweep_io.segy`       — SEG-Y readers (low-level byte-offset readers
                                are stdlib + numpy; the full-file
                                :func:`sweep_io.segy.read_segy` /
                                :func:`sweep_io.segy.write_segy` defer
                                ``segyio`` import until called)

Lazy-imported (require optional extras):
- :mod:`sweep_io.datasets`        — torch Dataset wrappers (extra: ``[torch]``)
- :mod:`sweep_io.cuda_prefetch`   — CUDA-aware prefetcher (extra: ``[torch]``)
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import geometry, models, plan, prefetch, segy, segy_index

__all__ = ["models", "geometry", "plan", "prefetch", "segy", "segy_index", "__version__"]


def __getattr__(name: str):
    if name == "datasets":
        from . import datasets as _m
        return _m
    if name == "cuda_prefetch":
        from . import cuda_prefetch as _m
        return _m
    raise AttributeError(f"module 'sweep_io' has no attribute {name!r}")
