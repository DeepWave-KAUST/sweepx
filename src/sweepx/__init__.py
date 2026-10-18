"""sweepx — the sweep toolkit.

The wave-equation engine is its own package (``import sweep``, from
``sweep-solver``). Everything built on it lives here, in three layers:

=================  ==========================================================
``sweepx.io``      SEG-Y, acquisition geometry, data and model plans
``sweepx.loss``    misfit functions for FWI
``sweepx.nn``      neural reparameterizations and priors (INR, hash, SIREN)
``sweepx.opt``     optimizers, preconditioners and parameterizations for FWI
``sweepx.tasks``   the YAML task runner: forward / FWI / LSRTM / RTM, CLI
``sweepx.agent``   natural-language control layer over the task runner
=================  ==========================================================

A layer imports only from the layers below it (checked by import-linter).
Subpackages load on first access, so ``import sweepx`` stays cheap.
"""

import importlib

try:
    from importlib.metadata import version as _version

    __version__ = _version("sweepx")
except Exception:  # not installed (a bare checkout)
    __version__ = "unknown"

_SUBPACKAGES = ("io", "loss", "nn", "opt", "tasks", "agent")

__all__ = ["__version__", *_SUBPACKAGES]


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBPACKAGES))
