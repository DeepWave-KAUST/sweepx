"""sweepx — install handle for the sweep engine.

`sweepx` itself has no code: ``pip install sweepx`` pulls in ``sweep-solver``
(the solver engine, with prebuilt CUDA cores), ``sweep-agent`` (the
natural-language control layer) and ``sweep-tasks`` (the YAML task runner,
which brings ``sweep-loss``, ``sweep-io`` and ``sweep-nn``).

**Use ``import sweep``, not ``import sweepx``.**

The Python module name is ``sweep`` (provided by the ``sweep-solver``
distribution). The PyPI distribution name had to change because ``sweep``
was already taken — this is the same pattern as ``scikit-learn`` →
``import sklearn``.
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("sweepx")
except Exception:  # not installed (a bare checkout)
    __version__ = "unknown"

__all__ = ["__version__"]
