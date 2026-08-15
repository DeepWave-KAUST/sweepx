"""sweepx — install handle for the sweep engine.

`sweepx` itself has no code: ``pip install sweepx`` pulls in ``sweep-solver``
(the CUDA-enabled solver engine). Companion packages (``sweep-io``, ``sweep-nn``,
…) are not published in this solver-only release.

**Use ``import sweep``, not ``import sweepx``.**

The Python module name is ``sweep`` (provided by the ``sweep-solver``
distribution). The PyPI distribution name had to change because ``sweep``
was already taken — this is the same pattern as ``scikit-learn`` →
``import sklearn``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
