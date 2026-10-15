"""sweep-opt - PyTorch optimizers tailored for Full Waveform Inversion (FWI)."""

from sweepx.opt import optim, precond, parameterization, linalg, linesearch

from sweepx import __version__  # one version for the whole toolkit

__all__ = [
    "optim",
    "precond",
    "parameterization",
    "linalg",
    "linesearch",
    "__version__",
]
