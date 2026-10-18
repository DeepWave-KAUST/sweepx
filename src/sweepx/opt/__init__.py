"""sweep-opt - PyTorch optimizers tailored for Full Waveform Inversion (FWI)."""

from sweepx.opt import optim, precond, parameterization, linalg, linesearch

__version__ = "0.1.0"

__all__ = [
    "optim",
    "precond",
    "parameterization",
    "linalg",
    "linesearch",
    "__version__",
]
