"""Small numerical helpers shared by several loss functions.

Keeping these in one place avoids subtle inconsistencies (e.g. two different
Hilbert transforms used by the envelope and instantaneous-phase losses).
"""

from __future__ import annotations

from typing import Optional

import torch


# ----------------------------------------------------------------------------
# Hilbert transform / analytic signal
# ----------------------------------------------------------------------------
def hilbert(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Analytic signal a = x + i * H(x) (Hilbert transform along ``dim``).

    Implements the standard FFT-based recipe of Marple (1999) - identical to
    :func:`scipy.signal.hilbert`.  Differentiable w.r.t. ``x``.
    """
    if not torch.is_floating_point(x):
        x = x.float()
    n = x.shape[dim]
    Xf = torch.fft.fft(x, dim=dim)
    h = torch.zeros(n, dtype=Xf.dtype, device=Xf.device)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    # Broadcast h along ``dim``.
    shape = [1] * x.ndim
    shape[dim] = n
    h = h.view(shape)

    return torch.fft.ifft(Xf * h, dim=dim)


def envelope(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Instantaneous amplitude E(t) = |a(t)| of the analytic signal."""
    a = hilbert(x, dim=dim)
    return torch.sqrt(a.real * a.real + a.imag * a.imag + eps)


def instantaneous_phase(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Instantaneous phase phi(t) = atan2( H(x), x )."""
    a = hilbert(x, dim=dim)
    return torch.atan2(a.imag, a.real)


# ----------------------------------------------------------------------------
# Normalisation helpers
# ----------------------------------------------------------------------------
def l2_normalize(
    x: torch.Tensor, dim: int = -1, eps: float = 1e-12
) -> torch.Tensor:
    """Return ``x / ||x||_2`` along ``dim``."""
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / norm


def trace_energy(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Return ``<x, x>`` along ``dim`` (squared L2 norm)."""
    return (x * x).sum(dim=dim, keepdim=True).clamp_min(eps)


# ----------------------------------------------------------------------------
# Positive-mass / probability transforms (used by Wasserstein-family losses)
# ----------------------------------------------------------------------------
def positive_transform(
    x: torch.Tensor, method: str = "square", c: Optional[float] = None
) -> torch.Tensor:
    """Map a signed signal into a non-negative one.

    Common choices (Engquist & Froese 2014; Yang & Engquist 2018):
        * ``"square"``:    x ↦ x^2
        * ``"abs"``:       x ↦ |x|
        * ``"linear"``:    x ↦ x + c   (c >= max(|x|))
        * ``"exp"``:       x ↦ exp(x)
    """
    if method == "square":
        return x * x
    if method == "abs":
        return x.abs()
    if method == "linear":
        if c is None:
            c = float(x.abs().max().item()) + 1e-6
        return x + c
    if method == "exp":
        return torch.exp(x)
    raise ValueError(f"unknown positive transform {method!r}")


def normalize_density(
    x: torch.Tensor, dim: int = -1, eps: float = 1e-12
) -> torch.Tensor:
    """Scale a non-negative signal so that it sums to one along ``dim``."""
    s = x.sum(dim=dim, keepdim=True).clamp_min(eps)
    return x / s


__all__ = [
    "hilbert",
    "envelope",
    "instantaneous_phase",
    "l2_normalize",
    "trace_energy",
    "positive_transform",
    "normalize_density",
]
