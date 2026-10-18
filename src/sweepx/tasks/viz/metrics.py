"""Velocity-model comparison metrics: SNR, PSNR, SSIM, relative L2.

Pure-numpy, no scikit-image dependency. SSIM is the standard Wang et al.
(2004) formulation with a constant 7×7 box window. For more elaborate
needs use ``skimage.metrics`` directly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def relative_l2(est, ref, *, eps: float = 1e-12) -> float:
    """``|| est - ref || / || ref ||`` (L2 norm)."""
    est = np.asarray(est, dtype="float64")
    ref = np.asarray(ref, dtype="float64")
    return float(np.linalg.norm(est - ref) / (np.linalg.norm(ref) + eps))


def snr_db(est, ref, *, eps: float = 1e-12) -> float:
    """Signal-to-noise ratio (dB), referenced to ``ref``."""
    est = np.asarray(est, dtype="float64")
    ref = np.asarray(ref, dtype="float64")
    num = float(np.sum(ref * ref))
    den = float(np.sum((est - ref) ** 2)) + eps
    return 10.0 * np.log10(num / den)


def psnr_db(est, ref, *, data_range: float | None = None) -> float:
    """Peak-SNR (dB). ``data_range`` defaults to ``ref.max() - ref.min()``."""
    est = np.asarray(est, dtype="float64")
    ref = np.asarray(ref, dtype="float64")
    if data_range is None:
        data_range = float(ref.max() - ref.min())
    mse = float(np.mean((est - ref) ** 2)) + 1e-30
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(est, ref, *, window: int = 7, K1: float = 0.01, K2: float = 0.03) -> float:
    """Wang et al. (2004) SSIM on the full image. Returns a scalar in [-1, 1]."""
    est = np.asarray(est, dtype="float64")
    ref = np.asarray(ref, dtype="float64")
    if est.shape != ref.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {ref.shape}")
    L = float(ref.max() - ref.min())
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_x = uniform_filter(est, window)
    mu_y = uniform_filter(ref, window)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = uniform_filter(est * est, window) - mu_x2
    sigma_y2 = uniform_filter(ref * ref, window) - mu_y2
    sigma_xy = uniform_filter(est * ref, window) - mu_xy
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


__all__ = ["relative_l2", "snr_db", "psnr_db", "ssim"]
