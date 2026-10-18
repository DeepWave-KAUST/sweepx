"""Shot-gather displays."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .colormaps import SEISMIC_CMAP


def plot_shot(
    data,
    *,
    dt: float | None = None,
    dh: float | None = None,
    ax: Axes | None = None,
    perc: float = 99.0,
    cmap: str = SEISMIC_CMAP,
    title: str | None = None,
    **imshow_kwargs: Any,
) -> Axes:
    """Image-style shot gather display.

    Parameters
    ----------
    data
        ``(nt, nrec)`` array.
    dt
        Time sample interval (s). Used for y-axis extent.
    dh
        Receiver spacing (m). Used for x-axis extent.
    perc
        Percentile for symmetric clipping; ``99.0`` is a good default.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"plot_shot expects (nt, nrec); got shape {data.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    s = np.percentile(np.abs(data), perc)
    nt, nrec = data.shape
    extent: tuple[float, float, float, float] | None = None
    if dt is not None and dh is not None:
        extent = (0.0, nrec * dh, nt * dt, 0.0)
        ax.set_xlabel("offset (m)")
        ax.set_ylabel("time (s)")
    elif dt is not None:
        extent = (0.0, float(nrec), nt * dt, 0.0)
        ax.set_xlabel("trace")
        ax.set_ylabel("time (s)")
    im = ax.imshow(
        data, cmap=cmap, vmin=-s, vmax=s, aspect="auto",
        extent=extent, **imshow_kwargs,
    )
    if title is not None:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return ax


def plot_wiggle(
    data,
    *,
    dt: float | None = None,
    skip: int = 1,
    ax: Axes | None = None,
    scale: float = 1.0,
    fill: bool = True,
) -> Axes:
    """Wiggle display — one trace at a time, with optional positive-lobe fill.

    For large gathers (more than ~50 traces) prefer :func:`plot_shot`.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"plot_wiggle expects (nt, nrec); got shape {data.shape}")
    nt, nrec = data.shape
    t = np.arange(nt) * (dt if dt is not None else 1.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    max_abs = np.max(np.abs(data)) + 1e-12
    for j in range(0, nrec, skip):
        tr = data[:, j] / max_abs * scale
        ax.plot(tr + j, t, color="k", linewidth=0.6)
        if fill:
            ax.fill_betweenx(t, j, tr + j, where=(tr > 0), color="k", linewidth=0)
    ax.set_xlabel("trace")
    ax.set_ylabel("time (s)" if dt is not None else "sample")
    ax.set_xlim(-1, nrec)
    ax.invert_yaxis()
    return ax


__all__ = ["plot_shot", "plot_wiggle"]
