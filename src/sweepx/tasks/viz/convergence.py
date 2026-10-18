"""FWI loss / misfit curves."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_loss(
    history: Sequence[float] | Mapping[str, Sequence[float]],
    *,
    ax: Axes | None = None,
    logy: bool = True,
    title: str | None = None,
    xlabel: str = "iteration",
    ylabel: str = "loss",
) -> Axes:
    """Plot one or more loss series.

    Parameters
    ----------
    history
        Either a 1-D sequence or a ``{label: sequence}`` mapping.
    logy
        Use a log-scale y-axis (default: ``True``).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if isinstance(history, Mapping):
        for label, vals in history.items():
            ax.plot(np.asarray(vals), label=label)
        ax.legend()
    else:
        ax.plot(np.asarray(history))
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_band_progress(
    history: Sequence[float],
    band_boundaries: Sequence[int],
    band_labels: Sequence[str] | None = None,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot loss with vertical bands marking frequency-band stages.

    ``band_boundaries[i]`` is the iteration where band ``i`` *ends*.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax = plot_loss(history, ax=ax, logy=True)
    for i, b in enumerate(band_boundaries):
        ax.axvline(b, color="0.5", linestyle="--", linewidth=0.8)
        if band_labels is not None and i < len(band_labels):
            ax.text(b, ax.get_ylim()[1], f" {band_labels[i]}", va="top", fontsize=8)
    return ax


__all__ = ["plot_loss", "plot_band_progress"]
