"""Static and animated wavefield displays."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_snapshot(
    field,
    *,
    dh: tuple[float, float] | None = None,
    ax: Axes | None = None,
    perc: float = 99.5,
    cmap: str = "RdBu_r",
    title: str | None = None,
    aspect: str = "equal",
) -> Axes:
    """Single wavefield snapshot.

    ``aspect="equal"`` keeps a circular wavefront circular; ``"auto"`` stretches
    the field to fill the axes (see :func:`compare_snapshots`)."""
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"plot_snapshot expects 2-D; got shape {field.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=_snapshot_figsize(field.shape))
    s = np.percentile(np.abs(field), perc) + 1e-30
    extent = None
    if dh is not None:
        dz, dx = dh
        nz, nx = field.shape
        extent = (0.0, nx * dx, nz * dz, 0.0)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    ax.imshow(field, cmap=cmap, vmin=-s, vmax=s, aspect=aspect, extent=extent)
    if title is not None:
        ax.set_title(title)
    return ax


def _snapshot_figsize(shape, width: float = 6.0) -> tuple[float, float]:
    """A figure whose axes follow the field's own nz:nx, so an equal-aspect
    snapshot fills it instead of floating in white margins."""
    nz, nx = shape[-2:]
    return width, max(2.5, min(width * nz / max(nx, 1) + 0.6, 9.0))


def animate_snapshots(
    frames: Sequence[np.ndarray],
    out_path: str,
    *,
    fps: int = 15,
    dh: tuple[float, float] | None = None,
    perc: float = 99.5,
    cmap: str = "RdBu_r",
) -> str:
    """Render an animation to a GIF or MP4. Requires ``imageio``.

    Returns the output path.
    """
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as e:
        raise ImportError(
            "wavefield animation requires imageio: `pip install imageio` "
            "(or `pip install 'sweepx[animate]'`)."
        ) from e

    rendered: list[np.ndarray] = []
    for frame in frames:
        fig, ax = plt.subplots(figsize=_snapshot_figsize(np.shape(frame)), dpi=100)
        plot_snapshot(frame, dh=dh, ax=ax, perc=perc, cmap=cmap)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        rendered.append(rgb)
        plt.close(fig)
    iio.imwrite(out_path, rendered, fps=fps)
    return out_path


def compare_snapshots(
    fields: Sequence[np.ndarray],
    labels: Sequence[str],
    out_path: str,
    *,
    perc: float = 98.0,
    cmap: str = "RdBu_r",
    dh: tuple[float, float] | None = None,
    suptitle: str | None = None,
    aspect: str = "equal",
) -> str:
    """Several wavefield snapshots side by side with a shared symmetric color
    scale (e.g. isotropic vs VTI vs TTI wavefronts). Saves to ``out_path`` and
    returns it.

    ``aspect`` defaults to ``"equal"`` so distances on screen represent equal
    physical distances — essential for wavefront-shape comparison. Without it,
    matplotlib stretches each panel to fill its slot and the isotropic case can
    look elliptical, confounding the comparison. Pass ``"auto"`` only if you
    deliberately want the panels to fill their slots."""
    arrs = [np.asarray(f) for f in fields]
    if not arrs:
        raise ValueError("compare_snapshots: need at least one field")
    if len(arrs) != len(labels):
        raise ValueError(f"fields ({len(arrs)}) and labels ({len(labels)}) must match")
    s = np.percentile(np.concatenate([np.abs(a).ravel() for a in arrs]), perc) + 1e-30
    n = len(arrs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, field, label in zip(axes, arrs, labels):
        extent = None
        if dh is not None:
            dz, dx = dh
            nz, nx = field.shape
            extent = (0.0, nx * dx, nz * dz, 0.0)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
        ax.imshow(field, cmap=cmap, vmin=-s, vmax=s, aspect=aspect, extent=extent)
        ax.set_title(label)
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _curvilinear_to_physical(field: np.ndarray, topo: np.ndarray) -> np.ndarray:
    """Resample a curvilinear-grid snapshot (surface flattened to row 0) back to
    the physical grid. Each column is stretched from the flat computational
    column onto physical rows [topo[ix], nz-1]; air above the surface is NaN."""
    field = np.asarray(field)
    nz, nx = field.shape
    topo = np.asarray(topo)
    phys = np.full((nz, nx), np.nan, dtype=float)
    eta_comp = np.linspace(0.0, 1.0, nz)
    for ix in range(nx):
        s = int(topo[ix])
        depth = nz - 1 - s
        if depth <= 0:
            continue
        eta_phys = np.linspace(0.0, 1.0, depth + 1)
        phys[s:nz, ix] = np.interp(eta_phys, eta_comp, field[:, ix])
    return phys


def _mask_air(field: np.ndarray, topo: np.ndarray) -> np.ndarray:
    """Mask cells above the surface (physical-grid field, e.g. image method)."""
    out = np.asarray(field).astype(float).copy()
    nz, nx = out.shape
    out[np.arange(nz)[:, None] < np.asarray(topo)[None, :]] = np.nan
    return out


def plot_snapshot_topography(
    field,
    topo,
    *,
    ax: Axes | None = None,
    sources=None,
    receivers=None,
    curvilinear: bool = False,
    perc: float = 99.0,
    cmap: str = "RdBu_r",
    title: str | None = None,
) -> Axes:
    """Wavefield snapshot under irregular topography: air masked white, topo line
    + source (red star) / receiver (yellow ▼) markers overlaid.

    sweep returns snapshots on the physical grid, curvilinear equations
    included, so by default the air is only masked. ``curvilinear=True``
    resamples a snapshot taken on the boundary-fitted grid
    (``snapshot_grid='computational'``) onto the physical grid first.
    Source/receiver positions are (x, depth_below_local_surface) pairs; the
    depth is added to topo[x]."""
    field = np.asarray(field)
    topo = np.asarray(topo)
    phys = _curvilinear_to_physical(field, topo) if curvilinear else _mask_air(field, topo)
    if ax is None:
        _, ax = plt.subplots(figsize=_snapshot_figsize(phys.shape, width=8.0))
    s = np.nanpercentile(np.abs(phys), perc) + 1e-30
    ax.imshow(phys, cmap=cmap, vmin=-s, vmax=s, aspect="equal")
    x = np.arange(len(topo))
    ax.plot(x, topo, "k-", lw=1.3)
    ax.fill_between(x, 0, topo, color="white", zorder=2)
    if sources is not None and len(sources):
        sx = [int(p[0]) for p in sources]
        sz = [float(topo[int(p[0])]) + float(p[1]) for p in sources]
        ax.plot(sx, sz, "r*", ms=13, zorder=3)
    if receivers is not None and len(receivers):
        rx = [int(p[0]) for p in receivers]
        rz = [float(topo[int(p[0])]) + float(p[1]) for p in receivers]
        ax.plot(rx, rz, "yv", ms=4, zorder=3)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")
    ax.set_ylim(len(topo) if False else phys.shape[0], 0)
    return ax


def animate_snapshots_topography(
    frames,
    topo,
    out_path: str,
    *,
    sources=None,
    receivers=None,
    curvilinear: bool = False,
    fps: int = 8,
    perc: float = 99.0,
    cmap: str = "RdBu_r",
) -> str:
    """Animate topography snapshots to GIF/MP4 (air masked, topo + geometry
    overlaid). Requires imageio. Returns the output path."""
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as e:
        raise ImportError(
            "topography animation requires imageio: `pip install imageio` "
            "(or `pip install 'sweepx[animate]'`)."
        ) from e
    rendered: list[np.ndarray] = []
    for frame in frames:
        fig, ax = plt.subplots(figsize=_snapshot_figsize(np.shape(frame), width=8.0), dpi=100)
        plot_snapshot_topography(frame, topo, ax=ax, sources=sources, receivers=receivers,
                                 curvilinear=curvilinear, perc=perc, cmap=cmap)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        rendered.append(rgb)
        plt.close(fig)
    iio.imwrite(out_path, rendered, fps=fps)
    return out_path


__all__ = [
    "plot_snapshot",
    "animate_snapshots",
    "compare_snapshots",
    "plot_snapshot_topography",
    "animate_snapshots_topography",
]
