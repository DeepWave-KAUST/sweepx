"""Velocity / impedance / parameter model maps."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .colormaps import IMAGE_CMAP, VP_CMAP


def plot_vp(
    vp,
    *,
    dh: tuple[float, float] | None = None,
    ax: Axes | None = None,
    cmap: str = VP_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    cbar: bool = True,
    cbar_label: str = "vp (m/s)",
    **imshow_kwargs: Any,
) -> Axes:
    """2-D velocity model display.

    Parameters
    ----------
    vp
        2-D array. Convention: rows are depth (z), columns are lateral (x).
    dh
        ``(dz, dx)`` spacing in meters; controls axis extent. If ``None``,
        axes are in samples.
    ax
        Matplotlib axes. Created if not provided.
    cmap, vmin, vmax
        Standard imshow controls.
    title, cbar, cbar_label
        Optional decorations.
    """
    vp = np.asarray(vp)
    if vp.ndim != 2:
        raise ValueError(f"plot_vp expects 2-D vp; got shape {vp.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    extent: tuple[float, float, float, float] | None = None
    if dh is not None:
        dz, dx = dh
        nz, nx = vp.shape
        extent = (0.0, nx * dx, nz * dz, 0.0)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    im = ax.imshow(
        vp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
        extent=extent, **imshow_kwargs,
    )
    if title is not None:
        ax.set_title(title)
    if cbar:
        ax.figure.colorbar(im, ax=ax, label=cbar_label, fraction=0.04, pad=0.02)
    return ax


def plot_vp_diff(
    vp_est,
    vp_true,
    *,
    dh: tuple[float, float] | None = None,
    ax: Axes | None = None,
    cmap: str = IMAGE_CMAP,
    perc: float = 99.0,
    title: str | None = "vp residual",
) -> Axes:
    """Plot ``vp_est - vp_true`` with a symmetric percentile-clipped scale.

    Defaults to the bundled diverging :data:`IMAGE_CMAP` (``"sweep_image"``)
    which is tuned for percentile-clipped kernel / residual maps. Pass
    ``cmap="RdBu_r"`` to recover the legacy red/blue diverging look.
    """
    diff = np.asarray(vp_est) - np.asarray(vp_true)
    s = np.percentile(np.abs(diff), perc)
    return plot_vp(
        diff, dh=dh, ax=ax, cmap=cmap, vmin=-s, vmax=s,
        title=title, cbar_label="Δvp (m/s)",
    )


def plot_vp_ortho_slices(
    vol_zyx,
    *,
    dh_xyz: tuple[float, float, float],
    fig=None,
    axes=None,
    slice_xy_depth_m: float | None = None,
    slice_xz_y_m: float | None = None,
    slice_yz_x_m: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = VP_CMAP,
    title_prefix: str = "vp",
    cbar_label: str = "vp (m/s)",
) -> tuple:
    """Plot a 3-D volume as three orthogonal slices (XY, XZ, YZ).

    Layout: a 1×3 row of subplots — depth slice (XY at fixed z), inline
    slice (XZ at fixed y), and crossline slice (YZ at fixed x). Slice
    positions default to the volume centers. Used by the 3-D FWI QC
    (§3.6 in the OBN porting plan).

    Parameters
    ----------
    vol_zyx
        3-D array of shape ``(nz, ny, nx)``.
    dh_xyz
        Grid spacing ``(dz, dy, dx)`` in meters.
    fig, axes
        Optional pre-existing figure + 3-element axes list. If ``None``,
        a new figure with 3 axes is created.
    slice_*_m
        Slice positions in meters from the model's top/left edge. ``None``
        defaults to the geometric center along that axis.
    vmin, vmax, cmap
        Standard imshow controls (cmap defaults to the bundled
        ``"sweep_vp"`` velocity LUT).

    Returns
    -------
    (fig, axes)
        The matplotlib figure and a length-3 axes list.
    """
    vol = np.asarray(vol_zyx)
    if vol.ndim != 3:
        raise ValueError(f"plot_vp_ortho_slices expects 3-D vol; got shape {vol.shape}")
    nz, ny, nx = vol.shape
    dz, dy, dx = (float(d) for d in dh_xyz)
    if axes is None or fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    z_center = slice_xy_depth_m if slice_xy_depth_m is not None else 0.5 * (nz - 1) * dz
    y_center = slice_xz_y_m if slice_xz_y_m is not None else 0.5 * (ny - 1) * dy
    x_center = slice_yz_x_m if slice_yz_x_m is not None else 0.5 * (nx - 1) * dx
    iz = int(np.clip(round(z_center / dz), 0, nz - 1))
    iy = int(np.clip(round(y_center / dy), 0, ny - 1))
    ix = int(np.clip(round(x_center / dx), 0, nx - 1))

    # XY slice (depth slice): rows=y, cols=x.
    xy = vol[iz, :, :]
    extent_xy = (0.0, nx * dx, ny * dy, 0.0)
    im0 = axes[0].imshow(xy, cmap=cmap, vmin=vmin, vmax=vmax,
                         aspect="auto", extent=extent_xy)
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].set_title(f"{title_prefix} XY @ z={iz * dz:.0f} m")

    # XZ slice (inline at fixed y): rows=z, cols=x.
    xz = vol[:, iy, :]
    extent_xz = (0.0, nx * dx, nz * dz, 0.0)
    axes[1].imshow(xz, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", extent=extent_xz)
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("z (m)")
    axes[1].set_title(f"{title_prefix} XZ @ y={iy * dy:.0f} m")

    # YZ slice (crossline at fixed x): rows=z, cols=y.
    yz = vol[:, :, ix]
    extent_yz = (0.0, ny * dy, nz * dz, 0.0)
    axes[2].imshow(yz, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", extent=extent_yz)
    axes[2].set_xlabel("y (m)"); axes[2].set_ylabel("z (m)")
    axes[2].set_title(f"{title_prefix} YZ @ x={ix * dx:.0f} m")

    fig.colorbar(im0, ax=list(axes), fraction=0.025, pad=0.02, label=cbar_label)
    return fig, axes


def compare_models(
    panels: list,
    labels: list[str],
    out_path: str,
    *,
    dh: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = VP_CMAP,
    perc: tuple[float, float] = (1.0, 99.0),
    residual: tuple | None = None,
    suptitle: str | None = None,
) -> str:
    """Several velocity models side by side on a shared colour scale, with an
    optional residual panel. Saves to ``out_path`` and returns it.

    Typical FWI use: ``panels=[vp_init, vp_inverted, vp_true]`` with
    ``residual=(vp_inverted, vp_true)`` to show how close the inversion got.
    The shared vmin/vmax (percentile-clipped across all panels unless given)
    makes the panels directly comparable.
    """
    arrs = [np.asarray(p) for p in panels]
    if not arrs:
        raise ValueError("compare_models: need at least one model")
    if len(arrs) != len(labels):
        raise ValueError(f"panels ({len(arrs)}) and labels ({len(labels)}) must match")
    if vmin is None or vmax is None:
        allv = np.concatenate([a.ravel() for a in arrs])
        lo, hi = np.percentile(allv, perc)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    n = len(arrs) + (1 if residual is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, a, lab in zip(axes, arrs, labels):
        plot_vp(a, dh=dh, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, title=lab)
    if residual is not None:
        est, tru = residual
        plot_vp_diff(est, tru, dh=dh, ax=axes[len(arrs)], title="inverted − true")
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


__all__ = ["plot_vp", "plot_vp_diff", "plot_vp_ortho_slices", "compare_models"]
