"""Test the 3-D orthogonal-slice plotter (§3.6)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from sweep_tasks.viz.model import plot_vp_ortho_slices


def _ramp_volume(shape=(8, 8, 16)):
    nz, ny, nx = shape
    z = np.linspace(0, 1, nz)[:, None, None]
    y = np.linspace(0, 0.5, ny)[None, :, None]
    x = np.linspace(0, 0.25, nx)[None, None, :]
    return (2000.0 + 1000.0 * (z + y + x)).astype(np.float32)


def test_ortho_slices_returns_axes_and_correct_extents():
    vol = _ramp_volume()
    fig, axes = plot_vp_ortho_slices(vol, dh_xyz=(25.0, 25.0, 25.0))
    assert len(axes) == 3
    # Each axis should display the slice with extent matching the
    # corresponding dh × dim. The first axis is XY (cols=x*dx, rows=y*dy).
    nz, ny, nx = vol.shape
    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()
    assert xlim[1] == pytest.approx(nx * 25.0)
    # imshow puts the y axis inverted when extent has the top-left convention;
    # check both ends.
    assert max(ylim) == pytest.approx(ny * 25.0)


def test_ortho_slices_respects_anisotropic_dh():
    vol = _ramp_volume(shape=(10, 6, 12))
    fig, axes = plot_vp_ortho_slices(vol, dh_xyz=(10.0, 50.0, 25.0))
    # XY axis extents: cols=x*dx=12*25, rows=y*dy=6*50.
    assert axes[0].get_xlim()[1] == pytest.approx(12 * 25.0)
    assert max(axes[0].get_ylim()) == pytest.approx(6 * 50.0)
    # YZ axis extents: cols=y*dy=6*50, rows=z*dz=10*10.
    assert axes[2].get_xlim()[1] == pytest.approx(6 * 50.0)
    assert max(axes[2].get_ylim()) == pytest.approx(10 * 10.0)


def test_ortho_slices_rejects_2d():
    with pytest.raises(ValueError, match="3-D"):
        plot_vp_ortho_slices(np.zeros((8, 8)), dh_xyz=(10.0, 10.0, 10.0))


def test_ortho_slices_saves_png(tmp_path: Path):
    import matplotlib.pyplot as plt

    vol = _ramp_volume()
    fig, _ = plot_vp_ortho_slices(vol, dh_xyz=(25.0, 25.0, 25.0))
    out = tmp_path / "ortho.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
