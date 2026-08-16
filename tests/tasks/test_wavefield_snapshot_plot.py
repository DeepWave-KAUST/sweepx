"""The snapshot figure: the model underneath, and 3-D reduced to one inline.

Before this, `_plot_wavefield_snapshots` drew the wavefield alone at a fixed
4x4 inches per panel, and handed a raw 3-D volume straight to `imshow` — which
raised "Invalid shape ... for image data", so a 3-D wavefield run wrote a
snapshots.npy and no figure at all.
"""
import numpy as np
import pytest

from sweep_tasks._helpers.plotting import _plot_wavefield_snapshots


def _snaps2d(nsnap=2, nz=40, nx=90, abcn=5, free_surface=False):
    """Raw snapshots in the (nsnap, nfield, 1, 1, z, x) binding layout."""
    pz = nz + (abcn if free_surface else 2 * abcn)
    vol = np.zeros((nsnap, 3, 1, 1, pz, nx + 2 * abcn), np.float32)
    vol[:, 0, 0, 0, pz // 2, (nx + 2 * abcn) // 2] = 1.0
    return vol


def _snaps3d(nsnap=2, nz=20, ny=24, nx=30, abcn=4, free_surface=False):
    pz = nz + (abcn if free_surface else 2 * abcn)
    vol = np.zeros((nsnap, 3, 1, 1, pz, ny + 2 * abcn, nx + 2 * abcn), np.float32)
    vol[:, 0, 0, 0, pz // 2, (ny + 2 * abcn) // 2, (nx + 2 * abcn) // 2] = 1.0
    return vol


@pytest.mark.parametrize("free_surface", [False, True])
def test_2d_snapshot_with_model_background(tmp_path, free_surface):
    nz, nx, abcn = 40, 90, 5
    out = _plot_wavefield_snapshots(
        _snaps2d(nz=nz, nx=nx, abcn=abcn, free_surface=free_surface),
        [10, 20], abcn, (nz, nx), tmp_path / "s.png", free_surface,
        model=np.linspace(1500, 4000, nz)[:, None] * np.ones((1, nx), np.float32),
        model_label="vp (m/s)", dh=12.5)
    assert out.exists() and out.stat().st_size > 0


def test_3d_gives_three_orthogonal_cuts(tmp_path):
    """A 3-D volume is shown as depth slice / inline / crossline, not refused.

    It used to reach `imshow` whole and die on "Invalid shape ... for image
    data", leaving a snapshots.npy and no figure.
    """
    from PIL import Image

    nz, ny, nx, abcn = 20, 24, 30, 4
    model = np.broadcast_to(
        np.linspace(1800, 3200, nz)[:, None, None], (nz, ny, nx)).astype(np.float32)
    out = _plot_wavefield_snapshots(
        _snaps3d(nz=nz, ny=ny, nx=nx, abcn=abcn), [10, 20], abcn,
        (nz, ny, nx), tmp_path / "s3.png", False,
        model=model, model_label="vp (m/s)", dh=20.0, slice_xyz=(nx // 3, ny // 3, nz // 3))
    assert out.exists() and out.stat().st_size > 0
    # three columns for a 3-D grid against one for 2-D, at the same row count
    two_d = _plot_wavefield_snapshots(
        _snaps2d(nz=nz, nx=nx, abcn=abcn), [10, 20], abcn, (nz, nx),
        tmp_path / "s2.png", False, model=np.ones((nz, nx), np.float32), dh=20.0)
    assert Image.open(out).size[0] > Image.open(two_d).size[0]


def test_3d_slice_indices_are_clamped(tmp_path):
    """Out-of-range cuts must not raise — it is a plot, not a computation."""
    nz, ny, nx, abcn = 20, 24, 30, 4
    for bad in ((-5, -5, -5), (10_000, 10_000, 10_000)):
        out = _plot_wavefield_snapshots(
            _snaps3d(nz=nz, ny=ny, nx=nx, abcn=abcn), [10, 20], abcn,
            (nz, ny, nx), tmp_path / f"s{bad[0]}.png", False,
            model=np.ones((nz, ny, nx), np.float32), dh=20.0, slice_xyz=bad)
        assert out.exists()


def test_model_is_optional_and_a_mismatched_one_is_dropped(tmp_path):
    """No model, or a model that is not the grid, still yields a figure."""
    nz, nx, abcn = 40, 90, 5
    snaps = _snaps2d(nz=nz, nx=nx, abcn=abcn)
    for model in (None, np.ones((7, 7), np.float32), np.ones((nz, nx, 2), np.float32)):
        out = _plot_wavefield_snapshots(
            snaps, [10, 20], abcn, (nz, nx), tmp_path / "m.png", False,
            model=model, dh=12.5)
        assert out.exists()


def test_figure_is_shaped_by_the_grid_not_a_fixed_size(tmp_path):
    """A wide model must not come out square — that squashed Marmousi ~5x."""
    from PIL import Image

    wide = _plot_wavefield_snapshots(
        _snaps2d(nsnap=1, nz=20, nx=200, abcn=5), [1], 5,
        (20, 200), tmp_path / "wide.png", False,
        model=np.ones((20, 200), np.float32))
    tall = _plot_wavefield_snapshots(
        _snaps2d(nsnap=1, nz=100, nx=110, abcn=5), [1], 5,
        (100, 110), tmp_path / "tall.png", False,
        model=np.ones((100, 110), np.float32))
    aw = Image.open(wide).size[0] / Image.open(wide).size[1]
    at = Image.open(tall).size[0] / Image.open(tall).size[1]
    assert aw > at, f"wide grid did not produce a wider figure ({aw:.2f} vs {at:.2f})"
