"""Smoke tests for sweep_tasks.viz (absorbed from the retired sweep-viz) —
render to a non-interactive backend."""

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np

import sweep_tasks.viz
from sweep_tasks.viz.convergence import plot_band_progress, plot_loss
from sweep_tasks.viz.metrics import psnr_db, relative_l2, snr_db, ssim
from sweep_tasks.viz.model import plot_vp, plot_vp_diff
from sweep_tasks.viz.seismic import plot_shot, plot_wiggle


def test_viz_package_exposes_submodules():
    assert set(sweep_tasks.viz.__all__) == {
        "colormaps", "convergence", "metrics", "model", "seismic", "wavefield"
    }


def test_plot_vp_returns_axes():
    vp = np.linspace(1500, 4500, 64 * 128, dtype="float32").reshape(64, 128)
    ax = plot_vp(vp, dh=(10.0, 10.0))
    assert ax is not None
    plt.close(ax.figure)


def test_plot_shot_validates_shape():
    import pytest
    with pytest.raises(ValueError):
        plot_shot(np.zeros((10, 10, 10)))


def test_plot_loss_dict():
    ax = plot_loss({"band1": [1.0, 0.5, 0.25], "band2": [0.2, 0.1, 0.05]})
    plt.close(ax.figure)


def test_plot_band_progress():
    hist = list(np.exp(-np.linspace(0, 4, 60)) + 0.01)
    ax = plot_band_progress(hist, band_boundaries=[20, 40], band_labels=["A", "B"])
    plt.close(ax.figure)


def test_plot_wiggle():
    nt, nrec = 100, 5
    arr = np.zeros((nt, nrec))
    arr[40:60, :] = np.hanning(20)[:, None]
    ax = plot_wiggle(arr, dt=0.001, scale=0.4)
    plt.close(ax.figure)


def test_plot_vp_diff():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((40, 80)) * 100 + 2500
    b = a + rng.standard_normal(a.shape) * 50
    ax = plot_vp_diff(a, b, dh=(10.0, 10.0))
    plt.close(ax.figure)


def test_metrics_self_compare_is_identity():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 64)) + 2500
    assert relative_l2(a, a) < 1e-12
    assert snr_db(a, a) > 100
    assert ssim(a, a) > 0.999
    assert psnr_db(a, a) > 100
