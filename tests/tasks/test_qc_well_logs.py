"""Unit tests for ``save_vp_well_logs_png`` (per-iter vp(z) at a set of
pseudo-wells).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _synth_vp(nz: int = 60, ny: int = 40, nx: int = 30):
    """Synthetic vp volume with a 1500 m/s water layer up top + linear
    gradient below."""
    vp = np.zeros((nz, ny, nx), dtype=np.float32)
    seabed = 5
    vp[:seabed] = 1500.0
    vp[seabed:] = (
        1600.0 + 30.0 * np.arange(nz - seabed)[:, None, None]
    ).astype(np.float32) * np.ones((ny, nx), dtype=np.float32)
    return vp


def test_save_vp_well_logs_png_writes(tmp_path):
    from sweep_tasks.qc import save_vp_well_logs_png

    vp_init = _synth_vp()
    vp_cur = vp_init.copy()
    vp_cur += 50.0 * np.random.default_rng(0).standard_normal(vp_init.shape).astype(np.float32)
    # 6 wells on a 3×2 grid inside the volume.
    wells = np.array([
        (10, 5), (10, 15), (10, 25),
        (30, 5), (30, 15), (30, 25),
    ], dtype=np.int64)
    labels = [f"y={iy} x={ix}" for iy, ix in wells]
    out = tmp_path / "wells" / "iter_0000.png"
    written = save_vp_well_logs_png(
        vp_cur, vp_init,
        well_grid_idx=wells,
        well_labels=labels,
        dz_m=10.0,
        out_path=out,
        epoch=0,
        nrows=2, ncols=3,
    )
    assert written == out
    assert out.exists() and out.stat().st_size > 5_000


def test_save_vp_well_logs_png_out_of_grid_well_renders_stub(tmp_path):
    """OUT-OF-GRID wells render an empty stub without crashing —
    important when the user picks well coords that miss the cropped
    grid by a couple of cells."""
    from sweep_tasks.qc import save_vp_well_logs_png

    vp = _synth_vp(ny=20, nx=20)
    wells = np.array([
        (10, 10),         # OK
        (500, 5),         # iy out of range
        (5, -2),          # ix out of range
        (-5, -5),         # both out
    ], dtype=np.int64)
    labels = ["ok", "iy_oob", "ix_oob", "both_oob"]
    out = tmp_path / "wells_oob.png"
    save_vp_well_logs_png(
        vp, vp,
        well_grid_idx=wells,
        well_labels=labels,
        dz_m=10.0,
        out_path=out,
        epoch=5,
        nrows=2, ncols=2,
    )
    assert out.exists()


def test_save_vp_well_logs_png_shape_mismatch_raises(tmp_path):
    from sweep_tasks.qc import save_vp_well_logs_png

    a = _synth_vp(nz=10, ny=5, nx=5)
    b = _synth_vp(nz=12, ny=5, nx=5)
    with pytest.raises(ValueError, match="vp_current.shape"):
        save_vp_well_logs_png(
            a, b,
            well_grid_idx=np.array([[0, 0]]),
            well_labels=["x"],
            dz_m=10.0,
            out_path=tmp_path / "x.png", epoch=0,
        )


def test_save_vp_well_logs_png_label_mismatch_raises(tmp_path):
    from sweep_tasks.qc import save_vp_well_logs_png

    vp = _synth_vp()
    with pytest.raises(ValueError, match="well_labels length"):
        save_vp_well_logs_png(
            vp, vp,
            well_grid_idx=np.array([[0, 0], [1, 1]]),
            well_labels=["only_one"],
            dz_m=10.0,
            out_path=tmp_path / "x.png", epoch=0,
        )
