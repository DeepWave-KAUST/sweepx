"""Unit tests for :class:`sweep_io.geometry.RotatedFrame` + JSON loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sweep_io.geometry import (
    PhysicalGeometry,
    RotatedFrame,
    load_rotation_metadata,
)


def test_rotation_roundtrip_90_deg():
    """to_model then to_utm must recover the original UTM xy."""
    origin = np.array([1000.0, 2000.0])
    frame = RotatedFrame.from_rotation_deg(
        origin_xy_utm=origin, rotation_deg=90.0,
    )
    utm = np.array([[1500.0, 2500.0], [1000.0, 2000.0], [800.0, 1500.0]])
    model = frame.to_model(utm)
    back = frame.to_utm(model)
    np.testing.assert_allclose(back, utm, atol=1e-9)


def test_rotation_zero_deg_is_translation():
    """Zero rotation: model == utm - origin (no rotation)."""
    origin = np.array([100.0, 200.0])
    frame = RotatedFrame.from_rotation_deg(
        origin_xy_utm=origin, rotation_deg=0.0,
    )
    utm = np.array([[105.0, 210.0]])
    model = frame.to_model(utm)
    np.testing.assert_allclose(model, np.array([[5.0, 10.0]]), atol=1e-9)


def test_inline_shift_adds_along_target_axis():
    origin = np.array([0.0, 0.0])
    frame_x = RotatedFrame.from_rotation_deg(
        origin_xy_utm=origin, rotation_deg=0.0,
        target_axis="x", inline_shift=50.0, crossline_shift=10.0,
    )
    frame_y = RotatedFrame.from_rotation_deg(
        origin_xy_utm=origin, rotation_deg=0.0,
        target_axis="y", inline_shift=50.0, crossline_shift=10.0,
    )
    utm = np.array([[100.0, 200.0]])
    # target_axis='x' → inline along x, crossline along y
    np.testing.assert_allclose(frame_x.to_model(utm), np.array([[150.0, 210.0]]))
    # target_axis='y' → inline along y, crossline along x
    np.testing.assert_allclose(frame_y.to_model(utm), np.array([[110.0, 250.0]]))


def test_from_metadata_json_roundtrip(tmp_path: Path):
    R = np.array([[0.0, 1.0], [-1.0, 0.0]])  # 90° CW
    meta = {
        "origin_xy": [1000.0, 2000.0],
        "rotation_matrix": R.tolist(),
        "target_axis": "x",
        "inline_shift": 5.0,
        "crossline_shift": 0.0,
    }
    path = tmp_path / "rotation_metadata.json"
    path.write_text(json.dumps(meta))
    frame = load_rotation_metadata(path)
    assert frame.target_axis == "x"
    np.testing.assert_allclose(frame.origin_xy_utm, [1000.0, 2000.0])
    np.testing.assert_allclose(frame.rotation_matrix, R)
    assert frame.inline_shift == 5.0


def test_from_metadata_missing_required_raises():
    with pytest.raises(KeyError):
        RotatedFrame.from_metadata({"foo": 1})


def test_apply_rotation_3d_geometry():
    """3-D PhysicalGeometry.apply_rotation: xy rotates, z passes through."""
    # 90° rotation around origin (1000, 2000); inline=along x; no shifts.
    origin = np.array([1000.0, 2000.0])
    frame = RotatedFrame.from_rotation_deg(
        origin_xy_utm=origin, rotation_deg=90.0,
    )
    # Sources at (utm_x, utm_y, z).
    src = np.array([[1000.0, 2000.0, 10.0],
                    [1500.0, 2000.0, 10.0]])  # along +x at origin level
    # Receivers per shot.
    rec = np.array([
        [[1000.0, 2050.0, 50.0], [1000.0, 2100.0, 50.0]],
        [[1500.0, 2050.0, 50.0], [1500.0, 2100.0, 50.0]],
    ])
    pg = PhysicalGeometry(
        sources_xyz_m=src, receivers_xyz_m=rec, dt=0.001, nt=100,
    )
    rotated = pg.apply_rotation(frame)
    # z stays the same.
    np.testing.assert_allclose(rotated.sources_xyz_m[:, 2], [10.0, 10.0])
    np.testing.assert_allclose(rotated.receivers_xyz_m[..., 2],
                               [[50.0, 50.0], [50.0, 50.0]])
    # XY rotated: source 1 at UTM (1000, 2000) → model (0, 0).
    np.testing.assert_allclose(rotated.sources_xyz_m[0, :2], [0.0, 0.0], atol=1e-9)
    # Meta records the frame.
    assert rotated.meta.get("rotation", {}).get("target_axis") == "x"


def test_apply_rotation_2d_geometry_rejected():
    src = np.array([[100.0, 10.0], [200.0, 10.0]])
    rec = np.array([[[150.0, 50.0]], [[250.0, 50.0]]])
    pg = PhysicalGeometry(
        sources_xyz_m=src, receivers_xyz_m=rec, dt=0.001, nt=100,
    )
    frame = RotatedFrame.from_rotation_deg(
        origin_xy_utm=(0.0, 0.0), rotation_deg=45.0,
    )
    with pytest.raises(ValueError, match=r"3-D"):
        pg.apply_rotation(frame)
