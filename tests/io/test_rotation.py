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
    principal_direction,
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


# ---------------------------------------------------------------------------
# principal_direction — PCA line-azimuth estimation
# ---------------------------------------------------------------------------
def _line_points(theta_deg: float, n: int = 20, jitter: float = 0.0,
                 origin=(1000.0, 2000.0)) -> np.ndarray:
    """Points along a line at azimuth ``theta_deg`` with optional crossline jitter."""
    theta = np.deg2rad(theta_deg)
    t = np.linspace(-500.0, 500.0, n)
    base = np.column_stack([
        origin[0] + t * np.cos(theta),
        origin[1] + t * np.sin(theta),
    ])
    if jitter:
        # offset along the crossline (perpendicular) direction
        rng = np.random.default_rng(0)
        perp = np.array([-np.sin(theta), np.cos(theta)])
        base = base + (rng.standard_normal(n) * jitter)[:, None] * perp[None, :]
    return base


@pytest.mark.parametrize("theta_deg", [0.0, 17.0, 35.0, 90.0, 123.0])
def test_principal_direction_recovers_azimuth(theta_deg):
    pts = _line_points(theta_deg, n=30)
    direction, angle = principal_direction(pts)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-12)
    # azimuth is only defined modulo pi (SVD sign is arbitrary)
    diff = (angle - np.deg2rad(theta_deg) + np.pi / 2) % np.pi - np.pi / 2
    assert abs(diff) < 1e-6


def test_principal_direction_requires_two_points():
    with pytest.raises(ValueError, match="two points"):
        principal_direction(np.array([[1.0, 2.0]]))


def test_principal_direction_identical_points_raises():
    with pytest.raises(ValueError, match="identical"):
        principal_direction(np.full((5, 2), 3.0))


def test_principal_direction_bad_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(npoints, 2\)"):
        principal_direction(np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# RotatedFrame.fit — derive a frame from scattered points
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [17.0, 35.0, 80.0, 130.0])
def test_fit_aligns_line_to_target_x(theta_deg):
    """After fit + to_model, a tilted line collapses onto the x (inline) axis."""
    pts = _line_points(theta_deg, n=40)
    frame = RotatedFrame.fit(pts, target_axis="x")
    model = frame.to_model(pts)
    # crossline (y) ~ constant; inline (x) spans the line length
    assert np.ptp(model[:, 1]) < 1e-6              # crossline nearly flat
    assert np.ptp(model[:, 0]) > 100.0             # inline carries the variance


@pytest.mark.parametrize("theta_deg", [17.0, 35.0, 130.0])
def test_fit_aligns_line_to_target_y(theta_deg):
    pts = _line_points(theta_deg, n=40)
    frame = RotatedFrame.fit(pts, target_axis="y")
    model = frame.to_model(pts)
    assert np.ptp(model[:, 0]) < 1e-6              # crossline (x) flat
    assert np.ptp(model[:, 1]) > 100.0             # inline (y) carries variance


def test_fit_shift_inline_to_zero():
    pts = _line_points(35.0, n=25)
    frame = RotatedFrame.fit(pts, target_axis="x", shift_inline_to_zero=True)
    model = frame.to_model(pts)
    np.testing.assert_allclose(model[:, 0].min(), 0.0, atol=1e-9)


def test_fit_first_quadrant_both_shifts():
    pts = _line_points(35.0, n=25, jitter=30.0)
    frame = RotatedFrame.fit(
        pts, target_axis="x",
        shift_inline_to_zero=True, shift_crossline_to_zero=True,
    )
    model = frame.to_model(pts)
    np.testing.assert_allclose(model[:, 0].min(), 0.0, atol=1e-9)
    np.testing.assert_allclose(model[:, 1].min(), 0.0, atol=1e-9)
    assert np.all(model >= -1e-9)                  # whole box in first quadrant


def test_fit_roundtrip_to_utm():
    pts = _line_points(48.0, n=15, jitter=20.0)
    frame = RotatedFrame.fit(pts, target_axis="x", shift_inline_to_zero=True)
    back = frame.to_utm(frame.to_model(pts))
    np.testing.assert_allclose(back, pts, atol=1e-7)


def test_fit_serialises_to_legacy_metadata_schema():
    """fit() → to_dict() → from_metadata() must round-trip."""
    pts = _line_points(35.0, n=20)
    frame = RotatedFrame.fit(pts, target_axis="x", shift_inline_to_zero=True)
    reloaded = RotatedFrame.from_metadata(frame.to_dict())
    np.testing.assert_allclose(reloaded.rotation_matrix, frame.rotation_matrix)
    np.testing.assert_allclose(reloaded.origin_xy_utm, frame.origin_xy_utm)
    assert reloaded.inline_shift == frame.inline_shift
    assert reloaded.target_axis == frame.target_axis


def test_fit_requires_two_points():
    with pytest.raises(ValueError, match="two points"):
        RotatedFrame.fit(np.array([[1.0, 2.0]]))


def test_fit_bad_target_axis_raises():
    with pytest.raises(ValueError, match="target_axis"):
        RotatedFrame.fit(_line_points(10.0), target_axis="z")


# ---------------------------------------------------------------------------
# Gold-standard: fit() reproduces fwi_workflow's legacy rotation bit-for-bit
# ---------------------------------------------------------------------------
def test_fit_matches_fwi_workflow_rotation_estimator():
    """fit's origin + rotation matrix must equal rotation_to_align_line()."""
    acq = pytest.importorskip("fwi_workflow.geometry.acquisition")
    pts = _line_points(35.0, n=30, jitter=15.0)
    for target_axis in ("x", "y"):
        origin, rot, _ = acq.rotation_to_align_line(pts, target_axis=target_axis)
        frame = RotatedFrame.fit(pts, target_axis=target_axis)
        np.testing.assert_allclose(frame.origin_xy_utm, origin, atol=0, rtol=0)
        np.testing.assert_allclose(frame.rotation_matrix, rot, atol=0, rtol=0)


def test_fit_matches_fwi_workflow_rotate_csg_index():
    """End-to-end: fit + to_model reproduces rotate_csg_index inline/crossline."""
    line2d = pytest.importorskip("fwi_workflow.geometry.line2d")

    # Build a minimal per-trace CSG index: shots along a 35° line, each with
    # a small receiver spread along the same azimuth.
    theta = np.deg2rad(35.0)
    nshots, nrec = 6, 5
    s_t = np.linspace(0.0, 1000.0, nshots)
    shot_xy = np.column_stack([1000.0 + s_t * np.cos(theta),
                               2000.0 + s_t * np.sin(theta)])
    rec_t = np.linspace(-200.0, 200.0, nrec)
    sx_all = np.repeat(shot_xy[:, 0], nrec)
    sy_all = np.repeat(shot_xy[:, 1], nrec)
    gx_all = np.concatenate([shot_xy[i, 0] + rec_t * np.cos(theta)
                             for i in range(nshots)])
    gy_all = np.concatenate([shot_xy[i, 1] + rec_t * np.sin(theta)
                             for i in range(nshots)])
    shot_start = np.arange(nshots, dtype=np.int64) * nrec
    index = {"sx": sx_all, "sy": sy_all, "gx": gx_all, "gy": gy_all,
             "shot_start": shot_start}

    rotated, meta = line2d.rotate_csg_index(
        index, fit_points="sources", target_axis="x",
        shift_inline_to_zero=True, shift_crossline_to_zero=True,
    )

    # Reproduce with sweep-io: rotation fit on per-shot sources; shift minima
    # over per-trace sources + receivers (what rotate_csg_index does).
    source_points = shot_xy  # == _source_points(index)
    all_xy = np.vstack([np.column_stack([sx_all, sy_all]),
                        np.column_stack([gx_all, gy_all])])
    frame = RotatedFrame.fit(
        source_points, target_axis="x",
        shift_inline_to_zero=True, shift_crossline_to_zero=True,
        shift_points_xy=all_xy,
    )

    # Frame parameters match the legacy metadata.
    np.testing.assert_allclose(frame.origin_xy_utm, meta["origin_xy"], atol=0, rtol=0)
    np.testing.assert_allclose(frame.rotation_matrix, meta["rotation_matrix"], atol=0, rtol=0)
    np.testing.assert_allclose(frame.inline_shift, meta["inline_shift"], atol=1e-9)
    np.testing.assert_allclose(frame.crossline_shift, meta["crossline_shift"], atol=1e-9)

    # End-to-end coordinates match bit-for-bit (same float ops).
    model_src = frame.to_model(np.column_stack([sx_all, sy_all]))
    np.testing.assert_allclose(model_src[:, 0], rotated["source_inline"], atol=1e-9)
    np.testing.assert_allclose(model_src[:, 1], rotated["source_crossline"], atol=1e-9)
    model_rec = frame.to_model(np.column_stack([gx_all, gy_all]))
    np.testing.assert_allclose(model_rec[:, 0], rotated["receiver_inline"], atol=1e-9)
    np.testing.assert_allclose(model_rec[:, 1], rotated["receiver_crossline"], atol=1e-9)
