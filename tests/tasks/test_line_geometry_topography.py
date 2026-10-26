"""`geometry.kind: line` under `physics.topography`: the line follows the surface.

sweep (0.3.6 on) takes sources and receivers as physical rows on every
equation, curvilinear ones included, and refuses a point in the air. A line's
`depth` therefore counts down from the surface of each column."""
import numpy as np
import pytest


def _line(src_depth=2, rec_depth=0, stop=None):
    from sweepx.tasks.schemas import LineGeometry, LineSet
    return LineGeometry(sources=LineSet(step=10, depth=src_depth, stop=stop),
                        receivers=LineSet(step=5, depth=rec_depth, stop=stop))


def test_line_follows_the_surface(tmp_path):
    from sweepx.tasks._helpers.geometry import _build_geometry

    topo = (8 - 6 * np.exp(-((np.arange(40) - 20) / 6.0) ** 2)).round().astype(np.int64)
    np.save(tmp_path / "topo.npy", topo)
    src, rec = _build_geometry(_line(), (30, 40), topography=tmp_path / "topo.npy")
    assert np.array_equal(src[:, 0], [0, 10, 20, 30])
    assert np.array_equal(src[:, 1], topo[src[:, 0]] + 2)
    assert np.array_equal(rec[0, :, 1], topo[rec[0, :, 0]])
    flat_src, flat_rec = _build_geometry(_line(), (30, 40))
    assert (flat_src[:, 1] == 2).all() and (flat_rec[0, :, 1] == 0).all()


def test_fractional_surface_rounds_into_the_medium(tmp_path):
    from sweepx.tasks._helpers.geometry import _build_geometry

    np.save(tmp_path / "topo.npy", np.array([3.0, 3.2, 3.5, 4.0] * 10))
    src, rec = _build_geometry(_line(src_depth=0), (30, 40), topography=tmp_path / "topo.npy")
    assert rec[0, :, 1].tolist() == [3, 4, 4, 4, 3, 4, 4, 4]      # x = 0, 5, ..., 35


def test_explicit_points_stay_physical_rows(tmp_path):
    from sweepx.tasks._helpers.geometry import _build_geometry
    from sweepx.tasks.schemas import ExplicitGeometry

    np.save(tmp_path / "topo.npy", np.full(40, 7))
    g = ExplicitGeometry(sources=[[10, 9]], receivers=[[0, 7], [5, 8]])
    src, rec = _build_geometry(g, (30, 40), topography=tmp_path / "topo.npy")
    assert src.tolist() == [[10, 9]] and rec.tolist() == [[[0, 7], [5, 8]]]


def test_line_wider_than_the_topography_is_refused(tmp_path):
    from sweepx.tasks._helpers.geometry import _build_geometry

    np.save(tmp_path / "topo.npy", np.zeros(20))
    with pytest.raises(ValueError, match="physics.topography has 20 columns"):
        _build_geometry(_line(), (30, 40), topography=tmp_path / "topo.npy")
