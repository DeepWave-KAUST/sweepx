"""`geometry.kind: grid` — the 3-D counterpart of `kind: line`."""
import numpy as np
import pytest

def test_grid_geometry_builds_the_outer_product():
    from sweep_tasks._helpers.geometry import _build_geometry
    from sweep_tasks.schemas import AxisSpan, GridGeometry, GridSet

    g = GridGeometry(
        sources=GridSet(x=AxisSpan(start=10, stop=40, step=10),
                        y=AxisSpan(start=5, stop=25, step=10), depth=3),
        receivers=GridSet(x=AxisSpan(start=0, step=25),
                          y=AxisSpan(start=0, step=25), depth=1),
    )
    src, rec = _build_geometry(g, (60, 80, 100))          # (nz, ny, nx)
    assert src.shape == (3 * 2, 3)                        # x:10,20,30  y:5,15
    assert np.array_equal(src[:, 0], [10, 10, 20, 20, 30, 30])   # x slowest
    assert np.array_equal(src[:, 1], [5, 15, 5, 15, 5, 15])
    assert (src[:, 2] == 3).all()
    # receivers default to the grid extent: x -> nx=100, y -> ny=80
    assert rec.shape == (6, 4 * 4, 3)
    assert rec[0, :, 0].max() == 75 and rec[0, :, 1].max() == 75


def test_grid_geometry_is_3d_only():
    from sweep_tasks._helpers.geometry import _build_geometry
    from sweep_tasks.schemas import AxisSpan, GridGeometry, GridSet

    g = GridGeometry(
        sources=GridSet(x=AxisSpan(step=10), y=AxisSpan(step=10), depth=1),
        receivers=GridSet(x=AxisSpan(step=10), y=AxisSpan(step=10), depth=1),
    )
    with pytest.raises(ValueError, match="only supports 3-D grids"):
        _build_geometry(g, (60, 100))


def test_grid_geometry_rejects_an_empty_span():
    from sweep_tasks._helpers.geometry import _build_geometry
    from sweep_tasks.schemas import AxisSpan, GridGeometry, GridSet

    g = GridGeometry(
        sources=GridSet(x=AxisSpan(start=50, stop=20, step=5),
                        y=AxisSpan(step=10), depth=1),
        receivers=GridSet(x=AxisSpan(step=10), y=AxisSpan(step=10), depth=1),
    )
    with pytest.raises(ValueError, match=r"GridSet.x stop"):
        _build_geometry(g, (60, 100, 100))
