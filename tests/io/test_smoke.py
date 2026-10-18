"""Smoke tests — exercise the always-available numpy core."""

import numpy as np
import pytest

import sweep_io
from sweep_io.geometry import Geometry
from sweep_io.models import load_velocity, save_velocity


def test_version():
    assert isinstance(sweep_io.__version__, str)
    assert sweep_io.__version__.count(".") >= 1


def test_models_roundtrip_npy(tmp_path):
    vp = np.linspace(1500.0, 4500.0, 64 * 128, dtype="float32").reshape(64, 128)
    path = tmp_path / "vp.npy"
    save_velocity(path, vp)
    loaded = load_velocity(path)
    np.testing.assert_array_equal(loaded, vp)


def test_models_roundtrip_raw(tmp_path):
    vp = np.linspace(1500.0, 4500.0, 64 * 128, dtype="float32").reshape(64, 128)
    path = tmp_path / "vp.bin"
    save_velocity(path, vp)
    loaded = load_velocity(path, shape=(64, 128), dtype="float32")
    np.testing.assert_array_equal(loaded, vp)


def test_geometry_validation():
    with pytest.raises(ValueError, match="nshots mismatch"):
        Geometry(
            sources=np.zeros((2, 2)),
            receivers=np.zeros((3, 10, 2)),
        )


def test_geometry_save_load_json(tmp_path):
    rec_one_shot = np.stack(
        [np.arange(0, 100, 10, dtype="int64"), np.zeros(10, dtype="int64")], axis=1
    )  # (10, 2)
    g = Geometry(
        sources=np.array([[10, 0], [20, 0]], dtype="int64"),
        receivers=np.stack([rec_one_shot, rec_one_shot], axis=0),  # (2, 10, 2)
        dt=0.001,
        nt=2000,
        dh=(10.0, 10.0),
    )
    path = tmp_path / "geom.json"
    g.save(path)
    g2 = Geometry.load(path)
    np.testing.assert_array_equal(g2.sources, g.sources)
    np.testing.assert_array_equal(g2.receivers, g.receivers)
    assert g2.dt == g.dt and g2.nt == g.nt and g2.dh == g.dh


def test_lazy_submodules():
    """`datasets` and `cuda_prefetch` should not auto-import (need torch).

    Note: `segy` IS eager-imported now because its core (SEGYReader,
    IBM<->IEEE codec) is pure stdlib + numpy; only the segyio-using
    helpers defer their import to call time.

    Uses a subprocess for isolation — other tests in this session may
    have already imported the lazy modules.
    """
    import subprocess
    import sys
    code = (
        "import sys, sweep_io; "
        "print(int('sweep_io.datasets' in sys.modules), "
        "int('sweep_io.cuda_prefetch' in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "0 0", f"expected '0 0', got {out!r}"
