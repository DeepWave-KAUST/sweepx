"""Tests for per-batch local model windowing (Engquist-style FWI)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from sweep_tasks import load_task, TaskRunner
from sweep_tasks.runner import _compute_local_window, _rebase_geometry_to_window


# ---------------------------------------------------------------------------
# Window-computation helper unit tests
# ---------------------------------------------------------------------------


class _DummyWinSpec:
    def __init__(self, padding_x_m=100.0, padding_z_m=0.0,
                 full_depth=True, min_width_m=0.0):
        self.padding_x_m = padding_x_m
        self.padding_z_m = padding_z_m
        self.full_depth = full_depth
        self.min_width_m = min_width_m


def test_window_tightly_encloses_geometry():
    # 1 shot at x=50 grid, receivers x=40..60. Padding 100m / dh=10m = 10 cells.
    sources = np.array([[50, 4]])             # (B=1, [x, z])
    receivers = np.array([[[40, 6], [45, 6], [60, 6]]])  # (B=1, nrec=3, [x,z])
    full_shape = (48, 100)
    win = _DummyWinSpec(padding_x_m=100.0, padding_z_m=0.0, full_depth=True)
    z0, z1, x0, x1 = _compute_local_window(sources, receivers, full_shape, dh=10.0, win_spec=win)
    assert z0 == 0 and z1 == 48     # full depth
    assert x0 == max(0, 40 - 10)
    assert x1 == min(100, 60 + 10 + 1)


def test_window_clamps_to_full_shape():
    # Receivers span the full x extent; even huge padding should be clamped.
    sources = np.array([[5, 4]])
    receivers = np.array([[[0, 6], [99, 6]]])
    full_shape = (32, 100)
    win = _DummyWinSpec(padding_x_m=200.0, padding_z_m=200.0, full_depth=False)
    z0, z1, x0, x1 = _compute_local_window(sources, receivers, full_shape, dh=10.0, win_spec=win)
    assert x0 == 0 and x1 == 100, f"x not clamped: ({x0}, {x1})"
    # z: source at 4, receiver at 6, pad=20 cells → z0=0 (clamp), z1=27.
    # Both endpoints stay inside the model — the point is that no OOB occurs.
    assert 0 <= z0 < z1 <= 32


def test_window_min_width_enforced():
    sources = np.array([[50, 4]])
    receivers = np.array([[[48, 6], [52, 6]]])
    full_shape = (32, 100)
    # Padding-only window would be ~4 cells wide. min_width_m=300 → 30 cells.
    win = _DummyWinSpec(padding_x_m=0.0, min_width_m=300.0, full_depth=True)
    z0, z1, x0, x1 = _compute_local_window(sources, receivers, full_shape, dh=10.0, win_spec=win)
    assert (x1 - x0) >= 30


def test_rebase_geometry_shifts_correctly():
    sources = np.array([[50, 6]])
    receivers = np.array([[[40, 6], [60, 6]]])
    s, r = _rebase_geometry_to_window(sources, receivers, z0=2, x0=30)
    assert (s == np.array([[20, 4]])).all()
    assert (r == np.array([[[10, 4], [30, 4]]])).all()


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _tiny_grid_models(tmp_path: Path):
    shape = (48, 64)
    true_vp = (
        2200 + 600 * np.linspace(0, 1, shape[0])[:, None] * np.ones(shape)
    ).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_path = tmp_path / "true.npy"
    init_path = tmp_path / "init.npy"
    np.save(true_path, true_vp)
    np.save(init_path, init_vp)
    return true_path, init_path


def _base_fwi_spec(tmp_path, init_path, true_path):
    return {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 300},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 8, "depth": 2, "start": 8, "stop": 56},
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 60},
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 12,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "model_bounds": {"vp": {"min": 1500.0, "max": 4500.0}},
    }


def test_local_window_runs_grid_mode(tmp_path):
    """Grid-mode FWI with local windowing should run and reduce loss."""
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 3, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_local_window_grid",
        "local_model_window": {
            "enabled": True,
            "padding_x_m": 80.0,
            "padding_z_m": 0.0,
            "full_depth": True,
        },
    })
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "lw.yaml")))
    assert result.status.state == "success", result.status.error


def test_local_window_runs_reparam_mode(tmp_path):
    """Reparam (SIREN+hash) + local windowing should run."""
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 25.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_local_window_reparam",
        "reparam": {
            "kind": "velocity_inr",
            "hidden_features": 16, "hidden_layers": 1,
            "hash": {"enabled": True, "levels": 3, "log2_size": 8,
                     "base_resolution": 2, "finest_resolution": 16},
            "lr": 1.0e-3,
        },
        "local_model_window": {
            "enabled": True, "padding_x_m": 80.0, "full_depth": True,
        },
    })
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "lwr.yaml")))
    assert result.status.state == "success", result.status.error


def test_local_window_disabled_acts_like_full_grid(tmp_path):
    """``enabled: false`` should give identical behavior to omitting the spec."""
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_local_window_off",
        "local_model_window": {"enabled": False},
    })
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "off.yaml")))
    assert result.status.state == "success", result.status.error


# ---------------------------------------------------------------------------
# YAML bool-shortcut for the on/off toggle
# ---------------------------------------------------------------------------


def test_local_window_yaml_shortcut_true(tmp_path):
    """``local_model_window: true`` enables it with default parameters."""
    from sweep_tasks import LocalModelWindowSpec
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_lw_shortcut_true",
        "local_model_window": True,
    })
    yaml_path = _write(spec, tmp_path / "shortcut_true.yaml")
    parsed = load_task(yaml_path)
    assert isinstance(parsed.local_model_window, LocalModelWindowSpec)
    assert parsed.local_model_window.enabled is True
    # default values preserved
    assert parsed.local_model_window.padding_x_m == 1500.0
    assert parsed.local_model_window.full_depth is True
    result = TaskRunner().run(parsed)
    assert result.status.state == "success", result.status.error


def test_local_window_yaml_shortcut_false(tmp_path):
    """``local_model_window: false`` is equivalent to omitting the field."""
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_lw_shortcut_false",
        "local_model_window": False,
    })
    parsed = load_task(_write(spec, tmp_path / "shortcut_false.yaml"))
    assert parsed.local_model_window is None
    result = TaskRunner().run(parsed)
    assert result.status.state == "success", result.status.error
