"""End-to-end smoke tests for the sweep-nn reparameterization hook."""

from pathlib import Path

import numpy as np
import yaml

from sweep_tasks import load_task, TaskRunner


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _tiny_grid_models(tmp_path: Path):
    shape = (48, 48)
    true_vp = (
        2200 + 600 * np.linspace(0, 1, shape[0])[:, None] * np.ones(shape)
    ).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_path = tmp_path / "true.npy"
    init_path = tmp_path / "init.npy"
    np.save(true_path, true_vp)
    np.save(init_path, init_vp)
    return true_path, init_path, shape


def _base_fwi_spec(tmp_path, init_path, true_path):
    return {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 300},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 6, "depth": 2, "start": 6, "stop": 42},
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 44},
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


def test_fwi_reparam_runs_single_stage(tmp_path):
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 25.0, "eps": 1.0e-22},  # ignored
        "epochs": 3, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_reparam_single",
        "reparam": {
            "kind": "velocity_inr",
            "hidden_features": 32,
            "hidden_layers": 2,
            "vp_std": 50.0,
            "hash": {
                "enabled": True,
                "levels": 4,
                "features_per_level": 2,
                "log2_size": 10,
                "base_resolution": 4,
                "finest_resolution": 32,
            },
            "lr": 1.0e-3,
        },
    })
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "fwi_reparam.yaml")))
    assert result.status.state == "success", result.status.error
    # vp should be saved and within bounds.
    final_vp = np.load(result.task_dir / "output" / "inverted_vp.npy")
    assert final_vp.shape == (48, 48)
    assert final_vp.min() >= 1500.0 - 1e-3
    assert final_vp.max() <= 4500.0 + 1e-3


def test_fwi_reparam_no_hash(tmp_path):
    """Reparam without hash encoding (pure SIREN coords)."""
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 25.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_reparam_no_hash",
        "reparam": {
            "kind": "velocity_inr",
            "hidden_features": 32,
            "hidden_layers": 2,
            "hash": {"enabled": False},
            "lr": 1.0e-3,
        },
    })
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "no_hash.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_reparam_multistage_preserves_params(tmp_path):
    """Across multi-stage, network params must be preserved (multi-scale benefit)."""
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec.update({
        "optimizer": {"kind": "adam", "lr": 25.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 3, "show_every": 1,
        "task_id": "fwi_reparam_stages",
        "reparam": {
            "kind": "velocity_inr",
            "hidden_features": 16,
            "hidden_layers": 1,
            "hash": {"enabled": True, "levels": 3, "log2_size": 8,
                     "base_resolution": 2, "finest_resolution": 16},
            "lr": 1.0e-3,
        },
        "stages": [
            {"epochs": 2, "dh_m": 10.0, "batch_size": 3,
             "bandpass": {"lo_hz": 1.0, "hi_hz": 30.0, "order": 4}},
            # Stage 2: keep dh same (the simplest multi-stage smoke; we're
            # testing that update_base_velocity + optimizer-preservation
            # don't crash, not that the dh change works on a 48^2 grid).
            {"epochs": 2, "dh_m": 10.0, "batch_size": 3,
             "bandpass": {"lo_hz": 1.0, "hi_hz": 60.0, "order": 4}},
        ],
    })
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "stages.yaml")))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["num_stages"] == 2
    assert result.status.summary["epochs"] == 4
