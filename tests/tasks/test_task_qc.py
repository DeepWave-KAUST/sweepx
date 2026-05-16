"""End-to-end smoke tests for the optional FWI QC outputs."""

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
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 3, "batchsize": 3, "show_every": 1,
    }


def test_qc_all_outputs(tmp_path):
    """All QC products enabled — files should land under qc/."""
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec["task_id"] = "fwi_qc_all"
    spec["qc"] = {
        "every_n_epochs": 1,
        "vp_png": True,
        "vp_diff_png": True,
        "gradient_png": True,
        "shot_gather": True,
        "shot_gather_n_shots": 2,
        "loss_curve": True,
    }
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "qc.yaml")))
    assert result.status.state == "success", result.status.error

    qc_root = result.task_dir / "qc"
    assert qc_root.is_dir()
    # Per-iter products: 3 epochs at every_n_epochs=1 -> 3 PNGs each.
    vp_pngs = sorted((qc_root / "vp").glob("iter_*.png"))
    diff_pngs = sorted((qc_root / "vp_diff").glob("iter_*.png"))
    grad_pngs = sorted((qc_root / "gradient").glob("iter_*.png"))
    shot_pngs = sorted((qc_root / "shot_gather").glob("iter_*.png"))
    assert len(vp_pngs) == 3, f"expected 3 vp PNGs, got {len(vp_pngs)}"
    assert len(diff_pngs) == 3
    assert len(grad_pngs) == 3
    assert len(shot_pngs) == 3
    assert (qc_root / "loss_curve.png").exists()


def test_qc_disabled_when_every_n_zero(tmp_path):
    """every_n_epochs=0 disables per-epoch QC; final loss_curve still emitted."""
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec["task_id"] = "fwi_qc_off"
    spec["qc"] = {"every_n_epochs": 0, "loss_curve": True}
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "off.yaml")))
    assert result.status.state == "success", result.status.error
    qc_root = result.task_dir / "qc"
    # Final loss curve still emitted (final-only product).
    assert (qc_root / "loss_curve.png").exists()
    # No per-epoch dirs created.
    assert not (qc_root / "vp").exists()
    assert not (qc_root / "gradient").exists()


def test_qc_with_reparam_skips_gradient(tmp_path):
    """In reparam mode the vp is a non-leaf tensor (no .grad). The QC hook
    should silently skip the gradient PNG without crashing."""
    true_path, init_path = _tiny_grid_models(tmp_path)
    spec = _base_fwi_spec(tmp_path, init_path, true_path)
    spec["task_id"] = "fwi_qc_reparam"
    spec["epochs"] = 2
    spec["reparam"] = {
        "kind": "velocity_inr",
        "hidden_features": 16, "hidden_layers": 1,
        "hash": {"enabled": True, "levels": 3, "log2_size": 8,
                 "base_resolution": 2, "finest_resolution": 16},
        "lr": 1.0e-3,
    }
    spec["qc"] = {
        "every_n_epochs": 1,
        "vp_png": True,
        "gradient_png": True,    # should be silently skipped in reparam mode
        "loss_curve": True,
    }
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "reparam_qc.yaml")))
    assert result.status.state == "success", result.status.error
    qc_root = result.task_dir / "qc"
    # vp PNGs land normally.
    assert len(sorted((qc_root / "vp").glob("iter_*.png"))) == 2
    # gradient dir may not exist (skipped) — that's the contract.
    grad_dir = qc_root / "gradient"
    if grad_dir.exists():
        assert len(sorted(grad_dir.glob("iter_*.png"))) == 0
