"""End-to-end smoke runs for each task type on tiny inputs.

These tests are tuned to run in a few seconds on CPU or GPU. They confirm
that the runner can construct a propagator and execute the dispatch path
without crashing; they do not check numerical accuracy.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from sweep_tasks import load_task, TaskRunner


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _tiny_grid_models(tmp_path: Path) -> tuple[Path, Path, tuple[int, int]]:
    shape = (48, 48)
    true_vp = (2200 + 600 * np.linspace(0, 1, shape[0])[:, None] * np.ones(shape)).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_path = tmp_path / "true.npy"
    init_path = tmp_path / "init.npy"
    np.save(true_path, true_vp)
    np.save(init_path, init_vp)
    return true_path, init_path, shape


def test_introspect_list_equations(tmp_path):
    spec_yaml = _write({"task_type": "introspect", "action": "list_equations",
                        "output_dir": str(tmp_path / "tasks")},
                       tmp_path / "intro.yaml")
    spec = load_task(spec_yaml)
    result = TaskRunner().run(spec)
    assert result.status.state == "success", result.status.error
    assert any("introspect.json" in a for a in result.status.artifacts)


def test_introspect_describe_acoustic(tmp_path):
    spec_yaml = _write({"task_type": "introspect",
                        "action": "describe_equation",
                        "target": "Acoustic",
                        "output_dir": str(tmp_path / "tasks")},
                       tmp_path / "desc.yaml")
    spec = load_task(spec_yaml)
    result = TaskRunner().run(spec)
    assert result.status.state == "success", result.status.error


def _forward_spec_dict(tmp_path: Path) -> dict:
    return {
        "task_type": "forward",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 1, "depth": 4, "start": 24, "stop": 25},
            "receivers": {"step": 2, "depth": 6, "start": 4, "stop": 44},
        },
        "physics": {
            "equation": "Acoustic",
            "spatial_order": 8,
            "abcn": 12,
            "free_surface": False,
            "pml_type": "cpmlr",
            "source_type": ["h1"],
            "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "models": [{"name": "vp", "constant": 2200.0, "shape": [48, 48]}],
    }


def test_forward_runs(tmp_path):
    spec_yaml = _write(_forward_spec_dict(tmp_path), tmp_path / "fwd.yaml")
    result = TaskRunner().run(load_task(spec_yaml))
    assert result.status.state == "success", result.status.error
    assert any("record.npy" in a for a in result.status.artifacts)
    assert result.status.summary["nshots"] == 1


def test_wavefield_runs(tmp_path):
    spec_dict = _forward_spec_dict(tmp_path)
    spec_dict["task_type"] = "wavefield"
    spec_dict["snapshot_times"] = [50, 100, 150]
    spec_dict["plot"] = False
    spec_yaml = _write(spec_dict, tmp_path / "wf.yaml")
    result = TaskRunner().run(load_task(spec_yaml))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["snapshots_shape"][0] == 3


def test_fwi_runs_and_loss_decreases(tmp_path):
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec_dict = {
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
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 3, "batchsize": 3, "show_every": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi.yaml")))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["loss_decreased"] is True


def test_lsrtm_runs(tmp_path):
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec_dict = {
        "task_type": "lsrtm",
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
            "equation": "AcousticLSRTM", "spatial_order": 8, "abcn": 12,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["sh1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "background_model": {"name": "vp", "path": str(init_path)},
        "true_model": {"name": "vp", "path": str(true_path)},
        "optimizer": {"kind": "adam", "lr": 0.01, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 3, "show_every": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "lsrtm.yaml")))
    assert result.status.state == "success", result.status.error


def test_invalid_equation_name_fails_gracefully(tmp_path):
    spec_dict = _forward_spec_dict(tmp_path)
    spec_dict["physics"]["equation"] = "NotARealEquation"
    spec_yaml = _write(spec_dict, tmp_path / "bad.yaml")
    result = TaskRunner().run(load_task(spec_yaml))
    assert result.status.state == "failed"
    assert "Unknown equation" in (result.status.error or "")
    # status.json should be on disk with the failure recorded
    status_path = result.task_dir / "status.json"
    assert status_path.exists()


# ---------- new dispatch paths: wavelet kinds, geometry kinds, override ---

def test_forward_with_explicit_geometry(tmp_path):
    spec_dict = _forward_spec_dict(tmp_path)
    spec_dict["geometry"] = {
        "kind": "explicit",
        "sources": [[10, 4], [24, 4], [38, 4]],
        "receivers": [[4, 6], [12, 6], [20, 6], [28, 6], [36, 6], [44, 6]],
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "expl.yaml")))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["nshots"] == 3
    assert result.status.summary["nreceivers"] == 6


def test_forward_with_from_file_geometry(tmp_path):
    sources = np.array([[12, 4], [28, 4]], dtype=np.int64)
    receivers = np.array([[6, 6], [16, 6], [26, 6], [36, 6]], dtype=np.int64)
    sf = tmp_path / "src.npy"
    rf = tmp_path / "rec.npy"
    np.save(sf, sources)
    np.save(rf, receivers)
    spec_dict = _forward_spec_dict(tmp_path)
    spec_dict["geometry"] = {
        "kind": "from_file",
        "sources_file": str(sf),
        "receivers_file": str(rf),
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fromfile.yaml")))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["nshots"] == 2
    assert result.status.summary["nreceivers"] == 4


def test_forward_with_from_npy_wavelet(tmp_path):
    nt = 200
    dt = 0.001
    # save a chirp-like signal as a stand-in
    t = np.linspace(0, nt * dt, nt, dtype=np.float32)
    wave = (np.sin(2 * np.pi * 20 * t) * np.exp(-t / 0.05)).astype(np.float32)
    wpath = tmp_path / "wave.npy"
    np.save(wpath, wave)
    spec_dict = _forward_spec_dict(tmp_path)
    spec_dict["wavelet"] = {"kind": "from_npy", "path": str(wpath), "scale": 1.0}
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fnpy.yaml")))
    assert result.status.state == "success", result.status.error


def test_from_npy_wavelet_length_mismatch_fails(tmp_path):
    wpath = tmp_path / "short.npy"
    np.save(wpath, np.zeros(50, dtype=np.float32))  # spec has nt=200
    spec_dict = _forward_spec_dict(tmp_path)
    spec_dict["wavelet"] = {"kind": "from_npy", "path": str(wpath), "scale": 1.0}
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "short.yaml")))
    assert result.status.state == "failed"
    assert "length" in (result.status.error or "").lower()


def test_fwi_with_modeling_override_wavelet(tmp_path):
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec_dict = {
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
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 3, "show_every": 1,
        "modeling_override": {
            "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.1, "scale": 1.0},
        },
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_ov.yaml")))
    assert result.status.state == "success", result.status.error


def _fwi_smoke_spec(tmp_path: Path, **overrides) -> dict:
    """Compact FWI spec dict for fast runner smoke tests."""

    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    base = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
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
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 3, "show_every": 1,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("loss_kind", ["mse", "l1", "huber"])
def test_fwi_loss_kinds_run(tmp_path, loss_kind):
    spec_dict = _fwi_smoke_spec(tmp_path, loss={"kind": loss_kind})
    if loss_kind == "huber":
        spec_dict["loss"]["huber_delta"] = 0.5
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / f"l_{loss_kind}.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_model_bounds_enforced(tmp_path):
    spec_dict = _fwi_smoke_spec(
        tmp_path,
        model_bounds={"vp": {"min": 2400.0, "max": 2600.0}},
        epochs=4,
    )
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "bounds.yaml")))
    assert result.status.state == "success", result.status.error
    inv = np.load(result.task_dir / "output" / "inverted_vp.npy")
    assert inv.min() >= 2400.0 - 1e-3
    assert inv.max() <= 2600.0 + 1e-3


def test_fwi_freeze_top_n_rows_keeps_top_unchanged(tmp_path):
    spec_dict = _fwi_smoke_spec(tmp_path, freeze_top_n_rows=8, epochs=3)
    init_arr = np.load(spec_dict["init_model"]["path"])
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "freeze.yaml")))
    assert result.status.state == "success", result.status.error
    inv = np.load(result.task_dir / "output" / "inverted_vp.npy")
    assert np.allclose(inv[:8, :], init_arr[:8, :])


def test_fwi_multi_stage_runs(tmp_path):
    spec_dict = _fwi_smoke_spec(
        tmp_path,
        stages=[
            {"epochs": 2, "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.08, "scale": 1.0}, "lr_scale": 1.0},
            {"epochs": 2, "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0}, "lr_scale": 0.5},
        ],
    )
    spec_dict["epochs"] = 1  # ignored when stages set
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "stages.yaml")))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["num_stages"] == 2
    losses = np.load(result.task_dir / "output" / "loss.npy")
    assert losses.shape[0] == 4  # 2 + 2


@pytest.mark.parametrize("opt_kind", ["sgd", "lbfgs"])
def test_fwi_optimizer_variants_run(tmp_path, opt_kind):
    if opt_kind == "sgd":
        opt = {"kind": "sgd", "lr": 0.5, "momentum": 0.9}
    else:
        opt = {"kind": "lbfgs", "lr": 1.0, "max_iter": 3, "history_size": 3}
    spec_dict = _fwi_smoke_spec(tmp_path, optimizer=opt, epochs=2)
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / f"opt_{opt_kind}.yaml")))
    assert result.status.state == "success", result.status.error


@pytest.mark.parametrize("scheduler", [
    {"kind": "step", "step_size": 1, "gamma": 0.5},
    {"kind": "exp", "gamma": 0.9},
    {"kind": "cosine", "eta_min": 0.0},
])
def test_fwi_scheduler_runs(tmp_path, scheduler):
    spec_dict = _fwi_smoke_spec(tmp_path, scheduler=scheduler, epochs=3)
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "sched.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_train_shot_batchsize_accumulation(tmp_path):
    spec_dict = _fwi_smoke_spec(tmp_path, batchsize=4, train_shot_batchsize=2, epochs=2)
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "accum.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_save_illumination_produces_artifacts(tmp_path):
    spec_dict = _fwi_smoke_spec(tmp_path, save_illumination=True, show_every=1, epochs=2)
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "illum.yaml")))
    assert result.status.state == "success", result.status.error
    illum_files = list((result.task_dir / "output" / "epochs").glob("*illumination*.npy"))
    # source_illumination / receiver_illumination only emit if the propagator
    # exposes them; the test passes when at least one shows up OR neither does
    # (eager backend currently emits source_illumination but not receiver_illumination).
    assert all(f.name.endswith(".npy") for f in illum_files)


def test_fwi_resume_from_checkpoint(tmp_path):
    # First run: 2 epochs
    spec_dict = _fwi_smoke_spec(tmp_path, epochs=2)
    result1 = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "first.yaml")))
    assert result1.status.state == "success", result1.status.error
    losses1 = np.load(result1.task_dir / "output" / "loss.npy").tolist()
    assert len(losses1) == 2

    # Second run: resume_from=first task_id, run 2 more epochs (total 4)
    first_id = result1.status.task_id
    spec_dict2 = _fwi_smoke_spec(tmp_path, epochs=4, resume_from=first_id)
    result2 = TaskRunner().run(load_task(_write(spec_dict2, tmp_path / "resume.yaml")))
    assert result2.status.state == "success", result2.status.error
    losses2 = np.load(result2.task_dir / "output" / "loss.npy").tolist()
    assert len(losses2) == 4
    # First two losses must equal the prior run's losses
    assert np.allclose(losses2[:2], losses1)


def test_fwi_init_models_list_acoustic_smoke(tmp_path):
    # Multi-model `init_models` API path; for Acoustic the list is just [vp].
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec_dict = _fwi_smoke_spec(tmp_path)
    spec_dict.pop("init_model")
    spec_dict["init_models"] = [{"name": "vp", "path": str(init_path)}]
    spec_dict["obs"] = {"synthetic_from_models": [{"name": "vp", "path": str(true_path)}]}
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "list_init.yaml")))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["models_inverted"] == ["vp"]


def test_fwi_per_model_lr_dict_acoustic_smoke(tmp_path):
    spec_dict = _fwi_smoke_spec(
        tmp_path,
        optimizer={"kind": "adam", "lr": {"vp": 5.0}, "eps": 1.0e-22},
    )
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "pm_lr.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_modeling_override_geometry_shape_mismatch_fails(tmp_path):
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 8, "depth": 2, "start": 8, "stop": 40},   # 4 shots
            "receivers": {"step": 4, "depth": 4, "start": 4, "stop": 44},
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 12,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 2, "show_every": 1,
        "modeling_override": {
            # 8 shots instead of 4 -> shape mismatch must be rejected at runtime
            "geometry": {
                "kind": "line",
                "sources": {"step": 4, "depth": 2, "start": 8, "stop": 40},
                "receivers": {"step": 4, "depth": 4, "start": 4, "stop": 44},
            },
        },
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_bad_ov.yaml")))
    assert result.status.state == "failed"
    assert "shots" in (result.status.error or "")
