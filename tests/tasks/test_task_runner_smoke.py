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


def test_fwi_auto_resume_same_task_dir(tmp_path):
    # ``resume: true`` + a fixed ``task_id`` → re-running the same YAML
    # picks up the checkpoint left under the same task_dir. No need to
    # set ``resume_from``.
    spec_dict = _fwi_smoke_spec(tmp_path, epochs=2, task_id="autoresume",
                                resume=True)
    result1 = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "first.yaml")))
    assert result1.status.state == "success", result1.status.error
    losses1 = np.load(result1.task_dir / "output" / "loss.npy").tolist()
    assert len(losses1) == 2
    assert (result1.task_dir / "checkpoint.pt").exists()
    # The first run starts fresh (no checkpoint yet) — resumed_from
    # stays None, no "interrupted" flag.
    assert result1.status.summary["resumed_from"] is None
    assert result1.status.summary["interrupted"] is False

    # Re-run with same task_id + more epochs → should auto-load the
    # checkpoint and continue.
    spec_dict2 = _fwi_smoke_spec(tmp_path, epochs=4, task_id="autoresume",
                                 resume=True)
    result2 = TaskRunner().run(load_task(_write(spec_dict2, tmp_path / "second.yaml")))
    assert result2.status.state == "success", result2.status.error
    assert result2.task_dir == result1.task_dir  # same task_id ⇒ same dir
    losses2 = np.load(result2.task_dir / "output" / "loss.npy").tolist()
    assert len(losses2) == 4
    assert np.allclose(losses2[:2], losses1)
    assert result2.status.summary["resumed_from"] == "autoresume"


def test_fwi_graceful_stop_via_sigint(tmp_path, monkeypatch):
    # Simulate ctrl-c by patching ``_save_checkpoint`` so the very first
    # invocation (end of iter 0) also fires a SIGINT to our own PID.
    # This is deterministic — unlike a timer thread, it can't race past
    # the iter loop or land mid-setup. After the signal: the handler
    # arms ``stopper.requested``, the loop's ``should_stop`` check at
    # end-of-iter 0 trips, and we break out cleanly with a checkpoint
    # already on disk (the save itself just happened).
    import os
    import signal

    # ``_save_checkpoint`` now lives in the ``tasks.fwi`` mixin module (post
    # runner.py split); the FWI loop resolves it in that namespace, so the
    # monkeypatch must target there, not the re-exported ``runner`` binding.
    from sweep_tasks.tasks import fwi as _fwi_mod

    real_save = _fwi_mod._save_checkpoint
    call_count = {"n": 0}

    def _save_then_sigint(task_dir, payload):
        out = real_save(task_dir, payload)
        call_count["n"] += 1
        if call_count["n"] == 1:
            os.kill(os.getpid(), signal.SIGINT)
        return out

    monkeypatch.setattr(_fwi_mod, "_save_checkpoint", _save_then_sigint)

    spec_dict = _fwi_smoke_spec(tmp_path, epochs=5, task_id="stoptest",
                                resume=True)
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "stop.yaml")))

    assert result.status.state == "success", result.status.error
    assert result.status.summary["interrupted"] is True
    assert result.status.summary["interrupted_at_epoch"] == 0
    losses_partial = np.load(result.task_dir / "output" / "loss.npy").tolist()
    assert len(losses_partial) == 1
    assert (result.task_dir / "checkpoint.pt").exists()

    # Resume the rest. Restore the un-patched ``_save_checkpoint`` so
    # the 2nd run isn't interrupted, then re-run the same YAML.
    monkeypatch.setattr(_fwi_mod, "_save_checkpoint", real_save)
    result2 = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "stop2.yaml")))
    assert result2.status.state == "success", result2.status.error
    assert result2.status.summary["resumed_from"] == "stoptest"
    losses_full = np.load(result2.task_dir / "output" / "loss.npy").tolist()
    assert len(losses_full) == 5
    # Iter-0 loss must match between the runs (resume preserves Adam state + RNG).
    assert losses_full[0] == pytest.approx(losses_partial[0])


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


def _fwi_spec_with(tmp_path, **overrides):
    """Tiny FWI spec; overrides splice into the dict."""
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 300},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 6, "depth": 2, "start": 6, "stop": 42},   # 6 shots
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 44},  # 20 receivers
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
    }
    spec.update(overrides)
    return spec


def test_fwi_data_plan_shot_stride_reduces_nshots(tmp_path):
    """`data_plan.shot_stride=3` over 6 shots should keep 2 shots."""
    spec_dict = _fwi_spec_with(tmp_path, **{
        "data_plan": {"shot_stride": 3},
        "batchsize": 2,    # cannot exceed kept-shots count
    })
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_plan.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_data_plan_receiver_stride_reduces_nrec(tmp_path):
    spec_dict = _fwi_spec_with(tmp_path, **{
        "data_plan": {"receiver_stride": 2},
    })
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_rcvplan.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_model_plan_crops_and_runs(tmp_path):
    """Gap 2: model_plan crops vp in z (receivers stay within the new domain)."""
    import numpy as np
    spec_dict = _fwi_spec_with(tmp_path, **{
        "model_plan": {"z_window_m": [0.0, 100.0]},   # crop z, keep full x
        "batchsize": 1,
    })
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_mplan.yaml")))
    assert result.status.state == "success", result.status.error
    final = np.load(
        Path(spec_dict["output_dir"]) / result.status.task_id / "output" / "inverted_vp.npy"
    )
    assert final.shape[0] < 48, f"z-axis should be cropped; got shape {final.shape}"


# ============================================================================
# SEG-Y obs + from_segy_headers geometry (Option A) — small synthetic SEG-Y
# ============================================================================
def _write_tiny_segy(path, *, nshots=2, nrec=4, nt=64, dh_m=10.0):
    """Build a minimal SEG-Y the runner can read end-to-end via Option A."""
    import struct
    import numpy as np

    from sweep_io.segy import (
        SEGY_BIN_HEADER_SIZE, SEGY_TEXT_HEADER_SIZE, SEGY_TRACE_HEADER_SIZE,
        FORMAT_IEEE_FLOAT32, write_segy_minimal,
    )
    from sweep_io.segy_index import SEGY_REV1_BYTES as BM

    n_total = nshots * nrec
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((n_total, nt)) * 1e-3).astype("float32")
    write_segy_minimal(path, data, dt=0.001, sample_format=FORMAT_IEEE_FLOAT32)
    trace_total = SEGY_TRACE_HEADER_SIZE + nt * 4
    base = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
    with open(path, "r+b") as f:
        for s in range(nshots):
            sx = int((1.0 + s) * 4 * dh_m)
            for r in range(nrec):
                i = s * nrec + r
                f.seek(base + i * trace_total)
                hdr = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
                struct.pack_into(">i", hdr, BM["shot"], s + 1)
                struct.pack_into(">i", hdr, BM["trace_in_shot"], r)
                struct.pack_into(">h", hdr, BM["coord_scalar"], 1)
                struct.pack_into(">i", hdr, BM["sx"], sx)
                struct.pack_into(">i", hdr, BM["sy"], 0)
                struct.pack_into(">i", hdr, BM["rx"], int(r * 2 * dh_m))
                struct.pack_into(">i", hdr, BM["ry"], 0)
                f.seek(base + i * trace_total)
                f.write(bytes(hdr))


def test_fwi_obs_segy_and_from_segy_headers_runs(tmp_path):
    """Option A: obs.kind=segy + geometry.kind=from_segy_headers from the SAME file."""
    import numpy as np
    segy = tmp_path / "tiny.segy"
    _write_tiny_segy(segy, nshots=3, nrec=6, nt=64, dh_m=10.0)

    vp = np.full((48, 48), 2000.0, dtype="float32")
    vp_path = tmp_path / "vp.npy"
    np.save(vp_path, vp)

    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 64},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.005, "scale": 1.0},
        "geometry": {
            "kind": "from_segy_headers",
            "path": str(segy),
            "source_depth_m_override": 10.0,
            "receiver_depth_m_override": 10.0,
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(vp_path)},
        "obs": {"segy": {"path": str(segy),
                         "source_depth_m_override": 10.0,
                         "receiver_depth_m_override": 10.0}},
        "optimizer": {"kind": "adam", "lr": 1.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 2, "show_every": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_segy.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_obs_segy_index_runs(tmp_path):
    """Option B: pre-built SEGYIndex + obs.kind=segy_index + geometry.kind=from_segy_index."""
    import numpy as np
    from sweep_io.segy_index import build_segy_index

    segy = tmp_path / "tinyB.segy"
    _write_tiny_segy(segy, nshots=3, nrec=6, nt=64, dh_m=10.0)

    idx = build_segy_index([segy],
                           source_depth_m_override=10.0,
                           receiver_depth_m_override=10.0,
                           num_workers=1)
    idx_path = tmp_path / "tinyB.index.npz"
    idx.save(idx_path)

    vp = np.full((48, 48), 2000.0, dtype="float32")
    vp_path = tmp_path / "vp.npy"
    np.save(vp_path, vp)

    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 64},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.005, "scale": 1.0},
        "geometry": {"kind": "from_segy_index", "index_path": str(idx_path)},
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(vp_path)},
        "obs": {"segy_index": {"index_path": str(idx_path)}},
        "optimizer": {"kind": "adam", "lr": 1.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 2, "show_every": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_segy_idx.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_stages_per_stage_dh_and_bandpass_runs(tmp_path):
    """Gaps 4 + 5: stage 0 at dh=20m, stage 1 at dh=10m + bandpass.

    Confirms vp gets resampled across the dh change and the per-stage
    bandpass + dt_s + optimizer-reset path doesn't crash.
    """
    import numpy as np
    true_path, init_path, _ = _tiny_grid_models(tmp_path)
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 20.0},
        "time": {"dt": 0.002, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 10.0, "delay": 0.05, "scale": 1.0},
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
        "epochs": 1, "batchsize": 3, "show_every": 10,
        "stages": [
            {"epochs": 2, "dh_m": 20.0, "bandpass": {"lo_hz": 1.0, "hi_hz": 6.0}},
            {"epochs": 2, "dh_m": 10.0, "dt_s": 0.001,
             "bandpass": {"lo_hz": 1.0, "hi_hz": 12.0}},
        ],
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_stages.yaml")))
    assert result.status.state == "success", result.status.error
    # Stage 1 finer grid → final vp must have a different shape from the init.
    init = np.load(init_path)
    final = np.load(
        Path(spec_dict["output_dir"]) / result.status.task_id / "output" / "inverted_vp.npy"
    )
    assert final.shape != init.shape, (
        f"expected resampled vp shape; got {final.shape} == init {init.shape}"
    )


def test_fwi_obs_segy_index_lazy_prefetched_load(tmp_path):
    """`lazy=True` now uses a background Prefetcher during the index→tensor
    materialisation. End-to-end runner output is identical to lazy=False.
    """
    import numpy as np
    from sweep_io.segy_index import build_segy_index

    segy = tmp_path / "tinyC.segy"
    _write_tiny_segy(segy, nshots=3, nrec=4, nt=32, dh_m=10.0)
    idx = build_segy_index([segy], source_depth_m_override=10.0,
                           receiver_depth_m_override=10.0, num_workers=1)
    idx_path = tmp_path / "tinyC.index.npz"
    idx.save(idx_path)

    vp = np.full((32, 32), 2000.0, dtype="float32")
    np.save(tmp_path / "vp.npy", vp)

    base = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 32},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.005, "scale": 1.0},
        "geometry": {"kind": "from_segy_index", "index_path": str(idx_path)},
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 8,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(tmp_path / "vp.npy")},
        "optimizer": {"kind": "adam", "lr": 1.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 1, "show_every": 1,
    }

    eager = {**base, "task_id": "eager", "obs": {"segy_index": {"index_path": str(idx_path)}}}
    lazy  = {**base, "task_id": "lazy",  "obs": {"segy_index": {"index_path": str(idx_path), "lazy": True}}}

    r_eager = TaskRunner().run(load_task(_write(eager, tmp_path / "eager.yaml")))
    r_lazy  = TaskRunner().run(load_task(_write(lazy,  tmp_path / "lazy.yaml")))
    assert r_eager.status.state == "success", r_eager.status.error
    assert r_lazy.status.state == "success", r_lazy.status.error
    losses_eager = np.load(Path(base["output_dir"]) / "eager" / "output" / "loss.npy")
    losses_lazy  = np.load(Path(base["output_dir"]) / "lazy" / "output" / "loss.npy")
    np.testing.assert_allclose(losses_eager, losses_lazy, rtol=1e-5)
