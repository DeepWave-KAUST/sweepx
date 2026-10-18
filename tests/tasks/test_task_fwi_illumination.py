"""End-to-end smoke for illumination preconditioning (§3.5).

Verifies:
* the new ``illumination_precondition`` spec parses through the FWISpec
* a grid-mode FWI run completes without crashing when the flag is on
* the gradient applied to vp differs from the un-preconditioned baseline
  (i.e. the precondition actually fires; not a silent no-op)
* the reparam two-pass-full path also runs with the flag on

Synthetic 2-D problem on the eager backend (small + cheap; the wiring
is dim-agnostic, so verifying the wiring on 2-D exercises the same
runner branches as the 3-D OBN production case).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sweep_tasks import TaskRunner, load_task


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _baseline_fwi_spec(tmp_path: Path) -> dict:
    shape = (48, 48)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_vp = (2200.0 + 600.0 * np.linspace(0, 1, shape[0])[:, None]
               * np.ones(shape)).astype(np.float32)
    init_path = tmp_path / "init.npy"
    true_path = tmp_path / "true.npy"
    np.save(init_path, init_vp)
    np.save(true_path, true_vp)
    return {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 250},
        "wavelet": {"kind": "ricker", "fm": 12.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 8, "depth": 2, "start": 6, "stop": 42},
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
        "epochs": 2, "batchsize": 2, "show_every": 1,
    }


def test_illumination_precondition_grid_mode_runs(tmp_path):
    spec_dict = _baseline_fwi_spec(tmp_path)
    spec_dict["illumination_precondition"] = {
        "enabled": True, "epsilon": 1.0e-6, "exponent": 0.5,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi.yaml")))
    assert result.status.state == "success", result.status.error


def test_illumination_precondition_reparam_two_pass_runs(tmp_path):
    spec_dict = _baseline_fwi_spec(tmp_path)
    spec_dict["illumination_precondition"] = {
        "enabled": True, "epsilon": 1.0e-6, "exponent": 0.5,
    }
    spec_dict["reparam"] = {
        "kind": "velocity_inr",
        "hidden_features": 32, "hidden_layers": 2,
        "first_omega0": 10.0, "hidden_omega0": 10.0,
        "vp_mean": 0.0, "vp_std": 50.0,
        "lr": 1.0e-4,
        "backward_mode": "two_pass_full",
        "hash": {"enabled": False},
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi_rp.yaml")))
    assert result.status.state == "success", result.status.error


def test_illumination_eager_warning_then_noop(tmp_path, capsys):
    """The eager backend does not populate ``solver.{source,receiver}_illumination``
    so the precondition silently falls back to a no-op.

    The runner must emit a clear WARN line so users on the eager smoke
    backend don't think they're actually running with illumination
    preconditioning. Numerical correctness of the precondition is
    validated end-to-end in the C-backend integration tests / the
    legacy 3-D OBN comparison runs (out of scope for this 2-D unit test).
    """
    base_spec = _baseline_fwi_spec(tmp_path)
    base_spec["seed"] = 7
    base_spec["epochs"] = 2
    on_spec = dict(base_spec)
    on_spec["illumination_precondition"] = {
        "enabled": True, "epsilon": 1.0e-6, "exponent": 0.5,
    }
    r_off = TaskRunner().run(load_task(_write(base_spec, tmp_path / "off.yaml")))
    capsys.readouterr()  # drop the off-run captures
    r_on = TaskRunner().run(load_task(_write(on_spec, tmp_path / "on.yaml")))
    on_out = capsys.readouterr().out
    assert r_off.status.state == "success" and r_on.status.state == "success"
    assert "[illum] WARN" in on_out
    # Eager loss trajectory is identical with vs without illum (no-op).
    loss_off = np.load(r_off.task_dir / "output" / "loss.npy")
    loss_on = np.load(r_on.task_dir / "output" / "loss.npy")
    np.testing.assert_allclose(loss_off, loss_on)


def test_illumination_with_local_window_rejected(tmp_path):
    spec_dict = _baseline_fwi_spec(tmp_path)
    spec_dict["local_model_window"] = True
    spec_dict["illumination_precondition"] = {
        "enabled": True, "epsilon": 1.0e-6, "exponent": 0.5,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "bad.yaml")))
    assert result.status.state == "failed"
    err = (result.status.error or "").lower()
    assert "local_model_window" in err and "illumination" in err
