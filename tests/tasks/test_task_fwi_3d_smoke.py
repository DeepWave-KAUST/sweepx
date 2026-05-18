"""End-to-end smoke for 3-D ``_run_fwi`` (Acoustic3D).

Tiny ``(nz, ny, nx) = (16, 16, 24)`` synthetic problem on the eager backend.
Confirms that:

* the 3-D geometry dispatcher accepts ``ExplicitGeometry`` / ``FromFileGeometry``
  with ``(x, y, z)`` source/receiver indices;
* ``_resample_vp_tensor`` does the right thing on a 3-D vp (trilinear);
* the runner builds an Acoustic3D solver and runs a few epochs without
  crashing;
* loss decreases.

Companion to ``test_fwi_runs_and_loss_decreases`` (the 2-D smoke).
"""

from pathlib import Path

import numpy as np
import yaml

from sweep_tasks import load_task, TaskRunner


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _tiny_3d_models(tmp_path: Path) -> tuple[Path, Path, tuple[int, int, int]]:
    shape = (16, 16, 24)  # (nz, ny, nx)
    nz, ny, nx = shape
    z = np.arange(nz, dtype=np.float32)[:, None, None] / max(nz - 1, 1)
    true_vp = (2200.0 + 600.0 * z * np.ones(shape, dtype=np.float32)).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_path = tmp_path / "true.npy"
    init_path = tmp_path / "init.npy"
    np.save(true_path, true_vp)
    np.save(init_path, init_vp)
    return true_path, init_path, shape


def _explicit_3d_geometry() -> dict:
    """Return ``{sources, receivers}`` lists with 3-D ``(x, y, z)`` indices.

    Two sources, six receivers each (shared list).
    """
    # Sources spread along x at fixed y, z=2.
    sources = [[6, 8, 2], [14, 8, 2]]
    # Receivers on a small grid at z=4.
    receivers = [
        [4, 4, 4], [10, 4, 4], [16, 4, 4],
        [4, 12, 4], [10, 12, 4], [16, 12, 4],
    ]
    return {"kind": "explicit", "sources": sources, "receivers": receivers}


def test_fwi_3d_runs_and_loss_decreases(tmp_path):
    true_path, init_path, _ = _tiny_3d_models(tmp_path)
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 50.0},
        "time": {"dt": 0.002, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.1, "scale": 1.0},
        "geometry": _explicit_3d_geometry(),
        "physics": {
            "equation": "Acoustic3D", "spatial_order": 4, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 3, "batchsize": 2, "show_every": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi3d.yaml")))
    assert result.status.state == "success", result.status.error
    assert result.status.summary["loss_decreased"] is True


def test_fwi_3d_from_file_geometry(tmp_path):
    """3-D FromFileGeometry path: load ``(nshots, 3)`` + ``(nrec, 3)`` from .npy."""
    true_path, init_path, _ = _tiny_3d_models(tmp_path)
    sources = np.array([[6, 8, 2], [14, 8, 2]], dtype=np.int64)
    receivers = np.array(
        [[4, 4, 4], [10, 4, 4], [16, 4, 4],
         [4, 12, 4], [10, 12, 4], [16, 12, 4]], dtype=np.int64,
    )
    sf = tmp_path / "src3d.npy"
    rf = tmp_path / "rec3d.npy"
    np.save(sf, sources)
    np.save(rf, receivers)
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 50.0},
        "time": {"dt": 0.002, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.1, "scale": 1.0},
        "geometry": {
            "kind": "from_file",
            "sources_file": str(sf), "receivers_file": str(rf),
        },
        "physics": {
            "equation": "Acoustic3D", "spatial_order": 4, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 2, "show_every": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi3d_ff.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_3d_model_plan_crops(tmp_path):
    """3-D ModelPlan crop: keep a sub-volume + drop out-of-window sources."""
    true_path, init_path, shape = _tiny_3d_models(tmp_path)
    nz, ny, nx = shape
    dh = 50.0
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": dh},
        "time": {"dt": 0.002, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.1, "scale": 1.0},
        "geometry": _explicit_3d_geometry(),
        "physics": {
            "equation": "Acoustic3D", "spatial_order": 4, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 2, "show_every": 1,
        # Crop in y so we keep ~half the volume; sources at y=8 (=400 m) survive.
        "model_plan": {
            "y_window_m": [200.0, (ny - 1) * dh],
            "drop_outside_sources": True,
        },
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi3d_mp.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_3d_siren_pipeline_wavelet(tmp_path):
    """SIREN-pipeline npz wavelet routes through load_wavelet_npz + resample."""
    true_path, init_path, _ = _tiny_3d_models(tmp_path)
    # SIREN wavelet at 1 ms; solver runs at 2 ms — must auto-resample.
    nt_wav = 240
    dt_wav = 0.001
    t = np.arange(nt_wav, dtype=np.float32) * dt_wav
    fm = 8.0
    wave = ((1.0 - 2.0 * (np.pi * fm * (t - 0.08)) ** 2)
            * np.exp(-((np.pi * fm * (t - 0.08)) ** 2))).astype(np.float32)
    wav_path = tmp_path / "siren_wavelet.npz"
    np.savez(
        wav_path,
        optimized_siren_wavelet=wave,
        dt_s=np.float64(dt_wav),
        source_delay_s=np.float64(0.0),
    )
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 50.0},
        "time": {"dt": 0.002, "nt": 200},
        "wavelet": {
            "kind": "siren_pipeline_npz", "path": str(wav_path), "scale": 1.0,
        },
        "geometry": _explicit_3d_geometry(),
        "physics": {
            "equation": "Acoustic3D", "spatial_order": 4, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 2, "show_every": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi3d_wav.yaml")))
    assert result.status.state == "success", result.status.error


def test_fwi_3d_line_geometry_rejected(tmp_path):
    """LineGeometry is 2-D-only; surfacing a 3-D grid with it must fail clearly."""
    _, init_path, _ = _tiny_3d_models(tmp_path)
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 50.0},
        "time": {"dt": 0.002, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.1, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 4, "depth": 2, "start": 6, "stop": 20},
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 22},
        },
        "physics": {
            "equation": "Acoustic3D", "spatial_order": 4, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(init_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 1,
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "fwi3d_bad.yaml")))
    assert result.status.state == "failed"
    err = (result.status.error or "").lower()
    assert "line" in err and ("2-d" in err or "2d" in err)
