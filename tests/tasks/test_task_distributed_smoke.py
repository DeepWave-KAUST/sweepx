"""Subprocess-based smoke test for the torchrun (shot-parallel) FWI path.

We can't comfortably exercise `torch.distributed` from inside a single pytest
process, so this test spawns ``torchrun --nproc_per_node=2 --standalone``
against the real ``sweep_tasks.cli run`` entry point with the gloo backend on CPU.

Skips automatically when ``torchrun`` isn't on PATH (e.g. minimal CI images).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_models(tmp_path: Path) -> tuple[Path, Path]:
    shape = (32, 32)
    true_vp = (2200 + 600 * np.linspace(0, 1, shape[0])[:, None] * np.ones(shape)).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_path = tmp_path / "true.npy"
    init_path = tmp_path / "init.npy"
    np.save(true_path, true_vp)
    np.save(init_path, init_vp)
    return true_path, init_path


def _write_dist_yaml(tmp_path: Path, true_path: Path, init_path: Path) -> Path:
    spec = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "seed": 0,
        "device": "cpu",
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 160},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 4, "depth": 2, "start": 4, "stop": 28},
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 28},
        },
        "physics": {
            "equation": "Acoustic",
            "spatial_order": 8,
            "abcn": 10,
            "free_surface": False,
            "pml_type": "cpmlr",
            "source_type": ["h1"],
            "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 2,
        "batchsize": 4,
        "show_every": 1,
    }
    path = tmp_path / "dist_fwi.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_torchrun_two_ranks_gloo_fwi(tmp_path):
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun is not on PATH")

    true_path, init_path = _write_tiny_models(tmp_path)
    yaml_path = _write_dist_yaml(tmp_path, true_path, init_path)

    env = os.environ.copy()
    extra = str(ROOT / "src")
    env["PYTHONPATH"] = (extra + os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else extra
    env["SWEEP_DIST_BACKEND"] = "gloo"
    # Force CPU even when CUDA is present, so we can run 2 ranks on a single GPU host.
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--standalone",
        "-m", "sweep_tasks.cli", "run", str(yaml_path),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"torchrun returned {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )

    tasks_root = tmp_path / "tasks"
    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir())
    assert task_dirs, f"No task dirs created under {tasks_root}"
    status = json.loads((task_dirs[-1] / "status.json").read_text())
    assert status["state"] == "success", f"Task failed: {status}"
    assert status["summary"]["world_size"] == 2
    # The loss list should contain exactly `epochs` entries (no rank-0 duplication)
    losses = np.load(task_dirs[-1] / "output" / "loss.npy")
    assert losses.shape[0] == 2


def test_torchrun_blocks_lbfgs(tmp_path):
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun is not on PATH")

    true_path, init_path = _write_tiny_models(tmp_path)
    yaml_path = _write_dist_yaml(tmp_path, true_path, init_path)
    spec = yaml.safe_load(yaml_path.read_text())
    spec["optimizer"] = {"kind": "lbfgs", "lr": 1.0, "max_iter": 3, "history_size": 3}
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False))

    env = os.environ.copy()
    extra = str(ROOT / "src")
    env["PYTHONPATH"] = (extra + os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else extra
    env["SWEEP_DIST_BACKEND"] = "gloo"
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--standalone",
        "-m", "sweep_tasks.cli", "run", str(yaml_path),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    # The task should land state=failed (with a clear error) — the CLI returns 1 in that case.
    assert proc.returncode == 1, (
        f"expected returncode 1 (task failed), got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )
    tasks_root = tmp_path / "tasks"
    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir())
    assert task_dirs, "no task dir created"
    status = json.loads((task_dirs[-1] / "status.json").read_text())
    assert status["state"] == "failed"
    assert "LBFGS" in (status.get("error") or "")
