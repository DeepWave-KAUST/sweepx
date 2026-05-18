"""Tests for the run-metadata dump (``_dump_run_metadata``) +
``save_gradient_ortho_slices_png``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml


def test_save_gradient_ortho_slices_png_writes(tmp_path):
    from sweep_tasks.qc import save_gradient_ortho_slices_png

    grad = (
        np.random.default_rng(0).standard_normal((20, 30, 40)).astype(np.float32)
    )
    out = tmp_path / "g" / "iter_0000.png"
    written = save_gradient_ortho_slices_png(
        grad, dh=(10.0, 10.0, 10.0), out_path=out, epoch=0,
    )
    assert written == out
    assert out.exists() and out.stat().st_size > 5_000


def test_save_gradient_ortho_slices_png_rejects_2d(tmp_path):
    from sweep_tasks.qc import save_gradient_ortho_slices_png

    with pytest.raises(ValueError, match="3-D grad"):
        save_gradient_ortho_slices_png(
            np.zeros((10, 10), dtype=np.float32),
            dh=10.0, out_path=tmp_path / "x.png", epoch=0,
        )


def _make_spec_dict(tmp_path, init_path, true_path, *, task_id="dump_test"):
    return {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "task_id": task_id,
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 6, "depth": 2, "start": 6, "stop": 42},
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 44},
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 4, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "model_bounds": {"vp": {"min": 1500.0, "max": 4500.0}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 1, "show_every": 1,
    }


def _write_tiny_models(tmp_path: Path):
    shape = (48, 48)
    true_vp = (2200 + 600 * np.linspace(0, 1, shape[0])[:, None]
               * np.ones(shape)).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    t = tmp_path / "true.npy"; i = tmp_path / "init.npy"
    np.save(t, true_vp); np.save(i, init_vp)
    return t, i


def test_dump_run_metadata_writes_yaml_and_json(tmp_path):
    """The dump produces a YAML mirror of the spec + a runtime json."""
    from sweep_tasks import load_task
    from sweep_tasks.runner import _dump_run_metadata

    true_path, init_path = _write_tiny_models(tmp_path)
    spec_dict = _make_spec_dict(tmp_path, init_path, true_path)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    spec = load_task(yaml_path)
    target_dir = tmp_path / "dumped"
    _dump_run_metadata(
        spec, target_dir,
        extras={"my_runtime_value": 42, "tag": "dump_test"},
    )
    cfg = (target_dir / "config_resolved.yaml")
    meta = (target_dir / "run_meta.json")
    assert cfg.exists(), "config_resolved.yaml should be written"
    assert meta.exists(), "run_meta.json should be written"

    # YAML round-trip: must be parseable + retain task_id.
    loaded_cfg = yaml.safe_load(cfg.read_text())
    assert loaded_cfg["task_type"] == "fwi"
    assert loaded_cfg["task_id"] == "dump_test"

    loaded_meta = json.loads(meta.read_text())
    # Required top-level keys
    for k in ("task_id", "host", "started_at", "package_versions",
              "git_repos", "cuda", "runtime", "env_vars_of_interest"):
        assert k in loaded_meta, f"missing key {k} in run_meta.json"
    # Extras propagated.
    assert loaded_meta["runtime"]["my_runtime_value"] == 42
    assert loaded_meta["runtime"]["tag"] == "dump_test"


def test_dump_run_metadata_resolved_yaml_expands_defaults(tmp_path):
    """The dumped YAML must contain pydantic defaults that the user
    didn't write — this is the whole point of dumping the resolved
    spec rather than echoing the raw input."""
    from sweep_tasks import load_task
    from sweep_tasks.runner import _dump_run_metadata

    true_path, init_path = _write_tiny_models(tmp_path)
    spec_dict = _make_spec_dict(tmp_path, init_path, true_path, task_id="dump2")
    yaml_path = tmp_path / "cfg2.yaml"
    yaml_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    spec = load_task(yaml_path)
    target = tmp_path / "out"
    _dump_run_metadata(spec, target)
    cfg = yaml.safe_load((target / "config_resolved.yaml").read_text())
    # ``seed`` is a pydantic default in BaseTaskSpec, not set in input.
    assert "seed" in cfg
    # Loss section default
    assert "loss" in cfg and cfg["loss"]["kind"] in ("mse", "l1", "huber", "trace_cosine")
