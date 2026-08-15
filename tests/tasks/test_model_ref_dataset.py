"""``ModelRef.dataset`` — reference a sweep.datasets benchmark straight from YAML.

Covers the source validator, the array resolution (preset / field pick /
smoothing) and one end-to-end run through ``TaskRunner`` so the new source is
exercised on a real task path rather than only in the helper.

Only EMBEDDED catalogue entries are used (``marmousi:2d-demo``,
``overthrust:2d-demo``) so the suite never touches the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sweep_tasks import TaskRunner, load_task
from sweep_tasks._helpers.model import _model_array, _infer_shape
from sweep_tasks.schemas import ModelRef


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


# ----- source validator ---------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {},                                                  # no source
    {"path": "x.npy", "dataset": "marmousi:2d-demo"},    # two sources
    {"constant": 1500.0, "shape": (4, 4), "dataset": "marmousi:2d-demo"},
    {"dataset": "marmousi:2d-demo", "path": "x.npy", "constant": 1.0},
])
def test_model_ref_requires_exactly_one_source(kwargs):
    with pytest.raises(ValueError, match="exactly one of"):
        ModelRef(name="vp", **kwargs)


@pytest.mark.parametrize("field", ["dataset_field", "preset", "downsample"])
def test_dataset_only_options_rejected_without_dataset(field):
    value = 2 if field == "downsample" else "x"
    with pytest.raises(ValueError, match="only applies together"):
        ModelRef(name="vp", path="x.npy", **{field: value})


def test_dataset_source_accepted():
    ref = ModelRef(name="vp", dataset="marmousi:2d-demo", preset="vp_smooth")
    assert ref.dataset == "marmousi:2d-demo" and ref.path is None


# ----- array resolution ---------------------------------------------------

def test_dataset_array_matches_sweep_datasets():
    from sweep.datasets import load_marmousi
    arr = _model_array(ModelRef(name="vp", dataset="marmousi:2d-demo"))
    np.testing.assert_allclose(arr, load_marmousi("vp_true"))


def test_preset_selects_a_different_model():
    true = _model_array(ModelRef(name="vp", dataset="marmousi:2d-demo"))
    smooth = _model_array(ModelRef(name="vp", dataset="marmousi:2d-demo",
                                   preset="vp_smooth"))
    assert true.shape == smooth.shape
    assert not np.array_equal(true, smooth)
    # smoothing a model can only reduce its lateral roughness
    assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(true, axis=1)).mean()


def test_sole_array_field_is_used_when_name_does_not_match():
    """Embedded demos return the SELECTED preset under the key "vp" whatever
    the preset was, so a ModelRef named "vs" must still resolve."""
    vs = _model_array(ModelRef(name="vs", dataset="marmousi:2d-demo",
                               preset="vs_true"))
    from sweep.datasets import load_marmousi
    np.testing.assert_allclose(vs, load_marmousi("vs_true"))


def test_smooth_sigma_cells_smooths_any_source(tmp_path):
    rng = np.random.default_rng(0)
    rough = rng.standard_normal((40, 60)).astype(np.float32) * 100 + 2500
    p = tmp_path / "rough.npy"
    np.save(p, rough)
    plain = _model_array(ModelRef(name="vp", path=p))
    smoothed = _model_array(ModelRef(name="vp", path=p, smooth_sigma_cells=3.0))
    assert smoothed.shape == plain.shape
    assert smoothed.std() < plain.std()


def test_infer_shape_from_dataset():
    shape = _infer_shape([ModelRef(name="vp", dataset="overthrust:2d-demo")], None)
    assert shape == (187, 801)


def test_unknown_dataset_field_is_a_clear_error():
    ref = ModelRef(name="vp", dataset="marmousi:2d-demo", dataset_field="nope")
    with pytest.raises(ValueError, match="has no field 'nope'"):
        _model_array(ref)


# ----- end to end ---------------------------------------------------------

def test_forward_task_runs_from_dataset_only_yaml(tmp_path):
    """A task YAML with no .npy anywhere: models come from sweep.datasets."""
    spec_dict = {
        "task_type": "forward",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 25.0},
        "time": {"dt": 0.002, "nt": 150},
        "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.12, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 300, "depth": 2, "start": 100, "stop": 700},
            "receivers": {"step": 20, "depth": 4, "start": 20, "stop": 780},
        },
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 12,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False},
        "models": [{"name": "vp", "dataset": "overthrust:2d-demo",
                    "preset": "true", "smooth_sigma_cells": 4.0}],
    }
    result = TaskRunner().run(load_task(_write(spec_dict, tmp_path / "ds.yaml")))
    assert result.status.state == "success", result.status.error
    rec = np.load(result.task_dir / "output" / "record.npy")
    assert rec.shape[0] == 2         # LineSet stop is exclusive: 100, 400
    assert np.all(np.isfinite(rec)) and np.abs(rec).max() > 0
    # grid.shape is unset, so the (187, 801) grid was inferred from the
    # dataset — receivers at x up to 780 would be out of bounds otherwise.
    assert rec.shape[2] == 38
