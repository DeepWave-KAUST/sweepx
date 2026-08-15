"""``ModelRef`` sources that need no file on disk.

``dataset`` pulls a benchmark out of :mod:`sweep.datasets`; ``linear_gradient``
builds a 1-D ramp in memory. Both exist so a task YAML is self-contained, and
both are covered here at three levels: the source validator, the array the
helper resolves, and one end-to-end ``TaskRunner`` run each — a source that
only works in the helper is the failure mode worth guarding against.

Also pins ``output/initial_vp.npy``: with an in-memory source there is no input
file to point at afterwards, so the run dir has to record its own start.

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


# ----- linear_gradient source ---------------------------------------------

def test_linear_gradient_builds_water_layer_over_a_ramp():
    ref = ModelRef(name="vp", shape=(281, 1361),
                   linear_gradient={"vmin": 1500.0, "vmax": 4000.0,
                                    "water_rows": 37, "water_vp": 1500.0})
    a = _model_array(ref)
    assert a.shape == (281, 1361) and a.dtype == np.float32
    assert np.allclose(a[:37], 1500.0)              # flat water column
    assert a[37, 0] == pytest.approx(1500.0)        # ramp starts below it
    assert a[-1, 0] == pytest.approx(4000.0)
    assert np.allclose(a, a[:, :1])                 # laterally constant
    below = a[37:, 0]
    assert np.all(np.diff(below) > 0)               # monotone ramp


def test_linear_gradient_is_3d_capable():
    a = _model_array(ModelRef(name="vp", shape=(40, 30, 50),
                              linear_gradient={"vmin": 1500.0, "vmax": 4000.0,
                                               "water_rows": 5}))
    assert a.shape == (40, 30, 50)
    assert np.allclose(a[:5], 1500.0)
    assert np.allclose(a, a[:, :1, :1])


def test_linear_gradient_requires_shape():
    with pytest.raises(ValueError, match="`shape` is required"):
        ModelRef(name="vp", linear_gradient={"vmin": 1500.0, "vmax": 4000.0})


def test_linear_gradient_rejects_inverted_range():
    with pytest.raises(ValueError, match="vmax"):
        ModelRef(name="vp", shape=(10, 10),
                 linear_gradient={"vmin": 4000.0, "vmax": 1500.0})


def test_linear_gradient_is_exclusive_with_other_sources():
    with pytest.raises(ValueError, match="exactly one of"):
        ModelRef(name="vp", shape=(10, 10), dataset="marmousi:2d-demo",
                 linear_gradient={"vmin": 1500.0, "vmax": 4000.0})



# ----- the run dir records where the inversion started ---------------------

def test_fwi_writes_the_resolved_initial_model(tmp_path):
    """`output/initial_vp.npy` must land next to the result.

    With an in-memory source (`dataset` / `linear_gradient`) there is no input
    file to point at afterwards, so without this dump a QC plot has to rebuild
    the starting model by hand and drifts the moment a parameter changes.
    """
    import yaml as _yaml

    true_path = tmp_path / "true.npy"
    np.save(true_path, (2200 + 600 * np.linspace(0, 1, 48)[:, None]
                        * np.ones((48, 48))).astype(np.float32))
    spec_dict = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "seed": 0,
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {"kind": "line",
                     "sources": {"step": 12, "depth": 2, "start": 6, "stop": 42},
                     "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 44}},
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 12,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "shape": [48, 48],
                       "linear_gradient": {"vmin": 2000.0, "vmax": 3000.0,
                                           "water_rows": 4}},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 2,
    }
    yaml_path = tmp_path / "init_dump.yaml"
    yaml_path.write_text(_yaml.safe_dump(spec_dict, sort_keys=False))

    result = TaskRunner().run(load_task(yaml_path))
    assert result.status.state == "success", result.status.error
    dumped = result.task_dir / "output" / "initial_vp.npy"
    assert dumped.exists(), "the run dir does not record its starting model"
    expected = _model_array(load_task(yaml_path).init_model)
    np.testing.assert_allclose(np.load(dumped), expected)
    # and it is genuinely the START, not a copy of the result
    assert not np.array_equal(np.load(dumped),
                              np.load(result.task_dir / "output" / "inverted_vp.npy"))
