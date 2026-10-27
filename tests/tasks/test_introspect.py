"""``sweepx.tasks.introspect`` and ``yaml_io.resolve_paths``: the public views
tools such as sweep-studio build forms, previews and pre-flight checks on.

Each view must agree with the code path the runner actually takes, so most
tests here compare a public wrapper against the private helper the runner
calls (or against a real run) rather than against hand-written expectations.
Only embedded datasets (``marmousi:2d-demo``) are used, so nothing touches the
network.
"""

from __future__ import annotations

import copy
import importlib.resources as ir
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import TypeAdapter

import sweep.equations as eq_mod

from sweepx.tasks import TaskRunner, introspect, load_task_from_dict
from sweepx.tasks._helpers.geometry import _build_geometry_2d
from sweepx.tasks._helpers.model import (
    _get_equation_class,
    _infer_shape,
    _model_array,
    _model_names_for_equation,
)
from sweepx.tasks._helpers.solver_build import _cfl_check
from sweepx.tasks._helpers.wavelet_build import _build_wavelet
from sweepx.tasks.schemas import ModelRef, TaskSpec
from sweepx.tasks.yaml_io import (
    RESOLVED_PATH_KEYS,
    RESOLVED_WATER_PATH_KEYS,
    _resolve_paths,
    resolve_paths,
)

ROW_KEYS = {
    "name", "ndim", "summary", "is_facade", "models", "fields",
    "default_source_fields", "default_receiver_fields", "pml", "free_surface",
    "impl", "boundary_saving_c", "topography", "note",
}
MODEL_KEYS = {"name", "aliases", "unit", "description", "required"}
FIELD_KEYS = {"name", "aliases", "description", "supports_source",
              "supports_receiver", "internal"}


@pytest.fixture(scope="module")
def catalog():
    return introspect.equation_catalog()


def _concrete(catalog):
    return {n: r for n, r in catalog.items() if not r["is_facade"]}


# ----- equation catalogue ---------------------------------------------------

def test_catalog_covers_every_runner_equation(catalog):
    assert set(_concrete(catalog)) == set(eq_mod._equation_classes())


def test_every_row_has_the_contract_keys_and_round_trips_json(catalog):
    for name, row in catalog.items():
        assert set(row) == ROW_KEYS, name
        assert row["name"] == name
        assert all(set(m) == MODEL_KEYS for m in row["models"]), name
        assert all(set(f) == FIELD_KEYS for f in row["fields"]), name
        assert set(row["pml"]) == {"default", "supported"}
        assert set(row["free_surface"]) == {"supported", "per_edge"}
        assert set(row["impl"]) == {"eager", "c", "c_available"}
        assert json.loads(json.dumps(row, allow_nan=False)) == row, name


def test_every_concrete_equation_is_fully_described(catalog):
    """A failure is recorded in ``note`` instead of raising; on the installed
    sweep there must be none, or the catalogue is hiding a gap."""
    for name, row in _concrete(catalog).items():
        assert "unavailable" not in (row["note"] or ""), (name, row["note"])
        assert row["models"] and row["fields"], name
        assert row["impl"]["eager"] is True
        assert row["pml"]["default"] in row["pml"]["supported"], name


def test_describe_equation_equals_the_catalogue_row(catalog):
    for name, row in catalog.items():
        assert introspect.describe_equation(name) == row


def test_models_are_in_runner_order(catalog):
    for name, row in _concrete(catalog).items():
        runner = _model_names_for_equation(_get_equation_class(name))
        assert [m["name"] for m in row["models"]] == runner, name


def test_acoustic_models_and_defaults():
    row = introspect.describe_equation("Acoustic")
    assert [m["name"] for m in row["models"]] == ["vp"]
    assert row["models"][0]["unit"] == "m/s"
    assert row["ndim"] == 2
    assert row["default_source_fields"] == ["h1"]
    assert row["pml"] == {"default": "cpmlr", "supported": ["cpmlr"]}


@pytest.mark.parametrize("name", [
    "Elastic", "Elastic3D", "ElasticCurvilinear",
    "DASElastic", "DASElastic3D", "DASMu", "DASMu3D", "DASZhao", "DASZhao3D",
])
def test_elastic_family_has_vp_vs_rho(catalog, name):
    assert {"vp", "vs", "rho"} <= {m["name"] for m in catalog[name]["models"]}


def test_default_fields_are_valid_for_their_role(catalog):
    for name, row in _concrete(catalog).items():
        src = {f["name"] for f in row["fields"] if f["supports_source"]}
        rec = {f["name"] for f in row["fields"] if f["supports_receiver"]}
        assert row["default_source_fields"], name
        assert set(row["default_source_fields"]) <= src, name
        assert set(row["default_receiver_fields"]) <= rec, name


def test_ndim_is_read_off_the_runner_constructed_instance(catalog):
    """The rule is the instance attribute; the ``3D`` suffix is only a cross
    check that holds on the installed sweep. ``Elastic3D``'s class is named
    "Elastic", so ``__name__`` would have said 2."""
    for name, row in _concrete(catalog).items():
        assert row["ndim"] == (3 if name.endswith("3D") else 2), name
    assert eq_mod.Elastic3D.__name__ == "Elastic"
    assert catalog["Elastic3D"]["note"] is None          # not an alias either


def test_aliases_are_flagged_by_identity(catalog):
    assert catalog["AcousticVTILiang"]["note"] == "alias of AcousticVTI (same class)"
    assert catalog["DASElastic"]["note"] == "alias of DASZhao (same class)"
    assert catalog["AcousticVTI"]["note"] is None


@pytest.mark.parametrize("name,targets", [
    ("DAS", ["DASMu", "DASMu3D", "DASZhao", "DASZhao3D"]),
    ("AcousticAniso", ["AcousticTTI", "AcousticTariq", "AcousticVTI",
                       "AcousticVTI1st", "AcousticVTI1st3D"]),
])
def test_facades_are_listed_but_not_selectable(catalog, name, targets):
    row = catalog[name]
    assert row["is_facade"] and row["models"] == [] and row["ndim"] is None
    assert row["impl"] == {"eager": False, "c": False, "c_available": False}
    assert "does not accept" in row["note"]
    assert f"instead: {targets}" in row["note"]
    assert all(not catalog[t]["is_facade"] for t in targets)


def test_facade_alias_is_flagged(catalog):
    assert "alias of DAS (same class)" in catalog["DASModeler"]["note"]


def test_topography_follows_the_step_that_reads_it(catalog):
    for name in ("Acoustic", "Elastic", "AcousticCurvilinear", "ElasticCurvilinear"):
        assert catalog[name]["topography"] is True, name
    for name in ("AcousticVTI", "AcousticBorn", "DASZhao"):
        assert catalog[name]["topography"] is False, name


def test_unknown_equation_lists_close_matches():
    with pytest.raises(KeyError, match="close matches: .*'Acoustic'"):
        introspect.describe_equation("Acoustik")


# ----- CFL --------------------------------------------------------------------

def _cfl_check_before_refactor(vmax_m_s, dh_m, dt_s, *, threshold=0.85):
    # Verbatim body of _helpers.solver_build._cfl_check before it moved its
    # number and limits into sweepx.tasks.introspect.
    cfl = float(vmax_m_s) * float(dt_s) / float(dh_m)
    if cfl > 1.0:
        raise ValueError(
            f"CFL violation: vmax({vmax_m_s:.0f}) * dt({dt_s}) / dh({dh_m}) = "
            f"{cfl:.3f} > 1.0 — the forward propagator will produce NaNs. "
            f"Lower `time.dt` (try {0.6 * dh_m / vmax_m_s:.4f}s) or coarsen "
            f"`grid.dh`."
        )
    if cfl > threshold:
        print(
            f"[cfl] WARNING: vmax * dt / dh = {cfl:.3f} > {threshold:.2f}. "
            f"Stable forward modeling is not guaranteed; consider dt <= "
            f"{0.6 * dh_m / vmax_m_s:.4f}s for vmax={vmax_m_s:.0f} m/s."
        )


def test_cfl_constants_and_number():
    assert (introspect.CFL_WARN, introspect.CFL_RAISE) == (0.85, 1.0)
    assert introspect.cfl_number(3000, 12.5, 0.001) == 3000.0 * 0.001 / 12.5
    assert introspect.cfl_number(3000, 10.0, 0.003) > introspect.CFL_WARN


def _outcome(fn, capsys, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        err = None
    except ValueError as exc:
        err = str(exc)
    return err, capsys.readouterr().out


@pytest.mark.parametrize("args,kwargs", [
    ((2000.0, 10.0, 0.001), {}),                  # 0.2: silent
    ((3000.0, 10.0, 0.003), {}),                  # 0.9: warns
    ((4000.0, 10.0, 0.003), {}),                  # 1.2: raises
    ((3333.0, 10.0, 0.003), {}),                  # 0.9999: warns, not raises
    ((2500, 12.5, 0.004), {}),                    # ints / exactly 0.8
    ((3000.0, 10.0, 0.0025), {"threshold": 0.5}), # 0.75 against a lower bar
    ((1500.0, 5.0, 0.002), {"threshold": 0.6}),
])
def test_cfl_check_messages_unchanged(capsys, args, kwargs):
    assert _outcome(_cfl_check, capsys, *args, **kwargs) == \
        _outcome(_cfl_check_before_refactor, capsys, *args, **kwargs)


# ----- resolve_paths ------------------------------------------------------------

def _bundled_templates() -> dict[str, dict]:
    root = ir.files("sweepx.tasks").joinpath("templates")
    return {p.name: yaml.safe_load(p.read_text()) for p in root.iterdir()
            if p.name.endswith(".yaml")}


def test_resolve_paths_equals_the_private_rule_on_every_template(tmp_path):
    templates = _bundled_templates()
    assert {"forward.yaml", "fwi.yaml", "rtm.yaml", "lsrtm.yaml"} <= set(templates)
    for name, raw in templates.items():
        before = copy.deepcopy(raw)
        assert resolve_paths(raw, tmp_path) == _resolve_paths(raw, tmp_path.resolve()), name
        assert raw == before, f"{name}: resolve_paths modified its input"


def test_load_task_from_dict_is_resolve_then_validate(tmp_path):
    adapter = TypeAdapter(TaskSpec)
    tasks = {n: r for n, r in _bundled_templates().items() if "task_type" in r}
    assert len(tasks) >= 6
    for name, raw in tasks.items():
        assert load_task_from_dict(raw, base_dir=tmp_path) == \
            adapter.validate_python(resolve_paths(raw, tmp_path)), name


def test_resolved_keys_are_exactly_the_published_set(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_HOOKS_ROOT", "/demo/root")
    unresolved = ["plan_path", "index_path", "topography", "data_mask_path",
                  "diving_window_db", "ckpt_path", "seabed_depth_path",
                  "init_from", "true_model_path", "rotation_metadata"]
    doc = {"a": [{k: "rel/x" for k in RESOLVED_PATH_KEYS | set(unresolved)}],
           "path": "${STUDIO_HOOKS_ROOT}/vp.npy", "npy_path": "/abs/obs.npy",
           "output_dir": 7}
    out = resolve_paths(doc, tmp_path)
    for k in RESOLVED_PATH_KEYS:
        assert out["a"][0][k] == str(tmp_path.resolve() / "rel/x"), k
    for k in unresolved:
        assert out["a"][0][k] == "rel/x", k
    assert out["path"] == "/demo/root/vp.npy"
    assert out["npy_path"] == "/abs/obs.npy"
    assert out["output_dir"] == 7                       # non-strings pass through


def test_water_files_resolve_only_inside_the_water_block(tmp_path):
    assert RESOLVED_WATER_PATH_KEYS == {"seabed", "mask"}
    doc = {"water": {"seabed": "sb.npz", "mask": "m.npy", "velocity": 1500.0},
           "stages": [{"water": {"seabed": 120.0}}],
           "seabed": "x", "reparam": {"mask": "y"}}
    out = resolve_paths(doc, tmp_path)
    base = tmp_path.resolve()
    assert out["water"] == {"seabed": str(base / "sb.npz"), "mask": str(base / "m.npy"),
                            "velocity": 1500.0}
    assert out["stages"][0]["water"] == {"seabed": 120.0}   # a flat depth stays a number
    assert out["seabed"] == "x" and out["reparam"] == {"mask": "y"}   # ordinary words elsewhere
    assert out == _resolve_paths(doc, base)


def test_resolve_paths_refuses_a_non_mapping(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        resolve_paths(["path", "x"], tmp_path)


# ----- load_model_array ----------------------------------------------------------

@pytest.mark.parametrize("ref", [
    {"name": "vp", "constant": 2000.0, "shape": [12, 20]},
    {"name": "vp", "constant": 2000.0, "shape": [6, 8, 10]},
    {"name": "vp", "linear_gradient": {"vmin": 1800, "vmax": 3500, "water_rows": 4},
     "shape": [30, 16]},
    {"name": "vp", "linear_gradient": {"vmin": 1800, "vmax": 3500}, "shape": [30, 16],
     "smooth_sigma_cells": 2.0},
    {"name": "vp", "dataset": "marmousi:2d-demo"},
    {"name": "vp", "dataset": "marmousi:2d-demo", "preset": "vp_smooth"},
    {"name": "vp", "dataset": "marmousi:2d-demo", "smooth_sigma_cells": 3.0},
])
def test_load_model_array_matches_the_runner_loader(tmp_path, ref):
    got = introspect.load_model_array(ref, tmp_path)
    want = _model_array(ModelRef(**ref))
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(introspect.load_model_array(ModelRef(**ref), tmp_path), want)


def test_load_model_array_resolves_a_relative_path_like_a_task_yaml(tmp_path):
    vp = np.linspace(1500, 3000, 6 * 9, dtype=np.float64).reshape(6, 9)
    (tmp_path / "models").mkdir()
    np.save(tmp_path / "models" / "vp.npy", vp)
    ref = {"name": "vp", "path": "models/vp.npy", "smooth_sigma_cells": 1.0}
    got = introspect.load_model_array(ref, tmp_path)
    want = _model_array(ModelRef(**{**ref, "path": str(tmp_path / "models" / "vp.npy")}))
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)


def test_load_model_array_hands_out_its_own_copy_of_a_dataset(tmp_path):
    ref = {"name": "vp", "dataset": "marmousi:2d-demo"}
    first = introspect.load_model_array(ref, tmp_path)
    first[0, 0] = -1.0
    assert introspect.load_model_array(ref, tmp_path)[0, 0] != -1.0
    assert _model_array(ModelRef(**ref))[0, 0] != -1.0


def test_load_model_array_validates_the_reference(tmp_path):
    with pytest.raises(ValueError, match="exactly one of"):
        introspect.load_model_array({"name": "vp", "constant": 1.0, "shape": [2, 2],
                                     "dataset": "marmousi:2d-demo"}, tmp_path)
    with pytest.raises(ValueError, match="Extra inputs"):
        introspect.load_model_array({"name": "vp", "constant": 1.0, "shape": [2, 2],
                                     "smoothing": 2}, tmp_path)


# ----- geometry / wavelet ----------------------------------------------------------

def _forward_template(tmp_path: Path, shape=(40, 160)) -> dict:
    """The bundled forward template on a small model: its `models:` swapped
    for the commented path alternative, ``./vp_true.npy``, placed beside it."""
    raw = _bundled_templates()["forward.yaml"]
    np.save(tmp_path / "vp_true.npy", np.full(shape, 2000.0, dtype=np.float32))
    raw["models"] = [{"name": "vp", "path": "./vp_true.npy"}]
    raw["output_dir"] = str(tmp_path / "runs")
    return raw


def test_geometry_and_wavelet_match_the_runner_helpers_on_the_forward_template(tmp_path):
    spec = load_task_from_dict(_forward_template(tmp_path), base_dir=tmp_path)
    shape = _infer_shape(spec.models, spec.grid.shape)
    assert introspect.grid_shape(spec) == shape == (40, 160)

    sources, receivers = introspect.build_geometry_arrays(spec)
    want_s, want_r = _build_geometry_2d(spec.geometry, shape)
    np.testing.assert_array_equal(sources, want_s)
    np.testing.assert_array_equal(receivers, want_r)
    assert sources.dtype == receivers.dtype == np.int64
    # sweep's (x, z) order: the template puts sources every 50 cells at depth 1
    np.testing.assert_array_equal(sources[:, 0], np.arange(0, 160, 50))
    assert set(sources[:, 1]) == {1} and set(receivers[..., 1].ravel()) == {18}
    assert receivers.shape == (sources.shape[0], 160, 2)

    wavelet = introspect.build_wavelet(spec)
    assert introspect.time_axis(spec) == (spec.time.dt, spec.time.nt)
    assert wavelet.dtype == np.float32 and wavelet.shape == (spec.time.nt,)
    np.testing.assert_array_equal(wavelet, _build_wavelet(spec.wavelet, spec.time))


def test_geometry_equals_what_a_forward_run_writes(tmp_path):
    raw = _forward_template(tmp_path, shape=(30, 120))
    raw["time"] = {"dt": 0.002, "nt": 60}
    raw["device"] = "cpu"
    # Uncompiled: torch.compile of the step (Inductor building C++ on CPU) is
    # most of this run's cost, and the geometry the runner writes does not
    # depend on it.
    raw["backend"] = {**raw["backend"], "eager_options": {"use_compile": False}}
    spec = load_task_from_dict(raw, base_dir=tmp_path)
    result = TaskRunner().run(spec)
    assert result.status.state == "success", result.status.error
    out = Path(spec.output_dir) / spec.task_id / "output"
    sources, receivers = introspect.build_geometry_arrays(spec)
    np.testing.assert_array_equal(np.load(out / "sources.npy"), sources)
    np.testing.assert_array_equal(np.load(out / "receivers.npy"), receivers)


def _fwi_doc(tmp_path: Path, **extra) -> dict:
    return {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "runs"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 300},
        "wavelet": {"kind": "ricker", "fm": 10.0, "delay": 0.1},
        "geometry": {"kind": "line", "sources": {"step": 20, "depth": 2},
                     "receivers": {"step": 2, "depth": 2}},
        "physics": {"equation": "Acoustic"},
        "init_model": {"name": "vp", "linear_gradient": {"vmin": 1500, "vmax": 2500},
                       "shape": [24, 60]},
        "obs": {"synthetic_from": {"name": "vp", "constant": 2000.0, "shape": [24, 60]}},
        "optimizer": {"kind": "adam", "lr": 1.0},
        "epochs": 1,
        **extra,
    }


def test_fwi_time_axis_follows_data_plan_dt_target(tmp_path):
    spec = load_task_from_dict(_fwi_doc(tmp_path, data_plan={"dt_target_s": 0.002}),
                               base_dir=tmp_path)
    assert introspect.time_axis(spec) == (0.002, 150)
    np.testing.assert_array_equal(
        introspect.build_wavelet(spec),
        _build_wavelet(spec.wavelet, spec.time, override_dt=0.002, override_nt=150))
    assert introspect.grid_shape(spec) == (24, 60)
    sources, _ = introspect.build_geometry_arrays(spec)
    want, _ = _build_geometry_2d(spec.geometry, (24, 60), dh=10.0, segy_cache={})
    np.testing.assert_array_equal(sources, want)


def test_plan_streaming_fwi_keeps_the_task_time_axis(tmp_path):
    # (a data_plan, which this path never reads, is refused when the YAML loads)
    doc = _fwi_doc(tmp_path)
    doc["obs"] = {"plan": {"plan_path": str(tmp_path / "plan.npz"), "sampling": {}}}
    spec = load_task_from_dict(doc, base_dir=tmp_path)
    assert introspect.time_axis(spec) == (0.001, 300)
    with pytest.raises(ValueError, match="obs.plan"):
        introspect.build_geometry_arrays(spec)


def test_frequency_selection_fwi_is_refused(tmp_path):
    doc = _fwi_doc(tmp_path, source_encoding={
        "enabled": True, "mode": "frequency_selection",
        "frequency": {"coeff_shards": str(tmp_path / "c*.npz")}})
    spec = load_task_from_dict(doc, base_dir=tmp_path)
    for fn, match in ((introspect.build_wavelet, "wavelet-free"),
                      (introspect.build_geometry_arrays, "coefficient shards"),
                      (introspect.time_axis, "comb")):
        with pytest.raises(ValueError, match=match):
            fn(spec)


def test_wrappers_refuse_a_task_without_a_solver(tmp_path):
    spec = load_task_from_dict({"task_type": "introspect", "action": "list_equations"},
                               base_dir=tmp_path)
    with pytest.raises(TypeError, match="forward / wavefield / fwi / rtm / lsrtm"):
        introspect.build_geometry_arrays(spec)
