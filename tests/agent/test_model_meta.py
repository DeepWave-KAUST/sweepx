"""Grid spacing recorded with a model file, and the tools' use of it.

An LLM that loaded Marmousi (dh 12.5 m) passed dh=20 to run_forward_sweep and got
a "successful" run on a stretched grid. Models written by get_benchmark_model /
make_synthetic_model now record dh in <stem>.meta.json; tools fill an omitted dh
from it and reject a different one. run_forward_sweep also reports the run in
physical units with warnings.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sweep")

from sweepx.agent.tools import registry  # noqa: E402
from sweepx.agent.tools._modelmeta import meta_path, read_model_meta  # noqa: E402


def _synth(tmp_path, dh=5.0, name="m.npy", **kw):
    r = registry.get("make_synthetic_model").invoke({"nz": 30, "nx": 40, "out_path": str(tmp_path / name), "dh": dh, **kw})
    assert r.ok, r.error
    return r.value


def _forward(**kw):
    return registry.get("run_forward_sweep").invoke({"nt": 60, "fm": 10.0, "device": "cpu", "plot": False, **kw})


def test_writers_record_dh(tmp_path):
    v = _synth(tmp_path, dh=7.5)
    assert v["dh"] == 7.5 and "dh=7.5 m" in v["summary"]
    assert read_model_meta(v["model_path"])["dh"] == 7.5
    el = _synth(tmp_path, dh=4.0, name="el.npy", elastic=True)
    for key in ("vp_path", "vs_path", "rho_path"):
        assert read_model_meta(el[key])["dh"] == 4.0
    b = registry.get("get_benchmark_model").invoke({"name": "marmousi", "out_dir": str(tmp_path)})
    assert b.ok, b.error
    assert read_model_meta(b.value["model_path"])["dh"] == b.value["dh"] == 12.5


def test_omitted_dh_is_filled_and_a_different_one_rejected(tmp_path):
    vp = _synth(tmp_path, dh=5.0)["model_path"]
    ok = _forward(vp_path=vp, out_dir=str(tmp_path / "run"))
    assert ok.ok, ok.error
    assert ok.value["physics"]["dh_m"] == 5.0
    bad = _forward(vp_path=vp, dh=20.0)
    assert not bad.ok and "dh=20 m disagrees with the 5 m" in bad.error
    same = _forward(vp_path=vp, dh=5.0, out_dir=str(tmp_path / "run2"))
    assert same.ok, same.error


def test_models_recording_different_dh_cannot_share_a_grid(tmp_path):
    a = _synth(tmp_path, dh=5.0, name="a.npy")["model_path"]
    b = _synth(tmp_path, dh=10.0, name="b.npy")["model_path"]
    r = registry.get("run_fwi").invoke({"init_model_path": a, "synthetic_true_vp_path": b})
    assert not r.ok and "different grid spacings" in r.error


def test_no_record_still_needs_dh(tmp_path):
    bare = tmp_path / "bare.npy"
    np.save(bare, np.full((30, 40), 2000.0, np.float32))
    r = _forward(vp_path=str(bare))
    assert not r.ok and "dh (grid spacing, m) is required" in r.error


def test_placeholder_path_reports_the_missing_file():
    r = _forward(vp_path="${VP_PATH}")
    assert not r.ok and "model file not found: ${VP_PATH}" in r.error


def test_stale_record_is_ignored(tmp_path):
    vp = _synth(tmp_path, dh=5.0)["model_path"]
    np.save(vp, np.full((10, 12), 2000.0, np.float32))  # overwritten with another shape
    assert read_model_meta(vp) is None
    assert meta_path(vp).is_file()
    r = _forward(vp_path=vp, dh=20.0, nt=10, out_dir=str(tmp_path / "run"))
    assert r.ok, r.error


def test_build_spec_fills_and_checks_grid_dh(tmp_path):
    vp = _synth(tmp_path, dh=5.0)["model_path"]
    body = {
        "time": {"dt": 0.001, "nt": 50}, "wavelet": {"kind": "ricker", "fm": 10.0},
        "geometry": {"kind": "line", "sources": {"step": 20, "depth": 1}, "receivers": {"step": 1, "depth": 1}},
        "physics": {"equation": "Acoustic"},
        "models": [{"name": "vp", "path": vp}],
    }
    bad = registry.get("build_spec").invoke({"task_type": "forward", "spec": {**body, "grid": {"dh": 10.0}}, "save_yaml": False})
    assert bad.ok and "grid.dh=10 m disagrees with the 5 m" in bad.value["error"]
    good = registry.get("build_spec").invoke({"task_type": "forward", "spec": dict(body), "save_yaml": False})
    assert good.ok and "error" not in good.value, good.value
    assert good.value["spec"]["grid"]["dh"] == 5.0


def test_schema_is_compact_and_dh_optional_where_a_model_records_it():
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in registry.openai_specs()}

    def keys(node, under_properties=False):
        if isinstance(node, dict):
            for k, v in node.items():
                if not under_properties:
                    yield k
                yield from keys(v, under_properties=(k == "properties" and not under_properties))
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert "title" not in set(keys(specs, under_properties=True))  # a parameter may be *named* title
    assert '{"type": "null"}' not in json.dumps(specs)
    fwd = specs["run_forward_sweep"]
    assert "dh" not in fwd.get("required", []) and "recorded" in fwd["properties"]["dh"]["description"]
    assert "vp_path" in fwd["required"]


def test_physics_report_warns_about_a_record_that_is_too_short(tmp_path):
    vp = _synth(tmp_path, dh=5.0)["model_path"]
    r = _forward(vp_path=vp, nt=20, source_x_frac=0.0, out_dir=str(tmp_path / "run"))
    assert r.ok, r.error
    phys = r.value["physics"]
    assert phys["source_m"] == [0.0, 10.0] and phys["model_extent_m"] == [200.0, 150.0]
    assert phys["record_length_s"] == pytest.approx(20 * r.value["dt"])
    assert any("ends before the direct wave" in w for w in phys["warnings"])
    assert r.value["impl"] == "eager"  # CPU
    assert "WARNINGS:" in r.value["summary"]


def test_forward_runs_on_the_c_backend(tmp_path):
    """impl='c' takes the geometry as numpy (it copies and shifts it on the host);
    torch tensors made it fail with "'Tensor' object has no attribute 'copy'"."""
    vp = _synth(tmp_path, dh=5.0)["model_path"]
    r = _forward(vp_path=vp, backend_impl="c", out_dir=str(tmp_path / "c"))
    if not r.ok and "not available" in r.error.lower():
        pytest.skip(f"no compiled sweep core here: {r.error}")
    assert r.ok, r.error
    assert r.value["impl"] == "c"
    e = _forward(vp_path=vp, backend_impl="eager", out_dir=str(tmp_path / "e"))
    a, b = np.load(tmp_path / "c" / "record.npy"), np.load(tmp_path / "e" / "record.npy")
    assert np.abs(a - b).max() <= 1e-3 * np.abs(b).max()


def test_returned_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vp = _synth(tmp_path, dh=5.0)["model_path"]
    r = registry.get("run_forward_sweep").invoke({"vp_path": vp, "nt": 30, "device": "cpu", "out_dir": "rel"})
    assert r.ok, r.error
    assert r.value["record_path"] == str(tmp_path / "rel" / "record.npy")
    m = registry.get("plot_model").invoke({"model_path": vp})
    assert m.ok and m.value["image_path"] == str(tmp_path / "sweep_runs" / "m.png"), m


def test_animate_reports_a_missing_model_file(tmp_path):
    r = registry.get("animate_wavefield").invoke({"vp_path": str(tmp_path / "nope_vp.npy")})
    assert r.ok and "model file not found" in r.value["error"]
