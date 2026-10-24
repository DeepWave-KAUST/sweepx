"""obs.plan (``sampling=None``) materialisation: cache key and stage geometry.

Tiny 2-D streamer-style CSG plan on a 12.5 m grid: 6 shots 25 m apart at
x = 112 + 25 k, i.e. 0.5 m off the native nodes (like the Viking line), and
8 fixed receivers 50 m apart.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from sweepx.io.segy import (
    FORMAT_IEEE_FLOAT32,
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    write_segy_minimal,
)
from sweepx.io.seismic_plan import SeismicPlan
from sweepx.tasks import TaskRunner, load_task, load_task_from_dict
from sweepx.tasks._helpers.plan_apply import _normalize_fwi_init_models
from sweepx.tasks.plan_materialize import _cache_key, materialize_plan_dataset

NT, DT = 160, 0.001
DH = 12.5
SHAPE = (20, 48)
SHOT_X = 112.0 + 25.0 * np.arange(6)
REC_X = 100.0 + 50.0 * np.arange(8)
DEPTH = 25.0


def _plan(tmp_path: Path) -> Path:
    nshot, nrec = SHOT_X.size, REC_X.size
    traces = np.random.default_rng(0).standard_normal((nshot * nrec, NT)).astype(np.float32)
    segy = tmp_path / "line.sgy"
    write_segy_minimal(segy, traces, dt=DT, sample_format=FORMAT_IEEE_FLOAT32)
    tr_bytes = SEGY_TRACE_HEADER_SIZE + NT * 4
    src = np.stack([SHOT_X, np.zeros(nshot), np.full(nshot, DEPTH)], axis=-1)
    rec = np.stack([REC_X, np.zeros(nrec), np.full(nrec, DEPTH)], axis=-1)
    plan = SeismicPlan(
        grouping="csg",
        files=[segy],
        trace_size_per_file=np.asarray([tr_bytes], dtype=np.int64),
        sample_format=FORMAT_IEEE_FLOAT32,
        samples_per_trace=NT,
        dt_s=DT,
        row_file_id=np.zeros(nshot * nrec, dtype=np.int64),
        row_trace_offset=(SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
                          + np.arange(nshot * nrec, dtype=np.int64) * tr_bytes),
        row_source_xyz=np.repeat(src, nrec, axis=0),
        row_receiver_xyz=np.tile(rec, (nshot, 1)),
        group_id=np.arange(nshot, dtype=np.int64),
        group_xyz=src,
        group_offsets=np.arange(nshot + 1, dtype=np.int64) * nrec,
        build_meta={"label": "tiny_2d_csg"},
    )
    path = tmp_path / "plan.npz"
    plan.save(path)
    return path


def _spec_dict(tmp_path: Path, plan_path: Path, *, vmax: float = 2500.0) -> dict:
    return {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"), "device": "cpu",
        "grid": {"dh": DH}, "time": {"dt": DT, "nt": NT},
        "wavelet": {"kind": "ricker", "fm": 5.0, "delay": 0.2, "scale": 1.0},
        "geometry": {"kind": "from_plan", "plan_path": str(plan_path),
                     "grid_origin_xyz_m": [0.0, 0.0, 0.0]},
        "physics": {"equation": "Acoustic", "spatial_order": 4, "abcn": 8,
                    "free_surface": False,
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "shape": list(SHAPE),
                       "linear_gradient": {"vmin": 1500.0, "vmax": vmax}},
        "obs": {"plan": {"plan_path": str(plan_path)}},
        "optimizer": {"kind": "adam", "lr": 1.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": SHOT_X.size, "show_every": 1,
    }


def _key(d: dict, tmp_path: Path) -> str:
    spec = load_task_from_dict(d, base_dir=tmp_path)
    return _cache_key(spec, _normalize_fwi_init_models(spec),
                      effective_dt=DT, effective_nt=NT, source_delay_s=0.0)


def _materialize(d: dict, tmp_path: Path) -> dict:
    spec = load_task_from_dict(d, base_dir=tmp_path)
    return materialize_plan_dataset(spec, _normalize_fwi_init_models(spec),
                                    effective_dt=DT, effective_nt=NT)


def test_cache_key_covers_every_starting_model_and_geometry_input(tmp_path):
    plan = _plan(tmp_path)
    base = _spec_dict(tmp_path, plan)
    k0 = _key(base, tmp_path)
    assert _key(_spec_dict(tmp_path, plan), tmp_path) == k0

    def variant(edit):
        d = yaml.safe_load(yaml.safe_dump(base))
        edit(d)
        return _key(d, tmp_path)

    lg = lambda d: d["init_model"]["linear_gradient"]            # noqa: E731
    changed = {
        "vmax": variant(lambda d: lg(d).update(vmax=3000.0)),
        "vmin": variant(lambda d: lg(d).update(vmin=1600.0)),
        "water_rows": variant(lambda d: lg(d).update(water_rows=3)),
        "shape": variant(lambda d: d["init_model"].update(shape=[22, 48])),
        "smooth": variant(lambda d: d["init_model"].update(smooth_sigma_cells=2.0)),
        "grid.dh": variant(lambda d: d["grid"].update(dh=10.0)),
        "lift": variant(lambda d: d["geometry"].update(lift_source_to_water_vp=1500.0)),
    }
    for what, k in changed.items():
        assert k != k0, f"{what} change still hits the old cache entry"

    # A constant model: its shape is part of the model too.
    def const(shape):
        def edit(d):
            d["init_model"] = {"name": "vp", "constant": 2000.0, "shape": shape}
        return edit
    assert variant(const([20, 48])) != variant(const([22, 48]))

    # An .npy rewritten in place (same path, same size, later write).
    npy = tmp_path / "init_vp.npy"
    np.save(npy, np.full(SHAPE, 2000.0, np.float32))

    def from_path(d):
        d["init_model"] = {"name": "vp", "path": str(npy)}
    k_file = variant(from_path)
    st = os.stat(npy)
    np.save(npy, np.full(SHAPE, 2100.0, np.float32))
    os.utime(npy, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert variant(from_path) != k_file


def test_materialize_changed_linear_gradient_is_not_served_stale(tmp_path, capsys):
    plan = _plan(tmp_path)
    m1 = _materialize(_spec_dict(tmp_path, plan, vmax=2500.0), tmp_path)
    assert m1["cropped_models"]["vp"].max() == pytest.approx(2500.0)
    m2 = _materialize(_spec_dict(tmp_path, plan, vmax=3000.0), tmp_path)
    assert m2["cropped_models"]["vp"].max() == pytest.approx(3000.0)
    capsys.readouterr()
    # The cache still serves an unchanged config.
    m3 = _materialize(_spec_dict(tmp_path, plan, vmax=2500.0), tmp_path)
    assert "CACHE HIT" in capsys.readouterr().out
    np.testing.assert_array_equal(m3["cropped_models"]["vp"], m1["cropped_models"]["vp"])
    for k in ("sources", "receivers", "sources_m", "receivers_m"):
        np.testing.assert_array_equal(m3[k], m1[k])


def test_materialize_returns_unrounded_positions(tmp_path):
    m = _materialize(_spec_dict(tmp_path, _plan(tmp_path)), tmp_path)
    np.testing.assert_allclose(m["sources_m"], np.stack([SHOT_X, np.full(6, DEPTH)], -1))
    np.testing.assert_allclose(
        m["receivers_m"], np.broadcast_to(np.stack([REC_X, np.full(8, DEPTH)], -1), (6, 8, 2)))
    # They snap back to exactly the materialised indices on the native grid.
    np.testing.assert_array_equal(np.rint(m["sources_m"] / DH), m["sources"])
    np.testing.assert_array_equal(np.rint(m["receivers_m"] / DH), m["receivers"])


def test_coarse_stage_snaps_unrounded_plan_positions(tmp_path, monkeypatch, capsys):
    # At 25 m the 12.5 m-rounded positions (x.5 m) all sit on half cells and
    # round-half-even merges neighbouring shots (6 -> 4); the true positions
    # land in 6 distinct cells, as the SEG-Y route gives.
    from sweepx.tasks.tasks.fwi import FWIRunnerMixin

    seen = []
    original = FWIRunnerMixin._fwi_train_step

    def _spy(self, spec_, solver, wavelet, sources, receivers, *args, **kwargs):
        seen.append(np.asarray(sources).copy())
        return original(self, spec_, solver, wavelet, sources, receivers, *args, **kwargs)

    monkeypatch.setattr(FWIRunnerMixin, "_fwi_train_step", _spy)
    d = _spec_dict(tmp_path, _plan(tmp_path))
    d["stages"] = [{"epochs": 1, "dh_m": 25.0}, {"epochs": 1, "dh_m": DH}]
    d["epochs"] = 2
    p = tmp_path / "staged.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    res = TaskRunner().run(load_task(p))
    assert res.status.state == "success", res.status.error
    assert "source dedupe" not in capsys.readouterr().out
    coarse, native = seen[0], seen[-1]
    np.testing.assert_array_equal(coarse[:, 0], np.round(SHOT_X / 25.0))
    np.testing.assert_array_equal(native[:, 0], np.round(SHOT_X / DH))
