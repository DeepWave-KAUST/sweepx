"""The `water:` block: one water layer for every inversion path.

Schema (exactly one source, ranges, unknown keys), the retired keys (a
migration error at any non-default value, silence at the defaults an older
run's config_resolved.yaml spells out), the region on every stage grid (the
same physical layer for seabed and mask inputs, 2-D and 3-D, buffer as a
depth), the two modes on the runner paths, and ``migrate-water``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import TypeAdapter, ValidationError

from sweep_tasks import TaskRunner, load_task
from sweep_tasks._helpers.water import WaterRegion, _nearest_native
from sweep_tasks.bathymetry import save_seabed_depth_npz, water_mask_from_seabed_depth
from sweep_tasks.schemas import FWISpec, LSRTMSpec, TaskSpec, WaterSpec

DATA = Path(__file__).parent / "data" / "resolved_before_water"
_TA = TypeAdapter(TaskSpec)


def _write(spec: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


# ------------------------------------------------------------- schema ----

def _fwi_doc(**extra):
    doc = dict(
        task_type="fwi", grid={"dh": 10.0}, time={"dt": 0.001, "nt": 10},
        wavelet={"kind": "ricker", "fm": 5.0},
        geometry={"kind": "line", "sources": {"step": 5, "depth": 1},
                  "receivers": {"step": 1, "depth": 1}},
        physics={"equation": "Acoustic"},
        init_model={"name": "vp", "constant": 2000.0, "shape": [10, 20]},
        obs={"synthetic_from": {"name": "vp", "constant": 2000.0, "shape": [10, 20]}},
        optimizer={"kind": "adam", "lr": 1.0}, epochs=1)
    doc.update(extra)
    return doc


def _lsrtm_doc(**extra):
    doc = dict(
        task_type="lsrtm", grid={"dh": 10.0}, time={"dt": 0.001, "nt": 10},
        wavelet={"kind": "ricker", "fm": 5.0},
        geometry={"kind": "line", "sources": {"step": 5, "depth": 1},
                  "receivers": {"step": 1, "depth": 1}},
        physics={"equation": "AcousticLSRTM"},
        background_model={"name": "vp", "constant": 2000.0, "shape": [10, 20]},
        true_model={"name": "vp", "constant": 2000.0, "shape": [10, 20]},
        optimizer={"kind": "adam", "lr": 1.0}, epochs=1)
    doc.update(extra)
    return doc


@pytest.mark.parametrize("water, msg", [
    ({}, "exactly one of `seabed`"),
    ({"seabed": "/a.npz", "mask": "/b.npy"}, "exactly one of `seabed`"),
    ({"mask": "/b.npy", "buffer_cells": 1}, "bake the buffer into the mask"),
    ({"seabed": "/a.npz", "buffer_cells": -1}, "greater than or equal to 0"),
    ({"seabed": "/a.npz", "velocity": 0.0}, "greater than 0"),
    ({"seabed": "/a.npz", "velocity": float("inf")}, "finite number"),
    ({"seabed": -5.0}, "a flat seabed is a finite depth"),
    ({"seabed": "/a.npz", "mode": "free"}, "'pin' or 'reset_each_stage'"),
    ({"seabed": "/a.npz", "depth": 3}, "Extra inputs are not permitted"),
    (True, "valid dictionary"),
])
def test_water_schema_refuses(water, msg):
    with pytest.raises(ValidationError, match=msg):
        _TA.validate_python(_fwi_doc(water=water))


def test_water_schema_accepts():
    spec = _TA.validate_python(_fwi_doc(water={"seabed": 350}))
    assert spec.water == WaterSpec(seabed=350.0)
    assert spec.water.mode == "pin" and spec.water.velocity == 1500.0
    spec = _TA.validate_python(_fwi_doc(water={"mask": "/m.npy", "mode":
                                               "reset_each_stage",
                                               "velocity": 1485.6}))
    assert spec.water.mask == Path("/m.npy")
    assert _TA.validate_python(_fwi_doc(water=False)).water is False
    assert _TA.validate_python(_fwi_doc()).water is None
    assert _TA.validate_python(_lsrtm_doc(water={"seabed": 100.0})).water.seabed == 100.0


def test_water_paths_resolve_against_the_yaml(tmp_path):
    doc = _fwi_doc(water={"mask": "sub/water.npy"})
    spec = load_task(_write(doc, tmp_path / "t.yaml"))
    assert spec.water.mask == (tmp_path / "sub" / "water.npy").resolve()
    doc = _fwi_doc(water={"seabed": 12.5})
    assert load_task(_write(doc, tmp_path / "f.yaml")).water.seabed == 12.5


def test_lsrtm_refuses_a_water_velocity():
    with pytest.raises(ValidationError, match="water.velocity=1480.0 does not apply"):
        _TA.validate_python(_lsrtm_doc(water={"seabed": 100.0, "velocity": 1480.0}))


def test_free_params_water_value_needs_the_water_block():
    fp = [{"name": "vp", "std": 500.0}, {"name": "z", "std": 2.0, "water_value": 1.5}]
    rp = {"kind": "velocity_inr", "free_params": fp}
    with pytest.raises(ValidationError, match="no `water:` block"):
        _TA.validate_python(_fwi_doc(reparam=rp))
    _TA.validate_python(_fwi_doc(reparam=rp, water={"seabed": 50.0}))
    bad = [{"name": "vp", "std": 500.0, "water_value": 1480.0}]
    with pytest.raises(ValidationError, match="disagrees with water.velocity"):
        _TA.validate_python(_fwi_doc(reparam={"kind": "velocity_inr",
                                              "free_params": bad},
                                     water={"seabed": 50.0}))


# ------------------------------------------------------ retired keys ----

@pytest.mark.parametrize("doc, hint", [
    (_fwi_doc(freeze_top_n_rows=37), "mask: <boolean .npy of the native grid, rows 0..36"),
    (_fwi_doc(freeze_water_layer={"enabled": True, "seabed_depth_path": "/s.npz",
                                  "buffer_cells": 2}),
     "seabed: /s.npz\n      buffer_cells: 2"),
    (_fwi_doc(reparam={"kind": "velocity_inr", "mask_water_layer": True,
                       "seabed_depth_path": "/s.npz", "water_vp_m_s": 1485.6}),
     "seabed: /s.npz\n      velocity: 1485.6"),
    (_fwi_doc(reparam={"kind": "velocity_inr", "mask_water_layer": True,
                       "seabed_depth_path": "/s.npz",
                       "water_reset_each_stage": True}),
     "mode: reset_each_stage"),
    (_fwi_doc(reparam={"kind": "velocity_inr", "mask_water_layer": True}),
     "the water is no longer read off velocities"),
    (_fwi_doc(reparam={"kind": "velocity_inr", "water_reset_each_stage": True}),
     "mode: reset_each_stage"),
    (_fwi_doc(reparam={"kind": "velocity_inr", "seabed_depth_path": "/s.npz"}),
     "did nothing without mask_water_layer"),
    (_fwi_doc(reparam={"kind": "velocity_inr", "water_vp_m_s": 1480.0}),
     "velocity: 1480.0"),
    (_lsrtm_doc(freeze_top_n_rows=10), "reflectivity at zero"),
])
def test_retired_keys_raise_with_their_water_block(doc, hint):
    with pytest.raises(ValidationError) as err:
        _TA.validate_python(doc)
    msg = str(err.value)
    assert "was retired" in msg or "were retired" in msg
    assert "water:" in msg and hint in msg, msg


@pytest.mark.parametrize("doc", [
    _fwi_doc(freeze_top_n_rows=0, freeze_water_layer=None),
    _fwi_doc(freeze_water_layer={"enabled": False, "seabed_depth_path": None,
                                 "buffer_cells": 0}),
    _fwi_doc(reparam={"kind": "velocity_inr", "mask_water_layer": False,
                      "water_vp_m_s": 1500.0, "seabed_depth_path": None,
                      "water_reset_each_stage": False}),
    _lsrtm_doc(freeze_top_n_rows=0),
])
def test_retired_keys_at_their_defaults_still_load(doc):
    spec = _TA.validate_python(doc)
    dumped = spec.model_dump(mode="json")
    for k in ("freeze_top_n_rows", "freeze_water_layer"):
        assert k not in dumped
    assert "mask_water_layer" not in (dumped.get("reparam") or {})
    assert dumped["water"] is None


@pytest.mark.parametrize("name", ["conv_reparam", "freqsel_reparam", "lsrtm"])
def test_resolved_configs_from_before_the_water_block_load(name, tmp_path):
    """config_resolved.yaml files written by the previous sweep-tasks (every
    retired key spelled at its default) load, and re-dump without them."""
    text = (DATA / f"{name}.yaml").read_text()
    for k in ("freeze_top_n_rows",):
        assert k in text
    spec = load_task(_write(yaml.safe_load(text), tmp_path / f"{name}.yaml"))
    assert spec.water is None
    redump = yaml.safe_dump(spec.model_dump(mode="json"))
    for k in ("freeze_top_n_rows", "freeze_water_layer", "mask_water_layer",
              "water_vp_m_s", "seabed_depth_path", "water_reset_each_stage"):
        assert k not in redump


# ------------------------------------------- the region on every grid ----

def _stage_shape(shape, dh0, dh):
    from sweep_tasks._helpers.stage_grid import _stage_grid_shape
    return _stage_grid_shape(shape, dh0, dh)


def _seabed_2d(nx, dh):
    x = np.arange(nx) * dh
    return (180.0 + 60.0 * np.sin(x / 300.0) + 0.37 * x / dh).astype(np.float32)


def _seabed_3d(ny, nx, dh):
    y = np.arange(ny)[:, None] * dh
    x = np.arange(nx)[None, :] * dh
    return (150.0 + 40.0 * np.cos(x / 250.0) + 25.0 * np.sin(y / 170.0)).astype(np.float32)


def _interp_lat(sd, dh0, coords):
    """Seabed at physical lateral positions: separable linear interpolation."""
    if sd.ndim == 1:
        return np.interp(coords[0], np.arange(sd.shape[0]) * dh0, sd)
    ys, xs = coords
    a = np.stack([np.interp(xs, np.arange(sd.shape[1]) * dh0, row) for row in sd])
    return np.stack([np.interp(ys, np.arange(sd.shape[0]) * dh0, col)
                     for col in a.T], axis=1)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dh", [10.0, 20.0, 25.0, 7.0, 33.3333])
@pytest.mark.parametrize("buffer_cells", [0, 2])
def test_seabed_layer_is_the_same_physical_layer_on_every_grid(tmp_path, ndim, dh,
                                                               buffer_cells):
    dh0 = 10.0
    shape = (40, 61) if ndim == 2 else (30, 17, 23)
    sd = _seabed_2d(shape[1], dh0) if ndim == 2 else _seabed_3d(shape[1], shape[2], dh0)
    p = save_seabed_depth_npz(tmp_path / "sb.npz", sd)
    reg = WaterRegion(WaterSpec(seabed=p, buffer_cells=buffer_cells), shape, dh0)
    st = _stage_shape(shape, dh0, dh)
    m = reg.mask_on_grid(st, dh)
    assert m.shape == st and m.dtype == np.bool_
    if dh == dh0:   # on the native grid: the runtime function, bit for bit
        assert np.array_equal(m, water_mask_from_seabed_depth(
            sd, nz=shape[0], dh_z_m=dh0, buffer_cells=buffer_cells))
    # Independently: each stage column samples the seabed at its own position
    # (clamped past the extent), and a cell is water iff its centre is above
    # seabed + buffer, the buffer a depth of native cells.
    lat = [np.minimum(np.arange(n) * dh, (n0 - 1) * dh0)
           for n, n0 in zip(st[1:], shape[1:])]
    depth = _interp_lat(sd, dh0, lat) + buffer_cells * dh0
    zc = (np.arange(st[0]) + 0.5) * dh
    expect = zc.reshape((-1,) + (1,) * (ndim - 1)) < depth[None]
    # float32 arithmetic in the runtime path may flip a cell whose centre sits
    # within a micro-metre of the seabed; there are none in this layout
    assert np.array_equal(m, expect)
    # the frozen depth per column is the seabed + buffer, to within one cell
    n_wet = m.sum(axis=0)
    assert np.all(n_wet * dh >= depth - dh) and np.all(n_wet * dh <= depth + dh)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dh", [10.0, 20.0, 25.0, 7.0, 15.0])
def test_mask_layer_is_the_nearest_native_cell_on_every_grid(tmp_path, ndim, dh):
    dh0 = 10.0
    shape = (24, 37) if ndim == 2 else (20, 13, 17)
    rng = np.random.default_rng(5)
    mask = rng.random(shape) < 0.4
    np.save(tmp_path / "w.npy", mask)
    reg = WaterRegion(WaterSpec(mask=tmp_path / "w.npy"), shape, dh0)
    st = _stage_shape(shape, dh0, dh)
    m = reg.mask_on_grid(st, dh)
    assert m.shape == st
    if dh == dh0:
        assert np.array_equal(m, mask)
    # Independently: the stage cell at physical position (i * dh) per axis is
    # the native cell whose interval [ (k - 1/2) dh0, (k + 1/2) dh0 ) holds it.
    idx = []
    for n, n0 in zip(st, shape):
        pos = np.arange(n) * dh
        k = np.floor(pos / dh0 + 0.5 + 1e-9).astype(int)
        idx.append(np.clip(k, 0, n0 - 1))
    assert np.array_equal(m, mask[np.ix_(*idx)])


def test_nearest_native_ties_go_up_and_coincident_samples_stay():
    assert _nearest_native(5, 5.0, 10.0, 4).tolist() == [0, 1, 1, 2, 2]
    assert _nearest_native(4, 30.0, 10.0, 12).tolist() == [0, 3, 6, 9]
    assert _nearest_native(4, 0.3, 0.1, 12).tolist() == [0, 3, 6, 9]
    # 15 * 0.7 is 10.499999999999998 in floating point: still a tie, still up
    assert _nearest_native(16, 7.0, 10.0, 20)[15] == 11


def test_flat_seabed_number_is_a_seabed_file_of_one_depth(tmp_path):
    p = save_seabed_depth_npz(tmp_path / "flat.npz", np.full(30, 462.5))
    a = WaterRegion(WaterSpec(seabed=462.5), (40, 30), 12.5)
    b = WaterRegion(WaterSpec(seabed=p), (40, 30), 12.5)
    for dh in (12.5, 25.0, 6.25):
        st = _stage_shape((40, 30), 12.5, dh)
        assert np.array_equal(a.mask_on_grid(st, dh), b.mask_on_grid(st, dh))
    # 462.5 m on 12.5 m cells: rows 0..36 are water (centres to 456.25 m)
    assert a.mask_on_grid((40, 30), 12.5)[:, 0].sum() == 37


def test_buffer_is_a_depth_not_a_cell_count():
    reg = WaterRegion(WaterSpec(seabed=100.0, buffer_cells=2), (30, 8), 10.0)
    # native: 100 m + 2 x 10 m = 120 m -> 12 rows
    assert reg.mask_on_grid((30, 8), 10.0)[:, 0].sum() == 12
    # 25 m stage: still 120 m (centres 12.5..112.5 -> 5 rows), not 100 + 50 m
    st = _stage_shape((30, 8), 10.0, 25.0)
    assert reg.mask_on_grid(st, 25.0)[:, 0].sum() == 5


def test_crop_keeps_the_physical_layer(tmp_path):
    shape, dh0 = (30, 40), 10.0
    sd = _seabed_2d(40, dh0)
    p = save_seabed_depth_npz(tmp_path / "sb.npz", sd)
    reg = WaterRegion(WaterSpec(seabed=p), shape, dh0)
    full = reg.mask_on_grid(shape, dh0)
    cut = reg.cropped((3, 5), (25, 31))
    assert np.array_equal(cut.mask_on_grid((22, 26), dh0), full[3:25, 5:31])
    np.save(tmp_path / "m.npy", full)
    regm = WaterRegion(WaterSpec(mask=tmp_path / "m.npy"), shape, dh0)
    assert np.array_equal(regm.cropped((3, 5), (25, 31)).mask_on_grid((22, 26), dh0),
                          full[3:25, 5:31])


def test_region_refuses_bad_inputs(tmp_path):
    np.save(tmp_path / "wrong_shape.npy", np.zeros((5, 5), bool))
    with pytest.raises(ValueError, match="native model grid is"):
        WaterRegion(WaterSpec(mask=tmp_path / "wrong_shape.npy"), (6, 5), 10.0)
    np.save(tmp_path / "values.npy", np.full((6, 5), 2.0))
    with pytest.raises(ValueError, match="booleans or 0/1"):
        WaterRegion(WaterSpec(mask=tmp_path / "values.npy"), (6, 5), 10.0)
    np.save(tmp_path / "plain.npy", np.full(5, 30.0))
    with pytest.raises(ValueError, match="plain array"):
        WaterRegion(WaterSpec(seabed=tmp_path / "plain.npy"), (6, 5), 10.0)
    np.savez(tmp_path / "nokey.npz", depth=np.full(5, 30.0))
    with pytest.raises(ValueError, match="seabed_depth"):
        WaterRegion(WaterSpec(seabed=tmp_path / "nokey.npz"), (6, 5), 10.0)
    p = save_seabed_depth_npz(tmp_path / "two_d.npz", np.full((4, 5), 30.0))
    with pytest.raises(ValueError, match=r"needs \(nx,\)"):
        WaterRegion(WaterSpec(seabed=p), (6, 5), 10.0)
    p = save_seabed_depth_npz(tmp_path / "survey.npz", np.full((40, 10), 30.0))
    with pytest.raises(ValueError, match="aspect ratio"):
        WaterRegion(WaterSpec(seabed=p), (6, 20, 20), 10.0)
    # 0/1 integers are a mask too
    np.save(tmp_path / "ints.npy", np.eye(6, 5, dtype=np.int8))
    assert WaterRegion(WaterSpec(mask=tmp_path / "ints.npy"), (6, 5), 10.0
                       ).mask_on_grid((6, 5), 10.0).sum() == 5


# ------------------------------------------------ the runner, every path ----

def _models(tmp_path, shape=(36, 48), water_rows=6, v_water=1500.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    z = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    true = (2000.0 + 25.0 * z).astype(np.float32)
    true[:water_rows] = v_water
    init = np.full(shape, 2600.0, np.float32)
    init[:water_rows] = v_water
    np.save(tmp_path / "true.npy", true)
    np.save(tmp_path / "init.npy", init)
    return tmp_path / "true.npy", tmp_path / "init.npy"


def _conv_spec(tmp_path, **extra):
    true_p, init_p = _models(tmp_path)
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"), "device": "cpu",
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 220},
        "wavelet": {"kind": "ricker", "fm": 12.0, "delay": 0.08, "scale": 1.0},
        "geometry": {"kind": "line",
                     "sources": {"step": 10, "depth": 2, "start": 4, "stop": 45},
                     "receivers": {"step": 2, "depth": 2, "start": 2, "stop": 46}},
        "physics": {"equation": "Acoustic", "spatial_order": 4, "abcn": 10,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "path": str(init_p)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_p)}},
        "optimizer": {"kind": "adam", "lr": 20.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 2, "show_every": 1,
        "water": {"seabed": 55.0},
    }
    spec.update(extra)
    if "stages" in spec:                         # a staged run runs the stages' epochs
        spec["epochs"] = sum(int(st["epochs"]) for st in spec["stages"])
    return spec


def _spy_train_steps(monkeypatch):
    from sweep_tasks.tasks.fwi import FWIRunnerMixin

    seen = []
    original = FWIRunnerMixin._fwi_train_step

    def _spy(self, spec_, solver, wavelet, sources, receivers, inv_in_order,
             inv_by_name, *args, **kwargs):
        wm = kwargs.get("water_mask")
        seen.append({"vp": inv_by_name["vp"].detach().cpu().numpy().copy(),
                     "mask": None if wm is None else wm.cpu().numpy().copy(),
                     "net": kwargs.get("reparam_net")})
        return original(self, spec_, solver, wavelet, sources, receivers,
                        inv_in_order, inv_by_name, *args, **kwargs)

    monkeypatch.setattr(FWIRunnerMixin, "_fwi_train_step", _spy)
    return seen


def _expected_layer(shape, dh, depth=55.0):
    zc = (np.arange(shape[0]) + 0.5) * dh
    return np.broadcast_to((zc < depth)[:, None], shape)


@pytest.mark.parametrize("source", ["seabed", "mask"])
def test_conventional_pin_holds_the_layer_on_every_stage(tmp_path, monkeypatch, source):
    seen = _spy_train_steps(monkeypatch)
    spec = _conv_spec(tmp_path, stages=[{"epochs": 2, "dh_m": 20.0},
                                        {"epochs": 2, "dh_m": 10.0}])
    native = np.zeros((36, 48), bool)
    native[:5, :30] = True                                     # a stepped seabed
    native[:4, 30:] = True
    np.save(tmp_path / "w.npy", native)
    spec["water"] = ({"seabed": 55.0} if source == "seabed"
                     else {"mask": str(tmp_path / "w.npy")})
    spec["water"]["velocity"] = 1490.0                         # != start (1500)
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "pin.yaml")))
    assert res.status.state == "success", res.status.error
    shapes = [(19, 25), (19, 25), (36, 48), (36, 48)]
    for s, shp in zip(seen, shapes):
        dh = 20.0 if shp == (19, 25) else 10.0
        assert s["mask"].shape == shp
        if source == "seabed":
            expect = _expected_layer(shp, dh)
        else:   # the native cell nearest to each stage cell
            idx = [np.clip(np.floor(np.arange(n) * dh / 10.0 + 0.5).astype(int),
                           0, n0 - 1) for n, n0 in zip(shp, (36, 48))]
            expect = native[np.ix_(*idx)]
        assert np.array_equal(s["mask"], expect)
        assert np.all(s["vp"][s["mask"]] == 1490.0)          # stamped, held
    inv = np.load(res.task_dir / "output" / "inverted_vp.npy")
    assert np.all(inv[seen[-1]["mask"]] == 1490.0)
    assert not np.all(inv[~seen[-1]["mask"]] == 2600.0)


def test_conventional_reset_restarts_the_water_each_stage(tmp_path, monkeypatch):
    seen = _spy_train_steps(monkeypatch)
    spec = _conv_spec(tmp_path, stages=[{"epochs": 2}, {"epochs": 2}])
    spec["water"] = {"seabed": 55.0, "mode": "reset_each_stage"}
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "reset.yaml")))
    assert res.status.state == "success", res.status.error
    wet = _expected_layer((36, 48), 10.0)
    assert all(s["mask"] is None for s in seen)        # nothing frozen in a step
    assert np.all(seen[0]["vp"][wet] == 1500.0)        # stage entries: reset
    assert np.all(seen[2]["vp"][wet] == 1500.0)
    assert not np.all(seen[1]["vp"][wet] == 1500.0)    # free within the stage
    inv = np.load(res.task_dir / "output" / "inverted_vp.npy")
    assert not np.all(inv[wet] == 1500.0)


def test_conventional_reparam_pin_and_reset(tmp_path, monkeypatch):
    rp = {"kind": "velocity_inr", "hidden_features": 8, "hidden_layers": 1,
          "hash": {"enabled": False}, "vp_std": 200.0, "lr": 1.0e-3}
    spec = _conv_spec(tmp_path, reparam=rp, stages=[{"epochs": 2}, {"epochs": 2}])
    spec["water"] = {"seabed": 55.0, "velocity": 1490.0}
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "rpin.yaml")))
    assert res.status.state == "success", res.status.error
    inv = np.load(res.task_dir / "output" / "inverted_vp.npy")
    wet = _expected_layer(inv.shape, 10.0)
    assert np.all(inv[wet] == 1490.0)

    seen = _spy_train_steps(monkeypatch)
    spec = _conv_spec(tmp_path / "r", reparam=rp, stages=[{"epochs": 2}, {"epochs": 2}])
    spec["water"] = {"seabed": 55.0, "mode": "reset_each_stage"}
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "rreset.yaml")))
    assert res.status.state == "success", res.status.error
    net = seen[0]["net"]
    assert net is not None and net.water_mask is None       # no render pin
    for i in (0, 2):                                          # stage entries
        np.testing.assert_allclose(seen[i]["vp"][wet], 1500.0, atol=1e-3)
    inv = np.load(res.task_dir / "output" / "inverted_vp.npy")
    assert np.abs(inv[wet] - 1500.0).max() > 1e-2            # free within stage


def test_conventional_model_plan_crops_the_water(tmp_path, monkeypatch):
    seen = _spy_train_steps(monkeypatch)
    mask = np.zeros((36, 48), bool)
    mask[:5, :20] = True
    mask[:3, 20:] = True
    np.save(tmp_path / "w.npy", mask)
    spec = _conv_spec(tmp_path, model_plan={"x_window_m": [100.0, 400.0],
                                            "drop_outside_sources": True})
    spec["water"] = {"mask": str(tmp_path / "w.npy")}
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "crop.yaml")))
    assert res.status.state == "success", res.status.error
    from sweep_tasks._helpers.plan_apply import _model_plan_crop_box
    box = _model_plan_crop_box(load_task(tmp_path / "crop.yaml"), (36, 48), 10.0)
    sl = tuple(slice(a, b) for a, b in zip(*box))
    assert box[0][1] == 10
    assert np.array_equal(seen[0]["mask"], mask[sl])


def test_lsrtm_holds_the_water_reflectivity_at_zero(tmp_path):
    true_p, init_p = _models(tmp_path)
    spec = {
        "task_type": "lsrtm", "output_dir": str(tmp_path / "tasks"), "device": "cpu",
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 220},
        "wavelet": {"kind": "ricker", "fm": 12.0, "delay": 0.08, "scale": 1.0},
        "geometry": {"kind": "line",
                     "sources": {"step": 10, "depth": 2, "start": 4, "stop": 45},
                     "receivers": {"step": 2, "depth": 2, "start": 2, "stop": 46}},
        "physics": {"equation": "AcousticLSRTM", "spatial_order": 4, "abcn": 10,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["sh1"]},
        "backend": {"impl": "eager", "use_ckpt": False},
        "background_model": {"name": "vp", "path": str(init_p)},
        "true_model": {"name": "vp", "path": str(true_p)},
        "optimizer": {"kind": "adam", "lr": 1.0e-2},
        "epochs": 3, "batchsize": 2, "show_every": 10,
        "water": {"seabed": 55.0},
    }
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "lsrtm.yaml")))
    assert res.status.state == "success", res.status.error
    ref = np.load(res.task_dir / "output" / "reflectivity.npy")
    wet = _expected_layer(ref.shape, 10.0)
    assert np.all(ref[wet] == 0.0)
    assert np.any(ref[~wet] != 0.0)


# ---------------------------------------------------- frequency selection ----

def _freqsel_spec(tmp_path, *, water, reparam=None, stages=None):
    true_p, init_p = _models(tmp_path, shape=(30, 61), water_rows=5)
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"),
        "task_id": "fs", "device": "cpu",
        "grid": {"dh": 10.0}, "time": {"dt": 0.002, "nt": 700},
        "physics": {"equation": "Acoustic", "spatial_order": 4, "abcn": 12,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "path": str(init_p)},
        "source_encoding": {"enabled": True, "mode": "frequency_selection",
                            "frequency": {
                                "synthesize_from_true": True,
                                "true_model_path": str(true_p),
                                "synth_n_nodes": 3, "synth_rec_stride": 3,
                                "synth_node_z": 2, "probe_samples": 200,
                                "k_lo": 2, "k_hi": 4, "steady_samples": 450,
                                "slack_samples": 50, "ramp_s": 0.3,
                                "n_pools": 1}},
        "optimizer": {"kind": "adam", "lr": 20.0},
        "epochs": 2, "batchsize": 1, "show_every": 100,
    }
    if water is not ...:
        spec["water"] = water
    if reparam is not None:
        spec["reparam"] = reparam
    if stages is not None:
        spec["stages"] = stages
        spec["epochs"] = sum(int(st["epochs"]) for st in stages)
    return spec


def test_freqsel_refuses_unsaid_water_on_a_model_it_used_to_pin(tmp_path):
    spec = _freqsel_spec(tmp_path, water=...)
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "unsaid.yaml")))
    assert res.status.state == "failed"
    assert "exactly 1500 m/s" in res.status.error
    assert "water: false" in res.status.error


def test_freqsel_water_false_inverts_the_old_pinned_cells(tmp_path):
    spec = _freqsel_spec(tmp_path, water=False)
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "free.yaml")))
    assert res.status.state == "success", res.status.error
    inv = np.load(Path(spec["output_dir"]) / "fs" / "inverted_vp.npy")
    assert not np.all(inv[:5] == 1500.0)


@pytest.mark.parametrize("mode", ["pin", "reset_each_stage"])
def test_freqsel_grid_modes(tmp_path, mode):
    spec = _freqsel_spec(tmp_path, water={"seabed": 50.0, "mode": mode,
                                          "velocity": 1490.0},
                         stages=[{"epochs": 2, "dh_m": 20.0}, {"epochs": 2}])
    res = TaskRunner().run(load_task(_write(spec, tmp_path / f"{mode}.yaml")))
    assert res.status.state == "success", res.status.error
    task = Path(spec["output_dir"]) / "fs"
    inv = np.load(task / "inverted_vp.npy")
    wet = _expected_layer(inv.shape, 10.0, depth=50.0)
    if mode == "pin":
        assert np.all(inv[wet] == 1490.0)
        coarse = np.load(task / "vp_iter0002.npy")           # end of the 20 m stage
        assert coarse.shape == (16, 31)
        assert np.all(coarse[_expected_layer(coarse.shape, 20.0, depth=50.0)] == 1490.0)
    else:
        assert not np.all(inv[wet] == 1490.0)


@pytest.mark.parametrize("mode", ["pin", "reset_each_stage"])
def test_freqsel_reparam_modes(tmp_path, mode):
    rp = {"kind": "velocity_inr", "hidden_features": 8, "hidden_layers": 1,
          "hash": {"enabled": False}, "vp_std": 200.0, "lr": 1.0e-3}
    spec = _freqsel_spec(tmp_path, water={"seabed": 50.0, "mode": mode},
                         reparam=rp, stages=[{"epochs": 2}, {"epochs": 2}])
    res = TaskRunner().run(load_task(_write(spec, tmp_path / f"r{mode}.yaml")))
    assert res.status.state == "success", res.status.error
    inv = np.load(Path(spec["output_dir"]) / "fs" / "inverted_vp.npy")
    wet = _expected_layer(inv.shape, 10.0, depth=50.0)
    if mode == "pin":
        assert np.all(inv[wet] == 1500.0)
    else:
        assert np.abs(inv[wet] - 1500.0).max() > 1e-2


# ------------------------------------------------------- migrate-water ----

def _migrate(tmp_path, doc_text: str, *, out=True):
    from sweep_tasks.cli import main

    src = tmp_path / "old.yaml"
    src.write_text(doc_text)
    dst = tmp_path / "new.yaml"
    argv = ["migrate-water", str(src)] + (["-o", str(dst)] if out else [])
    rc = main(argv)
    return rc, (dst.read_text() if (out and dst.exists()) else None)


_CONV_YAML = """\
task_type: fwi
output_dir: ./runs
grid: {dh: 12.5}
time: {dt: 0.001, nt: 100}
wavelet: {kind: ricker, fm: 8.0}
geometry:
  kind: line
  sources: {step: 12, depth: 1}
  receivers: {step: 1, depth: 18}
physics: {equation: Acoustic}
init_model:
  name: vp
  shape: [40, 30]
  linear_gradient: {vmin: 1600.0, vmax: 3000.0, water_rows: 8, water_vp: 1500.0}
obs:
  synthetic_from: {name: vp, constant: 2000.0, shape: [40, 30]}
optimizer: {kind: adam, lr: 25.0}
# Hold the water column fixed.
freeze_top_n_rows: 8
epochs: 3
"""


def test_migrate_freeze_top_n_rows_writes_the_mask(tmp_path, capsys):
    rc, new = _migrate(tmp_path, _CONV_YAML)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "freeze_top_n_rows" not in new and "# Hold the water column fixed." in new
    doc = yaml.safe_load(new)
    assert doc["water"] == {"mask": "new_water_mask.npy", "velocity": 1500.0}
    mask = np.load(tmp_path / "new_water_mask.npy")
    assert mask.shape == (40, 30) and mask[:8].all() and not mask[8:].any()
    assert "+water:" in out                                  # the diff
    spec = load_task(tmp_path / "new.yaml")
    assert spec.water.mask == (tmp_path / "new_water_mask.npy").resolve()


def test_migrate_dry_run_writes_nothing(tmp_path, capsys):
    rc, new = _migrate(tmp_path, _CONV_YAML, out=False)
    assert rc == 0 and new is None
    assert not (tmp_path / "old_water_mask.npy").exists()
    assert "dry run" in capsys.readouterr().out


def test_migrate_refuses_rows_that_hold_no_single_velocity(tmp_path, capsys):
    rc, new = _migrate(tmp_path, _CONV_YAML.replace("freeze_top_n_rows: 8",
                                                    "freeze_top_n_rows: 12"))
    out = capsys.readouterr().out
    assert rc == 1 and new is None
    assert "not one velocity" in out and "NOT migrated" in out


def test_migrate_reparam_seabed_pin(tmp_path, capsys):
    save_seabed_depth_npz(tmp_path / "sb.npz", np.full(30, 100.0))
    text = _CONV_YAML.replace("freeze_top_n_rows: 8\n", "").replace(
        "optimizer: {kind: adam, lr: 25.0}\n",
        "optimizer: {kind: adam, lr: 25.0}\nreparam:\n  kind: velocity_inr\n"
        "  mask_water_layer: true   # pin\n  water_vp_m_s: 1485.6\n"
        "  seabed_depth_path: sb.npz\n  water_reset_each_stage: true\n  lr: 1.0e-4\n")
    rc, new = _migrate(tmp_path, text)
    out = capsys.readouterr().out
    assert rc == 0, out
    doc = yaml.safe_load(new)
    assert doc["water"] == {"seabed": "sb.npz", "velocity": 1485.6,
                            "mode": "reset_each_stage"}
    assert doc["reparam"] == {"kind": "velocity_inr", "lr": 1.0e-4}
    assert "ignored reparam.seabed_depth_path" in out


@pytest.mark.parametrize("snippet, why", [
    ("reparam:\n  kind: velocity_inr\n  mask_water_layer: true\n",
     "no longer read off velocities"),
    ("freeze_water_layer:\n  enabled: true\n  seabed_depth_path: sb.npy\n",
     "is a plain array"),
])
def test_migrate_refuses_what_it_cannot_translate(tmp_path, capsys, snippet, why):
    np.save(tmp_path / "sb.npy", np.full(30, 100.0))
    text = _CONV_YAML.replace("freeze_top_n_rows: 8\n", snippet)
    rc, new = _migrate(tmp_path, text)
    out = capsys.readouterr().out
    assert rc == 1 and new is None
    assert why in out


@pytest.mark.parametrize("old_default", [True, False])
def test_migrate_freqsel_implicit_pin_needs_the_user(tmp_path, capsys, old_default):
    text = """\
task_type: fwi
grid: {dh: 12.5}
time: {dt: 0.001, nt: 100}
physics: {equation: Acoustic}
init_model:
  name: vp
  shape: [40, 30]
  linear_gradient: {vmin: 1600.0, vmax: 3000.0, water_rows: 8, water_vp: 1500.0}
source_encoding:
  enabled: true
  mode: frequency_selection
  frequency: {coeff_shards: s.npz, probe_samples: 100, k_lo: 4, k_hi: 8,
              steady_samples: 50}
optimizer: {kind: adam, lr: 25.0}
loss: {kind: steady_gcn}
freeze_top_n_rows: 0
epochs: 3
"""
    if not old_default:
        text = text.replace("freeze_top_n_rows: 0\n", "")
    rc, new = _migrate(tmp_path, text)
    out = capsys.readouterr().out
    assert rc == 1 and new is None
    assert "pinned 240 starting-model cells at exactly 1500 m/s" in out
    assert "water: false" in out


def test_migrate_drops_defaults_of_an_old_resolved_config(tmp_path, capsys):
    rc, new = _migrate(tmp_path, (DATA / "conv_reparam.yaml").read_text())
    out = capsys.readouterr().out
    assert rc == 0, out
    for k in ("freeze_top_n_rows", "freeze_water_layer", "mask_water_layer",
              "water_vp_m_s", "seabed_depth_path", "water_reset_each_stage"):
        assert k not in new
    assert "water" not in yaml.safe_load(new)


# ------------------------------------------- plan streaming, 3-D stages ----

@pytest.mark.parametrize("source", ["seabed", "mask"])
def test_plan_streaming_stage_grids_hold_the_same_layer(tmp_path, monkeypatch, source):
    """3-D, a 100 m stage then the 50 m base: every grid the reparam network
    renders on pins the same physical layer, rebuilt from the definition."""
    from sweep_nn import VelocityINR
    from test_task_fwi_plan_streaming import (
        _build_plan_streaming_spec, _build_tiny_plan_streaming_fixture)

    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    shape, dh0 = (16, 16, 24), 50.0
    yy, xx = np.meshgrid(np.arange(16), np.arange(24), indexing="ij")
    seabed = (110.0 + 4.0 * xx + 3.0 * yy).astype(np.float32)   # 110..293 m
    if source == "seabed":
        water = {"seabed": str(save_seabed_depth_npz(tmp_path / "sb.npz", seabed)),
                 "velocity": 1480.0}
    else:
        zc = (np.arange(16) + 0.5)[:, None, None] * dh0
        native = zc < seabed[None]
        np.save(tmp_path / "w.npy", native)
        water = {"mask": str(tmp_path / "w.npy"), "velocity": 1480.0}
    spec = _build_plan_streaming_spec(tmp_path, fixture, epochs=2, extra={
        "stages": [{"epochs": 1, "dh_m": 100.0}, {"epochs": 1, "dh_m": 50.0}],
        "water": water,
        "reparam": {"kind": "velocity_inr", "hidden_features": 8, "hidden_layers": 1,
                    "hash": {"enabled": False}, "lr": 1.0e-3},
    })
    spec["backend"]["eager_options"] = {"use_compile": False}
    masks = []
    orig = VelocityINR.update_base_velocity

    def _spy(self, new_base, water_mask=None):
        masks.append(None if water_mask is None else water_mask.cpu().numpy().copy())
        return orig(self, new_base, water_mask=water_mask)

    monkeypatch.setattr(VelocityINR, "update_base_velocity", _spy)
    res = TaskRunner().run(load_task(_write(spec, tmp_path / "ps.yaml")))
    assert res.status.state == "success", res.status.error
    assert [m.shape for m in masks] == [(9, 9, 13), (16, 16, 24)]
    for m, dh in zip(masks, (100.0, 50.0)):
        st = m.shape
        if source == "seabed":
            lat = [np.minimum(np.arange(n) * dh, (n0 - 1) * dh0)
                   for n, n0 in zip(st[1:], shape[1:])]
            depth = _interp_lat(seabed, dh0, lat)
            expect = ((np.arange(st[0]) + 0.5) * dh)[:, None, None] < depth[None]
        else:
            idx = [np.clip(np.floor(np.arange(n) * dh / dh0 + 0.5 + 1e-9).astype(int),
                           0, n0 - 1) for n, n0 in zip(st, shape)]
            expect = native[np.ix_(*idx)]
        assert np.array_equal(m, expect)
    inv = np.load(res.task_dir / "output" / "inverted_vp.npy")
    assert np.all(inv[masks[-1]] == 1480.0)
    assert not np.all(inv[~masks[-1]] == 2500.0)


# ------------------------------------------------------------ examples ----

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rel, old_rule", [
    ("examples/synthetic/03_fwi_marmousi_multiscale.yaml", "rows:37"),
    ("examples/synthetic/05_lsrtm_marmousi.yaml", "rows:10"),
    ("examples/synthetic/12_fwi_marmousi_freqsel.yaml", "value:1500.0"),
    ("examples/field/viking/viking_freqsel_8band.yaml", "value:1485.6"),
])
def test_examples_hold_exactly_what_they_held_before(rel, old_rule):
    """The example YAMLs moved to `water:` without changing what they hold:
    freeze_top_n_rows N -> the same N rows; the frequency-selection value pin
    (and reparam.mask_water_layer) -> the same cells, on the native grid that
    every one of these examples runs on."""
    from sweep_tasks._helpers.model import _model_array

    spec = load_task(_REPO / rel)
    assert spec.water is not None and not any(
        getattr(st, "dh_m", None) for st in (spec.stages or []))
    ref = getattr(spec, "init_model", None)
    shape = tuple(ref.shape) if ref is not None else (281, 1361)
    held = WaterRegion(spec.water, shape, spec.grid.dh).mask_on_grid(
        shape, spec.grid.dh)
    kind, val = old_rule.split(":")
    if kind == "rows":
        expect = np.zeros(shape, bool)
        expect[: int(val)] = True
    else:
        expect = _model_array(ref) == np.float32(float(val))
    assert np.array_equal(held, expect)
