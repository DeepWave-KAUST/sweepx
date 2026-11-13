"""``grad_smooth:`` on the conventional FWI path.

Each inverted grid model is smoothed with its own block just before an Adam /
SGD step, on the epochs that block's ``every`` divides, counted over the whole
run: a one-model run writes one block, a run inverting several gives each
model its own under ``per_model`` (a model left out is not smoothed). The loop
used to smooth vp alone with the one block, on every epoch whatever ``every``
said, and to accept the block under reparam and L-BFGS, which never smooth.
What a run would not apply is refused: when the YAML loads, reparam / L-BFGS,
one block for several models, a ``per_model`` entry for a model the run does
not invert, settings beside ``per_model`` and a taper without a mask row; when
the run starts, a 2-D model's ``sigma_y_cells`` and a mask row a stage grid
does not have.

The runs below read every model's gradient off each ``optimizer.step()``
through a global step pre-hook. A block's ``mask_below_row`` zeroes the
gradient below that row as the smoother's last step, so a model whose gradient
is zero below its row was smoothed with its block at that step, and one with
gradient there was not.
"""
from pathlib import Path

import numpy as np
import pytest
import torch.optim.optimizer as _toptim
import yaml

from sweepx.tasks import TaskRunner, load_task
from sweepx.tasks.schemas import GradSmoothSpec
from test_freqsel_optimizer_settings import _REPARAM, _conventional, _doc, _validate

# ------------------------------------------------------------ at load ----


def _smoothed(**extra) -> dict:
    """A conventional (per-shot, Adam) one-model spec with grad_smooth on."""
    return _conventional(_doc(grad_smooth={"sigma_x_cells": 2.0}, **extra))


def _vp_rho(grad_smooth, **extra) -> dict:
    """A conventional spec that inverts vp and rho (at load the equation is
    not resolved, so the init models say what the run inverts)."""
    doc = _conventional(_doc(grad_smooth=grad_smooth, **extra))
    shape = doc.pop("init_model")["shape"]
    doc["init_models"] = [{"name": n, "constant": c, "shape": shape}
                          for n, c in (("vp", 1500.0), ("rho", 2000.0))]
    return doc


@pytest.mark.parametrize("extra, why", [
    ({"reparam": _REPARAM}, r"with reparam \(the network renders vp"),
    ({"optimizer": {"kind": "lbfgs"}},
     r"with optimizer.kind='lbfgs' \(L-BFGS steps from its own closure"),
], ids=["reparam", "lbfgs"])
def test_refused_where_the_conventional_path_never_smooths(extra, why):
    with pytest.raises(Exception, match=r"the conventional path would ignore "
                                        r"grad_smooth " + why):
        _validate(_smoothed(**extra))
    # absent or switched off it asks for nothing
    for off in (None, {"enabled": False, "sigma_x_cells": 2.0}):
        _validate(_conventional(_doc(grad_smooth=off, **extra)))


def test_both_reasons_are_given():
    with pytest.raises(Exception, match=r"with reparam .* and with "
                                        r"optimizer.kind='lbfgs'"):
        _validate(_smoothed(reparam=_REPARAM, optimizer={"kind": "lbfgs"}))


@pytest.mark.parametrize("opt", [{"kind": "adam", "lr": 25.0},
                                 {"kind": "sgd", "lr": 1.0}],
                         ids=lambda o: o["kind"])
def test_adam_and_sgd_keep_it(opt):
    assert _validate(_smoothed(optimizer=opt)).grad_smooth.sigma_x_cells == 2.0


def test_several_models_take_a_block_each():
    """One block for vp and rho used to smooth vp alone."""
    with pytest.raises(Exception, match=(
            r"this run inverts vp, rho, so grad_smooth takes one block per "
            r"model: grad_smooth: \{per_model: \{vp: \{\.\.\.\}, rho: \{\.\.\.\}\}\}")):
        _validate(_vp_rho({"sigma_x_cells": 2.0}))
    with pytest.raises(Exception, match=r"grad_smooth.per_model names vs, which this "
                                        r"run does not invert \(it inverts vp, rho\)"):
        _validate(_vp_rho({"per_model": {"rho": {}, "vs": {}}}))
    spec = _validate(_vp_rho({"per_model": {"rho": {"sigma_x_cells": 1.0, "every": 3}}}))
    assert list(spec.grad_smooth.per_model) == ["rho"]
    assert spec.grad_smooth.per_model["rho"].every == 3
    _validate(_vp_rho({"enabled": False, "sigma_x_cells": 2.0}))   # off asks for nothing
    _validate(_conventional(_doc(grad_smooth={"per_model": {"vp": {}}})))  # one model


def test_per_model_stands_alone():
    for bad, msg in (
            ({"per_model": {}}, r"grad_smooth.per_model is empty"),
            ({"sigma_x_cells": 2.0, "every": 2, "per_model": {"vp": {}}},
             r"grad_smooth sets sigma_x_cells=2.0, every=2 beside per_model"),
            ({"per_model": {"vp": {"per_model": {"vp": {}}}}},
             r"grad_smooth.per_model.vp has a per_model of its own")):
        with pytest.raises(Exception, match=msg):
            GradSmoothSpec(**bad)
    # beside it, a config_resolved.yaml spells every other setting out at its
    # default
    GradSmoothSpec(**{n: f.default for n, f in GradSmoothSpec.model_fields.items()
                      if n != "per_model"}, per_model={"vp": {"sigma_x_cells": 2.0}})


def test_frequency_selection_does_not_read_per_model():
    with pytest.raises(Exception, match=r"grad_smooth.per_model is read only by the "
                                        r"conventional path"):
        _validate(_doc(grad_smooth={"per_model": {"vp": {}}}))


def test_a_record_of_a_run_loads_again(tmp_path):
    """config_resolved.yaml spells every setting out (per_model entries
    included) at its default, and a reparam / L-BFGS run's says null."""
    from sweepx.tasks._helpers.metadata import _dump_run_metadata

    docs = [_smoothed(), _vp_rho({"per_model": {"rho": {"sigma_x_cells": 1.0}}}),
            _conventional(_doc(reparam=_REPARAM)),
            _conventional(_doc(optimizer={"kind": "lbfgs"}))]
    for i, doc in enumerate(docs):
        spec = _validate(doc)
        (tmp_path / str(i)).mkdir()
        _dump_run_metadata(spec, tmp_path / str(i))
        cfg = yaml.safe_load((tmp_path / str(i) / "config_resolved.yaml").read_text())
        assert _validate(cfg).grad_smooth == spec.grad_smooth
    assert spec.grad_smooth is None


def test_a_taper_needs_a_mask_row():
    for rows in ({}, {"mask_above_row": 0}):          # rows < 0: no edge
        with pytest.raises(Exception, match=r"mask_taper_rows=3 tapers the edge "
                                            r"of the depth mask, but neither"):
            GradSmoothSpec(mask_taper_rows=3, **rows)
    GradSmoothSpec(mask_taper_rows=3, mask_above_row=5)
    GradSmoothSpec(mask_taper_rows=3, mask_below_row=0)
    GradSmoothSpec(mask_taper_rows=0)                 # what a record spells out
    # -1 is "no mask on that side"; lower rows and a negative taper used to
    # pass for the same
    for bad in ({"mask_above_row": -2}, {"mask_below_row": -2},
                {"mask_taper_rows": -1}):
        with pytest.raises(Exception, match="greater than or equal to"):
            GradSmoothSpec(**bad)


def test_which_block_smooths_which_model():
    from sweepx.tasks.tasks.fwi import _grad_smooth_blocks

    one = GradSmoothSpec(sigma_x_cells=2.0)
    assert _grad_smooth_blocks(one, ["vp"]) == {"vp": one}
    two = GradSmoothSpec(per_model={"vp": {"enabled": False},
                                    "rho": {"sigma_x_cells": 1.0}})
    assert _grad_smooth_blocks(two, ["vp", "rho"]) == {"rho": two.per_model["rho"]}
    off = GradSmoothSpec(enabled=False, per_model={"rho": {}})
    assert _grad_smooth_blocks(off, ["vp", "rho"]) == {}
    assert _grad_smooth_blocks(None, ["vp"]) == {}


def test_mask_rows_are_rows_of_each_stage_grid():
    from sweepx.tasks.tasks.fwi import _check_grad_smooth

    def check(gs, shape=(24, 32), **extra):
        if gs.get("per_model") and set(gs["per_model"]) - {"vp"}:
            spec, names = _validate(_vp_rho(gs, **extra)), ["vp", "rho"]
        else:
            spec, names = _validate(_conventional(_doc(grad_smooth=gs, **extra))), ["vp"]
        _check_grad_smooth(spec, shape, names)

    check({"mask_below_row": 22})                         # row 23 is below it
    check({"mask_above_row": 23})                         # row 23 is kept
    with pytest.raises(ValueError, match=r"grad_smooth: the model is a 12.5 m grid of "
                                         r"24 rows, so mask_below_row=23 has no row "
                                         r"below it"):
        check({"mask_below_row": 23})
    with pytest.raises(ValueError, match=r"of 24 rows, so mask_above_row=24 zeroes "
                                         r"every row"):
        check({"mask_above_row": 24})
    # 24 rows of 12.5 m span 287.5 m: 13 rows of 25 m
    staged = dict(epochs=2, stages=[{"epochs": 1}, {"epochs": 1, "dh_m": 25.0}])
    check({"mask_below_row": 11}, **staged)
    check({"mask_above_row": 12}, **staged)
    with pytest.raises(ValueError, match=r"stage 1 runs on a 25 m grid of 13 rows, "
                                         r"so mask_below_row=12 has no row below"):
        check({"mask_below_row": 12}, **staged)
    with pytest.raises(ValueError, match=r"stage 1 runs on a 25 m grid of 13 rows, "
                                         r"so mask_above_row=13 zeroes every row"):
        check({"mask_above_row": 13}, **staged)
    check({"sigma_y_cells": 2.0}, shape=(24, 8, 32))      # 3-D has a y axis
    # each per_model block is checked, under its own name
    with pytest.raises(ValueError, match=r"grad_smooth.per_model.rho.sigma_y_cells=2.0 "
                                         r"is for 3-D models"):
        check({"per_model": {"vp": {}, "rho": {"sigma_y_cells": 2.0}}})
    with pytest.raises(ValueError, match=r"grad_smooth.per_model.rho: the model is a "
                                         r"12.5 m grid of 24 rows, so mask_below_row=23"):
        check({"per_model": {"rho": {"mask_below_row": 23}}})
    check({"per_model": {"rho": {"enabled": False, "sigma_y_cells": 2.0}}})


# ----------------------------------------------------------- the runs ----

NZ, NX, DH = 24, 32, 10.0
ROW = 4                     # the blocks' mask_below_row unless a test says
_GS = {"sigma_z_cells": 0.0, "sigma_x_cells": 1.0, "mask_below_row": ROW}


def _run_spec(tmp_path: Path, task_id: str, *, vp_rho: bool = False, **extra) -> dict:
    """A small 2-D run on the CPU; ``vp_rho`` inverts vp and rho (Acoustic1st),
    each smoothed with the block ``_GS`` unless ``extra`` says otherwise."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    z = np.arange(NZ, dtype=np.float32)[:, None] * DH * np.ones((1, NX), np.float32)
    np.save(tmp_path / "vp.npy", 2000.0 + 1.5 * z)
    np.save(tmp_path / "rho.npy", 2000.0 + 0.5 * z)
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"),
        "task_id": task_id, "device": "cpu", "seed": 0,
        "grid": {"dh": DH}, "time": {"dt": 0.001, "nt": 150},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.06},
        "geometry": {"kind": "line",
                     "sources": {"step": 10, "depth": 2, "start": 6, "stop": 27},
                     "receivers": {"step": 2, "depth": 2, "start": 2, "stop": 30}},
        "physics": {"equation": "Acoustic", "spatial_order": 4, "abcn": 10},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "constant": 2100.0, "shape": [NZ, NX]},
        "obs": {"synthetic_from": {"name": "vp", "path": str(tmp_path / "vp.npy")}},
        "optimizer": {"kind": "adam", "lr": 1.0},
        "grad_smooth": dict(_GS),
        "epochs": 3, "batchsize": 3, "show_every": 100,
    }
    if vp_rho:
        spec["physics"].update(equation="Acoustic1st", source_type=["p"],
                               receiver_type=["p"])
        del spec["init_model"]
        spec["init_models"] = [
            {"name": "vp", "constant": 2100.0, "shape": [NZ, NX]},
            {"name": "rho", "constant": 2050.0, "shape": [NZ, NX]}]
        spec["obs"] = {"synthetic_from_models": [
            {"name": n, "path": str(tmp_path / f"{n}.npy")} for n in ("vp", "rho")]}
        spec["grad_smooth"] = {"per_model": {"vp": dict(_GS), "rho": dict(_GS)}}
    spec.update(extra)
    return spec


def _run(spec: dict, path: Path):
    """Run ``spec``; return the result and, per optimizer step, the gradient
    of every model it stepped (in the equation's model order)."""
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    steps = []
    hook = _toptim.register_optimizer_step_pre_hook(
        lambda opt, args, kwargs: steps.append(
            [p.grad.detach().clone() for g in opt.param_groups for p in g["params"]]))
    try:
        result = TaskRunner().run(load_task(path))
    finally:
        hook.remove()
    return result, steps


def _below(g, row: int = ROW) -> float:
    return float(g[row + 1:].abs().max())


@pytest.mark.parametrize("every, smoothed", [
    (1, [True] * 5),
    (2, [True, False, True, False, True]),   # the second stage starts at epoch 3
])
def test_every_counts_the_epochs_of_the_whole_run(tmp_path, every, smoothed):
    """It used to smooth every epoch whatever ``every`` said."""
    spec = _run_spec(tmp_path, f"every{every}", epochs=5,
                     stages=[{"epochs": 3}, {"epochs": 2}],
                     grad_smooth=dict(_GS, every=every))
    result, steps = _run(spec, tmp_path / f"every{every}.yaml")
    assert result.status.state == "success", result.status.error
    assert [_below(vp) == 0.0 for (vp,) in steps] == smoothed


def test_each_model_is_smoothed_with_its_own_block(tmp_path):
    """vp every second epoch with its mask at row 4, rho every epoch with its
    mask at row 8."""
    gs = {"per_model": {"vp": dict(_GS, every=2), "rho": dict(_GS, mask_below_row=8)}}
    result, steps = _run(_run_spec(tmp_path, "own", vp_rho=True, epochs=2,
                                   grad_smooth=gs), tmp_path / "own.yaml")
    assert result.status.state == "success", result.status.error
    (vp0, rho0), (vp1, rho1) = steps
    assert _below(vp0) == 0.0 and _below(vp1) > 0.0              # vp: epoch 0 only
    assert _below(rho0, 8) == 0.0 and _below(rho1, 8) == 0.0     # rho: both epochs,
    assert float(rho0[ROW + 1:9].abs().max()) > 0.0               # below its own row


def test_a_model_left_out_is_not_smoothed(tmp_path):
    result, steps = _run(_run_spec(tmp_path, "rho_only", vp_rho=True, epochs=1,
                                   grad_smooth={"per_model": {"rho": dict(_GS)}}),
                         tmp_path / "rho_only.yaml")
    assert result.status.state == "success", result.status.error
    ((vp, rho),) = steps
    assert _below(vp) > 0.0 and _below(rho) == 0.0


def test_smoothing_keeps_the_water_frozen(tmp_path):
    """A z-smoothing spreads gradient into the pinned water; it is zeroed
    again for every model before the step."""
    block = {"sigma_z_cells": 1.5, "sigma_x_cells": 1.0}
    result, steps = _run(_run_spec(tmp_path, "water", vp_rho=True, epochs=1,
                                   # rows 0-1: cell centres 5 and 15 m
                                   water={"seabed": 25.0},
                                   grad_smooth={"per_model": {"vp": block,
                                                              "rho": block}}),
                         tmp_path / "water.yaml")
    assert result.status.state == "success", result.status.error
    (grads,) = steps
    for g in grads:
        assert float(g[:2].abs().max()) == 0.0
        assert float(g[2].abs().max()) > 0.0


def _refused(tmp_path: Path, task_id: str, **extra) -> str:
    result, steps = _run(_run_spec(tmp_path, task_id, **extra),
                         tmp_path / f"{task_id}.yaml")
    assert result.status.state == "failed"
    assert steps == []                                          # before training
    return result.status.error or ""


def test_sigma_y_on_a_2d_model_is_refused_at_the_start(tmp_path):
    error = _refused(tmp_path, "sy", grad_smooth=dict(_GS, sigma_y_cells=2.0))
    assert ("grad_smooth.sigma_y_cells=2.0 is for 3-D models; this model is 2-D "
            "(24, 32) (z, x), so it would be ignored") in error


def test_a_mask_row_a_stage_grid_lacks_is_refused_at_the_start(tmp_path):
    """Row 12 is a row of the 10 m model; stage 0's 20 m grid has 13 rows, so
    the mask would zero nothing there."""
    error = _refused(tmp_path, "rows", epochs=2,
                     stages=[{"epochs": 1, "dh_m": 20.0}, {"epochs": 1}],
                     grad_smooth=dict(_GS, mask_below_row=12))
    assert ("grad_smooth: stage 0 runs on a 20 m grid of 13 rows, so "
            "mask_below_row=12 has no row below it to zero") in error
