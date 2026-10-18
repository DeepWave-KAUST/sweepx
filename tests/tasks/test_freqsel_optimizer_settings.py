"""The frequency-selection path's optimizer: what it runs, and what it refuses.

It builds its own Adam at torch's default eps / betas, runs every stage at a
constant lr (``reparam.lr x inr_lr_scale`` under reparam, ``optimizer.lr x
lr_scale`` otherwise) and never reads ``scheduler``, ``optimizer.kind`` /
``eps`` / ``betas`` or, under reparam, ``lr_scale``. Those are rejected, by
value against their schema default, so a ``config_resolved.yaml`` still loads.
Across stages the Adam and its state carry over, as on the conventional path:
a grid model's is rebuilt only by ``optimizer_reset`` or a grid change.

The lr and Adam's state are read off every ``optimizer.step()`` through a
global step pre-hook, so they are what the stepping optimizer actually used.
"""
from pathlib import Path

import numpy as np
import pytest
import torch.optim.optimizer as _toptim
import yaml
from pydantic import TypeAdapter

from sweepx.tasks import TaskRunner, load_task
from sweepx.tasks.schemas import TaskSpec

_REPARAM = {"kind": "velocity_inr", "hidden_features": 16, "hidden_layers": 1,
            "hash": {"enabled": False}, "lr": 1.0e-4}


def _doc(**extra):
    doc = dict(
        task_type="fwi", output_dir="./x", task_id="t",
        grid={"dh": 12.5}, time={"dt": 0.001, "nt": 100},
        physics={"equation": "Acoustic"},
        init_model={"name": "vp", "constant": 1500.0, "shape": [20, 30]},
        source_encoding={
            "enabled": True, "mode": "frequency_selection",
            "frequency": {"coeff_shards": "s.npz", "probe_samples": 100,
                          "k_lo": 4, "k_hi": 8, "steady_samples": 50},
        },
        optimizer={"kind": "adam", "lr": 25.0}, epochs=2,
    )
    doc.update(extra)
    return doc


def _validate(doc):
    return TypeAdapter(TaskSpec).validate_python(doc)


def _conventional(doc):
    """The same doc without source encoding: the per-shot path."""
    doc = dict(doc, wavelet={"kind": "ricker", "fm": 10.0},
               geometry={"kind": "line", "sources": {"step": 6, "depth": 2},
                         "receivers": {"step": 2, "depth": 2}},
               obs={"synthetic_from": {"name": "vp", "constant": 2000.0,
                                       "shape": [20, 30]}})
    del doc["source_encoding"]
    return doc


@pytest.mark.parametrize("sched", [
    {"kind": "step", "step_size": 1},
    {"kind": "exp", "gamma": 0.9},
    {"kind": "cosine"},
], ids=lambda s: s["kind"])
def test_freqsel_rejects_a_scheduler(sched):
    """It has none; written ``constant`` (what a resolved config says) loads."""
    assert _validate(_doc(scheduler={"kind": "constant"})).scheduler.kind == "constant"
    with pytest.raises(Exception, match=f"scheduler.kind='{sched['kind']}' "
                                        "would be ignored"):
        _validate(_doc(scheduler=sched))
    _validate(_conventional(_doc(scheduler=sched)))     # the conventional path has one


@pytest.mark.parametrize("opt", [
    {"kind": "sgd", "lr": 1.0},
    {"kind": "lbfgs"},
], ids=lambda o: o["kind"])
def test_freqsel_rejects_another_optimizer_kind(opt):
    with pytest.raises(Exception, match=f"optimizer.kind='{opt['kind']}' would "
                                        "be ignored: it always runs Adam"):
        _validate(_doc(optimizer=opt))
    _validate(_conventional(_doc(optimizer=opt)))


@pytest.mark.parametrize("opts, shown", [
    ({"eps": 1.0e-3}, "optimizer.eps=0.001"),
    # even the eps that runs: the path reads no optimizer option
    ({"eps": 1.0e-8}, "optimizer.eps=1e-08"),
    ({"betas": [0.5, 0.6]}, r"optimizer.betas=\(0.5, 0.6\)"),
], ids=["eps", "eps_torch_default", "betas"])
def test_freqsel_judges_adam_options_by_value(opts, shown):
    """The schema default is what a config_resolved.yaml writes out, and has
    to load; any other value would be dropped."""
    spec = _validate(_doc(optimizer={"kind": "adam", "lr": 25.0,
                                     "eps": 1.0e-22, "betas": [0.9, 0.999]}))
    assert spec.optimizer.eps == 1.0e-22
    with pytest.raises(Exception, match=f"{shown} would be ignored: it runs "
                                        r"Adam at torch's default eps=1e-8"):
        _validate(_doc(optimizer={"kind": "adam", "lr": 25.0, **opts}))
    _validate(_conventional(_doc(optimizer={"kind": "adam", "lr": 25.0, **opts})))


def test_freqsel_rejects_lr_scale_under_reparam():
    """Under reparam each stage runs at reparam.lr x inr_lr_scale only."""
    stages = [{"epochs": 1, "lr_scale": 1.0, "inr_lr_scale": 0.5},
              {"epochs": 1, "lr_scale": 0.5}]
    with pytest.raises(Exception, match=r"stages\[1\].lr_scale=0.5 would be "
                                        "ignored: under reparam"):
        _validate(_doc(reparam=_REPARAM, stages=stages))
    # at its default it loads, and without reparam it is the grid model's scale
    _validate(_doc(reparam=_REPARAM, stages=stages[:1], epochs=1))
    _validate(_doc(stages=[{"epochs": 1}, stages[1]]))
    # the conventional path multiplies lr_scale and inr_lr_scale
    _validate(_conventional(_doc(reparam=_REPARAM, stages=stages)))


def test_freqsel_resolved_config_reruns(tmp_path):
    """Every optimizer field written out at its default, every stage scale:
    the record a reparam ladder leaves behind has to load as itself."""
    from sweepx.tasks._helpers.metadata import _dump_run_metadata

    stages = [{"epochs": 1}, {"epochs": 1, "inr_lr_scale": 0.5}]
    spec = _validate(_doc(reparam=_REPARAM, stages=stages))
    _dump_run_metadata(spec, tmp_path)
    cfg = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())
    assert cfg["optimizer"]["eps"] == 1.0e-22 and cfg["scheduler"] == {"kind": "constant"}
    assert [s["lr_scale"] for s in cfg["stages"]] == [1.0, 1.0]
    again = _validate(cfg)
    assert [s.inr_lr_scale for s in again.stages] == [1.0, 0.5]
    # and an old record with an ignored setting in it is refused, not re-run
    cfg["stages"][1]["lr_scale"] = 0.85
    with pytest.raises(Exception, match=r"stages\[1\].lr_scale=0.85"):
        _validate(cfg)


# ------------------------------------------------------ runs, lr per step ----

NZ, NX, DH = 24, 64, 10.0


def _run_spec(tmp_path: Path, task_id: str, **extra) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    z = np.arange(NZ, dtype=np.float64)[:, None] * DH
    vp = tmp_path / "vp.npy"
    np.save(vp, ((1500.0 + 2.0 * z) * np.ones((1, NX))).astype(np.float32))
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"),
        "task_id": task_id, "device": "cpu", "seed": 0,
        "grid": {"dh": DH}, "time": {"dt": 0.002, "nt": 320},
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 16,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "constant": 1600.0, "shape": [NZ, NX]},
        "source_encoding": {"enabled": True, "mode": "frequency_selection",
                            "frequency": {
                                "synthesize_from_true": True,
                                "true_model_path": str(vp),
                                "synth_n_nodes": 3, "synth_rec_stride": 2,
                                "synth_node_z": 2,
                                "probe_samples": 100, "k_lo": 2, "k_hi": 4,
                                "steady_samples": 200, "slack_samples": 20,
                                "ramp_s": 0.1, "n_pools": 1}},
        "optimizer": {"kind": "adam", "lr": 1.0},
        "epochs": 3, "batchsize": 1, "show_every": 100,
    }
    spec.update(extra)
    return spec


def _run(spec: dict, path: Path):
    """Run ``spec``; return the task dir and (lr, eps, betas) of every step."""
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    steps = []
    hook = _toptim.register_optimizer_step_pre_hook(
        lambda opt, args, kwargs: steps.append(
            (type(opt).__name__,) + tuple(
                (g["lr"], g["eps"], tuple(g["betas"])) for g in opt.param_groups)))
    try:
        result = TaskRunner().run(load_task(path))
    finally:
        hook.remove()
    assert result.status.state == "success", result.status.error
    return Path(spec["output_dir"]) / spec["task_id"], steps


_ADAM = (1.0e-8, (0.9, 0.999))      # what the path runs: torch's defaults


def test_each_stage_runs_at_its_own_scale(tmp_path):
    """1.0 / 0.5 / 1.0: the third stage is back at the base lr. Both
    optimizers are carried across these stages, the reparam one always and
    the grid model's on one grid, and a 1.0 scale used to leave the 0.5 of
    the stage before in place under reparam."""
    for base, key, extra in ((1.0e-4, "inr_lr_scale", {"reparam": _REPARAM}),
                             (1.0, "lr_scale", {})):
        stages = [{"epochs": 1, key: 1.0}, {"epochs": 1, key: 0.5},
                  {"epochs": 1, key: 1.0}]
        _, steps = _run(_run_spec(tmp_path / key, "ladder", stages=stages, **extra),
                        tmp_path / f"{key}.yaml")
        assert steps == [("Adam", (base * s, *_ADAM)) for s in (1.0, 0.5, 1.0)], key


def test_a_restored_state_does_not_bring_its_lr(tmp_path):
    """A warm start restores Adam's moments; its lr is this run's base x the
    stage's scale, not the lr the state was saved at."""
    first = _run_spec(tmp_path, "first", epochs=1,
                      reparam=dict(_REPARAM, save_net=True),
                      stages=[{"epochs": 1, "inr_lr_scale": 0.5}])
    task, steps = _run(first, tmp_path / "first.yaml")
    assert steps == [("Adam", (0.5e-4, *_ADAM))]
    second = _run_spec(tmp_path, "second", epochs=2,
                       reparam=dict(_REPARAM, init_from=str(task / "reparam_net.pt"),
                                    init_optimizer_from="auto"))
    _, steps = _run(second, tmp_path / "second.yaml")
    assert steps == [("Adam", (1.0e-4, *_ADAM))] * 2


# --------------------------------------- Adam's state across stage entries ----

def _band(spec: dict, k_lo: int, k_hi: int) -> dict:
    """A stage's own frequency block: the top-level one at another band."""
    return dict(spec["source_encoding"]["frequency"], k_lo=k_lo, k_hi=k_hi)


def _states(spec: dict, path: Path):
    """Run ``spec``; for every step, read BEFORE it: which optimizer and which
    leaf it steps (by identity), Adam's step count and max|exp_avg| (0 and 0.0
    for a fresh Adam, which holds no state yet), and the leaf's values."""
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    opts, leaves, steps, values = [], [], [], []

    def _index(seen, obj):      # holding them keeps their identities unique
        i = next((i for i, o in enumerate(seen) if o is obj), None)
        if i is None:
            seen.append(obj)
            i = len(seen) - 1
        return i

    def _hook(opt, args, kwargs):
        (leaf,) = [p for g in opt.param_groups for p in g["params"]]
        st = opt.state.get(leaf, {})
        steps.append((_index(opts, opt), _index(leaves, leaf),
                      int(st["step"]) if "step" in st else 0,
                      float(st["exp_avg"].abs().max()) if "exp_avg" in st else 0.0))
        values.append(leaf.detach().cpu().numpy().copy())

    hook = _toptim.register_optimizer_step_pre_hook(_hook)
    try:
        result = TaskRunner().run(load_task(path))
    finally:
        hook.remove()
    assert result.status.state == "success", result.status.error
    return steps, values


@pytest.mark.parametrize("second, carried", [
    ({}, True),                             # same grid, optimizer_reset: false
    ({"optimizer_reset": True}, False),     # same grid, a reset asked for
    ({"dh_m": 20.0}, False),                # a grid change: the state is shape-bound
], ids=["same_grid", "optimizer_reset", "dh_change"])
def test_grid_model_keeps_its_adam_on_one_grid(tmp_path, capsys, second, carried):
    """Two bands, two steps each. On one grid the second stage steps the same
    leaf with the same Adam, its step count running on (2, 3) and its moments
    in place; it was rebuilt on every entry, so ``optimizer_reset: false``
    did nothing. A reset or a grid change starts a fresh Adam (0, 1)."""
    spec = _run_spec(tmp_path, "t", epochs=4)
    spec["stages"] = [{"epochs": 2},
                      {"epochs": 2, "frequency": _band(spec, 3, 5), **second}]
    steps, values = _states(spec, tmp_path / "t.yaml")
    opt, leaf, count, exp_avg = zip(*steps)
    if carried:
        assert opt == leaf == (0, 0, 0, 0) and count == (0, 1, 2, 3), steps
        assert exp_avg[2] > 0.0, steps
        assert "Adam moments dropped" not in capsys.readouterr().out
    else:
        assert opt == leaf == (0, 0, 1, 1) and count == (0, 1, 0, 1), steps
        assert exp_avg[2] == 0.0, steps
    if "dh_m" not in second:
        # either way the second stage starts from the first one's result
        task = Path(spec["output_dir"]) / spec["task_id"]
        np.testing.assert_array_equal(values[2], np.load(task / "vp_iter0002.npy"))


def test_a_dropped_state_says_why(tmp_path, capsys):
    """One line per rebuild, naming its cause, as the conventional path does."""
    spec = _run_spec(tmp_path, "t", epochs=3)
    spec["stages"] = [{"epochs": 1},
                      {"epochs": 1, "frequency": _band(spec, 3, 5),
                       "optimizer_reset": True},
                      {"epochs": 1, "frequency": _band(spec, 4, 6), "dh_m": 20.0}]
    _states(spec, tmp_path / "t.yaml")
    dropped = [ln for ln in capsys.readouterr().out.splitlines()
               if "Adam moments dropped" in ln]
    assert dropped == [
        "[freqsel] stage 1: optimizer_reset=True — Adam moments dropped",
        "[freqsel] stage 2: dh 10 -> 20 m — Adam moments dropped (its state is "
        "shape-bound)"], dropped
