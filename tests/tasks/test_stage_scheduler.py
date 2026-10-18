"""The top-level LR ``scheduler`` is one schedule over the whole run.

At global epoch ``t`` a stage steps at ``schedule(t)`` applied to its base lr
(``optimizer.lr x lr_scale``), on whichever optimizer it steps -- including
the new one a stage builds when it changes ``dh_m`` or sets
``optimizer_reset``. A resumed run continues the schedule bit-for-bit.

The lr is read off every ``optimizer.step()`` through a global step pre-hook,
so it is the lr the stepping optimizer actually used.
"""

import math
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim.optimizer as _toptim
import yaml

from sweepx.tasks import TaskRunner, load_task


def _models(tmp_path: Path):
    shape = (48, 48)
    true_vp = (2200 + 600 * np.linspace(0, 1, shape[0])[:, None]
               * np.ones(shape)).astype(np.float32)
    np.save(tmp_path / "true.npy", true_vp)
    np.save(tmp_path / "init.npy", np.full(shape, 2500.0, dtype=np.float32))


_GEOM = {"kind": "line",
         "sources": {"step": 24, "depth": 2, "start": 12, "stop": 42},
         "receivers": {"step": 4, "depth": 4, "start": 4, "stop": 44}}
_BACKEND = {"impl": "eager", "use_ckpt": False,
            "eager_options": {"use_compile": False}}


def _fwi(tmp_path, scheduler, stages=None, epochs=6, **extra):
    _models(tmp_path)
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"), "device": "cpu",
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 120},
        "wavelet": {"kind": "ricker", "fm": 10.0, "delay": 0.1, "scale": 1.0},
        "geometry": _GEOM,
        "physics": {"equation": "Acoustic", "spatial_order": 4, "abcn": 10,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": _BACKEND,
        "init_model": {"name": "vp", "path": str(tmp_path / "init.npy")},
        "obs": {"synthetic_from": {"name": "vp", "path": str(tmp_path / "true.npy")}},
        "optimizer": {"kind": "adam", "lr": 8.0, "eps": 1.0e-22},
        "scheduler": scheduler,
        "epochs": epochs, "batchsize": 1, "show_every": 1,
    }
    if stages is not None:
        spec["stages"] = stages
    spec.update(extra)
    return spec


def _lsrtm(tmp_path, scheduler, stages=None, epochs=6, **extra):
    _models(tmp_path)
    spec = {
        "task_type": "lsrtm", "output_dir": str(tmp_path / "tasks"), "device": "cpu",
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 120},
        "wavelet": {"kind": "ricker", "fm": 10.0, "delay": 0.1, "scale": 1.0},
        "geometry": _GEOM,
        "physics": {"equation": "AcousticLSRTM", "spatial_order": 4, "abcn": 10,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["sh1"]},
        "backend": _BACKEND,
        "background_model": {"name": "vp", "path": str(tmp_path / "init.npy")},
        "true_model": {"name": "vp", "path": str(tmp_path / "true.npy")},
        "optimizer": {"kind": "adam", "lr": 0.08, "eps": 1.0e-22},
        "scheduler": scheduler,
        "epochs": epochs, "batchsize": 1, "show_every": 1,
    }
    if stages is not None:
        spec["stages"] = stages
    spec.update(extra)
    return spec


def _run(spec, path: Path, monkeypatch=None, stop_after=None):
    """Run ``spec``; return (result, lr of every optimizer.step()).

    ``stop_after=k`` sends SIGINT after the k-th checkpoint, which stops the
    run cleanly after that epoch (the graceful-stop path)."""
    if stop_after is not None:
        from sweepx.tasks.tasks import fwi as _fwi_mod
        from sweepx.tasks.tasks import lsrtm as _lsrtm_mod
        mod = _fwi_mod if spec["task_type"] == "fwi" else _lsrtm_mod
        real = mod._save_checkpoint
        calls = {"n": 0}

        def _save_then_stop(task_dir, payload):
            out = real(task_dir, payload)
            calls["n"] += 1
            if calls["n"] == stop_after:
                os.kill(os.getpid(), signal.SIGINT)
            return out

        monkeypatch.setattr(mod, "_save_checkpoint", _save_then_stop)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    lrs = []
    hook = _toptim.register_optimizer_step_pre_hook(
        lambda opt, args, kwargs: lrs.append(opt.param_groups[0]["lr"]))
    try:
        result = TaskRunner().run(load_task(path))
    finally:
        hook.remove()
        if stop_after is not None:
            monkeypatch.undo()
    assert result.status.state == "success", result.status.error
    return result, lrs


_HALVE = {"kind": "step", "step_size": 1, "gamma": 0.5}   # lr(t) = base * 0.5**t


@pytest.mark.parametrize("stages, scales", [
    # stage 1 on a coarser grid: a new optimizer at stage 1
    ([{"epochs": 3}, {"epochs": 3, "dh_m": 20.0}], [1, 1]),
    # coarse -> native, the usual multiscale ladder: a new optimizer at both entries
    ([{"epochs": 3, "dh_m": 20.0}, {"epochs": 3, "dh_m": 10.0}], [1, 1]),
    # fixed grid, the same optimizer throughout, stage 1 halves the base lr
    ([{"epochs": 3}, {"epochs": 3, "lr_scale": 0.5}], [1, 0.5]),
    # fixed grid, stage 1 drops Adam's moments
    ([{"epochs": 3}, {"epochs": 3, "optimizer_reset": True}], [1, 1]),
], ids=["grid_change", "coarse_to_fine", "lr_scale", "optimizer_reset"])
def test_fwi_schedule_spans_stages(tmp_path, stages, scales):
    _, lrs = _run(_fwi(tmp_path, _HALVE, stages), tmp_path / "fwi.yaml")
    expected = [8.0 * scales[t // 3] * 0.5 ** t for t in range(6)]
    assert lrs == expected


def test_fwi_cosine_default_anneals_once_over_all_stages(tmp_path):
    # t_max defaults to all stages' epochs (6): one cosine over the run, on
    # each stage's base lr (8, then 8 * 0.5 on stage 1's new grid).
    sched = {"kind": "cosine", "eta_min": 0.5}
    stages = [{"epochs": 3}, {"epochs": 3, "dh_m": 20.0, "lr_scale": 0.5}]
    _, lrs = _run(_fwi(tmp_path, sched, stages), tmp_path / "fwi.yaml")
    expected = [0.5 + ((8.0 if t < 3 else 4.0) - 0.5) * (1 + math.cos(math.pi * t / 6)) / 2
                for t in range(6)]
    assert lrs == pytest.approx(expected, rel=1e-12)


_EXP = {"kind": "exp", "gamma": 0.9}


@pytest.mark.parametrize("stages, epochs, stop_after, base_at_stop", [
    # no stages: the resume lands at epoch 6
    (None, 8, 6, 8.0),
    # fixed grid: the resume lands inside stage 1 (epochs 3..7, base 8 * 0.5)
    ([{"epochs": 3}, {"epochs": 5, "lr_scale": 0.5}], 8, 6, 4.0),
    # stopped on the native grid, continues through stage 1's new grid/optimizer
    ([{"epochs": 3}, {"epochs": 3, "dh_m": 20.0}], 6, 2, None),
], ids=["single", "fixed_grid", "into_grid_change"])
def test_fwi_resume_continues_schedule_bit_identically(tmp_path, monkeypatch, stages,
                                                       epochs, stop_after, base_at_stop):
    cont, lr_cont = _run(_fwi(tmp_path, _EXP, stages, epochs, task_id="cont"),
                         tmp_path / "cont.yaml")
    if base_at_stop is not None:
        # The chained lr the resume lands on differs from the closed form in
        # the last bits, so matching lr_cont needs the checkpointed lr back.
        closed = base_at_stop * _EXP["gamma"] ** stop_after
        assert lr_cont[stop_after] != closed
        assert lr_cont[stop_after] == pytest.approx(closed, rel=1e-12)
    split = _fwi(tmp_path, _EXP, stages, epochs, task_id="split", resume=True)
    first, lr_first = _run(split, tmp_path / "split.yaml", monkeypatch, stop_after)
    assert first.status.summary["interrupted"] is True
    second, lr_second = _run(split, tmp_path / "split.yaml")
    assert second.status.summary["resumed_from"] == "split"

    assert len(lr_first) == stop_after
    assert lr_first + lr_second == lr_cont
    for name in ("inverted_vp.npy", "loss.npy"):
        assert np.array_equal(np.load(cont.task_dir / "output" / name),
                              np.load(second.task_dir / "output" / name)), name


def test_lsrtm_schedule_spans_stages_and_resumes(tmp_path, monkeypatch):
    stages = [{"epochs": 3}, {"epochs": 3, "lr_scale": 0.5}]
    _, lr_cont = _run(_lsrtm(tmp_path, _HALVE, stages, task_id="cont"),
                      tmp_path / "cont.yaml")
    assert lr_cont == [0.08 * (1 if t < 3 else 0.5) * 0.5 ** t for t in range(6)]

    split = _lsrtm(tmp_path, _HALVE, stages, task_id="split", resume=True)
    _, lr_first = _run(split, tmp_path / "split.yaml", monkeypatch, stop_after=4)
    _, lr_second = _run(split, tmp_path / "split.yaml")
    assert lr_first + lr_second == lr_cont


@pytest.mark.parametrize("sched", [
    {"kind": "exp", "gamma": 0.9},
    {"kind": "step", "step_size": 2, "gamma": 0.7},
    # t_max < epochs: the chained cosine's restart branch
    {"kind": "cosine", "eta_min": 0.3, "t_max": 3},
], ids=["exp", "step", "cosine_restart"])
def test_fwi_without_stages_steps_like_a_plain_torch_scheduler(tmp_path, sched):
    from pydantic import TypeAdapter
    from sweepx.tasks.runtime.scheduler import build
    from sweepx.tasks.schemas import Scheduler

    _, lrs = _run(_fwi(tmp_path, sched, epochs=7), tmp_path / "fwi.yaml")
    opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=8.0)
    plain = build(TypeAdapter(Scheduler).validate_python(sched), opt, 7)
    expected = []
    for _ in range(7):
        expected.append(opt.param_groups[0]["lr"])
        opt.step()
        plain.step()
    assert lrs == expected


def test_lr_at_is_the_closed_form_of_the_chained_schedulers():
    from sweepx.tasks.runtime.scheduler import build, lr_at
    from sweepx.tasks.schemas import SchedulerCosine, SchedulerExp, SchedulerStep

    for spec in (SchedulerStep(step_size=3, gamma=0.8), SchedulerExp(gamma=0.95),
                 SchedulerCosine(eta_min=0.2), SchedulerCosine(eta_min=0.2, t_max=7)):
        opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=5.0)
        sched = build(spec, opt, 40)
        for t in range(40):
            assert math.isclose(opt.param_groups[0]["lr"], lr_at(spec, 5.0, t, 40),
                                rel_tol=1e-12, abs_tol=1e-15), (spec, t)
            opt.step()
            sched.step()


def test_bind_restores_checkpointed_lr_only_for_the_same_schedule():
    from sweepx.tasks._helpers.optimizer import _bind_stage_scheduler
    from sweepx.tasks.runtime.scheduler import lr_at
    from sweepx.tasks.schemas import SchedulerConstant, SchedulerExp

    spec = SchedulerExp(gamma=0.9)
    closed = lr_at(spec, 8.0, 6, 10)
    chained = 8.0
    for _ in range(6):
        chained *= 0.9
    assert chained != closed and math.isclose(chained, closed, rel_tol=1e-12)

    def _opt():
        return torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=8.0)

    opt = _opt()
    sched = _bind_stage_scheduler(spec, opt, 10, 6, resume_lrs=[chained])
    assert opt.param_groups[0]["lr"] == chained
    assert sched.last_epoch == 6 and sched.base_lrs == [8.0]
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == chained * 0.9

    # A checkpoint written under another schedule or base lr: the current
    # schedule's value wins.
    opt = _opt()
    _bind_stage_scheduler(spec, opt, 10, 6, resume_lrs=[8.0])
    assert opt.param_groups[0]["lr"] == closed
    # No resume: the closed form at the stage's first epoch.
    opt = _opt()
    _bind_stage_scheduler(spec, opt, 10, 6)
    assert opt.param_groups[0]["lr"] == closed
    # constant: no scheduler, the optimizer is left alone.
    opt = _opt()
    assert _bind_stage_scheduler(SchedulerConstant(), opt, 10, 6, resume_lrs=[1.0]) is None
    assert opt.param_groups[0]["lr"] == 8.0 and "initial_lr" not in opt.param_groups[0]
