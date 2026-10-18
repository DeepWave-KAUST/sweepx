"""A level a growing hash grid adds runs at the lr of the rest of the network.

``hash.c2f.growing`` registers each grown level with the live optimizer as a
new param group. It joins at the lr the network's groups run at that moment
-- base lr x the stage's scale x the schedule's factor at the global epoch --
and from then on, stage entries included, it is treated like them. It used to
join at ``optimizer.defaults['lr']`` (the unscaled ``reparam.lr``), off the
schedule, and the next stage entry decayed it a second time.

The lr of every group is read off each ``optimizer.step()`` through a global
step pre-hook, so it is what the stepping optimizer used.
"""
import math
from pathlib import Path

import pytest
import torch.optim.optimizer as _toptim
import yaml

from sweepx.tasks import TaskRunner, load_task
from sweepx.tasks.runtime.scheduler import lr_at
from sweepx.tasks.schemas import SchedulerConstant, SchedulerCosine, SchedulerExp

import test_freqsel_optimizer_settings as _fs
from test_stage_resume import _REPARAM, _assert_resume_bit_identical, deterministic  # noqa: F401
from test_stage_scheduler import _fwi

# 4 levels, 2 at the start: with progress = epoch / (total - 1) the third
# level grows at 50 % of the run, the fourth at 100 %.
_GROWING = {**_REPARAM["reparam"],
            "hash": {**_REPARAM["reparam"]["hash"],
                     "c2f": {"enabled": True, "growing": True, "base_levels": 2}}}
_LR = _GROWING["lr"]


def _run_groups(spec: dict, path: Path):
    """Run ``spec``; return, per optimizer step, the lr of every param group."""
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    steps = []
    hook = _toptim.register_optimizer_step_pre_hook(
        lambda opt, args, kwargs: steps.append([g["lr"] for g in opt.param_groups]))
    try:
        result = TaskRunner().run(load_task(path))
    finally:
        hook.remove()
    assert result.status.state == "success", result.status.error
    return steps


def _expected(sched, scales, total):
    return [lr_at(sched, _LR * s, t, total) for t, s in enumerate(scales)]


# 9 epochs: levels grow at epochs 4 (inside stage 1, at scale 0.5) and 8 (at
# the start of stage 2, whose entry already re-scaled the level grown at 4).
_STAGES = [{"epochs": 2}, {"epochs": 6, "lr_scale": 0.5},
           {"epochs": 1, "inr_lr_scale": 0.25}]
_SCALES = [1.0] * 2 + [0.5] * 6 + [0.25]
_GROUPS = [1] * 4 + [2] * 4 + [3]


@pytest.mark.parametrize("sched", [
    {"kind": "constant"},
    {"kind": "exp", "gamma": 0.9},
    # t_max < epochs: the chained cosine's restart branch, at epoch 7, reads
    # the scheduler's base_lrs, which the level grown at 4 has to be in
    {"kind": "cosine", "eta_min": 1.0e-4, "t_max": 2},
], ids=lambda s: s["kind"])
def test_conventional_grown_levels_run_at_the_network_lr(tmp_path, sched):
    spec = _fwi(tmp_path, sched, _STAGES, 9, reparam=_GROWING,
                optimizer=_REPARAM["optimizer"], model_bounds=_REPARAM["model_bounds"])
    steps = _run_groups(spec, tmp_path / "fwi.yaml")
    assert [len(s) for s in steps] == _GROUPS, steps
    kind = {"constant": SchedulerConstant, "exp": SchedulerExp,
            "cosine": SchedulerCosine}[sched["kind"]]
    expected = _expected(kind(**sched), _SCALES, 9)
    for t, (lrs, want) in enumerate(zip(steps, expected)):
        assert lrs == [lrs[0]] * len(lrs), (t, lrs)          # one lr, bit for bit
        assert math.isclose(lrs[0], want, rel_tol=1e-12), (t, lrs, want)


@pytest.mark.parametrize("stop_after", [
    4,      # resumes at epoch 4, where the third level grows
    6,      # inside stage 1, with the grown level in the checkpoint
    8,      # enters stage 2 (and grows the fourth level) on the resume
], ids=["at_growth", "after_growth", "into_next_stage"])
def test_conventional_growing_resume_is_bit_identical(tmp_path, monkeypatch,
                                                      deterministic, stop_after):
    _assert_resume_bit_identical(tmp_path, monkeypatch, _STAGES, 9, stop_after,
                                 reparam=_GROWING, optimizer=_REPARAM["optimizer"],
                                 model_bounds=_REPARAM["model_bounds"])


def test_bind_keeps_each_groups_checkpointed_lr_on_its_own():
    """A resume keeps a group's checkpointed lr where it is the schedule's (up
    to the chained steps' rounding), and puts any other group on the schedule:
    a checkpoint written before grown levels followed it."""
    import torch

    from sweepx.tasks._helpers.optimizer import _bind_stage_scheduler

    spec = SchedulerExp(gamma=0.9)
    chained = 8.0
    for _ in range(6):
        chained *= 0.9
    closed = lr_at(spec, 8.0, 6, 10)
    assert chained != closed and math.isclose(chained, closed, rel_tol=1e-12)
    opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=8.0)
    opt.add_param_group({"params": [torch.zeros(1, requires_grad=True)], "lr": 8.0})
    sched = _bind_stage_scheduler(spec, opt, 10, 6, resume_lrs=[chained, 8.0e-3])
    assert [g["lr"] for g in opt.param_groups] == [chained, closed]
    assert sched.base_lrs == [8.0, 8.0]


def test_freqsel_grown_levels_run_at_the_stage_lr(tmp_path):
    """Each stage runs at reparam.lr x inr_lr_scale; a level grown inside it
    (epochs 3 and 5 of 6) as well."""
    reparam = {**_fs._REPARAM, "lr": _LR, "hash": _GROWING["hash"]}
    stages = [{"epochs": 2}, {"epochs": 2, "inr_lr_scale": 0.5},
              {"epochs": 2, "inr_lr_scale": 0.25}]
    spec = _fs._run_spec(tmp_path, "grow", epochs=6, stages=stages, reparam=reparam)
    steps = _run_groups(spec, tmp_path / "freqsel.yaml")
    assert [len(s) for s in steps] == [1, 1, 1, 2, 2, 3], steps
    for t, (lrs, s) in enumerate(zip(steps, [1.0, 1.0, 0.5, 0.5, 0.25, 0.25])):
        assert lrs == [_LR * s] * len(lrs), (t, lrs)
