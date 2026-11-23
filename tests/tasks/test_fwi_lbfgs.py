"""`optimizer.kind: lbfgs` is sweepx.opt's L-BFGS, and a non-finite loss fails the run.

torch's L-BFGS took its steps in gradient units: on a velocity model in m/s it
either stopped at once (tolerance on a tiny gradient) or, with a larger lr,
walked into NaN while the task still reported success."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from sweepx.tasks import TaskRunner, load_task


def _spec(tmp_path: Path, **overrides) -> Path:
    shape = (48, 48)
    true = (2200 + 600 * np.linspace(0, 1, shape[0])[:, None] * np.ones(shape)).astype(np.float32)
    np.save(tmp_path / "true.npy", true)
    np.save(tmp_path / "init.npy", np.full(shape, 2500.0, dtype=np.float32))
    spec = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 6, "depth": 2, "start": 6, "stop": 42},     # 6 shots
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 44},
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 12,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(tmp_path / "init.npy")},
        "obs": {"synthetic_from": {"name": "vp", "path": str(tmp_path / "true.npy")}},
        "optimizer": {"kind": "lbfgs"},
        "model_bounds": {"vp": {"min": 2150.0, "max": 2850.0}},
        "epochs": 4, "batchsize": 6, "show_every": 10,
    }
    spec.update(overrides)
    path = tmp_path / "fwi.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_lbfgs_fwi_lowers_the_loss_and_keeps_the_bounds(tmp_path):
    result = TaskRunner().run(load_task(_spec(tmp_path)))
    assert result.status.state == "success", result.status.error
    losses = np.load(result.task_dir / "output" / "loss.npy")
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.8 * losses[0], losses
    vp = np.load(result.task_dir / "output" / "inverted_vp.npy")
    assert np.abs(vp - 2500.0).max() > 1.0                      # it moved
    assert vp.min() >= 2150.0 - 1e-3 and vp.max() <= 2850.0 + 1e-3


def test_lbfgs_schema():
    from sweepx.tasks.schemas import OptimizerLBFGS

    opt = OptimizerLBFGS()
    assert (opt.initial_step, opt.max_iter, opt.line_search_fn) == ("auto", 1, "strong_wolfe")
    with pytest.raises(ValueError):
        OptimizerLBFGS(line_search_fn=None)          # torch's no-line-search mode is gone
    with pytest.raises(ValueError):
        OptimizerLBFGS(initial_step=0.0)


def test_lbfgs_auto_initial_step():
    import torch

    from sweepx.tasks._helpers.optimizer import _lbfgs_initial_step

    assert _lbfgs_initial_step("auto", [torch.full((4, 4), 2500.0)]) == pytest.approx(25.0)
    assert _lbfgs_initial_step(7.5, [torch.zeros(3)]) == 7.5
    with pytest.raises(ValueError, match="all zeros"):
        _lbfgs_initial_step("auto", [torch.zeros(3)])


def test_lbfgs_lr_lives_in_the_param_group():
    """Schedulers and per-stage lr scales write param_groups[0]['lr']; the
    task L-BFGS reads its step multiplier from there."""
    import torch

    from sweepx.tasks._helpers.optimizer import _build_optimizer
    from sweepx.tasks.schemas import OptimizerLBFGS

    vp = torch.full((4, 4), 2000.0, requires_grad=True)
    opt = _build_optimizer(OptimizerLBFGS(lr=0.5), {"vp": vp}, ["vp"])
    assert opt.param_groups[0]["lr"] == 0.5
    opt.param_groups[0]["lr"] = 0.25

    def closure():
        opt.zero_grad()
        loss = ((vp - 1990.0) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    assert opt.alpha_init == 0.25 and opt.alpha_max == 2.5


def test_reparam_lbfgs_refuses_initial_step():
    import torch

    from sweepx.tasks._helpers.optimizer import _build_reparam_optimizer
    from sweepx.tasks.schemas import OptimizerLBFGS

    w = [torch.zeros(3, requires_grad=True)]
    _build_reparam_optimizer(OptimizerLBFGS(), w, 1e-3)                  # fine without it
    with pytest.raises(ValueError, match="initial_step"):
        _build_reparam_optimizer(OptimizerLBFGS(initial_step=5.0), w, 1e-3)


def test_nonfinite_loss_fails_the_run(tmp_path, monkeypatch):
    from sweepx.tasks.tasks.fwi import FWIRunnerMixin

    monkeypatch.setattr(FWIRunnerMixin, "_fwi_train_step", lambda self, *a, **k: float("nan"))
    spec = _spec(tmp_path, optimizer={"kind": "adam", "lr": 5.0}, epochs=2)
    result = TaskRunner().run(load_task(spec))
    assert result.status.state == "failed"
    assert "diverged" in (result.status.error or "")


def test_lbfgs_refuses_a_shot_batch(tmp_path):
    """A batch of some shots is a different misfit each epoch: L-BFGS's curvature
    pairs then mean nothing (a run from the agent went to vp = -4240 m/s)."""
    result = TaskRunner().run(load_task(_spec(tmp_path, batchsize=3)))
    assert result.status.state == "failed"
    assert "Set batchsize to 6" in (result.status.error or "")


def _count_reuse(monkeypatch):
    """Count the L-BFGS steps that start from the previous line search's
    gradient instead of running every shot again."""
    from sweepx.opt.optim._wolfe_step import AcceptedGradCache

    hits = [0]
    load = AcceptedGradCache.load

    def counted(self, *a):
        out = load(self, *a)
        hits[0] += out is not None
        return out

    monkeypatch.setattr(AcceptedGradCache, "load", counted)
    return hits


def _losses(result):
    assert result.status.state == "success", result.status.error
    return np.load(result.task_dir / "output" / "loss.npy")


def test_lbfgs_fwi_reuses_the_accepted_gradient_across_epochs(tmp_path, monkeypatch):
    """Each epoch's step builds its own closure; within a stage the first
    epoch's closure is passed again, so every epoch after the first starts
    from the gradient its line search already computed: one forward and
    adjoint of all shots fewer per epoch. The run follows the same path as
    recomputing (only the shot summation order of later epochs differs)."""
    hits = _count_reuse(monkeypatch)
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    on = _losses(TaskRunner().run(load_task(_spec(tmp_path / "on"))))
    assert hits[0] == 3                              # epochs 2, 3, 4
    hits[0] = 0
    off = _losses(TaskRunner().run(load_task(_spec(
        tmp_path / "off",
        optimizer={"kind": "lbfgs", "reuse_accepted_grad": False}))))
    assert hits[0] == 0
    np.testing.assert_allclose(on, off, rtol=1e-4)


def test_lbfgs_fwi_drops_the_held_closure_at_each_stage(tmp_path, monkeypatch):
    """A new stage is a new misfit (band, grid, obs): its first epoch runs all
    shots at its start model, and only its later epochs reuse."""
    hits = _count_reuse(monkeypatch)
    _losses(TaskRunner().run(load_task(_spec(
        tmp_path, stages=[{"epochs": 2}, {"epochs": 2}]))))
    assert hits[0] == 2                              # epoch 2 of each stage


def test_lbfgs_max_iter_reuses_within_an_epoch(tmp_path, monkeypatch):
    hits = _count_reuse(monkeypatch)
    _losses(TaskRunner().run(load_task(_spec(
        tmp_path, epochs=2, optimizer={"kind": "lbfgs", "max_iter": 3}))))
    assert hits[0] == 5                              # 6 iterations, all but the first


def test_reuse_accepted_grad_schema_and_reparam_refusal():
    import torch

    from sweepx.tasks._helpers.optimizer import _build_optimizer, _build_reparam_optimizer
    from sweepx.tasks.schemas import OptimizerLBFGS

    assert OptimizerLBFGS().reuse_accepted_grad is True
    vp = torch.full((4, 4), 2000.0, requires_grad=True)
    opt = _build_optimizer(OptimizerLBFGS(reuse_accepted_grad=False), {"vp": vp}, ["vp"])
    assert opt.reuse_accepted_grad is False
    w = [torch.zeros(3, requires_grad=True)]
    with pytest.raises(ValueError, match="reuse_accepted_grad"):
        _build_reparam_optimizer(OptimizerLBFGS(reuse_accepted_grad=True), w, 1e-3)


def test_stage_closure_key():
    """The held closure comes back only for the same objects and values."""
    from sweepx.tasks._helpers.optimizer import _release_stage_closure, _stage_closure

    class Opt:
        reuse_accepted_grad = True

        def __init__(self):
            self.invalidated = 0

        def invalidate_grad_cache(self):
            self.invalidated += 1

    opt, obs = Opt(), object()
    built = []

    def build():
        built.append(lambda: None)
        return built[-1]

    a = _stage_closure(opt, build, (opt, obs), (0.001,))
    assert _stage_closure(opt, build, (opt, obs), (0.001,)) is a
    b = _stage_closure(opt, build, (opt, object()), (0.001,))     # other obs
    assert b is not a
    c = _stage_closure(opt, build, (opt, obs), (0.002,))          # other dt
    assert c is not b and len(built) == 3
    _release_stage_closure(opt)
    assert not hasattr(opt, "_task_closure") and opt.invalidated == 1
    opt.reuse_accepted_grad = False
    assert _stage_closure(opt, build, (opt, obs), (0.002,)) is not _stage_closure(
        opt, build, (opt, obs), (0.002,))
    assert not hasattr(opt, "_task_closure")


def _lsrtm_spec(tmp_path: Path, **overrides) -> Path:
    shape = (32, 32)
    true = np.full(shape, 2500.0, dtype=np.float32)
    true[12:] = 2800.0
    np.save(tmp_path / "true.npy", true)
    spec = {
        "task_type": "lsrtm",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 120},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08},
        "geometry": {"kind": "line", "sources": {"step": 14, "depth": 2, "start": 9, "stop": 24},
                     "receivers": {"step": 4, "depth": 4}},
        "physics": {"equation": "AcousticBorn", "receiver_type": ["sh1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "background_model": {"name": "vp", "constant": 2500.0, "shape": list(shape)},
        "true_model": {"name": "vp", "path": str(tmp_path / "true.npy")},
        "optimizer": {"kind": "lbfgs", "initial_step": 0.03},
        "epochs": 3, "batchsize": 2, "show_every": 10,
    }
    spec.update(overrides)
    path = tmp_path / "lsrtm.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_lbfgs_lsrtm_reuses_only_for_every_shot(tmp_path, monkeypatch):
    hits = _count_reuse(monkeypatch)
    (tmp_path / "all").mkdir()
    (tmp_path / "batch").mkdir()
    _losses(TaskRunner().run(load_task(_lsrtm_spec(tmp_path / "all"))))
    assert hits[0] == 2                              # epochs 2 and 3
    hits[0] = 0
    # A batch of 1 of the 2 shots is a new misfit each epoch: no reuse.
    _losses(TaskRunner().run(load_task(_lsrtm_spec(tmp_path / "batch", batchsize=1))))
    assert hits[0] == 0
