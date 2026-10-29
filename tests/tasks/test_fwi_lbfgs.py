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
