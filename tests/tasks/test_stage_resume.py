"""Stop + resume reproduces the uninterrupted run at any point of a staged run.

A checkpoint holds the models on the grid of the stage it was written in.
Resuming puts the run back on that grid -- models, reparam network, optimizer,
geometry, solver -- and then continues the stage, or enters the next one,
exactly as the uninterrupted run did. What only entering a stage does
(``optimizer_reset``, the reparam water reset) is not repeated when the
resume lands inside the stage.

Compared bit for bit: every ``.npy`` under ``output/`` (inverted model, loss,
every epoch snapshot) and the lr of every optimizer step. Same determinism
setting as the scheduler resume tests: eager backend, no compile, CPU. Under
reparam also ``torch.use_deterministic_algorithms``: on more than one CPU
thread the hash grid's backward (``index_put_`` with accumulate) sums with
atomics, so without it two uninterrupted runs already differ.

Runs are four to six epochs, two or three per stage: room to stop inside a
stage, at its last epoch, and inside the next one. A stop that resumes inside
a stage leaves at least two steps of that stage's optimizer in the checkpoint,
so a resume that kept Adam's moments but restarted its step count differs.
Each epoch draws one of two shots (1, 0, 0, 1, 0, ... under the default seed);
most stops are where the resumed epochs draw other shots than a fresh run's
first epochs would, so a resume that lost the numpy RNG differs. A stop inside
a stage entered at epoch 3 is at epoch 4 or 5, where the checkpointed lr
(chained, 0.9 x 0.9 x ...) is not the closed form, so a resume that lost it
differs too.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_stage_scheduler import _NT, _fwi, _run

_EXP = {"kind": "exp", "gamma": 0.9}
_REPARAM = {
    "reparam": {"kind": "velocity_inr", "hidden_features": 16, "hidden_layers": 2,
                "vp_std": 50.0, "lr": 1.0e-3,
                "hash": {"enabled": True, "levels": 4, "features_per_level": 2,
                         "log2_size": 10, "base_resolution": 4, "finest_resolution": 32}},
    "optimizer": {"kind": "adam", "lr": 25.0, "eps": 1.0e-22},
    "model_bounds": {"vp": {"min": 1500.0, "max": 4500.0}},
}
# Three epochs in stage 0: a stop inside it (after epoch 1) still leaves two
# optimizer steps in the checkpoint.
_COARSE = [{"epochs": 3, "dh_m": 20.0}, {"epochs": 2, "dh_m": 10.0}]
_INHERIT = [{"epochs": 2, "dh_m": 20.0}, {"epochs": 3, "lr_scale": 0.5}]
_COARSE_HALF_LR = [{"epochs": 3, "dh_m": 20.0}, {"epochs": 2, "dh_m": 10.0, "lr_scale": 0.5}]
# What a checkpoint held before it recorded its stage.
_OLD_KEYS = {"epoch", "models", "optimizer", "scheduler", "losses", "torch_rng",
             "numpy_rng"}


@pytest.fixture
def deterministic():
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(was)


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    """``(stages, epochs, **extra) -> (lr of every step, output hashes)`` of the
    uninterrupted run, run once per configuration in this module: the cases
    that stop the same run at different epochs compare against the same one."""
    runs = {}

    def run(stages, epochs, **extra):
        key = json.dumps([stages, epochs, extra,
                          torch.are_deterministic_algorithms_enabled()],
                         sort_keys=True, default=str)
        if key not in runs:
            d = tmp_path_factory.mktemp("cont")
            cont, lr_cont = _run(_fwi(d, _EXP, stages, epochs, task_id="cont", **extra),
                                 d / "cont.yaml")
            runs[key] = lr_cont, _outputs(cont.task_dir)
        return runs[key]

    return run


def _reparam(**changes):
    return {**_REPARAM, "reparam": {**_REPARAM["reparam"], **changes}}


def _as_old_checkpoint(ckpt):
    for key in set(ckpt) - _OLD_KEYS:
        ckpt.pop(key)


def _outputs(task_dir: Path) -> dict:
    out = task_dir / "output"
    return {str(p.relative_to(out)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*.npy"))}


def _stop(tmp_path, monkeypatch, stages, epochs, stop_after, **extra):
    """Run to the ``stop_after``-th checkpoint and stop; return the resumable spec."""
    split = _fwi(tmp_path, _EXP, stages, epochs, task_id="split", resume=True, **extra)
    first, lr_first = _run(split, tmp_path / "split.yaml", monkeypatch, stop_after)
    assert first.status.summary["interrupted"] is True
    assert len(lr_first) == stop_after
    return split, first, lr_first


def _assert_resume_bit_identical(tmp_path, monkeypatch, stages, epochs, stop_after,
                                 edit_checkpoint=None, uninterrupted=None, **extra):
    if uninterrupted is None:
        cont, lr_cont = _run(_fwi(tmp_path, _EXP, stages, epochs, task_id="cont", **extra),
                             tmp_path / "cont.yaml")
        expected = _outputs(cont.task_dir)
    else:
        lr_cont, expected = uninterrupted(stages, epochs, **extra)
    split, first, lr_first = _stop(tmp_path, monkeypatch, stages, epochs, stop_after,
                                   **extra)
    if edit_checkpoint is not None:
        path = first.task_dir / "checkpoint.pt"
        ckpt = torch.load(path, weights_only=False)
        edit_checkpoint(ckpt)
        torch.save(ckpt, path)
    second, lr_second = _run(split, tmp_path / "split.yaml")
    assert second.status.summary["resumed_from"] == "split"
    assert lr_first + lr_second == lr_cont
    assert len(expected) == epochs + 3        # snapshots + inverted, initial, loss
    assert _outputs(second.task_dir) == expected


@pytest.mark.parametrize("stages, stop_after", [
    # inside the coarse stage 0: its models are on the 20 m grid
    (_COARSE, 2),
    # at the last epoch of the coarse stage: the resume enters stage 1
    (_COARSE, 3),
    # inside stage 1, which inherits stage 0's 20 m grid
    (_INHERIT, 4),
    # entering stage 1: same grid as stage 0, so its entry rebuilds no
    # solver and re-snaps no geometry -- the resume has to
    (_INHERIT, 2),
    # each stage on its own time axis too
    ([{"epochs": 3, "dh_m": 20.0, "dt_s": 0.002},
      {"epochs": 2, "dh_m": 10.0, "dt_s": 0.001}], 2),
], ids=["inside_coarse", "end_of_coarse", "inside_inherited_grid",
        "into_inherited_grid", "inside_coarse_own_dt"])
def test_fwi_resume_on_a_stage_grid_is_bit_identical(tmp_path, monkeypatch, uninterrupted,
                                                     stages, stop_after):
    epochs = sum(s["epochs"] for s in stages)
    _assert_resume_bit_identical(tmp_path, monkeypatch, stages, epochs, stop_after,
                                 uninterrupted=uninterrupted, nt=_NT)


def test_fwi_resume_into_an_inherited_grid_resnaps_segy_geometry(tmp_path, monkeypatch):
    # With dedupe off, a stage on its predecessor's grid keeps that stage's
    # geometry instead of re-snapping it; resumed at that entry, the geometry
    # in hand is the native one, so the resume must re-snap it itself.
    from test_task_runner_smoke import _write_tiny_segy

    segy = tmp_path / "tiny.segy"
    _write_tiny_segy(segy, nshots=3, nrec=6, nt=64, dh_m=10.0)
    depths = {"source_depth_m_override": 10.0, "receiver_depth_m_override": 10.0}
    _assert_resume_bit_identical(
        tmp_path, monkeypatch, _INHERIT, 5, 2,
        geometry={"kind": "from_segy_headers", "path": str(segy), "dedupe": False,
                  **depths},
        obs={"segy": {"path": str(segy), **depths}}, time={"dt": 0.001, "nt": 64})


@pytest.mark.parametrize("stages, extra, epochs, stop_after", [
    # no stages: the network's weights are the model, so they must come back
    (None, _REPARAM, 4, 2),
    # inside the coarse stage 0 / at its last epoch
    (_COARSE_HALF_LR, _REPARAM, 5, 2),
    (_COARSE_HALF_LR, _REPARAM, 5, 3),
    # hash.c2f.growing: a level grows at epochs 2 and 4 (progress = epoch / 4
    # reaches 50 % and 100 %), each with its own optimizer param group and an
    # initialisation drawn from the torch RNG: the checkpoint holds the first,
    # the resumed run grows the second
    (_COARSE, _reparam(hash={**_REPARAM["reparam"]["hash"],
                             "c2f": {"enabled": True, "growing": True,
                                     "base_levels": 2}}), 5, 4),
], ids=["no_stages", "inside_coarse", "end_of_coarse", "growing_hash_grid"])
def test_fwi_reparam_resume_is_bit_identical(tmp_path, monkeypatch, deterministic,
                                             uninterrupted, stages, extra, epochs,
                                             stop_after):
    _assert_resume_bit_identical(tmp_path, monkeypatch, stages, epochs, stop_after,
                                 uninterrupted=uninterrupted, nt=_NT, **extra)


def test_fwi_reparam_water_reset_resume_is_bit_identical(tmp_path, monkeypatch,
                                                         deterministic):
    # The water reset re-bakes the network's base at each stage entry from the
    # render at that moment, which a resume cannot recompute: stop inside the
    # native stage 1, after its entry reset.
    shape = (48, 48)
    true_vp = (2200 + 600 * np.linspace(0, 1, shape[0])[:, None]
               * np.ones(shape)).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_vp[:6] = init_vp[:6] = 1500.0
    np.save(tmp_path / "true_water.npy", true_vp)
    np.save(tmp_path / "init_water.npy", init_vp)
    water = np.zeros(shape, dtype=bool)
    water[:6] = True
    np.save(tmp_path / "water_mask.npy", water)
    _assert_resume_bit_identical(
        tmp_path, monkeypatch, _COARSE, 5, 4,
        init_model={"name": "vp", "path": str(tmp_path / "init_water.npy")},
        obs={"synthetic_from": {"name": "vp", "path": str(tmp_path / "true_water.npy")}},
        water={"mask": str(tmp_path / "water_mask.npy"), "mode": "reset_each_stage"},
        nt=_NT, **_REPARAM)


def test_fwi_reparam_resume_keeps_the_init_optimizer_from_lr(tmp_path, monkeypatch,
                                                            deterministic):
    # init_optimizer_from starts the network's Adam from another run's state,
    # its decayed lr included; that lr stays the base lr across a resume.
    # (Stopped where the chained lr is not the closed form.)
    first, _ = _run(_fwi(tmp_path, _EXP, None, 1, task_id="first", nt=_NT, **_REPARAM),
                    tmp_path / "first.yaml")
    _assert_resume_bit_identical(
        tmp_path, monkeypatch, None, 4, 2, nt=_NT,
        **_reparam(init_optimizer_from=str(first.task_dir / "checkpoint.pt")))


@pytest.mark.parametrize("extra", [{}, _REPARAM], ids=["grid", "reparam"])
def test_resume_inside_an_optimizer_reset_stage_keeps_the_moments(tmp_path, monkeypatch,
                                                                  deterministic, extra):
    # The reset happens on entering stage 1 (epoch 3); the run stops after
    # epoch 4, two Adam steps after the reset, and must not drop the moments
    # a second time (nor restart their step count).
    stages = [{"epochs": 3}, {"epochs": 3, "optimizer_reset": True}]
    _assert_resume_bit_identical(tmp_path, monkeypatch, stages, 6, 5, nt=_NT, **extra)


def test_a_checkpoint_without_stage_entries_still_resumes(tmp_path, monkeypatch,
                                                          uninterrupted):
    # A checkpoint written before checkpoints recorded their stage: the stage
    # is the one holding the epoch, and the models are on its grid.
    _assert_resume_bit_identical(tmp_path, monkeypatch, _COARSE, 5, 2,
                                 edit_checkpoint=_as_old_checkpoint,
                                 uninterrupted=uninterrupted, nt=_NT)


def test_a_reparam_checkpoint_without_the_network_is_refused(tmp_path, monkeypatch):
    # Older checkpoints hold only the rendered vp: continuing from them would
    # restart the network from its initial weights.
    split, first, _ = _stop(tmp_path, monkeypatch, None, 2, 1, nt=_NT, **_REPARAM)
    path = first.task_dir / "checkpoint.pt"
    ckpt = torch.load(path, weights_only=False)
    _as_old_checkpoint(ckpt)
    torch.save(ckpt, path)
    with pytest.raises(AssertionError, match="holds no reparam network"):
        _run(split, tmp_path / "split.yaml")


def test_a_checkpoint_on_another_grid_is_refused(tmp_path, monkeypatch):
    # The stopped run was on a 20 m grid; the config now puts that epoch on
    # 15 m. The models cannot be continued there.
    split, _, _ = _stop(tmp_path, monkeypatch, _COARSE, 5, 1, nt=_NT)
    split["stages"] = [{"epochs": 3, "dh_m": 15.0}, {"epochs": 2, "dh_m": 10.0}]
    with pytest.raises(AssertionError, match="stages changed since the checkpoint"):
        _run(split, tmp_path / "split.yaml")
