"""Multi-rank plan-streaming runs land where one rank does.

Shot-parallel (DDP) runs split the batch across ranks and SUM the gradient, so
every rank has to normalise by the GLOBAL sample count: per shot it used its
own, which made gradient and loss world_size times too large. A rank whose
slice is empty re-solves a shot of another rank only to stay in the
collectives; it must add nothing. Under a shot cap
(``sampling.shared_shots_per_iter``) every rank has to draw the same shots. A domain-decomposed (DD) grid run hands
every rank the whole model, whose gradient comes back on that rank's tile
only; summed across the tiles, every rank steps the whole model, so the one
rank 0 saves and a dh_m stage resamples is fresh everywhere.

Every run here is a real torchrun job on CPU (gloo). DD uses
``_plan_stream_ranks.FakeModelParallel``, which keeps ModelParallel's gradient
contract without GPUs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from test_task_fwi_plan_streaming import (
    _BAND_HI,
    _BAND_LO,
    _SYN_SHOTS,
    _synthetic_stage_spec,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve().parent / "_plan_stream_ranks.py"
_E = 2                                            # epochs per stage
_ENCODED = {"enabled": True, "min_coverage": 0, "sign_seed": 7}
_INR = {"kind": "velocity_inr", "hidden_features": 16, "hidden_layers": 2,
        "hash": {"enabled": False}, "lr": 1.0e-3, "vp_std": 100.0}


def _run(tmp_path, tag, spec, nproc, dd=False, gpu=False):
    """Run ``spec`` on ``nproc`` ranks (CPU and gloo, or a GPU each and NCCL);
    return the per-rank npz + rank 0's dumps."""
    spec = dict(spec, task_id=tag)
    path = tmp_path / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    out = tmp_path / tag
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if gpu:
        env["SWEEP_DIST_BACKEND"] = "nccl"
    else:
        env.update(SWEEP_DIST_BACKEND="gloo", CUDA_VISIBLE_DEVICES="")
    if dd:
        env.update(SWEEP_DD_ENABLE="1", SWEEP_DD_PY="1", SWEEP_DD_PX=str(nproc),
                   SWEEP_DD_RENDER_CHUNK="4")
    cmd = [sys.executable, str(DRIVER), str(path), str(out)]
    if nproc > 1:
        cmd[1:1] = ["-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
                    "--standalone"]
    if dd:
        cmd.append("--fake-dd")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"{tag}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    ranks = [dict(np.load(out / f"rank{r}.npz")) for r in range(nproc)]
    assert all(str(ROOT / "src") in str(r["tree"]) for r in ranks), ranks[0]["tree"]
    task = Path(str(ranks[0]["task_dir"]))
    ranks[0]["loss"] = np.load(task / "output" / "loss.npy")
    ranks[0]["inverted_vp"] = np.load(task / "output" / "inverted_vp.npy")
    grad = out / "grad_raw.npy"
    ranks[0]["grad"] = np.load(grad) if grad.exists() else None
    for name in ("illum_sr", "grad_precond"):   # with the illumination precondition
        dump = out / f"{name}.npy"
        ranks[0][name] = np.load(dump) if dump.exists() else None
    return ranks


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _snapshot(rank, epoch):
    task = Path(str(rank["task_dir"]))
    return np.load(task / "output" / "epochs" / f"vp_epoch_{epoch:04d}.npy")


_DDP_CASES = {
    # (ranks, extra spec): B = 4 shots, so 3 ranks leave the last one empty
    "per_shot": (2, {}),
    "per_shot_empty_rank": (3, {}),
    "per_crg_empty_rank": (3, {"per_crg": True}),
    "single_step_inr": (2, {"reparam": dict(_INR, backward_mode="single_step")}),
    # 7 of the 12 shots per gather: every rank has to draw the same 7
    "per_shot_shot_cap": (2, {"cap": 7}),
    "per_crg_shot_cap": (3, {"per_crg": True, "cap": 7}),
}


@pytest.mark.parametrize("case", list(_DDP_CASES))
def test_ddp_gives_the_one_rank_gradient_and_loss(tmp_path, case):
    """Same shots, same seed: one rank vs several, gradient and logged loss."""
    nproc, extra = _DDP_CASES[case]
    extra = dict(extra)
    per_crg = extra.pop("per_crg", False)
    cap = extra.pop("cap", len(_SYN_SHOTS))
    spec = _synthetic_stage_spec(tmp_path, [{"epochs": 2 * _E}], **extra)
    spec["optimizer"] = {"kind": "sgd", "lr": 3.0e6, "momentum": 0.0}
    spec["obs"]["plan"]["sampling"]["shared_shots_per_iter"] = cap
    if per_crg:
        spec["obs"]["plan"]["sampling"]["per_crg_independent"] = True
    one = _run(tmp_path, "one", spec, 1)[0]
    ranks = _run(tmp_path, "many", spec, nproc)
    for r in ranks[1:]:                         # every rank steps the same model
        np.testing.assert_array_equal(r["model"], ranks[0]["model"])
    many = ranks[0]
    np.testing.assert_allclose(many["loss"], one["loss"], rtol=1e-5)
    if one["grad"] is not None:                 # grid / two-pass: the leaf gradient
        assert _rel(many["grad"], one["grad"]) < 1e-5
    init = np.load(tmp_path / "init_vp.npy")
    assert _rel(many["inverted_vp"] - init, one["inverted_vp"] - init) < 1e-4


@pytest.mark.parametrize("reparam", [False, True], ids=["grid", "inr"])
def test_dd_ranks_step_the_whole_model(tmp_path, reparam):
    """Two DD tiles (x-cut) against one rank, across a dh_m stage change.

    Grid mode: every rank ends with the one-rank model and so does what rank 0
    saves, at the end and at the end of the coarse stage, from which the fine
    stage starts. A reparam network was already summed across the tiles; it is
    pinned here as well (every rank holds the same network).
    """
    stages = [{"epochs": _E, "dh_m": 100.0, "bandpass": _BAND_LO},
              {"epochs": _E, "dh_m": 50.0, "bandpass": _BAND_HI}]
    extra = {"source_encoding": _ENCODED}
    if reparam:
        extra["reparam"] = _INR
    # nx = 18 at 50 m and 10 at 100 m: both split into two whole tiles, so the
    # DD grids are the one-rank grids (no edge pad).
    spec = _synthetic_stage_spec(tmp_path, stages, shape=(12, 12, 18), **extra)
    spec["show_every"] = 1
    one = _run(tmp_path, "one", spec, 1)
    tiles = _run(tmp_path, "tiles", spec, 2, dd=True)
    one, r0, r1 = one[0], tiles[0], tiles[1]
    init = np.load(tmp_path / "init_vp.npy")
    np.testing.assert_array_equal(r0["model"], r1["model"])
    if reparam:
        np.testing.assert_array_equal(r0["params"], r1["params"])
        tol = 1e-3
    else:
        tol = 1e-5
        assert _rel(r0["grad"], one["grad"]) < tol
        # what rank 0 saved at the end of the coarse stage
        s1, s1_one = _snapshot(r0, _E - 1), _snapshot(one, _E - 1)
        coarse = np.full(s1_one.shape, 2500.0)
        assert _rel(s1 - coarse, s1_one - coarse) < tol
        for r in (r0, r1):                        # what the fine stage starts from
            assert _rel(r[f"seen{_E}"] - init, one[f"seen{_E}"] - init) < tol
    assert _rel(r0["model"] - init, one["model"] - init) < tol
    assert _rel(r0["inverted_vp"] - init, one["inverted_vp"] - init) < tol
    np.testing.assert_allclose(r0["loss"], one["loss"], rtol=1e-4)


def _gpus() -> int:
    import torch
    return torch.cuda.device_count()


@pytest.mark.skipif(_gpus() < 2, reason="solver illumination needs impl='c', a GPU per rank")
def test_ddp_per_crg_precondition_sees_the_whole_batch(tmp_path):
    """Per CRG on two GPUs (NCCL) against one: the precondition divides by the
    illumination of all four nodes either way. The solver keeps only its last
    call's, so one GPU used its last node's and each of two ranks sent its own
    last node's into the SUM all-reduce."""
    spec = _synthetic_stage_spec(
        tmp_path, [{"epochs": _E}],
        illumination_precondition={"enabled": True, "relative_epsilon": 1.0e-3})
    spec["device"] = "cuda"
    spec["backend"] = {"impl": "c", "use_ckpt": False}
    spec["obs"]["plan"]["sampling"]["per_crg_independent"] = True
    one = _run(tmp_path, "one", spec, 1, gpu=True)[0]
    ranks = _run(tmp_path, "two", spec, 2, gpu=True)
    np.testing.assert_array_equal(ranks[1]["model"], ranks[0]["model"])
    two = ranks[0]
    for name in ("grad", "illum_sr", "grad_precond"):   # epoch 0, rank 0
        assert one[name] is not None and two[name] is not None, name
        assert _rel(two[name], one[name]) < 1e-5, name
    np.testing.assert_allclose(two["loss"][0], one["loss"][0], rtol=1e-5)
