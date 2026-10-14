"""Under domain decomposition a plan-streaming run refuses what its loop skips.

DD (sweep's ModelParallel) is switched on by SWEEP_DD_ENABLE, not by the YAML,
so the schema cannot see it: the run checks when it starts. The DD loop skips
the illumination precondition, the smoothness prior and the seabed gradient
freeze, and a run that asked for one of them went ahead without it, saying so
in one log line. Now it stops before the plan is loaded. Without DD the same
settings run; under DD a block switched off, as a config_resolved.yaml records
it, does not stop a run, and the network's own water pin
(``reparam.mask_water_layer``) works. DD also pushes each tile's gradient
through the network in z-slabs of SWEEP_DD_RENDER_CHUNK rows, never reading
``reparam.backward_mode`` / ``backward_chunk_rows``: a value other than the
default stops the run the same way.

The refusals emulate DD in this process through the environment, as the rank
tests do; the run that has to work under DD is a real two-rank job on CPU with
``_plan_stream_ranks.FakeModelParallel``.
"""
from __future__ import annotations

import numpy as np
import pytest

from sweep_tasks import TaskRunner, load_task
from test_plan_streaming_rank_sync import _ENCODED, _INR, _run
from test_task_fwi_plan_streaming import (
    _SYN_SHAPE,
    _build_plan_streaming_spec,
    _build_tiny_plan_streaming_fixture,
    _synthetic_stage_spec,
    _write,
)

_SKIPPED = ["illumination_precondition", "smooth_regularization",
            "freeze_water_layer"]


def _asking(key, tmp_path):
    """A setting of ``key`` that asks for the feature."""
    if key == "illumination_precondition":
        return {"enabled": True, "relative_epsilon": 1.0e-3}
    if key == "smooth_regularization":
        return {"weight": 1.0e-3}
    sb = tmp_path / "seabed_depth.npy"        # the tiny fixture's (ny, nx)
    np.save(sb, np.full((16, 24), 100.0))
    return {"enabled": True, "seabed_depth_path": str(sb)}


@pytest.mark.parametrize("key", _SKIPPED)
def test_dd_refuses_what_its_loop_skips(tmp_path, monkeypatch, key):
    from sweep_io.seismic_plan import SeismicPlan

    loaded = []
    load = SeismicPlan.load.__func__
    monkeypatch.setattr(SeismicPlan, "load", classmethod(
        lambda cls, path: loaded.append(path) or load(cls, path)))
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _build_plan_streaming_spec(tmp_path, fixture, epochs=1,
                                      extra={key: _asking(key, tmp_path)})
    path = _write(spec, tmp_path / f"{key}.yaml")

    for var, val in (("SWEEP_DD_ENABLE", "1"), ("SWEEP_DD_PY", "1"),
                     ("SWEEP_DD_PX", "1")):
        monkeypatch.setenv(var, val)
    result = TaskRunner().run(load_task(path))
    assert result.status.state == "failed"
    err = result.status.error or ""
    assert f"SWEEP_DD_ENABLE=1) does not support {key}:" in err, err
    assert f"Turn it off ({key}." in err and "or run without DD" in err, err
    assert loaded == []                          # refused before the plan loads

    monkeypatch.delenv("SWEEP_DD_ENABLE")        # the same YAML without DD
    result = TaskRunner().run(load_task(path))
    assert result.status.state == "success", result.status.error
    assert loaded


def test_dd_runs_with_those_blocks_off_and_the_inr_water_pin(tmp_path):
    """Two DD tiles (x-cut), an INR whose water pin comes from a seabed map, and
    the three blocks written out but switched off: the run goes through, and
    every tile and the saved model hold the water at water_vp_m_s."""
    nz, ny, nx = _SYN_SHAPE
    sb = tmp_path / "seabed_depth.npz"
    np.savez(sb, seabed_depth=np.full((ny, nx), 120.0, dtype=np.float32))
    spec = _synthetic_stage_spec(
        tmp_path, [{"epochs": 2}], source_encoding=_ENCODED,
        reparam=dict(_INR, mask_water_layer=True, seabed_depth_path=str(sb)),
        illumination_precondition={"enabled": False, "relative_epsilon": 1.0e-3},
        smooth_regularization={"weight": 0.0, "order": "second"},
        freeze_water_layer={"enabled": False})
    spec["show_every"] = 1
    tiles = _run(tmp_path, "dd", spec, 2, dd=True)
    water = slice(0, 2)                          # cell centres 25 and 75 m < 120 m
    for r in tiles:
        seen = [r[f"seen{e}"] for e in range(2)]   # this rank's tile, per epoch
        assert all(s.shape == (nz, ny, nx // 2) for s in seen)
        assert all(np.all(s[water] == 1500.0) for s in seen)
        assert not np.array_equal(seen[0][2:], seen[1][2:])   # stepped below it
    assert np.all(tiles[0]["inverted_vp"][water] == 1500.0)


_TINY_INR = {"kind": "velocity_inr", "hidden_features": 8, "hidden_layers": 1,
             "hash": {"enabled": False}, "lr": 1.0e-4}
# reparam options the DD loop never reads (its tile backward slabs by
# SWEEP_DD_RENDER_CHUNK), as the message names them
_UNREAD = {
    "reparam.backward_mode='single_step'": {"backward_mode": "single_step"},
    "reparam.backward_mode='two_pass_chunked'": {"backward_mode": "two_pass_chunked"},
    "reparam.backward_chunk_rows=32": {"backward_chunk_rows": 32},
}


@pytest.mark.parametrize("named", list(_UNREAD))
def test_dd_refuses_reparam_backward_options_it_never_reads(tmp_path, monkeypatch, named):
    from sweep_io.seismic_plan import SeismicPlan
    from sweep_tasks.tasks.fwi_plan_streaming import _dd_unread_reparam_options

    loaded = []
    load = SeismicPlan.load.__func__
    monkeypatch.setattr(SeismicPlan, "load", classmethod(
        lambda cls, path: loaded.append(path) or load(cls, path)))
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _build_plan_streaming_spec(
        tmp_path, fixture, epochs=1, extra={"reparam": dict(_TINY_INR, **_UNREAD[named])})
    path = _write(spec, tmp_path / "unread.yaml")

    for var, val in (("SWEEP_DD_ENABLE", "1"), ("SWEEP_DD_PY", "1"),
                     ("SWEEP_DD_PX", "1")):
        monkeypatch.setenv(var, val)
    result = TaskRunner().run(load_task(path))
    assert result.status.state == "failed"
    err = result.status.error or ""
    assert f"SWEEP_DD_ENABLE=1) never reads {named}:" in err, err
    assert "z-slabs of SWEEP_DD_RENDER_CHUNK" in err and "or run without DD" in err, err
    assert loaded == []                          # refused before the plan loads

    monkeypatch.delenv("SWEEP_DD_ENABLE")        # the same YAML without DD
    result = TaskRunner().run(load_task(path))
    assert result.status.state == "success", result.status.error
    assert loaded

    # the defaults, as a config_resolved.yaml writes them out, ask for nothing
    spec["reparam"] = dict(_TINY_INR, backward_mode="two_pass_full",
                           backward_chunk_rows=64)
    assert _dd_unread_reparam_options(load_task(_write(spec, tmp_path / "d.yaml"))) == []
