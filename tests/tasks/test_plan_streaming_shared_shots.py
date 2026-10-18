"""``obs.plan.sampling.shared_shots_per_iter`` sets the shots per gather.

The field was never passed to the sampler: every iteration used all the shots
the picked nodes share (after ``source_lines_per_group`` /
``max_traces_per_sourceline``), whatever it said, and its default of 1 meant
nothing. It is now a cap: at most that many shots per node gather and
iteration, a random subset of the same shots for every node of the batch, or
every shot when fewer are there (said once in the log). With
``per_crg_independent`` it caps each node's own receivers. ``null``, the new
default, keeps them all, as before; ``1``, the old default an older run's
config_resolved.yaml records, is refused.

The fixture has 4 nodes that all record the same 12 shots.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from sweepx.tasks import TaskRunner, load_task
from test_plan_streaming_schema_guards import _plan_streaming_doc, _validate
from test_task_fwi_plan_streaming import (
    _SYN_NODES,
    _SYN_SHOTS,
    _synthetic_stage_spec,
    _write,
)

_E = 2
_ENCODED = {"enabled": True, "min_coverage": 0, "sign_seed": 7}
_ALL = {tuple(c) for c in np.rint(_SYN_SHOTS / 50.0).astype(np.int64)}


def _spy_geometry(monkeypatch):
    """Record the (sources, receivers) of every solver call."""
    from sweep.propagator.torch import PropTorch

    calls = []
    forward = PropTorch.forward

    def spy(self, wavelet, sources, receivers, *args, **kwargs):
        calls.append((np.array(sources), np.array(receivers)))
        return forward(self, wavelet, sources, receivers, *args, **kwargs)

    monkeypatch.setattr(PropTorch, "forward", spy)
    return calls


def _shots(receivers) -> set:
    """The shot cells of one call; a per-shot batch shares them across rows."""
    rec = np.asarray(receivers)
    for row in rec[1:]:
        np.testing.assert_array_equal(row, rec[0])
    return {tuple(c) for c in rec[0]}


def _run(tmp_path, spec, tag, cap, calls):
    run = copy.deepcopy(spec)
    run["task_id"] = tag
    run["obs"]["plan"]["sampling"]["shared_shots_per_iter"] = cap
    calls.clear()
    result = TaskRunner().run(load_task(_write(run, tmp_path / f"{tag}.yaml")))
    assert result.status.state == "success", result.status.error
    return [(s.copy(), r.copy()) for s, r in calls]


@pytest.mark.parametrize("path", ["encoded", "per_shot"])
def test_the_cap_is_the_number_of_shots_per_iteration(tmp_path, monkeypatch, path):
    """5 and 8 of the 12 shared shots, every iteration; the same nodes as with
    no cap, which keeps all 12."""
    extra = {"source_encoding": _ENCODED} if path == "encoded" else {}
    spec = _synthetic_stage_spec(tmp_path, [{"epochs": _E}], **extra)
    calls = _spy_geometry(monkeypatch)
    seen = {}
    for cap in (5, 8, None):
        seen[cap] = run = _run(tmp_path, spec, f"cap{cap}", cap, calls)
        assert len(run) == _E, (cap, len(run))          # one solve per iteration
        for _, rec in run:
            shots = _shots(rec)
            assert shots <= _ALL
            assert len(shots) == (len(_ALL) if cap is None else cap), (cap, len(shots))
    for cap in (5, 8):           # the cap thins the shots, never the node draw
        for (src, _), (src_all, _) in zip(seen[cap], seen[None]):
            np.testing.assert_array_equal(src, src_all)
    assert [_shots(r) for _, r in seen[5]] != [_shots(r) for _, r in seen[8]]


def test_the_cap_applies_to_each_crg_node(tmp_path, monkeypatch):
    """per_crg_independent shares no shots: each node's own receivers (one per
    grid cell, here all 12) are capped instead."""
    spec = _synthetic_stage_spec(tmp_path, [{"epochs": _E}])
    spec["obs"]["plan"]["sampling"]["per_crg_independent"] = True
    calls = _spy_geometry(monkeypatch)
    seen = {}
    for cap in (5, 8, None):
        seen[cap] = run = _run(tmp_path, spec, f"crg{cap}", cap, calls)
        assert len(run) == _E * len(_SYN_NODES)          # a solve per node
        for _, rec in run:
            shots = _shots(rec)                          # (1, n, 3): one node
            assert shots <= _ALL
            assert len(shots) == (len(_ALL) if cap is None else cap), (cap, len(shots))
    assert [_shots(r) for _, r in seen[5]] != [_shots(r) for _, r in seen[8]]


def test_a_cap_above_the_shots_drawn_uses_them_all_and_says_so(tmp_path, monkeypatch,
                                                             capsys):
    spec = _synthetic_stage_spec(tmp_path, [{"epochs": _E}])
    calls = _spy_geometry(monkeypatch)
    run = _run(tmp_path, spec, "cap50", 50, calls)
    assert [len(_shots(r)) for _, r in run] == [len(_ALL)] * _E
    out = capsys.readouterr().out
    assert out.count("sampling.shared_shots_per_iter=50:") == 1, out[-3000:]
    assert f"share only {len(_ALL)} shots" in out


def test_the_cap_keeps_one_shot_set_across_the_nodes(tmp_path):
    """Row j of every node's gather is still the same physical shot."""
    from sweepx.io.seismic_plan import SeismicPlan, sample_shared_shots_from_plan
    from sweepx.tasks.tasks.fwi_plan_streaming import _cap_shared_shots

    spec = _synthetic_stage_spec(tmp_path, [{"epochs": 1}])
    plan = SeismicPlan.load(spec["obs"]["plan"]["plan_path"])
    batch = sample_shared_shots_from_plan(plan, np.random.default_rng(3),
                                          batch_size=len(_SYN_NODES))
    capped, n = _cap_shared_shots(batch, 5, np.random.default_rng(4))
    assert n == len(_ALL) and capped.n_shared == 5
    xyz = [plan.row_source_xyz[r] for r in capped.rows_per_group]
    for other in xyz[1:]:
        np.testing.assert_array_equal(other, xyz[0])
    np.testing.assert_array_equal(capped.shared_shot_xyz_m, xyz[0])
    same, n = _cap_shared_shots(batch, None, np.random.default_rng(4))
    assert same is batch and n == len(_ALL)


def test_null_is_the_default_and_one_is_refused(tmp_path):
    """null keeps every shot; 1, the default before the field was read, which
    an older run's config_resolved.yaml records, is refused."""
    from sweepx.tasks._helpers.metadata import _dump_run_metadata

    doc = _plan_streaming_doc()
    doc["obs"]["plan"]["sampling"] = {}
    spec = _validate(doc)
    assert spec.obs.plan.sampling.shared_shots_per_iter is None
    _dump_run_metadata(spec, tmp_path)
    cfg = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())
    assert cfg["obs"]["plan"]["sampling"]["shared_shots_per_iter"] is None
    _validate(cfg)                                # a run's own record loads again
    cfg["obs"]["plan"]["sampling"]["shared_shots_per_iter"] = 1
    with pytest.raises(Exception, match=r"(?s)shared_shots_per_iter: 1 is "
                                        r"refused.*Write null"):
        _validate(cfg)
    cfg["obs"]["plan"]["sampling"]["shared_shots_per_iter"] = 2
    _validate(cfg)
