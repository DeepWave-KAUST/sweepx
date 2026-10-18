"""What a plan-streaming FWI spec (obs.plan.sampling) must refuse.

The plan-streaming loop never reads the keys below, so a run that set them
used to go ahead without them. A set value is refused; the schema default --
what config_resolved.yaml spells out for every key a run did not set -- has
to keep loading, and the check stays off every other path.
"""
import copy
import re

import pytest
import yaml
from pydantic import TypeAdapter

from sweepx.tasks.schemas import TaskSpec


def _validate(doc):
    return TypeAdapter(TaskSpec).validate_python(doc)


def _plan_streaming_doc(**extra):
    doc = dict(
        task_type="fwi", output_dir="./x", task_id="t",
        grid={"dh": 50.0}, time={"dt": 0.002, "nt": 80},
        wavelet={"kind": "ricker", "fm": 5.0},
        geometry={"kind": "from_plan", "plan_path": "plan.npz",
                  "rotation_metadata": "rotation.json"},
        physics={"equation": "Acoustic3D"},
        init_model={"name": "vp", "constant": 2500.0, "shape": [8, 8, 8]},
        obs={"plan": {"plan_path": "plan.npz",
                      "sampling": {"shared_shots_per_iter": 2}}},
        optimizer={"kind": "adam", "lr": 1.0}, epochs=2,
    )
    doc.update(extra)
    return doc


def _resolved(tmp_path, **extra):
    """config_resolved.yaml of a plan-streaming run, as a dict."""
    from sweepx.tasks._helpers.metadata import _dump_run_metadata

    _dump_run_metadata(_validate(_plan_streaming_doc(**extra)), tmp_path)
    return yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())


# (key, a value the path would ignore, values that ask for nothing: the
# schema default first)
_KEYS = [
    ("scheduler", {"kind": "step", "step_size": 1}, [{"kind": "constant"}]),
    ("resume", True, [False]),
    ("resume_from", "earlier_task", [None]),
    ("train_shot_batchsize", 2, [None]),
    ("save_illumination", True, [False]),
    ("grad_smooth", {"sigma_z_cells": 2.0}, [None, {"enabled": False}]),
    ("diffusion_prior", {"ckpt_path": "prior.pt"},
     [None, {"enabled": False, "ckpt_path": "prior.pt"}]),
    ("local_model_window", True, [None, False, {"enabled": False}]),
    ("data_plan", {"shot_stride": 2}, [None]),
    ("modeling_override", {"wavelet": {"kind": "ricker", "fm": 3.0}}, [None]),
]


@pytest.mark.parametrize("key, value, unset", _KEYS, ids=[k[0] for k in _KEYS])
def test_plan_streaming_refuses_a_key_it_never_reads(tmp_path, key, value, unset):
    cfg = _resolved(tmp_path)
    assert cfg[key] == unset[0]                  # what a run records unset ...
    _validate(cfg)                               # ... loads again
    for v in unset[1:]:
        _validate(_plan_streaming_doc(**{key: v}))
    named = rf"(?s)plan-streaming path.*{key}|{key}.*plan-streaming path"
    with pytest.raises(Exception, match=named):
        _validate({**cfg, key: value})
    conventional = _plan_streaming_doc(**{key: value})
    conventional["obs"]["plan"].pop("sampling")  # another path: not this check
    if key == "data_plan":   # applied to a static obs; plan obs reads dt_target_s only
        conventional["obs"] = {"npy_path": "obs.npy"}
    _validate(conventional)


def test_plan_streaming_refuses_a_stage_boundary_override(tmp_path):
    """Only frequency selection merges a stage's boundary: block."""
    cfg = _resolved(tmp_path, stages=[{"epochs": 1}, {"epochs": 1}])
    assert [st["boundary"] for st in cfg["stages"]] == [None, None]
    _validate(cfg)
    cfg["stages"][1]["boundary"] = {"storage": "cpu"}
    with pytest.raises(Exception, match=r"never reads stages\[1\]\.boundary"):
        _validate(cfg)


_ENCODED = {"enabled": True, "min_coverage": 0, "sign_seed": 7}
_FREQ = {"coeff_shards": "shards/*.npz"}
_STAGES = [{"epochs": 1}, {"epochs": 1}]
_BAND = {"lo_hz": 2.0, "hi_hz": 8.0}
_BANDED = [{"epochs": 1}, {"epochs": 1, "bandpass": _BAND}]

# (where, a value the path would ignore, values that ask for nothing: the
# schema default first; the run's other settings; and whether the
# conventional path -- obs.plan without sampling -- reads the key)
_NESTED = [
    (("obs", "plan", "n_receivers_per_shot"), 16, [None], {}, True),
    (("obs", "plan", "receiver_select"), "random", ["nearest"], {}, True),
    (("obs", "plan", "max_shots"), 10, [None], {}, True),
    (("obs", "plan", "min_receivers"), 2, [8], {}, True),
    (("obs", "plan", "materialize_seed"), 3, [0], {}, True),
    (("obs", "plan", "sampling", "prefetch_factor"), 4, [2], {}, False),
    (("obs", "plan", "sampling", "per_crg_independent"), True, [False],
     {"source_encoding": _ENCODED}, False),
    (("source_encoding", "frequency"), _FREQ, [None],
     {"source_encoding": _ENCODED}, False),
    (("stages", 1, "frequency"), _FREQ, [None], {"stages": _STAGES}, False),
    # of a bandpass the loop reads lo_hz, hi_hz and order only
    (("bandpass", "target"), "wavelet", ["syn"], {"bandpass": _BAND}, False),
    (("bandpass", "padtype"), "even", ["odd"], {"bandpass": _BAND}, False),
    (("stages", 1, "bandpass", "target"), "wavelet", ["syn"], {"stages": _BANDED}, True),
    (("stages", 1, "bandpass", "padtype"), None, ["odd"], {"stages": _BANDED}, True),
]


def _name(where):
    out = ""
    for k in where:
        out += f"[{k}]" if isinstance(k, int) else (f".{k}" if out else k)
    return out


def _at(doc, where, value=None, *, put=False):
    for k in where[:-1]:
        doc = doc[k]
    if put:
        doc[where[-1]] = value
    return doc[where[-1]]


@pytest.mark.parametrize("where, value, unset, extra, conventional_reads", _NESTED,
                         ids=[_name(k[0]) for k in _NESTED])
def test_plan_streaming_refuses_a_nested_key_it_never_reads(
        tmp_path, where, value, unset, extra, conventional_reads):
    cfg = _resolved(tmp_path, **copy.deepcopy(extra))
    assert _at(cfg, where) == unset[0]           # what a run records unset ...
    _validate(cfg)                               # ... loads again
    _at(cfg, where, value, put=True)
    with pytest.raises(Exception, match=rf"(?s)plan-streaming path.*never reads.*"
                                        rf"{re.escape(_name(where))}"):
        _validate(cfg)
    if conventional_reads:                       # another path: not this check
        conventional = _plan_streaming_doc(**copy.deepcopy(extra))
        _at(conventional, where, value, put=True)
        conventional["obs"]["plan"].pop("sampling")
        _validate(conventional)


def test_plan_streaming_reads_per_crg_and_frequency_when_encoding_is_off(tmp_path):
    """per_crg_independent is what the per-shot loop (encoding off) runs on, and
    a frequency block in a source_encoding block switched off asks for nothing."""
    doc = _plan_streaming_doc(source_encoding=dict(_ENCODED, enabled=False,
                                                   frequency=_FREQ))
    doc["obs"]["plan"]["sampling"]["per_crg_independent"] = True
    _validate(doc)
    doc["source_encoding"]["enabled"] = True
    with pytest.raises(Exception, match=r"source_encoding\.frequency.*"
                                        r"per_crg_independent=True"):
        _validate(doc)


def _sampled(doc, **sampling):
    doc["obs"]["plan"]["sampling"].update(sampling)
    return doc


_NOT_APPLIED = (r"(?s)plan-streaming path.*never reads.*{}.*applied only by the "
                r"receiver-first sampler")


def test_plan_streaming_min_coverage_needs_the_receiver_first_sampler(tmp_path):
    """min_coverage reaches the receiver-first sampler's node index only: the
    random shared-shot and per-CRG samplers draw from every node in the grid,
    so a value there is refused, and so is any in a source_encoding block
    switched off, which has no effect; it used to reach that index anyway."""
    cfg = _resolved(tmp_path)
    assert cfg["obs"]["plan"]["sampling"]["min_coverage"] == 0
    _validate(cfg)
    for name, doc in (
            ("obs.plan.sampling.min_coverage=5",
             _sampled(_plan_streaming_doc(), min_coverage=5)),
            ("obs.plan.sampling.min_coverage=5",
             _sampled(_plan_streaming_doc(), min_coverage=5, per_crg_independent=True)),
            ("source_encoding.min_coverage=5",
             _plan_streaming_doc(source_encoding=dict(_ENCODED, min_coverage=5)))):
        with pytest.raises(Exception, match=_NOT_APPLIED.format(re.escape(name))):
            _validate(doc)
        _validate(_sampled(doc, receiver_first=True, per_crg_independent=False))
    off = dict(_ENCODED, enabled=False, min_coverage=5)
    with pytest.raises(Exception, match=r"(?s)never reads.*source_encoding\."
                                        r"min_coverage=5.*switched off has no effect"):
        _validate(_sampled(_plan_streaming_doc(source_encoding=off), receiver_first=True))
    _validate(_plan_streaming_doc(source_encoding=dict(off, min_coverage=0)))
    conventional = _plan_streaming_doc(source_encoding=off)
    conventional["obs"]["plan"].pop("sampling")  # another path: not this check
    _validate(conventional)


def test_plan_streaming_per_crg_never_uses_receiver_first():
    doc = _sampled(_plan_streaming_doc(), per_crg_independent=True, receiver_first=True)
    with pytest.raises(Exception, match=r"(?s)never reads.*obs\.plan\.sampling\."
                                        r"receiver_first=True.*draws its nodes at random"):
        _validate(doc)
    _validate(_sampled(_plan_streaming_doc(), per_crg_independent=True))
    _validate(_sampled(_plan_streaming_doc(), receiver_first=True))
    # with encoding on, per_crg_independent is what is refused, not receiver_first
    with pytest.raises(Exception, match=r"per_crg_independent=True") as err:
        _validate(_sampled(_plan_streaming_doc(source_encoding=_ENCODED),
                           per_crg_independent=True, receiver_first=True))
    assert "receiver_first=True" not in str(err.value)


# what each block skipped under backward_mode single_step asks for, and its off form
_LEAF = {
    "illumination_precondition": ({"enabled": True}, {"enabled": False}),
    "smooth_regularization": ({"weight": 1.0e-3}, {"weight": 0.0}),
}
_SINGLE_STEP = {"kind": "velocity_inr", "backward_mode": "single_step"}


def _off_text(off):
    """The off value as the message writes it: ``enabled: false`` / ``weight: 0``."""
    return "enabled: false" if "enabled" in off else "weight: 0"


@pytest.mark.parametrize("key", list(_LEAF))
def test_plan_streaming_single_step_refuses_what_the_loop_skips(tmp_path, key):
    """single_step leaves the loop no velocity gradient, so it skips these: a
    block that asks for one is refused, one switched off (or the two-pass
    modes, or another path) is not."""
    on, off = _LEAF[key]
    cfg = _resolved(tmp_path, reparam=_SINGLE_STEP)
    assert cfg[key] is None
    _validate(cfg)
    _validate({**cfg, key: off})
    with pytest.raises(Exception, match=rf"(?s)backward_mode='single_step' does not "
                                        rf"support {key}.*two_pass_full or "
                                        rf"two_pass_chunked.*{key}\.{_off_text(off)}"):
        _validate({**cfg, key: on})
    for mode in ("two_pass_full", "two_pass_chunked"):
        _validate(_plan_streaming_doc(reparam=dict(_SINGLE_STEP, backward_mode=mode),
                                      **{key: on}))
    conventional = _plan_streaming_doc(reparam=_SINGLE_STEP, **{key: on})
    conventional["obs"]["plan"].pop("sampling")  # another path: not this check
    _validate(conventional)

