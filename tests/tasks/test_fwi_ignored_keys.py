"""The conventional and frequency-selection FWI paths refuse what they never read.

A stage is read differently on each path: the conventional loop reads every
key but ``boundary`` and ``frequency`` (frequency selection's), and frequency
selection never reads a stage's ``wavelet``, ``nt``, ``batch_size`` or
``bandpass``. Neither reads ``inr_lr_scale`` without ``reparam``, nor, with
stages, the top-level ``epochs``: the stages' epochs run. With ``obs.plan``
and no sampling the conventional path materialises the dataset from the plan
and reads only ``data_plan.dt_target_s``. A set value is refused at load; the
schema default, which ``config_resolved.yaml`` spells out, still loads.
"""
import pytest
import yaml

from sweep_tasks.schemas import DataPlanSpec, StageSpec
from test_freqsel_optimizer_settings import _REPARAM, _conventional, _doc, _validate

# Per stage key: a value a path could ignore, and the schema default.
_VALUES = {
    "wavelet": ({"kind": "ricker", "fm": 4.0}, None),
    "lr_scale": (0.5, 1.0),
    "inr_lr_scale": (0.5, 1.0),
    "optimizer_reset": (True, False),
    "dh_m": (25.0, None),
    "dt_s": (0.002, None),
    "nt": (60, None),
    "batch_size": (2, None),
    "bandpass": ({"lo_hz": 2.0, "hi_hz": 8.0}, None),
    "frequency": ({"coeff_shards": "s2.npz", "probe_samples": 100, "k_lo": 5,
                   "k_hi": 9, "steady_samples": 50}, None),
    "boundary": ({"storage": "cpu"}, None),
}
# What each path never reads with a reparam network (and without one, also
# inr_lr_scale).
_IGNORED = {"conventional": {"boundary", "frequency"},
            "freqsel": {"wavelet", "nt", "batch_size", "bandpass"}}
_PATH_NAME = {"conventional": "the conventional path",
              "freqsel": "source_encoding.mode='frequency_selection'"}


def _staged(path: str, stage: dict, **extra) -> dict:
    doc = _doc(epochs=2, stages=[{"epochs": 1}, {"epochs": 1, **stage}], **extra)
    return _conventional(doc) if path == "conventional" else doc


@pytest.mark.parametrize("path", sorted(_IGNORED))
@pytest.mark.parametrize("key", sorted(StageSpec.model_fields.keys() - {"epochs"}))
def test_a_stage_key_the_path_never_reads_is_refused(path, key):
    bad, default = _VALUES[key]        # a new StageSpec key needs a decision here
    reparam = {} if key == "lr_scale" else {"reparam": _REPARAM}
    if key in _IGNORED[path]:
        with pytest.raises(Exception, match=(
                rf"stages\[1\] sets .*\b{key}\b.*, which {_PATH_NAME[path]}.* "
                r"never reads")):
            _validate(_staged(path, {key: bad}, **reparam))
    else:
        _validate(_staged(path, {key: bad}, **reparam))
    _validate(_staged(path, {key: default}, **reparam))


@pytest.mark.parametrize("path", sorted(_IGNORED))
def test_inr_lr_scale_needs_a_reparam_network(path):
    """It scales the network's lr; a grid model has lr_scale."""
    with pytest.raises(Exception, match=r"stages\[1\] sets inr_lr_scale=0.5, "
                                        r"which .* never reads.*use lr_scale"):
        _validate(_staged(path, {"inr_lr_scale": 0.5}))
    _validate(_staged(path, {"inr_lr_scale": 1.0}))
    _validate(_staged(path, {"inr_lr_scale": 0.5}, reparam=_REPARAM))


@pytest.mark.parametrize("path", sorted(_IGNORED))
def test_top_level_epochs_must_be_the_stages_sum(path):
    """The stages' epochs run; the top-level one says so or is refused."""
    doc = _staged(path, {})
    doc["stages"] = [{"epochs": 3}, {"epochs": 4}]
    for wrong in (1, 6, 70):
        with pytest.raises(Exception, match=rf"epochs={wrong}, but a run with stages "
                                            r"runs the stages' epochs, 3 \+ 4 = 7"):
            _validate(dict(doc, epochs=wrong))
    assert _validate(dict(doc, epochs=7)).epochs == 7
    doc.pop("stages")
    _validate(dict(doc, epochs=70))      # without stages, epochs is what runs


def _materialised(**extra) -> dict:
    """A conventional spec whose obs is materialised from a plan (no sampling)."""
    doc = _conventional(_doc(**extra))
    doc["geometry"] = {"kind": "from_plan", "plan_path": "plan.npz"}
    doc["obs"] = {"plan": {"plan_path": "plan.npz", "cache_all": True}}
    return doc


# Per data_plan field: a value the plan path would ignore.
_DATA_PLAN = {"shot_start": 1, "shot_stop": 4, "shot_stride": 2,
              "shot_indices": [0, 2], "receiver_stride": 2, "offset_min_m": 100.0,
              "offset_max_m": 900.0, "abs_offset": False, "time_decimate": 2,
              "t_start_s": 0.1, "t_end_s": 0.5}


@pytest.mark.parametrize("field", sorted(DataPlanSpec.model_fields.keys()
                                         - {"dt_target_s"}))
def test_a_materialised_plan_reads_only_dt_target_s(field):
    value = _DATA_PLAN[field]          # a new DataPlanSpec field needs a decision
    with pytest.raises(Exception, match=(
            rf"obs.plan without sampling .* reads only data_plan.dt_target_s, "
            rf"so data_plan.{field}=.* would be ignored")):
        _validate(_materialised(data_plan={field: value}))
    # a static obs tensor (here synthetic) is where data_plan applies
    _validate(_conventional(_doc(data_plan={field: value})))


def test_a_materialised_plan_keeps_dt_target_s_and_its_record_reruns(tmp_path):
    from sweep_tasks._helpers.metadata import _dump_run_metadata

    spec = _validate(_materialised(
        data_plan={"dt_target_s": 0.002}, epochs=3,
        stages=[{"epochs": 1}, {"epochs": 2, "dh_m": 25.0, "inr_lr_scale": 0.5}],
        reparam=_REPARAM))
    _dump_run_metadata(spec, tmp_path)
    cfg = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())
    assert cfg["data_plan"]["shot_stride"] == 1 and cfg["epochs"] == 3
    assert [s["boundary"] for s in cfg["stages"]] == [None, None]
    assert _validate(cfg).data_plan.dt_target_s == 0.002
    # an older record with an ignored setting in it is refused, not re-run
    cfg["data_plan"]["shot_stride"] = 3
    with pytest.raises(Exception, match=r"data_plan.shot_stride=3 would be ignored"):
        _validate(cfg)
