"""What a frequency-selection FWI spec accepts, and what it must refuse."""
from typing import get_args

import numpy as np
import pytest

from sweep_tasks.schemas import LossSpec

def _freqsel_fwi_doc(**extra):
    doc = dict(
        task_type="fwi", output_dir="./x", task_id="t",
        grid={"dh": 12.5}, time={"dt": 0.001, "nt": 100},
        physics={"equation": "Acoustic"},
        init_model={"name": "vp", "constant": 1500.0, "shape": [20, 30]},
        source_encoding={
            "enabled": True, "mode": "frequency_selection",
            "frequency": {"coeff_shards": "s.npz", "probe_samples": 100,
                          "k_lo": 4, "k_hi": 8, "steady_samples": 50},
        },
        optimizer={"kind": "adam", "lr": 25.0}, epochs=1,
    )
    doc.update(extra)
    return doc


def test_freqsel_rejects_freeze_top_n_rows():
    """The freqsel loop pins water by value, so a row count would be a no-op."""
    from pydantic import TypeAdapter
    from sweep_tasks.schemas import TaskSpec

    ta = TypeAdapter(TaskSpec)
    ta.validate_python(_freqsel_fwi_doc())                    # baseline parses
    with pytest.raises(Exception, match="freeze_top_n_rows is not supported"):
        ta.validate_python(_freqsel_fwi_doc(freeze_top_n_rows=37))


def test_freqsel_makes_wavelet_geometry_obs_optional():
    """The doc above has none of the three and must still validate."""
    from pydantic import TypeAdapter
    from sweep_tasks.schemas import TaskSpec

    spec = TypeAdapter(TaskSpec).validate_python(_freqsel_fwi_doc())
    assert spec.wavelet is None and spec.geometry is None and spec.obs is None


def _validate(doc):
    from pydantic import TypeAdapter
    from sweep_tasks.schemas import TaskSpec

    return TypeAdapter(TaskSpec).validate_python(doc)


def _plain_doc(task):
    """A minimal per-shot ``task`` doc: no source encoding, so no freqsel."""
    doc = _freqsel_fwi_doc(
        wavelet={"kind": "ricker", "fm": 10.0},
        geometry={"kind": "line", "sources": {"step": 6, "depth": 2},
                  "receivers": {"step": 2, "depth": 2}},
        obs={"synthetic_from": {"name": "vp", "constant": 2000.0,
                                "shape": [20, 30]}})
    del doc["source_encoding"]
    model = doc.pop("init_model")
    if task == "fwi":
        doc["init_model"] = model
    elif task == "lsrtm":
        del doc["obs"]
        doc.update(task_type="lsrtm", background_model=model, true_model=model)
    else:
        del doc["optimizer"], doc["epochs"]
        doc.update(task_type="rtm", velocity_model=model)
    return doc


def test_freqsel_records_the_loss_it_runs(tmp_path):
    """Run records name the GCN misfit the freqsel path builds, not ``mse``.

    ``_dump_run_metadata`` is the dump ``TaskRunner.run`` writes at startup,
    and the only one a freqsel run gets.
    """
    import json

    import yaml

    from sweep_tasks._helpers.metadata import _dump_run_metadata

    spec = _validate(_freqsel_fwi_doc())
    assert spec.loss.kind == "steady_gcn"
    _dump_run_metadata(spec, tmp_path)
    cfg = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert cfg["loss"]["kind"] == "steady_gcn"
    assert meta["loss_kind"] == "steady_gcn"
    # the record re-runs as itself, with every loss option written out
    assert _validate(cfg).loss.kind == "steady_gcn"


@pytest.mark.parametrize(
    "kind", sorted(set(get_args(LossSpec.model_fields["kind"].annotation))
                   - {"steady_gcn"}))
def test_freqsel_rejects_any_other_loss_kind(kind):
    """An explicit ``mse`` included: the path never evaluates it."""
    with pytest.raises(Exception, match=f"loss.kind='{kind}' would be ignored"):
        _validate(_freqsel_fwi_doc(loss={"kind": kind}))


def test_freqsel_judges_loss_options_by_value():
    """A set option would be dropped; one at its default is what a resolved
    config spells out, and has to load."""
    spec = _validate(_freqsel_fwi_doc(
        loss={"kind": "steady_gcn", "time_mute_samples": 0}))
    assert spec.loss.kind == "steady_gcn"
    with pytest.raises(Exception, match="loss.time_mute_samples=200 would"):
        _validate(_freqsel_fwi_doc(loss={"time_mute_samples": 200}))


@pytest.mark.parametrize("task", ["fwi", "lsrtm", "rtm"])
def test_steady_gcn_needs_frequency_selection(task):
    """No other path can evaluate it (LSRTM would crash at its first misfit)."""
    doc = _plain_doc(task)
    assert _validate(doc).loss.kind == "mse"             # baseline parses
    with pytest.raises(Exception, match=f"{task}: loss.kind='steady_gcn'"):
        _validate({**doc, "loss": {"kind": "steady_gcn"}})
