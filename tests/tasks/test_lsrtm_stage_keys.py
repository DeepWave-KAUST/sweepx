"""LSRTM refuses the stage keys it does not read.

The LSRTM stage loop reads only ``epochs``, ``lr_scale`` and ``wavelet``: one
grid, time axis, shot batch, solver and optimizer serve every stage. Any other
stage key is refused at load instead of being dropped. Keys are compared by
value with the schema default, so a ``config_resolved.yaml`` (every key
spelled out at its default) still loads.
"""
import itertools
from pathlib import Path

import pydantic
import pytest
import yaml

import sweepx.tasks
from sweepx.tasks.schemas import LSRTMSpec, StageSpec

_READ = ("epochs", "lr_scale", "wavelet")
# Per key: a value LSRTM would have ignored, and the schema default.
_VALUES = {
    "inr_lr_scale": (0.5, 1.0),
    "optimizer_reset": (True, False),
    "dh_m": (20.0, None),
    "dt_s": (0.002, None),
    "nt": (60, None),
    "batch_size": (2, None),
    "bandpass": ({"lo_hz": 2.0, "hi_hz": 8.0}, None),
    "frequency": ({"synthesize_from_true": True}, None),
    "boundary": ({"storage": "cpu"}, None),
}


def _spec(stage: dict) -> dict:
    return {
        "task_type": "lsrtm",
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 120},
        "wavelet": {"kind": "ricker", "fm": 10.0},
        "geometry": {"kind": "line", "sources": {"step": 24, "depth": 2},
                     "receivers": {"step": 4, "depth": 4}},
        "physics": {"equation": "AcousticBorn", "receiver_type": ["sh1"]},
        "background_model": {"name": "vp", "constant": 2500.0, "shape": [48, 48]},
        "true_model": {"name": "vp", "constant": 2500.0, "shape": [48, 48]},
        "optimizer": {"kind": "adam", "lr": 0.08},
        "epochs": 4,
        "stages": [{"epochs": 2}, {"epochs": 2, **stage}],
    }


@pytest.mark.parametrize("key", sorted(set(StageSpec.model_fields) - set(_READ)))
def test_lsrtm_refuses_a_stage_key_it_does_not_read(key):
    bad, default = _VALUES[key]          # a new StageSpec key needs a decision here
    with pytest.raises(pydantic.ValidationError,
                       match=rf"stages\[1\] sets .*\b{key}\b.*which LSRTM does not support"):
        LSRTMSpec(**_spec({key: bad}))
    # The default written out, as config_resolved.yaml writes every key, loads;
    # and so does the full dump of the result.
    spec = LSRTMSpec(**_spec({key: default}))
    assert LSRTMSpec(**yaml.safe_load(yaml.safe_dump(spec.model_dump(mode="json")))) == spec


def test_lsrtm_template_stage_example_uses_only_keys_lsrtm_reads():
    path = Path(sweepx.tasks.__file__).parent / "templates" / "lsrtm.yaml"
    lines = path.read_text().splitlines()
    start = lines.index("# stages:") + 1
    example = [line[1:] for line in
               itertools.takewhile(lambda s: s.startswith("#   - "), lines[start:])]
    assert example, "the template's commented stage example is gone"
    cfg = yaml.safe_load("\n".join(lines))
    cfg["stages"] = yaml.safe_load("\n".join(["stages:", *example]))["stages"]
    LSRTMSpec(**cfg)
    assert set().union(*cfg["stages"]) <= set(_READ)
