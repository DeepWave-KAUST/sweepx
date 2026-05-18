"""Schema + dispatch smoke tests for the new ``task_type: rtm``.

Covers:
  * RTMSpec parses a minimal YAML and round-trips through dump/load.
  * The TaskRunner dispatcher table includes ``"rtm"`` and routes it to
    ``_run_rtm``.

No end-to-end runner exercise — RTM needs the c-backend's ``solver.rtm``
which requires a CUDA-capable build; that path is validated by the
ibex smoke run, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sweep_tasks import (
    RTMImagingSpec,
    RTMSpec,
    TaskRunner,
    dump_task,
    load_task,
    load_task_from_dict,
    new_template,
)


def _minimal_rtm_dict(tmp_path: Path) -> dict:
    # The path doesn't need to point at a real file for schema validation —
    # the loader just resolves it. We use a tmp path so YAML round-trips
    # produce a consistent absolute string.
    vp_path = tmp_path / "inverted_vp.npy"
    vp_path.write_bytes(b"placeholder")
    plan_path = tmp_path / "plan.npz"
    plan_path.write_bytes(b"placeholder")
    return {
        "task_type": "rtm",
        "output_dir": str(tmp_path / "rtm_runs"),
        "device": "cpu",
        "grid": {"dh": 12.5, "shape": [401, 2305]},
        "time": {"dt": 0.001, "nt": 6000},
        "wavelet": {"kind": "ricker", "fm": 8.0, "delay": 0.18, "scale": 1.0},
        "geometry": {
            "kind": "from_plan",
            "plan_path": str(plan_path),
            "dedupe": True,
        },
        "obs": {
            "plan": {"plan_path": str(plan_path), "cache_all": True},
        },
        "physics": {
            "equation": "Acoustic",
            "spatial_order": 8,
            "abcn": 20,
            "free_surface": True,
            "pml_type": "cpmlr",
            "source_type": ["h1"],
            "receiver_type": ["h1"],
        },
        "backend": {"impl": "c", "use_ckpt": False},
        "velocity_model": {"name": "vp", "path": str(vp_path)},
        "loss": {"kind": "trace_cosine", "trace_cosine_demean": True},
        "imaging": {
            "shots_per_batch": 1,
            "filter_lowcut_hz": 3.0,
            "filter_highcut_hz": 25.0,
            "filter_order": 4,
            "filter_padtype": "odd",
            "illumination_epsilon": 1.0e-6,
            "normalize_by_illumination": True,
            "save_per_shot": False,
            "live_update_every_batches": 20,
            "loss_kind": "trace_cosine",
            "trace_cosine_demean": True,
        },
    }


def test_rtm_minimal_dict_parses(tmp_path):
    spec_dict = _minimal_rtm_dict(tmp_path)
    spec = load_task_from_dict(spec_dict, base_dir=tmp_path)
    assert isinstance(spec, RTMSpec)
    assert spec.task_type == "rtm"
    assert spec.imaging.shots_per_batch == 1
    assert spec.imaging.loss_kind == "trace_cosine"
    assert spec.velocity_model.name == "vp"
    assert spec.local_model_window is None


def test_rtm_yaml_round_trip(tmp_path):
    spec_dict = _minimal_rtm_dict(tmp_path)
    yaml_path = tmp_path / "rtm.yaml"
    yaml_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))

    spec = load_task(yaml_path)
    assert isinstance(spec, RTMSpec)

    dumped_path = tmp_path / "rtm_dumped.yaml"
    dump_task(spec, dumped_path)
    spec2 = load_task(dumped_path)
    assert type(spec) is type(spec2)
    assert spec.task_type == spec2.task_type == "rtm"
    assert spec.imaging.filter_lowcut_hz == spec2.imaging.filter_lowcut_hz
    assert spec.imaging.shots_per_batch == spec2.imaging.shots_per_batch


def test_rtm_template_round_trip(tmp_path):
    """The shipped ``new_template('rtm')`` validates."""
    template = new_template("rtm")
    yaml_path = tmp_path / "rtm_template.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert isinstance(spec, RTMSpec)
    assert spec.task_type == "rtm"


def test_rtm_dispatcher_registered(tmp_path):
    """The TaskRunner.run() dispatcher must include 'rtm' → _run_rtm.

    We can't exercise _run_rtm end-to-end without a GPU + plan files, so the
    test inspects the dispatcher contract directly. This mirrors how the FWI
    smoke + e2e tests are split.
    """
    runner = TaskRunner()
    # The dispatcher is built inside .run(); we test it indirectly by asserting
    # the bound method exists on the runner.
    assert hasattr(runner, "_run_rtm"), (
        "TaskRunner._run_rtm is missing — dispatcher will fail at run-time."
    )
    # Touch the underlying helper too so a typo would surface here, not in
    # an end-to-end run.
    from sweep_tasks.runner import _crop_padded_volume_to_model, _save_rtm_qc_pngs
    assert callable(_crop_padded_volume_to_model)
    assert callable(_save_rtm_qc_pngs)

    # Also exercise the registry path that the runner uses to look up the spec
    # class: it must include "rtm" mapped to RTMSpec.
    from sweep_tasks.registry import TASK_TYPES
    assert TASK_TYPES["rtm"] is RTMSpec


def test_rtm_velocity_model_required(tmp_path):
    spec_dict = _minimal_rtm_dict(tmp_path)
    spec_dict.pop("velocity_model")
    with pytest.raises(ValidationError):
        load_task_from_dict(spec_dict, base_dir=tmp_path)


def test_rtm_imaging_defaults():
    """RTMImagingSpec's defaults survive a no-args construction."""
    spec = RTMImagingSpec()
    assert spec.shots_per_batch == 1
    assert spec.filter_lowcut_hz is None
    assert spec.filter_highcut_hz is None
    assert spec.illumination_epsilon == 1.0e-6
    assert spec.normalize_by_illumination is True
    assert spec.save_per_shot is False
    assert spec.live_update_every_batches == 10
    assert spec.loss_kind == "trace_cosine"


def test_rtm_local_window_shorthand(tmp_path):
    """The ``true`` / ``false`` shorthand for local_model_window mirrors FWI."""
    spec_dict = _minimal_rtm_dict(tmp_path)
    spec_dict["local_model_window"] = True
    spec = load_task_from_dict(spec_dict, base_dir=tmp_path)
    assert spec.local_model_window is not None
    assert spec.local_model_window.enabled is True

    spec_dict["local_model_window"] = False
    spec2 = load_task_from_dict(spec_dict, base_dir=tmp_path)
    assert spec2.local_model_window is None
