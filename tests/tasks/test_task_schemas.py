"""Schema round-trip and validation tests for sweep_tasks."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sweep_tasks import (
    ForwardSpec,
    FWISpec,
    IntrospectSpec,
    LSRTMSpec,
    RTMSpec,
    WavefieldSpec,
    dump_task,
    load_task,
    new_template,
)


@pytest.mark.parametrize(
    "task_type, expected_cls",
    [
        ("introspect", IntrospectSpec),
        ("forward", ForwardSpec),
        ("wavefield", WavefieldSpec),
        ("fwi", FWISpec),
        ("lsrtm", LSRTMSpec),
        ("rtm", RTMSpec),
    ],
)
def test_template_yaml_round_trip(tmp_path, task_type, expected_cls):
    template = new_template(task_type)
    yaml_path = tmp_path / f"{task_type}.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))

    spec = load_task(yaml_path)
    assert isinstance(spec, expected_cls)
    assert spec.task_type == task_type


def test_dump_then_load_idempotent(tmp_path):
    template = new_template("forward")
    yaml_path = tmp_path / "fwd.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec1 = load_task(yaml_path)

    dump_target = tmp_path / "fwd_dumped.yaml"
    dump_task(spec1, dump_target)
    spec2 = load_task(dump_target)

    assert type(spec1) is type(spec2)
    assert spec1.task_type == spec2.task_type
    assert spec1.physics.equation == spec2.physics.equation
    assert spec1.time.nt == spec2.time.nt


def test_invalid_nt_rejected(tmp_path):
    template = new_template("forward")
    template["time"]["nt"] = 0  # must be >= 1
    yaml_path = tmp_path / "bad_nt.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_missing_required_field_rejected(tmp_path):
    template = new_template("fwi")
    template.pop("init_model")
    yaml_path = tmp_path / "no_init.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_extra_field_rejected(tmp_path):
    template = new_template("forward")
    template["unknown_key"] = 1
    yaml_path = tmp_path / "extra.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_obs_requires_exactly_one_source(tmp_path):
    template = new_template("fwi")
    template["obs"] = {
        "synthetic_from": {"name": "vp", "path": "true.npy"},
        "npy_path": "obs.npy",
    }
    yaml_path = tmp_path / "bad_obs.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_eager_options_rejected_when_impl_is_c(tmp_path):
    template = new_template("forward")
    template["backend"] = {
        "impl": "c",
        "eager_options": {"use_compile": True},  # conflict
    }
    yaml_path = tmp_path / "bad_backend.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


# ---------- new sub-structures: wavelet / geometry / modeling_override ----

def test_from_npy_wavelet_round_trip(tmp_path):
    template = new_template("forward")
    template["wavelet"] = {"kind": "from_npy", "path": "wavelet.npy", "scale": 0.5}
    yaml_path = tmp_path / "fw_npy.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.wavelet.kind == "from_npy"
    assert spec.wavelet.scale == 0.5


def test_explicit_geometry_shared_receivers(tmp_path):
    template = new_template("forward")
    template["geometry"] = {
        "kind": "explicit",
        "sources": [[10, 4], [50, 4]],
        "receivers": [[8, 8], [9, 8], [10, 8]],  # shared
    }
    yaml_path = tmp_path / "expl.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.geometry.kind == "explicit"
    assert len(spec.geometry.sources) == 2


def test_explicit_geometry_per_shot_receivers(tmp_path):
    template = new_template("forward")
    template["geometry"] = {
        "kind": "explicit",
        "sources": [[10, 4], [50, 4]],
        "receivers": [
            [[8, 8], [9, 8]],
            [[10, 8], [11, 8]],
        ],
    }
    yaml_path = tmp_path / "expl_pershot.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.geometry.kind == "explicit"


def test_explicit_geometry_per_shot_length_mismatch_rejected(tmp_path):
    template = new_template("forward")
    template["geometry"] = {
        "kind": "explicit",
        "sources": [[10, 4], [50, 4], [90, 4]],
        "receivers": [
            [[8, 8]],
            [[10, 8]],
        ],
    }
    yaml_path = tmp_path / "expl_bad.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_from_file_geometry_round_trip(tmp_path):
    template = new_template("forward")
    template["geometry"] = {
        "kind": "from_file",
        "sources_file": "sources.npy",
        "receivers_file": "receivers.npy",
    }
    yaml_path = tmp_path / "fromfile.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.geometry.kind == "from_file"


def test_modeling_override_round_trip(tmp_path):
    template = new_template("fwi")
    template["modeling_override"] = {
        "wavelet": {"kind": "ricker", "fm": 3.0, "delay": 0.5, "scale": 1.0},
    }
    yaml_path = tmp_path / "fwi_override.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.modeling_override is not None
    assert spec.modeling_override.wavelet.kind == "ricker"
    assert spec.modeling_override.wavelet.fm == 3.0


def test_modeling_override_empty_rejected(tmp_path):
    template = new_template("fwi")
    template["modeling_override"] = {}  # neither wavelet nor geometry
    yaml_path = tmp_path / "fwi_empty_override.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_unknown_wavelet_kind_rejected(tmp_path):
    template = new_template("forward")
    template["wavelet"] = {"kind": "morlet", "fm": 5.0}  # not supported
    yaml_path = tmp_path / "morlet.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


# ---------- backend / memory template scaffolding -----------------------

def test_new_template_eager_compile(tmp_path):
    template = new_template("fwi", backend="eager", compile=True)
    yaml_path = tmp_path / "eager_compile.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.backend.impl == "eager"
    assert spec.backend.eager_options is not None
    assert spec.backend.eager_options.use_compile is True


def test_new_template_cuda_full(tmp_path):
    template = new_template("fwi", backend="c", memory="full")
    yaml_path = tmp_path / "cuda_full.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.backend.impl == "c"
    assert spec.backend.cuda_options.memory is None


def test_new_template_cuda_boundary_gpu(tmp_path):
    template = new_template("fwi", backend="c", memory="boundary", storage="gpu")
    yaml_path = tmp_path / "cuda_bs_gpu.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    mem = spec.backend.cuda_options.memory
    assert mem.strategy == "boundary"
    assert mem.boundary.storage == "gpu"
    # gpu storage forbids transfer_interval/pinned_memory by design
    assert mem.boundary.transfer_interval is None
    assert mem.boundary.pinned_memory is None


def test_new_template_cuda_boundary_cpu(tmp_path):
    template = new_template("fwi", backend="c", memory="boundary", storage="cpu")
    yaml_path = tmp_path / "cuda_bs_cpu.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    mem = spec.backend.cuda_options.memory
    assert mem.strategy == "boundary"
    assert mem.boundary.storage == "cpu"
    assert mem.boundary.transfer_interval == 10
    assert mem.boundary.pinned_memory is True


def test_new_template_cuda_boundary_disk(tmp_path):
    template = new_template("fwi", backend="c", memory="boundary", storage="disk")
    yaml_path = tmp_path / "cuda_bs_disk.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    mem = spec.backend.cuda_options.memory
    assert mem.strategy == "boundary"
    assert mem.boundary.storage == "disk"
    assert mem.boundary.disk_async_read is True


def test_new_template_cuda_ckpt_chunk(tmp_path):
    template = new_template("fwi", backend="c", memory="ckpt", storage="gpu")
    yaml_path = tmp_path / "cuda_ckpt.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    mem = spec.backend.cuda_options.memory
    assert mem.strategy == "ckpt"
    assert mem.ckpt.mode == "chunk"
    assert mem.ckpt.chunks == 64
    assert mem.ckpt.storage == "gpu"


def test_new_template_rejects_eager_with_memory():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="--memory"):
        new_template("fwi", backend="eager", memory="boundary")


def test_new_template_rejects_c_with_compile():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="--compile"):
        new_template("fwi", backend="c", memory="full", compile=True)


def test_new_template_rejects_ckpt_disk():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="(?i)ckpt"):
        new_template("fwi", backend="c", memory="ckpt", storage="disk")


@pytest.mark.parametrize(
    "yaml_name",
    [
        "fwi_marmousi_eager_compile.yaml",
        "fwi_marmousi_cuda_boundary_gpu.yaml",
        "fwi_marmousi_cuda_boundary_cpu.yaml",
        "fwi_marmousi_cuda_ckpt_chunk.yaml",
        "fwi_marmousi_multistage_advanced.yaml",
    ],
)
def test_shipped_backend_variant_yaml_validates(yaml_name):
    """The 4 backend-variant sample YAMLs under examples/tasks/ must validate."""

    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "examples" / "tasks" / yaml_name
    assert yaml_path.exists(), f"missing sample yaml: {yaml_path}"
    spec = load_task(yaml_path)
    assert spec.task_type == "fwi"


# ---------- A/B/C-tier FWI feature schemas -------------------------------

@pytest.mark.parametrize("loss_kind", ["mse", "l1", "huber"])
def test_loss_kinds_validate(tmp_path, loss_kind):
    template = new_template("fwi")
    template["loss"] = {"kind": loss_kind}
    if loss_kind == "huber":
        template["loss"]["huber_delta"] = 0.5
    yaml_path = tmp_path / "loss.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.loss.kind == loss_kind


def test_model_bounds_rejected_when_both_none(tmp_path):
    template = new_template("fwi")
    template["model_bounds"] = {"vp": {"min": None, "max": None}}
    yaml_path = tmp_path / "bad_bounds.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_model_bounds_round_trip(tmp_path):
    template = new_template("fwi")
    template["model_bounds"] = {"vp": {"min": 1500.0, "max": 5500.0}}
    yaml_path = tmp_path / "bounds.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.model_bounds["vp"].min == 1500.0
    assert spec.model_bounds["vp"].max == 5500.0


def test_stages_round_trip(tmp_path):
    template = new_template("fwi")
    template["epochs"] = 1  # ignored when stages set
    template["stages"] = [
        {"epochs": 5, "wavelet": {"kind": "ricker", "fm": 3.0}, "lr_scale": 1.0},
        {"epochs": 7, "wavelet": {"kind": "ricker", "fm": 5.0}, "lr_scale": 0.5},
    ]
    yaml_path = tmp_path / "stages.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert len(spec.stages) == 2
    assert spec.stages[0].epochs == 5
    assert spec.stages[1].lr_scale == 0.5


def test_optimizer_sgd_round_trip(tmp_path):
    template = new_template("fwi")
    template["optimizer"] = {"kind": "sgd", "lr": 0.5, "momentum": 0.9, "nesterov": True}
    yaml_path = tmp_path / "sgd.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.optimizer.kind == "sgd"
    assert spec.optimizer.momentum == 0.9
    assert spec.optimizer.nesterov is True


def test_optimizer_lbfgs_round_trip(tmp_path):
    template = new_template("fwi")
    template["optimizer"] = {"kind": "lbfgs", "lr": 1.0, "max_iter": 10, "history_size": 5}
    yaml_path = tmp_path / "lbfgs.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.optimizer.kind == "lbfgs"
    assert spec.optimizer.max_iter == 10


def test_optimizer_per_model_lr_dict_round_trip(tmp_path):
    template = new_template("fwi")
    template["optimizer"] = {"kind": "adam", "lr": {"vp": 25.0}, "eps": 1.0e-22}
    yaml_path = tmp_path / "pm_lr.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert isinstance(spec.optimizer.lr, dict)
    assert spec.optimizer.lr["vp"] == 25.0


@pytest.mark.parametrize(
    "scheduler",
    [
        {"kind": "constant"},
        {"kind": "step", "step_size": 5, "gamma": 0.5},
        {"kind": "exp", "gamma": 0.9},
        {"kind": "cosine", "eta_min": 0.0},
    ],
)
def test_scheduler_kinds_validate(tmp_path, scheduler):
    template = new_template("fwi")
    template["scheduler"] = scheduler
    yaml_path = tmp_path / "sched.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.scheduler.kind == scheduler["kind"]


def test_fwi_rejects_both_init_model_and_init_models(tmp_path):
    template = new_template("fwi")
    template["init_models"] = [{"name": "vp", "path": "true.npy"}]  # both set is illegal
    yaml_path = tmp_path / "both_init.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    with pytest.raises(ValidationError):
        load_task(yaml_path)


def test_fwi_accepts_init_models_list(tmp_path):
    template = new_template("fwi")
    template.pop("init_model")
    template["init_models"] = [{"name": "vp", "path": "smooth.npy"}]
    yaml_path = tmp_path / "list_init.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.init_model is None
    assert spec.init_models is not None
    assert len(spec.init_models) == 1


def test_lsrtm_reflectivity_bounds_round_trip(tmp_path):
    template = new_template("lsrtm")
    template["reflectivity_bounds"] = {"min": -0.1, "max": 0.1}
    yaml_path = tmp_path / "lsrtm_b.yaml"
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False))
    spec = load_task(yaml_path)
    assert spec.reflectivity_bounds.min == -0.1
    assert spec.reflectivity_bounds.max == 0.1
