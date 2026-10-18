"""``config_resolved.yaml`` must re-run as the run it recorded.

The file's own header prints ``sweep-tasks run config_resolved.yaml`` as the
way to reproduce a run, so re-running it has to give the same run. A full
pydantic dump does not: a stage ``boundary:`` block that named ``storage``
alone comes back naming every field, and the ones the author never wrote now
override the global block with their schema defaults.

That is not hypothetical. A production band cascade was derived from a
``config_resolved.yaml`` whose stage blocks had gained ``storage_dtype: fp32``;
the global block said ``int8``. The boundary buffer grew 4x and the
coarsest band's peak nearly tripled, in a config nobody had edited.
"""
import yaml
from pydantic import BaseModel

from sweep_tasks._helpers.metadata import (_dump_run_metadata,
                                           _prune_setwise_overrides)
from sweep_tasks.schemas import (BackendSpec, BoundaryOptionsModel,
                                 CUDAOptionsModel, MemoryOptionsModel,
                                 StageSpec)
from sweep_tasks.tasks.fwi_freqsel import backend_with_stage_boundary


class _Spec(BaseModel):
    """The two blocks the override semantics live in, nothing else."""

    backend: BackendSpec
    stages: list[StageSpec]


def _spec():
    """Global block stages on the host in int8; stage 0 moves it to the card.

    This is the production shape: the coarse band's strips fit on the card, so
    the stage names ``storage`` (and clears the cpu-only knobs, which the merge
    rejects on the card) and inherits dtype from the global block.
    """
    backend = BackendSpec(
        impl="c",
        cuda_options=CUDAOptionsModel(
            memory=MemoryOptionsModel(
                strategy="boundary",
                boundary=BoundaryOptionsModel(
                    storage="cpu", storage_dtype="int8", transfer_interval=32,
                    pinned_memory=True, ring_buffers=4))))
    stage = StageSpec(
        epochs=30,
        boundary=BoundaryOptionsModel(storage="gpu", transfer_interval=None,
                                      pinned_memory=None, ring_buffers=None))
    return _Spec(backend=backend, stages=[stage])


def _effective_dtype(spec):
    """The dtype that actually reaches the solver for stage 0."""
    merged = backend_with_stage_boundary(spec.backend, spec.stages[0].boundary)
    return merged.cuda_options.memory.boundary.storage_dtype


def _reload(cfg: dict) -> _Spec:
    return _Spec.model_validate(yaml.safe_load(yaml.safe_dump(cfg)))


def test_the_original_inherits_int8():
    """Guard the premise: without any round trip the stage inherits int8."""
    assert _effective_dtype(_spec()) == "int8"


def test_full_dump_would_flip_the_dtype():
    """The bug, reproduced: an unpruned dump does not round-trip.

    This is the discrimination check for the test below -- if this ever stops
    failing to inherit, the round-trip assertion has become vacuous and is no
    longer evidence of anything.
    """
    spec = _spec()
    assert _effective_dtype(_reload(spec.model_dump(mode="json"))) == "fp32"


def test_pruned_dump_round_trips():
    spec = _spec()
    cfg = _prune_setwise_overrides(spec, spec.model_dump(mode="json"))
    assert _effective_dtype(_reload(cfg)) == "int8"
    # only the fields the stage named survive in the override
    assert set(cfg["stages"][0]["boundary"]) == {
        "storage", "transfer_interval", "pinned_memory", "ring_buffers"}


def test_the_global_block_is_still_dumped_in_full():
    """Pruning is for overrides only; the block being overridden stays whole."""
    spec = _spec()
    cfg = _prune_setwise_overrides(spec, spec.model_dump(mode="json"))
    gb = cfg["backend"]["cuda_options"]["memory"]["boundary"]
    assert gb["storage_dtype"] == "int8"
    assert set(gb) == set(BoundaryOptionsModel.model_fields)


def test_written_file_round_trips(tmp_path):
    """End to end: the wiring, not just the helper.

    A helper that is never called is the failure mode this catches -- the
    assertion is on the bytes ``_dump_run_metadata`` actually wrote.
    """
    spec = _spec()
    _dump_run_metadata(spec, tmp_path)
    on_disk = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())
    assert _effective_dtype(_Spec.model_validate(on_disk)) == "int8"


def test_stage_without_an_override_is_untouched():
    spec = _Spec(backend=_spec().backend, stages=[StageSpec(epochs=1)])
    cfg = _prune_setwise_overrides(spec, spec.model_dump(mode="json"))
    assert cfg["stages"][0]["boundary"] is None


def test_a_spec_without_stages_is_a_no_op():
    class _NoStages(BaseModel):
        backend: BackendSpec

    spec = _NoStages(backend=_spec().backend)
    cfg = spec.model_dump(mode="json")
    assert _prune_setwise_overrides(spec, cfg) == cfg
