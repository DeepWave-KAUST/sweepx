"""A stage may override only part of the global boundary block.

The band cascade wants the strips on the card while they fit and on the host
once they do not. Restating the whole block per stage would silently revert
dtype / interval / ring buffers to schema defaults, so an override copies only
the fields the stage actually named -- and a stage that moves storage back to
the card while the global block still carries cpu-only knobs is rejected here,
where the YAML vocabulary still exists to name them.
"""
import pytest

from sweep_tasks.schemas import (BackendSpec, BoundaryOptionsModel,
                                 CUDAOptionsModel, MemoryOptionsModel, StageSpec)
from sweep_tasks.tasks.fwi_freqsel import backend_with_stage_boundary


def _backend(**bnd):
    base = dict(storage="cpu", storage_dtype="int8", transfer_interval=32,
                pinned_memory=True, ring_buffers=4)
    base.update(bnd)
    return BackendSpec(
        impl="c",
        cuda_options=CUDAOptionsModel(
            memory=MemoryOptionsModel(
                strategy="boundary", boundary=BoundaryOptionsModel(**base))))


def _bnd(be):
    return be.cuda_options.memory.boundary


def test_unset_override_is_the_same_object_state():
    be = _backend()
    assert backend_with_stage_boundary(be, None) is be
    assert backend_with_stage_boundary(be, BoundaryOptionsModel()) is be


def test_only_named_fields_are_overridden():
    """The whole point: moving storage must not reset the other knobs."""
    be = _backend()
    out = backend_with_stage_boundary(
        be, BoundaryOptionsModel(storage="gpu", transfer_interval=None,
                                 pinned_memory=None, ring_buffers=None))
    assert _bnd(out).storage == "gpu"
    assert _bnd(out).storage_dtype == "int8", "unnamed field must be inherited"
    assert _bnd(be).storage == "cpu", "the global block must not be mutated"


def test_storage_dtype_alone_keeps_the_host_staging():
    be = _backend()
    out = backend_with_stage_boundary(be, BoundaryOptionsModel(storage_dtype="fp16"))
    assert _bnd(out).storage_dtype == "fp16"
    assert (_bnd(out).storage, _bnd(out).transfer_interval,
            _bnd(out).ring_buffers) == ("cpu", 32, 4)


def test_gpu_with_inherited_staging_knobs_is_rejected_by_name():
    """The core rejects the pair; say which knobs while the YAML names exist."""
    be = _backend()
    with pytest.raises(ValueError) as e:
        backend_with_stage_boundary(be, BoundaryOptionsModel(storage="gpu"))
    msg = str(e.value)
    assert "transfer_interval" in msg and "ring_buffers" in msg
    assert "null" in msg, "must say how to clear them"


def test_override_needs_a_global_boundary_block():
    be = BackendSpec(impl="c", cuda_options=CUDAOptionsModel(
        memory=MemoryOptionsModel(strategy="ckpt")))
    with pytest.raises(ValueError, match="cuda_options.memory.boundary"):
        backend_with_stage_boundary(be, BoundaryOptionsModel(storage="cpu"))


def test_stage_accepts_the_block():
    s = StageSpec(epochs=1, boundary=BoundaryOptionsModel(storage="gpu"))
    assert s.boundary.storage == "gpu"
    assert StageSpec(epochs=1).boundary is None


def test_full_field_override_warns_about_what_it_replaced(capsys):
    """A block naming every field inherits nothing -- say what that costs.

    This is the shape a pre-2026-09 ``config_resolved.yaml`` produces: the
    author wrote ``storage`` only, the dump expanded it to all eight fields,
    and ``storage_dtype`` came back as the schema default. Legal value, wrong
    run -- so the only way to notice is to be told.
    """
    be = _backend()
    ov = BoundaryOptionsModel(storage="gpu", storage_dtype="fp32",
                              transfer_interval=None, pinned_memory=None,
                              disk_dir=None, ring_buffers=None,
                              disk_async_read=False, tail_steps=None)
    assert set(ov.model_fields_set) == set(BoundaryOptionsModel.model_fields)
    out = backend_with_stage_boundary(be, ov)
    msg = capsys.readouterr().out
    assert "inherits nothing" in msg
    assert "storage_dtype: 'int8' -> 'fp32'" in msg
    assert _bnd(out).storage_dtype == "fp32", "the warning must not change it"


def test_partial_override_does_not_warn(capsys):
    """The normal case stays quiet, or the warning is noise nobody reads."""
    be = _backend()
    backend_with_stage_boundary(be, BoundaryOptionsModel(storage="gpu",
                                                         transfer_interval=None,
                                                         pinned_memory=None,
                                                         ring_buffers=None))
    assert capsys.readouterr().out == ""
