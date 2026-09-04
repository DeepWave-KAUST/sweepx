"""Boundary storage falls back to the host, and only under duress.

The card is where the strips belong while they fit -- they are written and
read every step, and staging through the host costs a PCIe round trip each
way. The fine bands do not fit: a fine rung can exhaust an 80 GB card even split four ways;
so the run starts on the card and moves off it on the first
out-of-memory, for the rest of that stage.

The switch is collective in the caller (every rank votes each iteration, and
they move together or not at all); what is checkable here is the override
itself.
"""
import pytest

from sweep_tasks.schemas import (BackendSpec, BoundaryOptionsModel,
                                 CUDAOptionsModel, MemoryOptionsModel)
from sweep_tasks.tasks.fwi_freqsel import backend_with_boundary_storage


def _backend(storage="gpu", strategy="boundary"):
    return BackendSpec(
        impl="c",
        cuda_options=CUDAOptionsModel(
            memory=MemoryOptionsModel(
                strategy=strategy,
                boundary=BoundaryOptionsModel(storage=storage,
                                              storage_dtype="int8"))))


def test_storage_is_overridden():
    out = backend_with_boundary_storage(_backend("gpu"), "cpu")
    assert out.cuda_options.memory.boundary.storage == "cpu"


def test_the_original_is_left_alone():
    """The stage keeps its own backend; a fallback must not leak upward."""
    be = _backend("gpu")
    backend_with_boundary_storage(be, "cpu")
    assert be.cuda_options.memory.boundary.storage == "gpu"


def test_everything_else_survives_the_copy():
    be = _backend("gpu")
    be.cuda_options.memory.boundary.tail_steps = 12345
    out = backend_with_boundary_storage(be, "cpu")
    assert out.cuda_options.memory.boundary.tail_steps == 12345
    assert out.cuda_options.memory.boundary.storage_dtype == "int8"
    assert out.impl == "c"


def test_no_boundary_config_is_a_clear_error():
    """Silently doing nothing here would look like the fallback had worked."""
    be = BackendSpec(impl="c")
    with pytest.raises(ValueError, match="cuda_options.memory.boundary"):
        backend_with_boundary_storage(be, "cpu")
