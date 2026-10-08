"""impl='c' defaults to boundary saving, as sweep itself does.

With no memory block the runner passed ``use_ckpt=False`` and nothing else,
which sweep reads as "no memory trick at all": every time step was stored, and
a 2-D Marmousi FWI at nt=10000 asked for 276 GiB. The default now lands in the
spec, so config_resolved.yaml records it; 'full' is a value you can ask for;
the legacy ``use_ckpt`` flag still selects checkpointing.
"""
import pytest
import torch
import yaml

from sweep_tasks._helpers.solver_build import _build_solver
from sweep_tasks.schemas import BackendSpec, BoundaryOptionsModel, MemoryOptionsModel, PhysicsSpec
from sweep_tasks.yaml_io import load_task, new_template


def test_c_defaults_to_boundary():
    mem = BackendSpec(impl="c").cuda_options.memory
    assert mem.strategy == "boundary"
    assert mem.boundary.storage == "gpu"


def test_eager_is_untouched():
    assert BackendSpec(impl="eager").cuda_options is None


def test_null_reads_as_the_default():
    assert MemoryOptionsModel(strategy=None).strategy == "boundary"
    assert BackendSpec(impl="c", cuda_options={"memory": None}).cuda_options.memory.strategy == "boundary"


def test_full_takes_no_options():
    assert MemoryOptionsModel(strategy="full").boundary is None
    with pytest.raises(ValueError, match="only read with strategy='boundary'"):
        MemoryOptionsModel(strategy="full", boundary=BoundaryOptionsModel())


def test_legacy_use_ckpt_is_left_alone():
    assert BackendSpec(impl="c", use_ckpt=True).cuda_options is None


def test_new_template_default_is_boundary(tmp_path):
    path = tmp_path / "fwi.yaml"
    path.write_text(yaml.safe_dump(new_template("fwi", backend="c"), sort_keys=False))
    assert load_task(path).backend.cuda_options.memory.strategy == "boundary"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="impl='c' needs a GPU")
@pytest.mark.parametrize("backend, expected", [
    (BackendSpec(impl="c"), "boundary"),
    (BackendSpec(impl="c", cuda_options={"memory": {"strategy": "full"}}), "full"),
    (BackendSpec(impl="c", cuda_options={"memory": {"strategy": "ckpt"}}), "ckpt"),
    (BackendSpec(impl="c", use_ckpt=True), "ckpt"),
])
def test_solver_runs_the_strategy_the_spec_names(backend, expected):
    solver = _build_solver(PhysicsSpec(equation="Acoustic"), backend, shape=(32, 32), dh=10.0,
                           dt=1e-3, nt=200, dev=torch.device("cuda"))
    assert solver.memory_strategy == expected
