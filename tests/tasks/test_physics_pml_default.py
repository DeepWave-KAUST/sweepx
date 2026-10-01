"""physics.pml_type defaults to the equation's own PML, not 'cpmlr'.

The default used to be 'cpmlr', which only the collocated equations ship.
Every staggered one (Elastic, the DAS family, ...) needs 'cpmls', so a config
that left pml_type out handed Elastic the wrong profiles: an unpack error on
impl='eager', a process abort in the compiled core on impl='c'.
"""
import torch

from sweep_tasks._helpers.solver_build import _build_solver
from sweep_tasks.schemas import BackendSpec, PhysicsSpec


def test_default_is_unset():
    assert PhysicsSpec(equation="Elastic").pml_type is None


def _solver(physics):
    return _build_solver(physics, BackendSpec(impl="eager"), shape=(32, 32), dh=10.0,
                         dt=1e-3, nt=50, dev=torch.device("cpu"))


def test_unset_gives_each_equation_its_own_pml():
    elastic = PhysicsSpec(equation="Elastic", source_type=["sxx", "szz"], receiver_type=["vx", "vz"])
    assert _solver(elastic).pml_type == "cpmls"
    assert _solver(PhysicsSpec(equation="Acoustic")).pml_type == "cpmlr"


def test_explicit_matching_value_still_accepted():
    assert _solver(PhysicsSpec(equation="Acoustic", pml_type="cpmlr")).pml_type == "cpmlr"
