"""Truncated-backward wiring: frequency.bwd_tail_margin -> boundary tail_steps."""
import pytest

from sweep_tasks.schemas import (BackendSpec, BoundaryOptionsModel,
                                 CUDAOptionsModel, FreqSelectionSpec,
                                 MemoryOptionsModel)
from sweep_tasks.tasks.fwi_freqsel import backend_with_tail_steps


def _fspec(**kw):
    return FreqSelectionSpec(**{"coeff_shards": "x.npz",
                                "probe_samples": 32000, **kw})


def _backend(**kw):
    return BackendSpec(**{"impl": "c", **kw})


def test_unset_margin_returns_the_very_same_object():
    b = _backend()
    assert backend_with_tail_steps(b, _fspec(), dd_on=False) is b


def test_tail_is_probe_plus_margin_and_original_is_untouched():
    b = _backend(cuda_options=CUDAOptionsModel(
        memory=MemoryOptionsModel(strategy="boundary",
                                  boundary=BoundaryOptionsModel(storage="cpu"))))
    out = backend_with_tail_steps(b, _fspec(bwd_tail_margin=2000), dd_on=False)
    assert out is not b
    assert out.cuda_options.memory.boundary.tail_steps == 34000
    assert out.cuda_options.memory.boundary.storage == "cpu"     # preserved
    assert b.cuda_options.memory.boundary.tail_steps is None     # deep copy


def test_missing_cuda_options_are_constructed():
    out = backend_with_tail_steps(_backend(), _fspec(bwd_tail_margin=1),
                                  dd_on=False)
    assert out.cuda_options.memory.strategy == "boundary"
    assert out.cuda_options.memory.boundary.tail_steps == 32001


def test_per_stage_probe_gives_per_stage_tail():
    b = _backend()
    for probe, margin, want in ((32000, 2000, 34000), (10000, 2000, 12000)):
        f = FreqSelectionSpec(coeff_shards="x.npz", probe_samples=probe,
                              bwd_tail_margin=margin)
        assert (backend_with_tail_steps(b, f, dd_on=False)
                .cuda_options.memory.boundary.tail_steps == want)


@pytest.mark.parametrize("kw,dd,msg", [
    (dict(impl="eager"), False, "impl='c'"),
    (dict(use_ckpt=True), False, "use_ckpt"),
])
def test_guards_raise_with_yaml_vocabulary(kw, dd, msg):
    with pytest.raises(ValueError, match=msg):
        backend_with_tail_steps(_backend(**kw), _fspec(bwd_tail_margin=1), dd)


def test_non_boundary_strategy_is_refused():
    b = _backend(cuda_options=CUDAOptionsModel(
        memory=MemoryOptionsModel(strategy="ckpt")))
    with pytest.raises(ValueError, match="strategy='boundary'"):
        backend_with_tail_steps(b, _fspec(bwd_tail_margin=1), dd_on=False)


def test_stage_override_warning_covers_the_new_key():
    """The wholesale-replacement trap applies to bwd_tail_margin too."""
    from sweep_tasks.tasks.fwi_freqsel import stage_freq_override_warnings
    g = _fspec(bwd_tail_margin=2000)
    stage = _fspec(coeff_shards="y.npz")
    assert any("bwd_tail_margin" in w
               for w in stage_freq_override_warnings(g, stage, si=1))


def test_margin_in_seconds_scales_with_the_stage_dt():
    """A flat step count silently rescales with dt; seconds must not.

    The second dt is deliberately one that does NOT divide the margin evenly
    (2.0 / 0.003 = 666.7): the conversion ceils, and an even divisor would let
    a round() regression through unnoticed.
    """
    b = _backend()
    f = _fspec(bwd_tail_margin_s=2.0)
    for dt, want in ((0.001, 32000 + 2000), (0.003, 32000 + 667)):
        out = backend_with_tail_steps(b, f, dd_on=False, dt=dt)
        assert out.cuda_options.memory.boundary.tail_steps == want


def test_both_margin_conventions_together_are_refused():
    import pydantic
    with pytest.raises(pydantic.ValidationError, match="not both"):
        _fspec(bwd_tail_margin=2000, bwd_tail_margin_s=2.0)


def test_unset_tail_steps_still_drives_an_old_sweep():
    """model_dump() always carries tail_steps=None; the dataclass hop must
    drop it, or this schema cannot drive any sweep released before the
    feature existed (measured: broke a cross-version baseline on ORIX)."""
    BoundaryOptionsModel(storage="gpu").to_dataclass()   # old installed sweep


def test_set_tail_steps_is_a_capability_check_on_old_sweep():
    import sweep.propagator.options as _o
    import inspect
    has_field = "tail_steps" in inspect.signature(
        _o.BoundaryOptions.__init__).parameters
    if has_field:
        assert BoundaryOptionsModel(tail_steps=9).to_dataclass().tail_steps == 9
    else:
        with pytest.raises(TypeError, match="tail_steps"):
            BoundaryOptionsModel(tail_steps=9).to_dataclass()


def test_dd_passes_through_with_the_same_tail():
    """DD tiles inherit tail_steps via the boundary config; same tail on
    every rank falls out of config inheritance, so the wiring no longer
    guards dd_on (the core guards equation coverage instead)."""
    out = backend_with_tail_steps(_backend(), _fspec(bwd_tail_margin=2000),
                                  dd_on=True)
    assert out.cuda_options.memory.boundary.tail_steps == 34000


def test_mixed_cascade_truncates_only_the_stages_that_ask():
    """The production recipe: margin on the expensive high bands only.
    Per-band control falls out of the field living on each stage's spec."""
    b = _backend()
    low = FreqSelectionSpec(coeff_shards="lo.npz", probe_samples=50000)
    high = FreqSelectionSpec(coeff_shards="hi.npz", probe_samples=10000,
                             bwd_tail_margin_s=6.0)
    assert backend_with_tail_steps(b, low, dd_on=True, dt=0.004) is b
    out = backend_with_tail_steps(b, high, dd_on=True, dt=0.002)
    assert out.cuda_options.memory.boundary.tail_steps == 10000 + 3000
