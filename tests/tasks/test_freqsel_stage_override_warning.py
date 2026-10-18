"""A stage `frequency:` block replaces the global one — warn about what it drops."""
from sweep_tasks.schemas import FreqSelectionSpec
from sweep_tasks.tasks.fwi_freqsel import stage_freq_override_warnings


def _spec(**kw):
    base = dict(coeff_shards="x.npz", probe_samples=32000, k_lo=96, k_hi=192)
    return FreqSelectionSpec(**{**base, **kw})


def test_warns_for_each_dropped_global_setting():
    g = _spec(random_batch=96, n_pools=12, steady_samples=20000)
    s = _spec(coeff_shards="y.npz", probe_samples=20000, k_lo=60, k_hi=160)
    msgs = stage_freq_override_warnings(g, s, si=3)
    joined = "\n".join(msgs)
    assert len(msgs) == 3, joined
    assert "stage 3" in joined
    for name, gv, sv in (("random_batch", "96", "None"),
                         ("n_pools", "12", repr(s.n_pools)),
                         ("steady_samples", "20000", repr(s.steady_samples))):
        assert f"frequency.{name}" in joined
        assert gv in joined and sv in joined


def test_silent_when_the_stage_repeats_them():
    g = _spec(random_batch=96, n_pools=12, steady_samples=20000)
    s = _spec(coeff_shards="y.npz", probe_samples=20000, k_lo=60, k_hi=160,
              random_batch=96, n_pools=12, steady_samples=20000)
    assert stage_freq_override_warnings(g, s, si=0) == []


def test_silent_when_values_agree_even_if_unset():
    """Only a CHANGE in effective value is worth a warning."""
    g = _spec(ramp_s=0.5)                      # 0.5 happens to be the default
    s = _spec(coeff_shards="y.npz")
    assert stage_freq_override_warnings(g, s, si=0) == []


def test_no_warning_when_the_stage_has_no_block():
    g = _spec(random_batch=96)
    assert stage_freq_override_warnings(g, g, si=0) == []
    assert stage_freq_override_warnings(g, None, si=0) == []
