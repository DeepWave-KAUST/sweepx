"""An extraction shard cannot be time-windowed after the fact.

The shard is one coefficient per (node, cell, bin) over the WHOLE record, so an
unwanted arrival contaminates every comb bin.  The mute has to run on the
gathers, before the DTFT, or not at all.
"""

import json

import numpy as np
import pytest

from sweep_tasks.freqsel import (FrequencyComb, extract_shard_gathers, offset_top_mute)


def _comb(dt=0.001, n_p=6000, k_lo=18, k_hi=36):
    return FrequencyComb(dt=dt, n_p=n_p, ks=np.arange(k_lo, k_hi + 1))


def _cells(xs):
    return np.stack([np.asarray(xs), np.ones(len(xs), int)], -1)



def test_mute_is_zero_before_the_cut_and_one_well_after():
    t = np.arange(1000) * 0.002                  # 2 s at 2 ms
    rec = np.ones((1, 1000))
    offset = np.array([2000.0])
    offset_top_mute(rec, offset, t, v=4000.0, taper=0.10)
    tcut = 2000.0 / 4000.0                       # 0.5 s
    assert rec[0, t < tcut].max() == 0.0
    np.testing.assert_allclose(rec[0, t > tcut + 0.10], 1.0, atol=1e-12)
    mid = (t >= tcut) & (t <= tcut + 0.10)
    assert np.all(np.diff(rec[0, mid]) >= -1e-12)          # monotone ramp


def test_the_cut_follows_the_offset_not_the_node():
    t = np.arange(1500) * 0.002                  # 3 s: long enough for BOTH cuts
    rec = np.ones((2, 1500))
    offset_top_mute(rec, np.array([1000.0, 4000.0]), t, v=2000.0)
    first = [np.flatnonzero(r > 0)[0] for r in rec]
    assert t[first[0]] == pytest.approx(0.5, abs=0.002)
    assert t[first[1]] == pytest.approx(2.0, abs=0.002)


def test_taper_zero_is_a_step_and_taper_widens_it():
    t = np.arange(500) * 0.002
    step = np.ones((1, 500))
    offset_top_mute(step, np.array([1000.0]), t, v=2000.0, taper=0.0)
    assert set(np.unique(step)) <= {0.0, 1.0}

    ramped = np.ones((1, 500))
    offset_top_mute(ramped, np.array([1000.0]), t, v=2000.0, taper=0.20)
    partial = (ramped > 0) & (ramped < 1)
    assert partial.sum() > 50                    # a real ramp, not a step


def test_guard_and_sqrt_term_move_the_curve():
    t = np.arange(800) * 0.002
    base = np.ones((1, 800))
    offset_top_mute(base, np.array([1000.0]), t, v=2000.0)
    plain = t[np.flatnonzero(base[0] > 0)[0]]

    early = np.ones((1, 800))
    offset_top_mute(early, np.array([1000.0]), t, v=2000.0, guard=0.3)
    assert t[np.flatnonzero(early[0] > 0)[0]] == pytest.approx(plain - 0.3, abs=0.002)

    curved = np.ones((1, 800))
    offset_top_mute(curved, np.array([1000.0]), t, v=2000.0, a=0.01)
    assert t[np.flatnonzero(curved[0] > 0)[0]] == pytest.approx(
        plain + 0.01 * np.sqrt(1000.0), abs=0.002)


@pytest.mark.parametrize("kw, msg", [
    (dict(v=0.0), "must be positive"),
    (dict(v=1500.0, taper=-0.1), "must not be negative"),
])
def test_mute_refuses_nonsense(kw, msg):
    with pytest.raises(ValueError, match=msg):
        offset_top_mute(np.ones((1, 10)), np.array([100.0]),
                        np.arange(10) * 0.001, **kw)


def test_mute_checks_its_shapes():
    with pytest.raises(ValueError, match="offsets"):
        offset_top_mute(np.ones((2, 10)), np.array([1.0]),
                        np.arange(10) * 0.001, v=1500.0)
    with pytest.raises(ValueError, match="t_axis"):
        offset_top_mute(np.ones((1, 10)), np.array([1.0]),
                        np.arange(9) * 0.001, v=1500.0)


def test_mute_settings_are_recorded_in_the_shard(tmp_path):
    """Without this the shard cannot be told apart from an unmuted one."""
    comb = _comb()
    gath = [(np.ones((100, 2)), np.array([1, 1]), _cells([1, 2]))]
    mute = dict(t0=-0.2, v=3000.0, a=0.01, guard=1.0, taper=0.4,
                form="t0 + x/v + a*sqrt(x) - guard")
    out = extract_shard_gathers(str(tmp_path / "m.npz"), gath, comb,
                                meta_extra={"mute": mute})
    with np.load(out) as p:
        assert json.loads(str(p["meta"]))["mute"] == mute
