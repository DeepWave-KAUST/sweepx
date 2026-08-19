"""Field-acquisition extraction: ragged fold, and a record clock != solver clock."""
import json

import numpy as np
import pytest

from sweep_tasks.freqsel import (FrequencyComb, FreqSelTargets, extract_shard,
                                 extract_shard_gathers)


def _comb(dt=0.001, n_p=6000, k_lo=18, k_hi=36):
    return FrequencyComb(dt=dt, n_p=n_p, ks=np.arange(k_lo, k_hi + 1))


def _tone(freqs, amps, phases, nt, dt):
    t = np.arange(nt) * dt
    return sum(a * np.cos(2 * np.pi * f * t + p)
               for f, a, p in zip(freqs, amps, phases))


def test_dt_record_is_the_records_clock_not_the_solvers(tmp_path):
    """A 4 ms record on a 1 ms comb must land on the same physical frequency.

    Without `dt_record` the DTFT walks the time axis at the solver's dt, which
    rescales every comb frequency by dt_record/comb.dt — here a factor of 4.
    """
    comb = _comb()
    f0 = float(comb.freqs[5])
    nt_slow, dt_slow = 1500, 0.004
    rec = _tone([f0], [1.0], [0.3], nt_slow, dt_slow)[:, None]      # (nt, 1)

    out = extract_shard(str(tmp_path / "slow.npz"), rec[None], np.array([[10, 1]]),
                        np.array([[20, 1]]), comb, dt_record=dt_slow)
    D = np.load(out)["D"]
    peak = int(np.argmax(np.abs(D[0])))
    assert peak == 5, f"tone at comb bin 5 landed on bin {peak}"

    # The same record read on the solver's clock: the tone is 4x off, so bin 5
    # is no longer the maximum. This is the bug the parameter exists to stop.
    out_bad = extract_shard(str(tmp_path / "bad.npz"), rec[None],
                            np.array([[10, 1]]), np.array([[20, 1]]), comb)
    assert int(np.argmax(np.abs(np.load(out_bad)["D"][0]))) != 5


def test_dt_record_above_nyquist_is_refused():
    comb = FrequencyComb(dt=0.001, n_p=1000, ks=np.arange(200, 260))  # to 259 Hz
    with pytest.raises(ValueError, match="Nyquist"):
        extract_shard("/dev/null", np.zeros((1, 10, 1)), np.array([[1, 1]]),
                      np.array([[2, 1]]), comb, dt_record=0.004)


def test_ragged_matches_rectangular_when_folds_are_equal(tmp_path):
    """The ragged writer is the same DTFT — equal folds must reproduce it bit for bit."""
    rng = np.random.default_rng(0)
    comb = _comb()
    nt, n_rec, n_nodes = 400, 6, 3
    rec = rng.standard_normal((n_nodes, nt, n_rec))
    nodes = np.array([[10, 1], [20, 1], [30, 1]])
    traces = np.stack([np.stack([np.arange(n_rec) + 40, np.ones(n_rec)], -1)] * n_nodes)

    a = np.load(extract_shard(str(tmp_path / "rect.npz"), rec, nodes, traces, comb))
    b = np.load(extract_shard_gathers(
        str(tmp_path / "ragg.npz"),
        ((rec[i], nodes[i], traces[i]) for i in range(n_nodes)), comb))
    assert np.array_equal(a["D"], b["D"])
    assert np.array_equal(a["node_ptr"], b["node_ptr"])
    assert np.array_equal(a["trace_grid_xyz"], b["trace_grid_xyz"])


def test_ragged_folds_survive_the_round_trip(tmp_path):
    """Varying fold per node is what field CRGs actually look like."""
    rng = np.random.default_rng(1)
    comb = _comb()
    nt = 300
    folds = [7, 3, 11]
    gathers = [(rng.standard_normal((nt, f)),
                np.array([10 * (i + 1), 1]),
                np.stack([np.arange(f) + 50, np.ones(f)], -1))
               for i, f in enumerate(folds)]
    out = extract_shard_gathers(str(tmp_path / "f.npz"), list(gathers), comb)
    with np.load(out) as p:
        assert np.array_equal(p["node_ptr"], np.array([0, 7, 10, 21]))
        assert len(p["D"]) == 21
        assert json.loads(str(p["meta"]))["nt_record"] == nt

    # and FreqSelTargets must accept it — the format claims CSR, so prove it
    tg = FreqSelTargets(str(out), comb, ny=1)
    assert tg.n_nodes == 3 and tg.n_items == 21
    assert np.array_equal(np.bincount(tg.node_of_item), np.array(folds))


def test_ragged_refuses_a_mixed_time_axis(tmp_path):
    comb = _comb()
    bad = [(np.zeros((100, 2)), np.array([1, 1]), np.stack([np.arange(2), np.ones(2)], -1)),
           (np.zeros((200, 2)), np.array([2, 1]), np.stack([np.arange(2), np.ones(2)], -1))]
    with pytest.raises(ValueError, match="every gather must share one time axis"):
        extract_shard_gathers(str(tmp_path / "x.npz"), bad, comb)
