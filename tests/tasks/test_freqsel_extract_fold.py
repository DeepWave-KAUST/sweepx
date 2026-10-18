"""Real acquisition puts several traces in one receiver cell.

Whenever the shot interval is finer than the grid a cell collects more than one
trace.  :class:`FreqSelTargets` requires a node's cells to be distinct — it
scatters them into a per-node column map — so a shard that keeps the duplicates
is refused at load time, after the extraction has already been paid for.
"""

import json

import numpy as np
import pytest

from sweep_tasks.freqsel import (FrequencyComb, FreqSelTargets,
                                 extract_shard_gathers)


def _comb(dt=0.001, n_p=6000, k_lo=18, k_hi=36):
    return FrequencyComb(dt=dt, n_p=n_p, ks=np.arange(k_lo, k_hi + 1))


def _cells(xs):
    return np.stack([np.asarray(xs), np.ones(len(xs), int)], -1)



def test_duplicate_cells_are_averaged_not_kept(tmp_path):
    """Three traces, two of them in one cell -> two items, fold [2, 1]."""
    comb = _comb()
    rng = np.random.default_rng(0)
    rec = rng.standard_normal((200, 3))
    gath = [(rec, np.array([5, 1]), _cells([10, 10, 20]))]
    out = extract_shard_gathers(str(tmp_path / "f.npz"), gath, comb)
    with np.load(out) as p:
        D, fold, trc = p["D"], p["fold"], p["trace_grid_xyz"]
    assert len(D) == 2
    assert np.array_equal(fold, np.array([2, 1]))
    assert np.array_equal(trc[:, 0], np.array([10, 20]))

    # the kept coefficient is the MEAN of the two, not their sum: fold varies
    # across cells and a sum would imprint the trace count as amplitude
    E = np.exp(-2j * np.pi * comb.freqs[None, :]
               * (np.arange(200) * comb.dt)[:, None])
    ref = (rec.T @ E).astype(np.complex64)
    np.testing.assert_allclose(D[0], (ref[0] + ref[1]) / 2, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D[1], ref[2], rtol=1e-5, atol=1e-6)


def test_folding_is_a_noop_when_cells_are_distinct(tmp_path):
    comb = _comb()
    rng = np.random.default_rng(1)
    gath = [(rng.standard_normal((150, 4)), np.array([2, 1]), _cells([1, 2, 3, 4]))]
    out = extract_shard_gathers(str(tmp_path / "d.npz"), gath, comb)
    with np.load(out) as p:
        assert len(p["D"]) == 4
        assert np.array_equal(p["fold"], np.ones(4, np.int32))


def test_a_folded_shard_is_accepted_by_the_loader(tmp_path):
    """The reason folding belongs in the writer: unfolded shards are refused."""
    comb = _comb()
    rng = np.random.default_rng(2)
    gath = [(rng.standard_normal((120, 5)), np.array([3, 1]), _cells([7, 7, 7, 8, 9])),
            (rng.standard_normal((120, 2)), np.array([9, 1]), _cells([8, 8]))]
    out = extract_shard_gathers(str(tmp_path / "l.npz"), gath, comb)
    tg = FreqSelTargets(out, comb, ny=1)
    assert tg.n_nodes == 2
    assert tg.n_items == 4                       # 3 + 1, not 5 + 2
    assert np.array_equal(np.bincount(tg.node_of_item), np.array([3, 1]))


def test_fold_counts_survive_across_nodes(tmp_path):
    comb = _comb()
    rng = np.random.default_rng(3)
    gath = [(rng.standard_normal((100, 4)), np.array([1, 1]), _cells([5, 5, 5, 6])),
            (rng.standard_normal((100, 3)), np.array([2, 1]), _cells([5, 6, 7]))]
    out = extract_shard_gathers(str(tmp_path / "n.npz"), gath, comb)
    with np.load(out) as p:
        assert np.array_equal(p["node_ptr"], np.array([0, 2, 5]))
        assert np.array_equal(p["fold"], np.array([3, 1, 1, 1, 1]))
