"""One source per grid cell.

Two sources in the same cell are injected into the same ``u[]`` element, and
the CUDA source kernel does that with ``atomicAdd`` — the summation order then
varies between runs, so the wavefield stops being reproducible. It is also a
geometry statement: at that dh the grid cannot separate the two nodes, so the
duplicate would be modelled from a position it is not at.

A field node array found this: at a coarse dh, consecutive node ids
collapse onto shared cells in pairs.

"""
import numpy as np
import pytest

from sweep_tasks.freqsel import PoolScheduler


def _grid(n=20, dup=()):
    """``n`` nodes on distinct cells, then ``dup`` pairs forced onto one cell."""
    g = np.stack([np.arange(n), np.arange(n) * 2, np.ones(n, int)], 1)
    for a, b in dup:
        g[b] = g[a]
    return g.astype(np.int32)


def test_distinct_cells_keep_every_node():
    s = PoolScheduler(_grid(20), n_pools=4, n_bins=32, seed=0)
    assert len(s.dropped_nodes) == 0
    assert sum(len(p) for p in s.pools) == 20


@pytest.mark.parametrize("random_batch", [None, 8])
def test_duplicate_cells_are_dropped(random_batch):
    g = _grid(20, dup=[(3, 4), (10, 11), (15, 16)])
    s = PoolScheduler(g, n_pools=4, n_bins=32, seed=0,
                      random_batch=random_batch)
    assert sorted(s.dropped_nodes.tolist()) == [4, 11, 16]
    for it in range(12):
        pool, bins = s.draw(it)
        assert len(np.unique(g[pool], axis=0)) == len(pool)
        assert not set(pool) & {4, 11, 16}
        assert len(bins) == len(pool)


def test_lowest_index_of_each_cell_is_the_one_kept():
    g = _grid(10, dup=[(2, 7)])
    s = PoolScheduler(g, n_pools=2, n_bins=16, seed=0)
    assert s.dropped_nodes.tolist() == [7]
    assert 2 in np.concatenate(s.pools)


def test_three_nodes_on_one_cell_leave_one():
    g = _grid(10, dup=[(1, 5), (1, 8)])
    s = PoolScheduler(g, n_pools=2, n_bins=16, seed=0)
    assert sorted(s.dropped_nodes.tolist()) == [5, 8]
    assert len(np.unique(g[np.concatenate(s.pools)], axis=0)) == 8


def test_random_batch_beyond_distinct_cells_is_an_error():
    g = _grid(10, dup=[(0, 1), (2, 3)])          # 10 nodes -> 8 cells
    PoolScheduler(g, n_pools=1, n_bins=16, seed=0, random_batch=8)
    with pytest.raises(ValueError, match="distinct source cells"):
        PoolScheduler(g, n_pools=1, n_bins=16, seed=0, random_batch=9)


def test_batch_beyond_comb_bins_still_errors_first():
    with pytest.raises(ValueError, match="exceeds .* comb bins"):
        PoolScheduler(_grid(20), n_pools=1, n_bins=4, seed=0, random_batch=8)


def test_draw_is_reproducible_for_a_seed():
    g = _grid(20, dup=[(3, 4)])
    a = [PoolScheduler(g, 4, 32, seed=7, random_batch=6).draw(i) for i in range(3)]
    b = [PoolScheduler(g, 4, 32, seed=7, random_batch=6).draw(i) for i in range(3)]
    for (pa, ba), (pb, bb) in zip(a, b):
        assert np.array_equal(pa, pb) and np.array_equal(ba, bb)


def test_extract_reports_cell_collisions_on_this_grid(tmp_path, capsys):
    """Collisions belong to a grid — the shard build is where you learn of them."""
    from sweep_tasks.freqsel import FrequencyComb, extract_shard

    rng = np.random.default_rng(0)
    rec = rng.standard_normal((4, 128, 3)).astype(np.float32)
    comb = FrequencyComb(dt=0.001, n_p=128, ks=np.arange(4, 9))
    traces = np.stack([np.arange(3), np.arange(3), np.ones(3, int)], 1)

    coarse = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 1], [4, 0, 1]])   # 2 share
    extract_shard(str(tmp_path / "coarse.npz"), rec, coarse, traces, comb)
    assert "1 node(s) share a cell" in capsys.readouterr().out

    fine = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1], [4, 0, 1]])     # none do
    extract_shard(str(tmp_path / "fine.npz"), rec, fine, traces, comb)
    assert "share a cell" not in capsys.readouterr().out
