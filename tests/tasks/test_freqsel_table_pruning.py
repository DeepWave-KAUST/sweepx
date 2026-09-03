"""The coefficient table only needs the columns the schedule will read.

The loss reads one frequency column per node per iteration, and which one is a
pure function of (seed, call count) -- so the set is knowable before the run
starts. ``plan()`` replays the draws behind an RNG snapshot, and the table is
built with only those columns.

What is checked here is that the pruned table holds the SAME numbers as the
full one (a wrong column map would still run, and would silently invert a
different dataset), that the plan really is the sequence ``draw`` then serves,
and that pruning composes with comb decimation -- the two features both rewrite
which columns of the shard are read.
"""
import numpy as np
import pytest
import torch

from sweep_tasks.freqsel import (FreqSelTargets, FrequencyComb, PoolScheduler,
                                 SteadyGCNLoss, extract_shard)

NT, K_LO, K_HI, N_NODES, N_REC, NY = 256, 8, 40, 6, 5, 64


def _shard(tmp_path, dup=False, seed=0):
    rng = np.random.default_rng(seed)
    rec = rng.standard_normal((N_NODES, NT, N_REC)).astype(np.float32)
    nodes = np.stack([np.arange(N_NODES) + 1, np.arange(N_NODES) * 2 + 1,
                      np.ones(N_NODES, int)], 1)
    xs = np.arange(N_REC) + 1
    if dup:                       # two traces of every node on ONE cell
        xs[1] = xs[0]
    traces = np.stack([xs, np.zeros(N_REC, int), np.zeros(N_REC, int)], 1)
    comb = FrequencyComb(dt=0.001, n_p=NT, ks=np.arange(K_LO, K_HI + 1))
    p = str(tmp_path / "obs_part0.npz")
    extract_shard(p, rec, nodes, traces, comb)
    return p, comb


def _sched(nodes_grid, n_bins, seed=11):
    return PoolScheduler(nodes_grid, 3, n_bins, seed=seed, random_batch=2)


def _bin_use(plan, n_nodes, n_bins):
    u = np.zeros((n_nodes, n_bins), bool)
    for pool, bins in plan:
        u[np.asarray(pool, np.int64), np.asarray(bins, np.int64)] = True
    return u


# ---------------------------------------------------------------- plan()
def test_plan_is_the_sequence_draw_then_serves(tmp_path):
    p, comb = _shard(tmp_path)
    t = FreqSelTargets(p, comb, ny=NY)
    s = _sched(t.node_grid, comb.n_bins)
    plan = s.plan(7)
    after = [s.draw(i) for i in range(7)]
    for (pa, ba), (pb, bb) in zip(plan, after):
        assert np.array_equal(pa, pb) and np.array_equal(ba, bb)


def test_plan_does_not_advance_the_rng(tmp_path):
    """A side effect here would desynchronise the run from its own table."""
    p, comb = _shard(tmp_path)
    t = FreqSelTargets(p, comb, ny=NY)
    a, b = _sched(t.node_grid, comb.n_bins), _sched(t.node_grid, comb.n_bins)
    a.plan(5)                                    # only difference between them
    for i in range(5):
        pa, ba = a.draw(i)
        pb, bb = b.draw(i)
        assert np.array_equal(pa, pb) and np.array_equal(ba, bb)


# ------------------------------------------------------------- pruned table
def _bound_pair(path, comb, bin_use):
    full = FreqSelTargets(path, comb, ny=NY)
    full.bind_ownership(np.arange(full.n_union), "cpu")
    pru = FreqSelTargets(path, comb, ny=NY)
    pru.bind_ownership(np.arange(pru.n_union), "cpu", bin_use=bin_use)
    return full, pru


def test_pruned_columns_are_bit_identical_to_the_full_table(tmp_path):
    p, comb = _shard(tmp_path)
    t = FreqSelTargets(p, comb, ny=NY)
    use = _bin_use(_sched(t.node_grid, comb.n_bins).plan(6), t.n_nodes, comb.n_bins)
    full, pru = _bound_pair(p, comb, use)
    bf, bp = full.bound, pru.bound
    assert bp["D"].shape[1] < bf["D"].shape[1], "nothing was pruned"
    for s in range(t.n_nodes):
        for k in np.nonzero(use[s])[0]:
            c = int(bp["colmap"][s, int(k)])
            assert c >= 0
            assert torch.equal(bf["D"][bf["dsel"][s], int(k)],
                               bp["D"][bp["dsel"][s], c])


def test_pruning_composes_with_comb_decimation(tmp_path):
    """Both features rewrite which shard columns are read; check them together."""
    p, full_comb = _shard(tmp_path)
    half = FrequencyComb(dt=full_comb.dt, n_p=NT // 2,
                         ks=np.arange((K_LO + 1) // 2, K_HI // 2 + 1))
    t = FreqSelTargets(p, half, ny=NY)
    assert t._bin_cols is not None, "this test needs the decimated path"
    use = _bin_use(_sched(t.node_grid, half.n_bins).plan(6), t.n_nodes, half.n_bins)
    full, pru = _bound_pair(p, half, use)
    bf, bp = full.bound, pru.bound
    for s in range(t.n_nodes):
        for k in np.nonzero(use[s])[0]:
            c = int(bp["colmap"][s, int(k)])
            assert torch.equal(bf["D"][bf["dsel"][s], int(k)],
                               bp["D"][bp["dsel"][s], c])


def test_unscheduled_pair_raises_instead_of_reading_a_wrong_column(tmp_path):
    from sweep_tasks.freqsel import _dcol
    p, comb = _shard(tmp_path)
    t = FreqSelTargets(p, comb, ny=NY)
    use = np.zeros((t.n_nodes, comb.n_bins), bool)
    use[:, 0] = True
    t.bind_ownership(np.arange(t.n_union), "cpu", bin_use=use)
    assert _dcol(t.bound, 0, 0) == 0
    with pytest.raises(RuntimeError, match="pruned D has no column"):
        _dcol(t.bound, 0, 1)


# ----------------------------------------------------------------- dup_free
@pytest.mark.parametrize("dup", [False, True])
def test_dup_free_reflects_the_geometry(tmp_path, dup):
    p, comb = _shard(tmp_path, dup=dup)
    t = FreqSelTargets(p, comb, ny=NY)
    t.bind_ownership(np.arange(t.n_union), "cpu")
    assert t.bound["dup_free"] is (not dup)


# -------------------------------------------------------------- batched loss
def _loss_value(monkeypatch, targets, comb, rec, pool, bins, batched, gpu_probes):
    monkeypatch.setenv("SWEEP_FREQSEL_BATCHED_LOSS", "1" if batched else "0")
    monkeypatch.setenv("SWEEP_FREQSEL_GPU_PROBES", "1" if gpu_probes else "0")
    r = torch.tensor(rec, requires_grad=True)
    lf = SteadyGCNLoss(comb, targets, 8, 4, torch.device("cpu"), distributed=False)
    J, _ = lf(r, pool, bins)
    J.backward()
    return float(J.detach()), r.grad.clone()


def _rec(n_union, nt, seed=3):
    return (np.random.default_rng(seed).standard_normal((1, nt, n_union, 1))
            * 1e-2).astype(np.float32)


def test_batched_loss_agrees_with_the_pool_loop(tmp_path, monkeypatch):
    p, comb = _shard(tmp_path)
    t = FreqSelTargets(p, comb, ny=NY)
    t.bind_ownership(np.arange(t.n_union), "cpu")
    assert t.bound["dup_free"]
    pool, bins = _sched(t.node_grid, comb.n_bins).draw(0)
    rec = _rec(t.n_union, 8 + 4 + comb.n_p)
    va, ga = _loss_value(monkeypatch, t, comb, rec, pool, bins, False, False)
    vb, gb = _loss_value(monkeypatch, t, comb, rec, pool, bins, True, True)
    assert abs(vb - va) <= 1e-5 * max(abs(va), 1e-12)
    assert float((gb - ga).abs().max()) <= 1e-4 * max(float(ga.abs().max()), 1e-12)


def test_batched_loss_falls_back_bit_exactly_when_cells_repeat(tmp_path, monkeypatch, capsys):
    """A dense scatter would drop a node's repeated cell, so it must not run."""
    p, comb = _shard(tmp_path, dup=True)
    t = FreqSelTargets(p, comb, ny=NY)
    t.bind_ownership(np.arange(t.n_union), "cpu")
    assert not t.bound["dup_free"]
    pool, bins = _sched(t.node_grid, comb.n_bins).draw(0)
    rec = _rec(t.n_union, 8 + 4 + comb.n_p)
    va, ga = _loss_value(monkeypatch, t, comb, rec, pool, bins, False, False)
    vb, gb = _loss_value(monkeypatch, t, comb, rec, pool, bins, True, False)
    assert vb == va and torch.equal(gb, ga)
    assert "dup_free=False" in capsys.readouterr().out


def test_device_probes_match_the_host_table(tmp_path, monkeypatch):
    p, comb = _shard(tmp_path)
    t = FreqSelTargets(p, comb, ny=NY)
    t.bind_ownership(np.arange(t.n_union), "cpu")
    lf = SteadyGCNLoss(comb, t, 8, 4, torch.device("cpu"), distributed=False)
    bins = np.arange(3)
    monkeypatch.setenv("SWEEP_FREQSEL_GPU_PROBES", "0")
    hr, hi = lf._probes(bins, 12)
    monkeypatch.setenv("SWEEP_FREQSEL_GPU_PROBES", "1")
    dr, di = lf._probes(bins, 12)
    s = 2.0 / comb.n_p
    assert float((hr - dr).abs().max()) < 1e-3 * s
    assert float((hi - di).abs().max()) < 1e-3 * s
