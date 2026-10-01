"""DD record layout: sweep-solver 0.3.0 made ModelParallel return the single-card
``(B, nt, nrec, nfield)`` record; before, it returned the raw ``(B, nrec, nt)``.
The loss read the DD record as raw unconditionally, so on 0.3.0 the time axis came
out one sample long and the first steady-state check died on a 0-column matmul.
Both layouts must give the single-domain answer.
"""
import numpy as np
import torch

import sweep_tasks.freqsel as fsl
from sweep_tasks.freqsel import FreqSelTargets, FrequencyComb, SteadyGCNLoss, extract_shard

NT, N_NODES, N_REC, NY = 192, 4, 24, 64
N_SS, SLACK = 8, 4


def _setup(tmp_path):
    rng = np.random.default_rng(1)
    rec = rng.standard_normal((N_NODES, NT, N_REC)).astype(np.float32)
    nodes = np.stack([np.arange(N_NODES) + 1, np.arange(N_NODES) * 2 + 1, np.ones(N_NODES, int)], 1)
    traces = np.stack([np.arange(N_REC) + 1, np.zeros(N_REC, int), np.zeros(N_REC, int)], 1)
    comb = FrequencyComb(dt=0.001, n_p=NT, ks=np.arange(6, 20))
    p = str(tmp_path / "obs_part0.npz")
    extract_shard(p, rec, nodes, traces, comb)
    t = FreqSelTargets(p, comb, ny=NY)
    t.bind_ownership(np.arange(t.n_union), "cpu")
    syn = (np.random.default_rng(5).standard_normal(
        (1, N_SS + SLACK + comb.n_p, t.n_union, 1)) * 1e-2).astype(np.float32)
    return t, comb, torch.tensor(syn)


def _loss(t, comb, distributed):
    return SteadyGCNLoss(comb, t, N_SS, SLACK, torch.device("cpu"), distributed=distributed)


def test_both_dd_layouts_match_single_domain(tmp_path, monkeypatch):
    t, comb, canon = _setup(tmp_path)
    raw = canon[..., 0].permute(0, 2, 1).contiguous()           # (B, nrec, nt), pre-0.3.0 DD
    pool = bins = np.arange(N_NODES)
    ref_y = _loss(t, comb, False)._as_rec_time(canon)
    ref = _loss(t, comb, False).two_window_check(canon, pool, bins, N_SS, SLACK)

    monkeypatch.setattr(fsl, "_dd_record_is_canonical", lambda: True)
    new = _loss(t, comb, True)
    assert torch.equal(new._as_rec_time(canon), ref_y)
    assert new.two_window_check(canon, pool, bins, N_SS, SLACK) == ref

    monkeypatch.setattr(fsl, "_dd_record_is_canonical", lambda: False)
    old = _loss(t, comb, True)
    assert torch.equal(old._as_rec_time(raw), ref_y)
    assert old.two_window_check(raw, pool, bins, N_SS, SLACK) == ref


def test_detection_reads_the_installed_solver():
    try:
        from sweep.parallel import dd_propagator
    except ImportError:
        assert fsl._dd_record_is_canonical() is False
        return
    assert fsl._dd_record_is_canonical() == hasattr(dd_propagator, "_cuda_record_to_canonical")
