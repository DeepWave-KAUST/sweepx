"""The steady-state QC must not need the whole record resident to gather.

``y[gi]`` copied the full nt-long record for every owned receiver before
slicing it down to the analysis window: on a full field survey that throws
away as much as it keeps, and it exhausted an 80 GB card on a
statistic that is only printed. Slicing first and gathering in row chunks
bounds the peak at one chunk.

Chunking is not bit-identical -- cuBLAS picks its kernel from the matrix shape,
so a large product and a small one reduce in different orders -- so what
is checked is that the returned median is the same number to well inside the
~1e-2 level the statistic is read at.
"""
import numpy as np
import pytest
import torch

import sweep_tasks.freqsel as fsl
from sweep_tasks.freqsel import (FreqSelTargets, FrequencyComb, SteadyGCNLoss,
                                 extract_shard)

NT, N_NODES, N_REC, NY = 192, 4, 24, 64
N_SS, SLACK = 8, 4


def _targets(tmp_path):
    rng = np.random.default_rng(1)
    rec = rng.standard_normal((N_NODES, NT, N_REC)).astype(np.float32)
    nodes = np.stack([np.arange(N_NODES) + 1, np.arange(N_NODES) * 2 + 1,
                      np.ones(N_NODES, int)], 1)
    traces = np.stack([np.arange(N_REC) + 1, np.zeros(N_REC, int),
                       np.zeros(N_REC, int)], 1)
    comb = FrequencyComb(dt=0.001, n_p=NT, ks=np.arange(6, 20))
    p = str(tmp_path / "obs_part0.npz")
    extract_shard(p, rec, nodes, traces, comb)
    t = FreqSelTargets(p, comb, ny=NY)
    t.bind_ownership(np.arange(t.n_union), "cpu")
    return t, comb


def _check(t, comb, rec, pool, bins):
    lf = SteadyGCNLoss(comb, t, N_SS, SLACK, torch.device("cpu"), distributed=False)
    return lf.two_window_check(torch.tensor(rec), pool, bins, N_SS, SLACK)


def test_chunking_does_not_move_the_statistic(tmp_path, monkeypatch):
    t, comb = _targets(tmp_path)
    rec = (np.random.default_rng(5).standard_normal(
        (1, N_SS + SLACK + comb.n_p, t.n_union, 1)) * 1e-2).astype(np.float32)
    pool, bins = np.arange(N_NODES), np.arange(N_NODES)

    whole = _check(t, comb, rec, pool, bins)          # 512 MB budget: one gather
    # Force the chunk loop: a budget of one row per chunk.
    monkeypatch.setattr(fsl, "_TWC_CHUNK_BYTES", comb.n_p * 4)
    chunked = _check(t, comb, rec, pool, bins)

    assert np.isfinite(whole) and np.isfinite(chunked)
    assert abs(chunked - whole) <= 1e-4 * max(abs(whole), 1e-12), (whole, chunked)


def test_the_row_budget_is_at_least_one_row(tmp_path, monkeypatch):
    """A budget smaller than a single row must not produce rows=0."""
    t, comb = _targets(tmp_path)
    rec = (np.random.default_rng(6).standard_normal(
        (1, N_SS + SLACK + comb.n_p, t.n_union, 1)) * 1e-2).astype(np.float32)
    monkeypatch.setattr(fsl, "_TWC_CHUNK_BYTES", 1)
    v = _check(t, comb, rec, np.arange(N_NODES), np.arange(N_NODES))
    assert np.isfinite(v)
