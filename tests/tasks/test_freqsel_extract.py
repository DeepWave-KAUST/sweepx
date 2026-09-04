"""`extract_shard` + the 3-D `kind: grid` geometry that the freqsel examples need.

The extractor is the observed-data half of frequency-selection FWI: it turns
recorded common-node gathers into the coefficient shard the inversion reads.
Nothing here touches a solver — the whole point of the shard is that the
inversion never sees a gather again.
"""
import numpy as np
import pytest

from sweep_tasks.freqsel import FreqSelTargets, FrequencyComb, extract_shard


def _comb(dt=0.001, n_p=200, k_lo=4, k_hi=8):
    return FrequencyComb(dt=dt, n_p=n_p, ks=np.arange(k_lo, k_hi + 1))


def _record(n_nodes=3, nt=200, n_rec=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_nodes, nt, n_rec)).astype(np.float32)


def _nodes(n_nodes=3, ndim=2):
    if ndim == 2:
        return np.stack([np.arange(n_nodes) * 7 + 3,
                         np.full(n_nodes, 2)], -1)
    return np.stack([np.arange(n_nodes) * 7 + 3, np.arange(n_nodes) * 3 + 1,
                     np.full(n_nodes, 2)], -1)


def _traces(n_rec=5, ndim=2):
    if ndim == 2:
        return np.stack([np.arange(n_rec) * 4 + 1, np.zeros(n_rec, int)], -1)
    return np.stack([np.arange(n_rec) * 4 + 1, np.full(n_rec, 6),
                     np.zeros(n_rec, int)], -1)


# --- the coefficients themselves ----------------------------------------

def test_extract_shard_matches_hand_dtft(tmp_path):
    comb, rec = _comb(), _record()
    out = extract_shard(str(tmp_path / "s.npz"), rec, _nodes(), _traces(), comb)
    with np.load(out) as p:
        D = p["D"]
    t = np.arange(rec.shape[1]) * comb.dt
    for s in range(rec.shape[0]):
        for r in range(rec.shape[2]):
            want = [np.sum(rec[s, :, r] * np.exp(-2j * np.pi * f * t))
                    for f in comb.freqs]
            got = D[s * rec.shape[2] + r]
            assert np.allclose(got, want, rtol=2e-4, atol=2e-4)


def test_extract_shard_accepts_trailing_field_axis(tmp_path):
    """A forward run writes (n_shots, nt, n_rec, 1) — take it as-is."""
    comb, rec = _comb(), _record()
    a = extract_shard(str(tmp_path / "a.npz"), rec, _nodes(), _traces(), comb)
    b = extract_shard(str(tmp_path / "b.npz"), rec[..., None], _nodes(),
                      _traces(), comb)
    with np.load(a) as pa, np.load(b) as pb:
        assert np.array_equal(pa["D"], pb["D"])


def test_extract_shard_per_node_traces(tmp_path):
    """Per-node receiver tables are kept per node, not silently broadcast."""
    comb, rec = _comb(), _record()
    per_node = np.stack([_traces() + 100 * s for s in range(rec.shape[0])])
    out = extract_shard(str(tmp_path / "s.npz"), rec, _nodes(), per_node, comb)
    with np.load(out) as p:
        assert np.array_equal(p["trace_grid_xyz"],
                              per_node.reshape(-1, 2).astype(np.int32))


# --- what FreqSelTargets makes of it ------------------------------------

def test_shard_loads_as_targets(tmp_path):
    comb, rec = _comb(), _record()
    n_nodes, _, n_rec = rec.shape
    out = extract_shard(str(tmp_path / "s.npz"), rec, _nodes(), _traces(), comb)
    t = FreqSelTargets(out, comb, ny=1)
    assert (t.n_nodes, t.n_items) == (n_nodes, n_nodes * n_rec)
    assert t.ndim == 2
    assert t.n_union == n_rec           # every node shares the same 5 cells
    assert np.array_equal(t.node_grid, _nodes())


def test_shard_loads_as_targets_3d(tmp_path):
    comb, rec = _comb(), _record()
    out = extract_shard(str(tmp_path / "s.npz"), rec, _nodes(ndim=3),
                        _traces(ndim=3), comb)
    t = FreqSelTargets(out, comb, ny=32)
    assert t.ndim == 3
    assert t.n_union == rec.shape[2]
    assert (t.union_xyz[:, 1] == 6).all()          # the y of _traces(ndim=3)


def test_targets_reject_a_comb_that_is_not_the_shards(tmp_path):
    """The comb check is what replaces 'did you use the right wavelet'.

    A configured comb may be an integer decimation of the extracted one (see
    test_freqsel_comb_decimation), so the check is no longer plain equality --
    but a comb asking for bins that were never extracted, or a window the
    extraction does not divide, must still be refused rather than truncated.
    """
    out = extract_shard(str(tmp_path / "s.npz"), _record(), _nodes(),
                        _traces(), _comb())
    # k_hi=9 was never extracted (shard holds k in [4, 8])
    with pytest.raises(ValueError, match="absent from the shards"):
        FreqSelTargets(out, _comb(k_lo=5, k_hi=9), ny=1)
    # a LONGER window than the extraction cannot be a decimation of it
    with pytest.raises(ValueError, match="not whole"):
        FreqSelTargets(out, _comb(n_p=400), ny=1)


# --- input validation ----------------------------------------------------

@pytest.mark.parametrize("bad,match", [
    (dict(record=np.zeros((3, 200))), "record must be"),
    (dict(node_grid_xyz=np.zeros((2, 2), int)), "node_grid_xyz must be"),
    (dict(node_grid_xyz=np.zeros((3, 4), int)), "ndim must be 2 or 3"),
    (dict(trace_grid_xyz=np.zeros((9, 2), int)), "trace_grid_xyz must be"),
    (dict(trace_grid_xyz=np.zeros((5, 3), int)), "ndim 3 != node_grid_xyz"),
    (dict(fold=np.ones(4)), "fold must have"),
])
def test_extract_shard_rejects_mismatched_inputs(tmp_path, bad, match):
    kw = dict(record=_record(), node_grid_xyz=_nodes(),
              trace_grid_xyz=_traces(), fold=None)
    kw.update(bad)
    with pytest.raises(ValueError, match=match):
        extract_shard(str(tmp_path / "s.npz"), kw["record"],
                      kw["node_grid_xyz"], kw["trace_grid_xyz"], _comb(),
                      fold=kw["fold"])
