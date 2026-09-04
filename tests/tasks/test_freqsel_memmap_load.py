"""The observed table is read by mapping the shard, not by materialising it.

``np.load`` returns a whole member; at production comb sizes these run to hundreds of GB, and four
DD ranks reading one each is what OOM-killed a cascade. The members are
written STORED, so they can be mapped and only the owned rows touched.

That is a change to the path the observed data arrives on, so it is checked
against the obvious implementation rather than just for not crashing.
"""
import numpy as np
import pytest

from sweep_tasks.freqsel import FreqSelTargets, FrequencyComb, extract_shard


def _shard(tmp_path, n_nodes=5, n_rec=4, nt=256, seed=0):
    rng = np.random.default_rng(seed)
    rec = rng.standard_normal((n_nodes, nt, n_rec)).astype(np.float32)
    nodes = np.stack([np.arange(n_nodes), np.arange(n_nodes)*2,
                      np.ones(n_nodes, int)], 1)
    traces = np.stack([np.arange(n_rec)+1, np.arange(n_rec), np.zeros(n_rec, int)], 1)
    comb = FrequencyComb(dt=0.001, n_p=nt, ks=np.arange(4, 10))
    p = str(tmp_path / "obs_part0.npz")
    extract_shard(p, rec, nodes, traces, comb)
    return p, comb


def test_member_is_stored_so_it_can_be_mapped(tmp_path):
    """If the writer ever switches to compression the mapping must not lie."""
    import zipfile
    p, _ = _shard(tmp_path)
    with zipfile.ZipFile(p) as z:
        assert z.getinfo("D.npy").compress_type == zipfile.ZIP_STORED


def test_mapped_member_matches_np_load(tmp_path):
    p, _ = _shard(tmp_path)
    mapped = FreqSelTargets._memmap_member(p, "D.npy")
    assert mapped is not None, "STORED member should map"
    with np.load(p) as z:
        assert np.array_equal(np.asarray(mapped), z["D"])


def test_bound_table_matches_the_obvious_implementation(tmp_path):
    """Whole point: same numbers, whichever way the rows are fetched."""
    p, comb = _shard(tmp_path)
    t = FreqSelTargets(p, comb, ny=64)
    t.bind_ownership(np.arange(t.n_union), device="cpu")
    got = t.bound["D"].numpy()

    with np.load(p) as z:
        D, ptr = z["D"][:, :comb.n_bins], z["node_ptr"]
    sc = np.ones(len(D), np.float32)
    for s in range(len(ptr)-1):
        a = float(np.abs(D[ptr[s]:ptr[s+1]]).max(initial=0.0))
        sc[ptr[s]:ptr[s+1]] = a if a > 0 else 1.0
    ref = np.empty((len(D), comb.n_bins, 2), np.float16)
    ref[:, :, 0] = D.real / sc[:, None]
    ref[:, :, 1] = D.imag / sc[:, None]
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_a_row_subset_gets_the_same_values(tmp_path):
    """A rank owning half the columns must still see its own rows unchanged."""
    p, comb = _shard(tmp_path)
    full = FreqSelTargets(p, comb, ny=64)
    full.bind_ownership(np.arange(full.n_union), device="cpu")
    ref = full.bound["D"].numpy()

    half = FreqSelTargets(p, comb, ny=64)
    own = np.arange(0, half.n_union, 2)
    half.bind_ownership(own, device="cpu")
    got = half.bound["D"].numpy()
    keep = np.isin(full.union_col, own)
    assert np.array_equal(got, ref[keep])
