"""A run may use an integer decimation of the comb its shards were extracted at.

The comb frequencies are baked into the observed coefficients, so changing the
bin count normally means re-running the extraction over the field data. But the
bins whose index is divisible by ``m`` stay mutually orthogonal over a window
``n_p / m``, at unchanged frequencies -- so a SHORTER record can reuse the long
extraction. That matters because the record length drives ``tail_steps``, which
drives the boundary buffer, which is what decides how many cards a band needs.

These check the mapping is right, not merely that it loads: a wrong column
selection would still run and would silently invert the node-to-frequency
assignment.
"""
import numpy as np
import pytest

from sweep_tasks.freqsel import FreqSelTargets, FrequencyComb, extract_shard


def _shard(tmp_path, nt=256, k_lo=8, k_hi=40, n_nodes=4, n_rec=3, seed=0):
    rng = np.random.default_rng(seed)
    rec = rng.standard_normal((n_nodes, nt, n_rec)).astype(np.float32)
    nodes = np.stack([np.arange(n_nodes), np.arange(n_nodes) * 2,
                      np.ones(n_nodes, int)], 1)
    traces = np.stack([np.arange(n_rec) + 1, np.arange(n_rec),
                       np.zeros(n_rec, int)], 1)
    comb = FrequencyComb(dt=0.001, n_p=nt, ks=np.arange(k_lo, k_hi + 1))
    p = str(tmp_path / "obs_part0.npz")
    extract_shard(p, rec, nodes, traces, comb)
    return p, comb


def _targets(path, comb):
    return FreqSelTargets(path, comb, ny=64)


def test_exact_comb_selects_every_column(tmp_path):
    p, comb = _shard(tmp_path)
    t = _targets(p, comb)
    assert t._bin_cols is None, "the ordinary case must stay a plain slice"
    assert t._nb == len(comb.ks)


def test_decimated_comb_picks_the_divisible_bins(tmp_path):
    """m=2: half the window, and the columns must be the EVEN extracted bins."""
    p, full = _shard(tmp_path, nt=256, k_lo=8, k_hi=40)
    half = FrequencyComb(dt=full.dt, n_p=full.n_p // 2,
                         ks=np.arange(4, 21))            # 8..40 even, halved
    t = _targets(p, half)
    assert t._bin_cols is not None
    assert t._nb == len(half.ks)
    with np.load(p) as z:
        ks = z["ks"].astype(np.int64)
        freqs = z["freqs"]
    # the selected shard bins are exactly the even ones ...
    assert np.array_equal(ks[t._bin_cols], np.asarray(half.ks) * 2)
    # ... and they sit at the SAME physical frequencies as the short comb
    np.testing.assert_allclose(
        freqs[t._bin_cols], np.asarray(half.ks) / (half.n_p * half.dt), rtol=1e-6)


def test_decimation_actually_reaches_the_right_coefficients(tmp_path):
    """A wrong column map would still run; compare against the shard itself."""
    p, full = _shard(tmp_path, nt=256, k_lo=8, k_hi=40)
    half = FrequencyComb(dt=full.dt, n_p=128, ks=np.arange(4, 21))
    t = _targets(p, half)
    with np.load(p) as z:
        D = z["D"]
    blk = t._cols(D[:5])
    assert blk.shape == (5, len(half.ks))
    np.testing.assert_array_equal(blk, D[:5][:, t._bin_cols])


def test_window_that_does_not_divide_is_refused(tmp_path):
    p, full = _shard(tmp_path, nt=256, k_lo=8, k_hi=40)
    bad = FrequencyComb(dt=full.dt, n_p=100, ks=np.arange(4, 21))   # 256 % 100
    with pytest.raises(ValueError, match="not whole"):
        _targets(p, bad)


def test_missing_bin_is_refused_not_silently_dropped(tmp_path):
    """Decimation is only valid if every k*m was actually extracted."""
    p, full = _shard(tmp_path, nt=256, k_lo=8, k_hi=40)
    # k=2 -> shard bin 4, below the extracted range [8, 40]
    bad = FrequencyComb(dt=full.dt, n_p=128, ks=np.arange(2, 21))
    with pytest.raises(ValueError, match="absent from the shards"):
        _targets(p, bad)


def test_dt_mismatch_still_refused(tmp_path):
    p, full = _shard(tmp_path)
    bad = FrequencyComb(dt=full.dt * 2, n_p=full.n_p, ks=full.ks)
    with pytest.raises(ValueError, match="dt"):
        _targets(p, bad)
