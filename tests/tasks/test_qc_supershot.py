"""Unit tests for ``save_supershot_qc_panel`` — the per-iter multisource
QC product (encoded-supershot obs/syn interleave + spectrum + survey map).

The helper is pure (just plots numpy arrays) so these tests don't spin
up the runner; they just feed it synthetic obs/syn + geometry and check
it writes a valid PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _synth_obs_syn(nt: int = 256, n_shared: int = 80, seed: int = 0):
    """Synthetic encoded-supershot pair: syn = obs + small noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(nt) / 100.0  # 0..2.56 s @ dt=0.01
    # Ricker-like wavelet shifted per receiver to make a moveout
    obs = np.zeros((nt, n_shared), dtype=np.float32)
    for r in range(n_shared):
        t0 = 0.5 + 0.02 * r
        f = 5.0
        w = (1 - 2 * (np.pi * f * (t - t0)) ** 2) * np.exp(
            -((np.pi * f * (t - t0)) ** 2)
        )
        obs[:, r] = w.astype(np.float32)
    syn = obs + 0.1 * rng.standard_normal(obs.shape).astype(np.float32)
    # The helper accepts (1, nt, n_shared, 1) — wrap to that shape.
    obs4 = obs[None, :, :, None]
    syn4 = syn[None, :, :, None]
    return obs4, syn4


def _synth_survey(B: int = 24, n_shared: int = 80, seed: int = 1):
    """Synthetic UTM geometry: random clusters of OBN + shot positions."""
    rng = np.random.default_rng(seed)
    all_groups = rng.uniform(
        low=[400_000, 7_700_000], high=[420_000, 7_720_000], size=(668, 2),
    )
    all_shots = rng.uniform(
        low=[395_000, 7_695_000], high=[425_000, 7_725_000], size=(20_000, 2),
    )
    picked = all_groups[rng.choice(all_groups.shape[0], size=B, replace=False)]
    used = all_shots[rng.choice(all_shots.shape[0], size=n_shared, replace=False)]
    return picked, used, all_groups, all_shots


def test_save_supershot_qc_panel_writes_png(tmp_path):
    from sweep_tasks.qc import save_supershot_qc_panel

    obs4, syn4 = _synth_obs_syn()
    picked, used, all_groups, all_shots = _synth_survey()
    out = tmp_path / "supershot" / "iter_0000.png"
    written = save_supershot_qc_panel(
        obs4, syn4,
        picked_group_utm_xy=picked,
        used_shot_utm_xy=used,
        all_groups_utm_xy=all_groups,
        all_shots_utm_xy=all_shots,
        dt=0.01, out_path=out, epoch=0,
        interleave_block=20, f_hi_hz=8.0,
    )
    assert written == out
    assert out.exists() and out.stat().st_size > 5_000  # non-trivial PNG


def test_save_supershot_qc_panel_accepts_3d_obs(tmp_path):
    """``obs_super`` from the CRG prefetcher is (1, n_shared, nt) — the
    helper must auto-detect and transpose to (nt, n_shared)."""
    from sweep_tasks.qc import save_supershot_qc_panel

    obs4, syn4 = _synth_obs_syn()
    # Convert obs to (1, n_shared, nt) instead of (1, nt, n_shared, 1).
    obs_3d = obs4[0, :, :, 0].T[None, :, :]  # (1, n_shared, nt)
    out = tmp_path / "iter_0000.png"
    picked, used, all_groups, _ = _synth_survey()
    save_supershot_qc_panel(
        obs_3d, syn4,
        picked_group_utm_xy=picked,
        used_shot_utm_xy=used,
        all_groups_utm_xy=all_groups,
        all_shots_utm_xy=None,  # also test the no-background-shots branch
        dt=0.01, out_path=out, epoch=42,
        interleave_block=15,
    )
    assert out.exists()


def test_save_supershot_qc_panel_sourceline_sort(tmp_path):
    """When sourceline_ids is given, traces are reordered by
    (line_id, sx_utm) — the PNG must still write successfully and the
    sort must accept a deliberately scrambled input."""
    from sweep_tasks.qc import save_supershot_qc_panel

    n_shared = 80
    obs4, syn4 = _synth_obs_syn(n_shared=n_shared)
    picked, used, all_groups, all_shots = _synth_survey(n_shared=n_shared)
    # Build a deliberately interleaved sourceline_ids array (3 lines,
    # rows distributed round-robin) and a deterministic sx_utm sequence
    # to verify the lexsort runs.
    sl_ids = np.tile(np.array([10, 20, 30]),
                     (n_shared + 2) // 3)[:n_shared]
    rng = np.random.default_rng(7)
    used_scrambled = used.copy()
    used_scrambled[:, 0] = 400_000 + rng.permutation(n_shared) * 50.0

    out = tmp_path / "sorted.png"
    save_supershot_qc_panel(
        obs4, syn4,
        picked_group_utm_xy=picked,
        used_shot_utm_xy=used_scrambled,
        all_groups_utm_xy=all_groups,
        all_shots_utm_xy=all_shots,
        sourceline_ids=sl_ids,
        dt=0.01, out_path=out, epoch=7,
        interleave_block=20, f_hi_hz=8.0,
    )
    assert out.exists() and out.stat().st_size > 5_000


def test_save_supershot_qc_panel_within_sourceline_key(tmp_path):
    """When ``within_sourceline_sort_key`` is provided, lexsort uses it
    as secondary key instead of the x-coord — must still write the PNG.
    """
    from sweep_tasks.qc import save_supershot_qc_panel

    n_shared = 80
    obs4, syn4 = _synth_obs_syn(n_shared=n_shared)
    picked, used, all_groups, _ = _synth_survey(n_shared=n_shared)
    sl_ids = np.tile(np.array([10, 20, 30]),
                     (n_shared + 2) // 3)[:n_shared]
    # Synthetic abs-row index: monotone increasing per line is what
    # matters; pass plain arange and let the lexsort apply.
    rng = np.random.default_rng(11)
    within_key = rng.permutation(n_shared).astype(np.int64)
    out = tmp_path / "within_key.png"
    save_supershot_qc_panel(
        obs4, syn4,
        picked_group_utm_xy=picked,
        used_shot_utm_xy=used,
        all_groups_utm_xy=all_groups,
        sourceline_ids=sl_ids,
        within_sourceline_sort_key=within_key,
        dt=0.01, out_path=out, epoch=0,
    )
    assert out.exists()


def test_save_supershot_qc_panel_within_sourceline_key_length_mismatch(tmp_path):
    from sweep_tasks.qc import save_supershot_qc_panel

    obs4, syn4 = _synth_obs_syn(n_shared=80)
    picked, used, all_groups, _ = _synth_survey(n_shared=80)
    sl_ids = np.zeros(80, dtype=np.int64)
    bad_key = np.arange(50, dtype=np.int64)
    with pytest.raises(ValueError, match="within_sourceline_sort_key length"):
        save_supershot_qc_panel(
            obs4, syn4,
            picked_group_utm_xy=picked,
            used_shot_utm_xy=used,
            all_groups_utm_xy=all_groups,
            sourceline_ids=sl_ids,
            within_sourceline_sort_key=bad_key,
            dt=0.01, out_path=tmp_path / "x.png", epoch=0,
        )


def test_save_supershot_qc_panel_sourceline_length_mismatch_raises(tmp_path):
    from sweep_tasks.qc import save_supershot_qc_panel

    obs4, syn4 = _synth_obs_syn(n_shared=80)
    picked, used, all_groups, _ = _synth_survey(n_shared=80)
    bad_ids = np.zeros(50, dtype=np.int64)  # length mismatch
    with pytest.raises(ValueError, match="sourceline_ids length"):
        save_supershot_qc_panel(
            obs4, syn4,
            picked_group_utm_xy=picked,
            used_shot_utm_xy=used,
            all_groups_utm_xy=all_groups,
            sourceline_ids=bad_ids,
            dt=0.01, out_path=tmp_path / "bad.png", epoch=0,
        )


def test_save_supershot_qc_panel_model_frame_with_extent(tmp_path):
    """Caller passes model-frame coords + inversion grid extent — the
    PNG must still write and the extent rectangle must be drawn."""
    from sweep_tasks.qc import save_supershot_qc_panel

    obs4, syn4 = _synth_obs_syn()
    # Simulate model-frame coords (small km-scale numbers, not UTM-scale).
    rng = np.random.default_rng(99)
    picked = rng.uniform(low=[2000, 3000], high=[18000, 17000], size=(24, 2))
    used = rng.uniform(low=[1500, 2500], high=[18500, 17500], size=(80, 2))
    all_groups = rng.uniform(low=[0, 0], high=[20000, 20000], size=(668, 2))
    all_shots = rng.uniform(low=[-1000, -1000], high=[21000, 21000], size=(20000, 2))
    extent = (0.0, 20000.0, 0.0, 20000.0)  # model-frame meters

    out = tmp_path / "model_frame.png"
    save_supershot_qc_panel(
        obs4, syn4,
        picked_group_utm_xy=picked,
        used_shot_utm_xy=used,
        all_groups_utm_xy=all_groups,
        all_shots_utm_xy=all_shots,
        frame_label="model",
        inversion_extent_xy_m=extent,
        dt=0.01, out_path=out, epoch=100,
        f_hi_hz=8.0,
    )
    assert out.exists() and out.stat().st_size > 5_000


def test_save_supershot_qc_panel_shape_mismatch_raises(tmp_path):
    from sweep_tasks.qc import save_supershot_qc_panel

    obs4, _ = _synth_obs_syn(n_shared=80)
    _, syn4_bad = _synth_obs_syn(n_shared=90)
    picked, used, all_groups, _ = _synth_survey()
    with pytest.raises(ValueError, match="obs.shape"):
        save_supershot_qc_panel(
            obs4, syn4_bad,
            picked_group_utm_xy=picked,
            used_shot_utm_xy=used,
            all_groups_utm_xy=all_groups,
            dt=0.01, out_path=tmp_path / "x.png", epoch=0,
        )
