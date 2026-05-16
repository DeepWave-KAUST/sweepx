"""Tests for PhysicalGeometry, to_grid, DataPlan, ModelPlan."""

import numpy as np
import pytest

from sweep_io.geometry import Geometry, PhysicalGeometry
from sweep_io.plan import (
    DataPlan,
    ModelPlan,
    apply_data_plan,
    apply_model_plan,
)


def _make_physical(nshots=3, nrec=10, ndim=2, dt=0.001, nt=512):
    """Make a streamer-like 2-D PhysicalGeometry with regular spacing."""
    sources = np.stack(
        [np.linspace(100.0, 900.0, nshots), np.zeros(nshots)], axis=1
    )  # x increasing, z=0
    recv_one_shot = np.stack(
        [np.arange(nrec, dtype="float64") * 25.0, np.zeros(nrec)], axis=1
    )
    receivers = np.broadcast_to(recv_one_shot, (nshots, nrec, ndim)).copy()
    return PhysicalGeometry(sources, receivers, dt=dt, nt=nt)


# ====================== PhysicalGeometry / to_grid =========================
def test_physical_geometry_shapes():
    pg = _make_physical(nshots=4, nrec=8)
    assert pg.nshots == 4
    assert pg.nreceivers == 8
    assert pg.ndim == 2


def test_physical_geometry_validation():
    with pytest.raises(ValueError, match="nshots mismatch"):
        PhysicalGeometry(
            sources_xyz_m=np.zeros((2, 2)),
            receivers_xyz_m=np.zeros((3, 5, 2)),
            dt=0.001, nt=100,
        )


def test_to_grid_no_dedupe_when_rec_spacing_equals_dh():
    # Receivers at 25 m, dh = 25 m → one receiver per cell, no dedupe needed.
    pg = _make_physical(nshots=2, nrec=10)
    gg, mask = pg.to_grid(dh=(25.0, 25.0), dedupe=True)
    assert isinstance(gg, Geometry)
    assert mask.shape == (2, 10)
    assert mask.all(), "no dedupe should occur when spacing matches dh"
    # receiver indices should be 0..9
    np.testing.assert_array_equal(gg.receivers[0, :, 0], np.arange(10))


def test_to_grid_dedupes_when_dh_larger_than_rec_spacing():
    # Receivers at x = 0..200 step 25 (9 receivers), dh = 75 m.
    # np.round(x/75) maps them to cells {0,0,1,1,1,2,2,2,3} -> 4 unique cells.
    pg = _make_physical(nshots=1, nrec=9)
    gg, mask = pg.to_grid(dh=(75.0, 75.0), dedupe=True)
    n_kept = int(mask.sum())
    assert n_kept == 4, f"expected 4 surviving receivers; got {n_kept}"
    # Each surviving receiver should map to a distinct grid x-index
    kept_idx = gg.receivers[0, mask[0], 0]
    assert len(set(kept_idx.tolist())) == n_kept


def test_to_grid_dedup_nearest_picks_center():
    """Receivers at 0, 25, 50, 75, 100 m, dh = 50 m, origin = 0.
    Cells centred at 0, 50, 100; receivers at 25 and 75 are equidistant from
    two cells — `np.round` puts them in cells 0 and 2 respectively. So we
    expect the cells {0, 1, 2}, and each cell's chosen rep is the one whose
    physical x is closest to the cell center."""
    sources = np.array([[0.0, 0.0]])
    recv = np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0], [100.0, 0.0]])[None, :, :]
    pg = PhysicalGeometry(sources, recv, dt=0.001, nt=100)
    gg, mask = pg.to_grid(dh=(50.0, 50.0), dedupe=True, dedup_method="nearest")
    # x=0 -> cell 0, x=25 -> cell 0/1 (np.round picks 0 — banker's rounding -> 0)
    # x=50 -> cell 1, x=75 -> cell 2 (np.round 1.5 -> 2 banker's)
    # x=100 -> cell 2
    # So cells visited: 0, 0, 1, 2, 2 → dedupe → 3 kept (one per cell).
    assert mask.sum() == 3


def test_to_grid_dedup_first_keeps_input_order():
    sources = np.array([[0.0, 0.0]])
    recv = np.array([[10.0, 0.0], [30.0, 0.0], [40.0, 0.0]])[None, :, :]  # all in cell 0 of dh=100
    pg = PhysicalGeometry(sources, recv, dt=0.001, nt=100)
    _, mask_first = pg.to_grid(dh=(100.0, 100.0), dedupe=True, dedup_method="first")
    _, mask_nearest = pg.to_grid(dh=(100.0, 100.0), dedupe=True, dedup_method="nearest")
    # "first" keeps index 0
    assert bool(mask_first[0, 0]) and not bool(mask_first[0, 1]) and not bool(mask_first[0, 2])
    # "nearest" keeps whichever is closest to cell center 0 — receiver at 10
    assert bool(mask_nearest[0, 0])


def test_to_grid_no_dedupe_returns_all_true_mask():
    pg = _make_physical()
    _, mask = pg.to_grid(dh=(25.0, 25.0), dedupe=False)
    assert mask.all()


def test_to_grid_scalar_dh_broadcasts():
    pg = _make_physical()
    gg_s, _ = pg.to_grid(dh=25.0)
    gg_t, _ = pg.to_grid(dh=(25.0, 25.0))
    np.testing.assert_array_equal(gg_s.sources, gg_t.sources)


def test_to_grid_origin_shifts_indices():
    sources = np.array([[100.0, 0.0]])
    recv = np.array([[100.0, 0.0]])[None, :, :]
    pg = PhysicalGeometry(sources, recv, dt=0.001, nt=100)
    gg, _ = pg.to_grid(dh=(50.0, 50.0), origin_xyz_m=(100.0, 0.0))
    # With origin at x=100, x=100 lands at index 0.
    assert gg.sources[0, 0] == 0


# =============================== DataPlan ==================================
def test_data_plan_shot_stride():
    pg = _make_physical(nshots=10, nrec=4)
    obs = np.arange(10 * 100 * 4, dtype="float32").reshape(10, 100, 4)
    plan = DataPlan(shot_stride=3)
    geom_out, obs_out, mask = apply_data_plan(plan, pg, obs)
    assert geom_out.nshots == 4    # shots [0, 3, 6, 9]
    assert obs_out.shape[0] == 4
    np.testing.assert_array_equal(geom_out.sources_xyz_m[1], pg.sources_xyz_m[3])
    assert mask.shape == (4, 4)


def test_data_plan_explicit_shot_indices_overrides_stride():
    pg = _make_physical(nshots=10, nrec=4)
    obs = np.zeros((10, 50, 4), dtype="float32")
    plan = DataPlan(shot_stride=99, shot_indices=[1, 5, 7])
    geom_out, obs_out, _ = apply_data_plan(plan, pg, obs)
    assert geom_out.nshots == 3
    assert obs_out.shape[0] == 3


def test_data_plan_offset_max_drops_far_receivers():
    # Source at x=100, receivers at x=0..1000 step 100 → offsets |0..900|.
    sources = np.array([[100.0, 0.0]])
    recv = np.stack(
        [np.arange(0.0, 1001.0, 100.0), np.zeros(11)], axis=1
    )[None, :, :]
    pg = PhysicalGeometry(sources, recv, dt=0.001, nt=100)
    obs = np.zeros((1, 100, 11), dtype="float32")
    plan = DataPlan(offset_max_m=250)
    _, _, mask = apply_data_plan(plan, pg, obs)
    # |0..900| ≤ 250 → receivers at x = 0, 100, 200, 300 (wait, |300-100|=200) and
    # x = -100..300 → here only x in [0, 100, 200, 300] survive (offsets 100, 0, 100, 200).
    assert mask[0].tolist() == [True, True, True, True, False, False, False, False, False, False, False]


def test_data_plan_offset_min_drops_near_receivers():
    sources = np.array([[0.0, 0.0]])
    recv = np.array([[0.0, 0.0], [100.0, 0.0], [500.0, 0.0]])[None, :, :]
    pg = PhysicalGeometry(sources, recv, dt=0.001, nt=10)
    obs = np.zeros((1, 10, 3), dtype="float32")
    _, _, mask = apply_data_plan(DataPlan(offset_min_m=200), pg, obs)
    # offsets 0, 100, 500 ≥ 200 → only 500 survives
    assert mask[0].tolist() == [False, False, True]


def test_data_plan_receiver_stride():
    pg = _make_physical(nshots=2, nrec=8)
    obs = np.zeros((2, 50, 8), dtype="float32")
    _, _, mask = apply_data_plan(DataPlan(receiver_stride=2), pg, obs)
    assert mask[0].tolist() == [True, False] * 4


def test_data_plan_dt_target_resamples():
    pg = _make_physical(nshots=1, nrec=2, dt=0.001, nt=200)
    obs = np.random.RandomState(0).standard_normal((1, 200, 2)).astype("float32")
    plan = DataPlan(dt_target_s=0.002)
    geom_out, obs_out, _ = apply_data_plan(plan, pg, obs)
    assert geom_out.dt == 0.002
    assert obs_out.shape[1] == 100   # halved


def test_data_plan_time_decimate():
    pg = _make_physical(nshots=1, nrec=2, dt=0.001, nt=200)
    obs = np.arange(1 * 200 * 2, dtype="float32").reshape(1, 200, 2)
    plan = DataPlan(time_decimate=2)
    geom_out, obs_out, _ = apply_data_plan(plan, pg, obs)
    assert geom_out.dt == pytest.approx(0.002)
    assert obs_out.shape[1] == 100


def test_data_plan_invalid_combinations():
    with pytest.raises(ValueError, match="mutually exclusive"):
        DataPlan(dt_target_s=0.002, time_decimate=2)
    with pytest.raises(ValueError):
        DataPlan(offset_min_m=500, offset_max_m=200)
    with pytest.raises(ValueError):
        DataPlan(shot_stride=0)


def test_data_plan_t_window_trims():
    pg = _make_physical(nshots=1, nrec=2, dt=0.001, nt=1000)
    obs = np.zeros((1, 1000, 2), dtype="float32")
    plan = DataPlan(t_start_s=0.1, t_end_s=0.5)
    geom_out, obs_out, _ = apply_data_plan(plan, pg, obs)
    assert obs_out.shape[1] == 400
    assert geom_out.nt == 400


# =============================== ModelPlan ==================================
def test_model_plan_crops_2d_vp():
    vp = np.linspace(1500.0, 4500.0, 100 * 200, dtype="float32").reshape(100, 200)
    plan = ModelPlan(x_window_m=(500.0, 1500.0), z_window_m=(0.0, 500.0))
    vp_out, _, _ = apply_model_plan(plan, vp, dh=(10.0, 10.0))
    # x: 50..151 (inclusive ceil), z: 0..51
    assert vp_out.shape == (51, 101)


def test_model_plan_no_window_returns_full():
    vp = np.zeros((50, 80), dtype="float32")
    vp_out, _, _ = apply_model_plan(ModelPlan(), vp, dh=(10.0, 10.0))
    assert vp_out.shape == vp.shape


def test_model_plan_drops_outside_sources():
    pg = _make_physical(nshots=4)
    # Sources at x = 100, 366.6, 633.3, 900 (linspace).
    vp = np.zeros((50, 80), dtype="float32")
    plan = ModelPlan(x_window_m=(300.0, 700.0), drop_outside_sources=True)
    _, geom_out, src_keep = apply_model_plan(plan, vp, dh=(10.0, 10.0), geom=pg)
    assert geom_out is not None
    # Only sources at x ≈ 366.6 and 633.3 fall in [300, 700]
    assert int(src_keep.sum()) == 2
    assert geom_out.nshots == 2


def test_model_plan_rebases_origin():
    pg = _make_physical(nshots=2)
    vp = np.zeros((50, 100), dtype="float32")
    plan = ModelPlan(x_window_m=(200.0, 800.0))
    vp_out, geom_out, _ = apply_model_plan(plan, vp, dh=(10.0, 10.0), geom=pg)
    # Sources at x = 100 / 900 are dropped (outside); rebase shifts the
    # surviving sources by -200 m.
    # First test that the rebase metadata records the new origin.
    assert geom_out is not None
    assert "model_plan_origin_m" in geom_out.meta


def test_model_plan_rejects_collapsing_window():
    vp = np.zeros((50, 50), dtype="float32")
    with pytest.raises(ValueError, match="collapses"):
        apply_model_plan(
            ModelPlan(x_window_m=(1e6, 1e6 + 1)), vp, dh=(10.0, 10.0)
        )


def test_model_plan_validation():
    with pytest.raises(ValueError):
        ModelPlan(x_window_m=(500.0, 100.0))
