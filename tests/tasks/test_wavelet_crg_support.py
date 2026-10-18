"""Tests for CRG-plan support in the wavelet analyze/estimate pipeline.

The wavelet modules historically refused any plan with
``grouping != 'csg'``. Under OBN reciprocity, ``|rec - src|`` is
symmetric and CRG ⇄ CSG are dual, so analyze and estimate should
accept both. These tests build a synthetic CRG plan (2 OBN receivers
each seeing 4 physical shots) and verify:

* ``analyze_direct_wavelet_batch`` accepts the plan (grouping check
  passes; SVD failure on synthetic noise traces is tolerated and
  caught separately).
* ``prepare_wavelet_inversion_data`` accepts the plan and returns
  geometry in the *propagator frame* — i.e. the propagator source is
  the physical OBN receiver (one per CRG group), and the
  propagator-frame receivers are the physical shots (multiple per
  group). This is the reciprocity remapping the FWI runner relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_io.segy import (
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    FORMAT_IEEE_FLOAT32,
    write_segy_minimal,
)
from sweep_io.seismic_plan import SeismicPlan


SAMPLES_PER_TRACE = 256
DT_S = 0.002


def _trace_byte_offset(idx: int) -> int:
    return (
        SEGY_TEXT_HEADER_SIZE
        + SEGY_BIN_HEADER_SIZE
        + idx * (SEGY_TRACE_HEADER_SIZE + SAMPLES_PER_TRACE * 4)
    )


def _build_crg_plan_with_traces(tmp_path: Path) -> Path:
    """Build a minimal CRG plan: 2 OBN receivers × 4 physical shots.

    Layout (UTM-style coords, depths in m):
      OBN_recv_0 = (200, 100, 50)
      OBN_recv_1 = (300, 200, 50)   # x differs so 2D dedup keeps both
      shots @ x in {500, 600, 700, 800}, y=0, z=5

    Each CRG group lists the 4 shots in the same order; the SEG-Y has
    8 traces with a synthetic Ricker pulse at sample 80, offset
    according to |rec - src| / v_water.
    """
    rng = np.random.default_rng(2026_05_22)
    n_recv = 2
    n_shot = 4
    n_row = n_recv * n_shot

    shot_x = np.array([500.0, 600.0, 700.0, 800.0], dtype=np.float64)
    recv_xy = np.array([[200.0, 100.0], [300.0, 200.0]], dtype=np.float64)
    z_src = 5.0
    z_rec = 50.0

    # rows in CRG order: (recv 0, shot 0..3), (recv 1, shot 0..3)
    row_source_xyz = np.zeros((n_row, 3), dtype=np.float64)
    row_receiver_xyz = np.zeros((n_row, 3), dtype=np.float64)
    for ir, rxy in enumerate(recv_xy):
        for ish, sx in enumerate(shot_x):
            r = ir * n_shot + ish
            row_source_xyz[r] = [sx, 0.0, z_src]
            row_receiver_xyz[r, :2] = rxy
            row_receiver_xyz[r, 2] = z_rec

    # Synthesize traces: Ricker at predicted direct-arrival, plus
    # background noise. Predicted direct-arrival time = |rec - src| / 1500.
    traces = rng.standard_normal((n_row, SAMPLES_PER_TRACE)).astype(np.float32) * 0.05
    fm = 8.0
    t = np.arange(SAMPLES_PER_TRACE, dtype=np.float32) * DT_S
    for r in range(n_row):
        offset = np.linalg.norm(row_receiver_xyz[r] - row_source_xyz[r])
        tau = float(offset / 1500.0)
        arg = np.pi * fm * (t - tau)
        ricker = (1.0 - 2.0 * arg ** 2) * np.exp(-(arg ** 2))
        traces[r] += ricker.astype(np.float32)

    segy_path = tmp_path / "crg_dual.sgy"
    write_segy_minimal(segy_path, traces, dt=DT_S, sample_format=FORMAT_IEEE_FLOAT32)

    plan = SeismicPlan(
        grouping="crg",
        files=[segy_path],
        trace_size_per_file=np.asarray(
            [SEGY_TRACE_HEADER_SIZE + SAMPLES_PER_TRACE * 4], dtype=np.int64
        ),
        sample_format=FORMAT_IEEE_FLOAT32,
        samples_per_trace=SAMPLES_PER_TRACE,
        dt_s=DT_S,
        row_file_id=np.zeros(n_row, dtype=np.int64),
        row_trace_offset=np.array(
            [_trace_byte_offset(i) for i in range(n_row)], dtype=np.int64
        ),
        row_source_xyz=row_source_xyz,
        row_receiver_xyz=row_receiver_xyz,
        group_id=np.array([100, 101], dtype=np.int64),
        group_xyz=np.array(
            [[*recv_xy[0], z_rec], [*recv_xy[1], z_rec]], dtype=np.float64
        ),
        group_offsets=np.array([0, n_shot, n_row], dtype=np.int64),
        build_meta={"label": "crg_dual_test_fixture"},
    )
    plan_path = tmp_path / "crg_plan.npz"
    plan.save(plan_path)
    return plan_path


def test_analyze_wavelet_accepts_crg_plan(tmp_path):
    """``analyze_direct_wavelet_batch`` must accept grouping='crg'.

    The synthetic traces are clean Rickers so the SVD should succeed.
    """
    from sweep_tasks.wavelet import AnalyzeWaveletConfig, analyze_direct_wavelet_batch

    plan_path = _build_crg_plan_with_traces(tmp_path)
    cfg = AnalyzeWaveletConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "analyze_out",
        shot_start=1,
        shot_stop=2,
        shot_stride=1,
        nearest_receivers=4,
        max_abs_offset_m=None,
        rel_t_min_s=-0.05,
        rel_t_max_s=0.30,
        iterations=10,
        min_residual_ratio=2.0,           # permissive — synthetic Ricker
        min_shape_corr_to_average=0.3,    # permissive
        no_plot=True,
    )
    out_path = analyze_direct_wavelet_batch(cfg)
    assert out_path.exists(), f"analyze output not written: {out_path}"

    with np.load(out_path) as nz:
        assert "wavelet" in nz.files
        # Causal wavelet should be non-trivial.
        assert float(np.max(np.abs(nz["wavelet"]))) > 0.0


def test_analyze_wavelet_rejects_supershot_grouping(tmp_path):
    """Sanity: only CSG/CRG are accepted — supershot must still fail."""
    from sweep_tasks.wavelet import AnalyzeWaveletConfig, analyze_direct_wavelet_batch

    plan_path = _build_crg_plan_with_traces(tmp_path)
    plan = SeismicPlan.load(plan_path)
    bogus_path = tmp_path / "supershot_plan.npz"
    object.__setattr__(plan, "grouping", "supershot")
    plan.save(bogus_path)

    cfg = AnalyzeWaveletConfig(
        plan_path=bogus_path,
        output_dir=tmp_path / "bogus_out",
        shot_start=1, shot_stop=1,
    )
    with pytest.raises(ValueError, match="CSG or CRG"):
        analyze_direct_wavelet_batch(cfg)


def test_prepare_wavelet_inversion_data_accepts_crg(tmp_path):
    """``prepare_wavelet_inversion_data`` must remap to the propagator
    frame under CRG: propagator source = physical receiver (one per
    group), propagator receivers = physical shots (multiple per group).
    """
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out",
        shot_start=1,
        shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * DT_S),
        velocity_m_s=1500.0,
        dx_m=25.0,
        dz_m=25.0,
        x_padding_m=200.0,
        z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        max_abs_offset_m=None,
        backend="cuda",  # unused: only prepare is called
        mode="discrete",
    )
    data = prepare_wavelet_inversion_data(cfg)

    # Two CRG groups → two propagator sources.
    assert data.sources.shape == (2, 2), data.sources.shape  # (nshots, [x,z])
    # Each CRG group has 4 physical shots → 4 propagator receivers
    # (subject to grid-dedup which can collapse coincident shots).
    assert data.receivers.shape[0] == 2
    assert data.receivers.shape[1] >= 1
    assert data.receivers.shape[-1] == 2
    assert data.observed.shape == (
        data.sources.shape[0],
        data.receivers.shape[1],
        data.nt,
    )
    # Reciprocity check: the propagator source positions should map back
    # to the physical OBN receiver coords (x ∈ {200, 300}) — not the
    # shot row at x ∈ {500..800}. We test the grid x-index range is
    # below what a shot at x=500 would land on.
    xmin = data.x_origin_m
    src_x_m = xmin + data.sources[:, 0].astype(np.float64) * cfg.dx_m
    assert src_x_m.max() < 450.0, (
        f"Under CRG the propagator source must be the physical receiver "
        f"(x ∈ 200..300 m); got src_x_m={src_x_m} which looks like physical shots."
    )


def test_prepare_wavelet_inversion_data_3d_geometry(tmp_path):
    """With ``equation='Acoustic3D'`` the prepare step must produce
    3D geometry: sources / receivers are (..., 3) with order (x, y, z);
    model is (nz, ny, nx); summary carries ny / dy_m / y_origin_m.
    """
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out_3d",
        shot_start=1,
        shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * DT_S),
        velocity_m_s=1500.0,
        dx_m=25.0,
        dy_m=25.0,
        dz_m=25.0,
        x_padding_m=200.0,
        y_padding_m=200.0,
        z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        max_abs_offset_m=None,
        backend="cuda",  # unused: only prepare is called
        mode="discrete",
        equation="Acoustic3D",
    )
    data = prepare_wavelet_inversion_data(cfg)

    # 3D sources: last axis = 3 (x, y, z).
    assert data.sources.shape[-1] == 3, data.sources.shape
    assert data.sources.shape[0] == 2  # two CRG groups
    # 3D receivers: (nshots, nreceivers, 3).
    assert data.receivers.shape[0] == 2
    assert data.receivers.shape[-1] == 3
    assert data.receivers.shape[1] >= 1
    # 3D model: (nz, ny, nx).
    assert data.model.ndim == 3, data.model.shape
    nz, ny, nx = data.model.shape
    assert nz > 1 and ny > 1 and nx > 1
    # observed remains (nshots, nreceivers, nt).
    assert data.observed.shape == (
        data.sources.shape[0],
        data.receivers.shape[1],
        data.nt,
    )
    # Summary must expose the 3D-specific fields.
    summary = data.geometry_summary
    assert summary["equation"] == "Acoustic3D"
    assert "ny" in summary and summary["ny"] == ny
    assert "dy_m" in summary and summary["dy_m"] == cfg.dy_m
    assert "y_origin_m" in summary
    # Reciprocity: propagator-frame source y should match physical receiver y
    # (OBN nodes at y=100, y=200) — not the shot row at y=0.
    src_y_m = summary["y_origin_m"] + data.sources[:, 1].astype(np.float64) * cfg.dy_m
    assert src_y_m.min() >= 50.0, (
        f"Under CRG+3D, propagator source y must come from the physical "
        f"receiver y (∈ 100..200); got {src_y_m}."
    )


def test_prepare_wavelet_inversion_data_simulation_dt(tmp_path):
    """``simulation_dt_s`` must decouple the propagator dt from plan dt.

    plan.dt_s is 2ms; setting simulation_dt_s=1ms gives stride=2 and
    simulation_nt = (plan_nt - 1) * 2 + 1.
    """
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    plan_dt = DT_S  # 0.002
    sim_dt = DT_S / 2  # 0.001 → stride 2

    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out_simdt",
        shot_start=1,
        shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * plan_dt),
        velocity_m_s=1500.0,
        dx_m=25.0, dz_m=25.0,
        x_padding_m=200.0, z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        max_abs_offset_m=None,
        backend="cuda", mode="discrete",
        simulation_dt_s=sim_dt,
    )
    data = prepare_wavelet_inversion_data(cfg)

    # obs side is unchanged: plan dt / nt.
    assert data.dt_s == pytest.approx(plan_dt, abs=1e-12)
    assert data.observed.shape[-1] == data.nt  # plan_nt

    # simulation side: dt halved, stride = 2, simulation_nt = (plan_nt-1)*2+1.
    assert data.simulation_dt_s == pytest.approx(sim_dt, abs=1e-12)
    assert data.simulation_to_plan_stride == 2
    assert data.simulation_nt == (data.nt - 1) * 2 + 1

    # Summary mirrors the new fields for downstream consumers.
    summary = data.geometry_summary
    assert summary["simulation_dt_s"] == pytest.approx(sim_dt, abs=1e-12)
    assert summary["simulation_nt"] == data.simulation_nt
    assert summary["simulation_to_plan_stride"] == 2


def test_prepare_wavelet_inversion_data_rejects_non_integer_simulation_stride(tmp_path):
    """Non-integer stride must error out — stride sampling requires it."""
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out_bad",
        shot_start=1, shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * DT_S),
        velocity_m_s=1500.0,
        dx_m=25.0, dz_m=25.0,
        x_padding_m=200.0, z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        backend="cuda", mode="discrete",
        # plan.dt_s = 0.002; ratio = 0.002 / 0.0007 ≈ 2.857 → not integer.
        simulation_dt_s=0.0007,
    )
    with pytest.raises(ValueError, match="integer multiple"):
        prepare_wavelet_inversion_data(cfg)


def test_prepare_wavelet_inversion_data_observed_delay_shift(tmp_path):
    """Explicit ``observed_delay_s`` must right-shift the obs trace by the
    matching sample count (zero-fill prepad). Matches the legacy
    fwi_workflow-dev ``delay_traces`` semantics.
    """
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    delay_s = 0.040  # 40 ms = 20 samples at 2 ms
    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out_delay",
        shot_start=1, shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * DT_S),
        velocity_m_s=1500.0,
        dx_m=25.0, dz_m=25.0,
        x_padding_m=200.0, z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        backend="cuda", mode="discrete",
        observed_delay_s=delay_s,
    )
    data = prepare_wavelet_inversion_data(cfg)

    # Round-trip the requested shift in seconds + samples.
    delay_samples = int(round(delay_s / DT_S))
    assert data.observed_delay_s == pytest.approx(delay_s, abs=1e-9)
    assert data.geometry_summary["observed_delay_s"] == pytest.approx(delay_s, abs=1e-9)

    # First ``delay_samples`` of obs must be zero after the legacy shift.
    assert np.allclose(data.observed[..., :delay_samples], 0.0), (
        "obs prepad samples should be zeroed by the legacy right-shift"
    )


def test_prepare_wavelet_inversion_data_no_shift_when_not_configured(tmp_path):
    """If neither ``config.observed_delay_s`` nor an initial-wavelet prepad
    metadata are available, prepare must NOT shift obs and report
    observed_delay_s == 0.
    """
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out_noshift",
        shot_start=1, shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * DT_S),
        velocity_m_s=1500.0,
        dx_m=25.0, dz_m=25.0,
        x_padding_m=200.0, z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        backend="cuda", mode="discrete",
        # observed_delay_s and initial_wavelet_path both omitted.
    )
    data = prepare_wavelet_inversion_data(cfg)
    assert data.observed_delay_s == pytest.approx(0.0, abs=1e-12)


def test_prepare_wavelet_inversion_data_direct_window_mask(tmp_path):
    """When ``direct_rel_t_min/max_s`` are set, prepare must build a 0/1
    mask whose support is around the predicted direct arrival per
    (source, receiver) pair. Mirrors legacy ``direct_window_mask``.
    """
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out_mask",
        shot_start=1, shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * DT_S),
        velocity_m_s=1500.0,
        dx_m=25.0, dz_m=25.0,
        x_padding_m=200.0, z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        backend="cuda", mode="discrete",
        observed_delay_s=0.05,
        direct_rel_t_min_s=-0.02,
        direct_rel_t_max_s=0.05,
        direct_water_velocity_m_s=1500.0,
    )
    data = prepare_wavelet_inversion_data(cfg)

    assert data.direct_window_mask is not None
    assert data.direct_window_mask.shape == data.observed.shape
    assert data.direct_window_mask.dtype == np.float32
    # Mask values are exactly 0 or 1 (no antialiasing).
    unique_vals = np.unique(data.direct_window_mask)
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})
    # At least one sample is kept (otherwise prepare would have raised).
    assert float(data.direct_window_mask.sum()) > 0.0
    # Summary mirrors the mask state.
    s = data.geometry_summary
    assert s["direct_window_active"] is True
    assert s["direct_window_rel_t_min_s"] == pytest.approx(-0.02, abs=1e-9)
    assert s["direct_window_rel_t_max_s"] == pytest.approx(0.05, abs=1e-9)
    assert 0.0 < s["direct_window_active_fraction"] < 1.0


def test_prepare_wavelet_inversion_data_direct_window_mask_disabled_by_default(tmp_path):
    """Mask must stay None when either rel_t bound is omitted —
    sweep-tasks' default loss path is full-trace.
    """
    from sweep_tasks.wavelet.sweep_torch import (
        WaveletInversionConfig,
        prepare_wavelet_inversion_data,
    )

    plan_path = _build_crg_plan_with_traces(tmp_path)
    cfg = WaveletInversionConfig(
        plan_path=plan_path,
        output_dir=tmp_path / "estimate_out_nomask",
        shot_start=1, shot_stop=2,
        tmax_s=float(SAMPLES_PER_TRACE * DT_S),
        velocity_m_s=1500.0,
        dx_m=25.0, dz_m=25.0,
        x_padding_m=200.0, z_padding_m=50.0,
        model_depth_m=400.0,
        nearest_receivers=4,
        backend="cuda", mode="discrete",
        # Only one bound supplied → mask should NOT be built.
        direct_rel_t_min_s=-0.02,
    )
    data = prepare_wavelet_inversion_data(cfg)
    assert data.direct_window_mask is None
    assert data.geometry_summary["direct_window_active"] is False
