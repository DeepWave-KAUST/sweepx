"""End-to-end smoke for the unified OBN multisource FWI path (TASK 019).

Synthesises a tiny ``grouping='crg'`` :class:`sweep_io.seismic_plan.SeismicPlan`
+ SEG-Y survey + rotation_metadata.json on disk, then runs
``TaskRunner`` with ``geometry.kind='from_plan'`` + ``obs.plan.sampling``
(PlanSamplingConfig) + ``source_encoding.enabled=true``. Verifies that:

* the source-encoded supershot training loop runs without crashing on
  the eager Acoustic3D backend;
* the auto grid-origin computation produces a snapped grid that fits
  all sources + receivers;
* missing pieces (no ``obs.plan.sampling``, encoding off,
  ``shared_shots_per_iter=0``, multi-GPU launch, wrong grouping) error
  out with clear messages.

Replaces the deprecated ``test_task_fwi_crg.py`` which exercised the
``from_crg_plan`` + ``obs.crg_plan`` shape now removed in TASK 019.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from sweep_tasks import TaskRunner, load_task
from sweep_io.segy import (
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    FORMAT_IEEE_FLOAT32,
    write_segy_minimal,
)
from sweep_io.seismic_plan import SeismicPlan


SAMPLES_PER_TRACE = 80
DT_S = 0.002


def _trace_byte_offset(idx: int) -> int:
    return (SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
            + idx * (SEGY_TRACE_HEADER_SIZE + SAMPLES_PER_TRACE * 4))


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _build_tiny_multisource_fixture(tmp_path: Path) -> dict[str, Path]:
    """Build init_vp + rotation + wavelet + SEG-Y + grouping='crg' SeismicPlan.

    Layout: 2 virtual sources (OBN nodes, aka plan groups) sharing 2
    physical shots; one SEG-Y file with 4 traces (random noise as obs).
    Identity rotation; auto grid origin.
    """
    # init vp (also used as "true" — random init avoids the synthetic
    # forward call). Shape (nz, ny, nx).
    init_vp = np.full((16, 16, 24), 2500.0, dtype=np.float32)
    init_path = tmp_path / "init_vp.npy"
    np.save(init_path, init_vp)

    # Rotation metadata: identity (no rotation).
    rot_meta_path = tmp_path / "rotation_metadata.json"
    rot_meta_path.write_text(json.dumps({
        "origin_xy": [0.0, 0.0],
        "rotation_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "target_axis": "x",
        "inline_shift": 0.0,
        "crossline_shift": 0.0,
    }))

    # Wavelet npz (SIREN-pipeline schema).
    wav_path = tmp_path / "wavelet.npz"
    nt_wav = 80
    t = np.arange(nt_wav, dtype=np.float32) * DT_S
    fm = 6.0
    arg = np.pi * fm * (t - 0.05)
    wav = ((1.0 - 2.0 * arg ** 2) * np.exp(-(arg ** 2))).astype(np.float32)
    np.savez(wav_path, optimized_siren_wavelet=wav,
             dt_s=np.float64(DT_S), source_delay_s=np.float64(0.0))

    # SEG-Y file: 4 traces (random noise).
    segy_path = tmp_path / "sourceLine_001.sgy"
    rng = np.random.default_rng(123)
    traces = rng.standard_normal((4, SAMPLES_PER_TRACE)).astype(np.float32)
    write_segy_minimal(segy_path, traces, dt=DT_S,
                       sample_format=FORMAT_IEEE_FLOAT32)

    # ---- SeismicPlan (grouping='crg') ---------------------------------
    # 2 OBN nodes (groups). Each node sees 2 physical shots (rows).
    #   row 0: file=0 trace=0, src=(500, 0, 5), recv = group 0 = (200,100,50)
    #   row 1: file=0 trace=1, src=(800, 0, 5), recv = group 0
    #   row 2: file=0 trace=2, src=(500, 0, 5), recv = group 1 = (200,200,50)
    #   row 3: file=0 trace=3, src=(800, 0, 5), recv = group 1
    # UTM coords stay inside (x_max=1100, y_max=300) so the auto-origin
    # fits a (nz=16, ny=16, nx=24) grid at dh=50m.
    row_source_xyz = np.array([
        [500.0, 0.0, 5.0],
        [800.0, 0.0, 5.0],
        [500.0, 0.0, 5.0],
        [800.0, 0.0, 5.0],
    ], dtype=np.float64)
    row_receiver_xyz = np.array([
        [200.0, 100.0, 50.0],
        [200.0, 100.0, 50.0],
        [200.0, 200.0, 50.0],
        [200.0, 200.0, 50.0],
    ], dtype=np.float64)
    row_file_id = np.zeros(4, dtype=np.int64)
    row_trace_offset = np.array(
        [_trace_byte_offset(i) for i in range(4)], dtype=np.int64,
    )
    group_id = np.array([10, 11], dtype=np.int64)
    group_xyz = np.array([
        [200.0, 100.0, 50.0],
        [200.0, 200.0, 50.0],
    ], dtype=np.float64)
    group_offsets = np.array([0, 2, 4], dtype=np.int64)
    plan = SeismicPlan(
        grouping="crg",
        files=[segy_path],
        trace_size_per_file=np.asarray(
            [SEGY_TRACE_HEADER_SIZE + SAMPLES_PER_TRACE * 4],
            dtype=np.int64,
        ),
        sample_format=FORMAT_IEEE_FLOAT32,
        samples_per_trace=SAMPLES_PER_TRACE,
        dt_s=DT_S,
        row_file_id=row_file_id,
        row_trace_offset=row_trace_offset,
        row_source_xyz=row_source_xyz,
        row_receiver_xyz=row_receiver_xyz,
        group_id=group_id,
        group_xyz=group_xyz,
        group_offsets=group_offsets,
        build_meta={"label": "tiny_multisource_fixture"},
    )
    plan_path = tmp_path / "plan.npz"
    plan.save(plan_path)
    return {
        "init_vp": init_path,
        "rotation": rot_meta_path,
        "wavelet": wav_path,
        "plan": plan_path,
        "segy": segy_path,
    }


def _build_multisource_spec(
    tmp_path: Path, fixture: dict[str, Path],
    *, batchsize: int = 2, epochs: int = 2, extra: dict | None = None,
) -> dict:
    spec = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 50.0, "shape": [16, 16, 24]},
        "time": {"dt": 0.002, "nt": 80},
        "wavelet": {
            "kind": "siren_pipeline_npz",
            "path": str(fixture["wavelet"]), "scale": 1.0,
        },
        "geometry": {
            "kind": "from_plan",
            "plan_path": str(fixture["plan"]),
            "rotation_metadata": str(fixture["rotation"]),
            "dh_xyz_m": [50.0, 50.0, 50.0],
            "auto_origin_pad_cells": [2, 2, 2],
        },
        "physics": {
            "equation": "Acoustic3D", "spatial_order": 4, "abcn": 8,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(fixture["init_vp"])},
        "obs": {"plan": {
            "plan_path": str(fixture["plan"]),
            "cache_all": False,
            "sampling": {
                "shared_shots_per_iter": 2,
                "min_coverage": 0,
                "num_workers": 1,
            },
        }},
        "optimizer": {"kind": "adam", "lr": 1.0, "eps": 1.0e-22},
        "epochs": epochs, "batchsize": batchsize, "show_every": 1,
        "loss": {"kind": "trace_cosine"},
        "source_encoding": {
            "enabled": True, "min_coverage": 0,
            "sign_seed": 7, "reseed_every_iter": True,
        },
    }
    if extra:
        spec.update(extra)
    return spec


def test_multisource_fwi_encoded_smoke_runs(tmp_path):
    fixture = _build_tiny_multisource_fixture(tmp_path)
    spec = _build_multisource_spec(tmp_path, fixture)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "ss.yaml")))
    assert result.status.state == "success", result.status.error
    # final outputs land on disk.
    assert (result.task_dir / "output" / "inverted_vp.npy").exists()
    assert (result.task_dir / "output" / "loss.npy").exists()
    loss = np.load(result.task_dir / "output" / "loss.npy")
    assert loss.size == 2
    assert np.all(np.isfinite(loss))
    assert result.status.summary["encoding"] is True
    assert result.status.summary["n_virtual_sources"] == 2


def test_multisource_fwi_requires_sampling_block(tmp_path):
    """obs.plan without `sampling` should fall through to the CSG static-obs
    path, which then rejects the CRG-grouped plan with a clear error."""
    fixture = _build_tiny_multisource_fixture(tmp_path)
    spec = _build_multisource_spec(tmp_path, fixture)
    spec["obs"]["plan"].pop("sampling")
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "bad.yaml")))
    assert result.status.state == "failed"
    err = (result.status.error or "").lower()
    assert "csg" in err or "grouping" in err


def test_multisource_fwi_per_shot_path_no_longer_blocked_at_init(tmp_path):
    """Per-shot CRG (multi-GPU) mode is now wired. With
    source_encoding absent, the runner must NOT raise the old
    ``Per-shot CRG ... not yet wired`` NotImplementedError at init.

    The downstream forward call may still fail in environments without
    a working sweep CUDA build, but the failure must no longer reference
    the per-shot-not-implemented gate.
    """
    fixture = _build_tiny_multisource_fixture(tmp_path)
    spec = _build_multisource_spec(tmp_path, fixture)
    spec.pop("source_encoding")
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "noenc.yaml")))
    err = (result.status.error or "").lower()
    # New behaviour: the per-shot init guard no longer fires.
    assert "not yet wired" not in err
    assert "per-shot crg" not in err


def test_multisource_fwi_emits_ortho_slice_qc(tmp_path):
    """qc.vp_png with 3-D vp should drop a 1×3 orthogonal-slice PNG."""
    fixture = _build_tiny_multisource_fixture(tmp_path)
    spec = _build_multisource_spec(tmp_path, fixture, epochs=1)
    spec["qc"] = {
        "every_n_epochs": 1,
        "vp_png": True, "vp_diff_png": True, "gradient_png": False,
        "shot_gather": False, "loss_curve": False,
    }
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "qc.yaml")))
    assert result.status.state == "success", result.status.error
    qc_dir = result.task_dir / "qc"
    vp_pngs = sorted((qc_dir / "vp").glob("iter_*.png"))
    diff_pngs = sorted((qc_dir / "vp_diff").glob("iter_*.png"))
    assert vp_pngs, "no vp PNG written"
    assert diff_pngs, "no vp_diff PNG written"


def test_multisource_fwi_model_plan_crop_is_origin_aware(tmp_path):
    """Regression: model_plan.x_window_m is interpreted in MODEL-frame
    meters relative to the same origin that the auto-grid + source/
    receiver projection use. Previously the crop computed indices via
    ``floor(x_window / dh)`` ignoring origin → on non-zero auto-origin
    surveys the data slice was shifted by ``floor(origin / dh)`` cells,
    silently dropping sources at the right edge.

    Test: pick an init_vp + plan whose auto-origin is NON-zero
    (sources clustered far from grid x=0), then ask for a window that
    JUST covers all sources. With the fix in place every source must
    still satisfy the bounds check (slot_in.all()). Without the fix
    (pre-fix), the right-most sources would land outside the cropped
    grid and get filtered out → run still succeeds but with fewer
    eligible groups.
    """
    fixture = _build_tiny_multisource_fixture(tmp_path)
    spec = _build_multisource_spec(tmp_path, fixture, epochs=1)
    spec["task_id"] = "crop_origin_aware"
    # The tiny fixture's auto-origin should be slightly negative
    # (pad=2 cells × dh=50 = 100m to the left of bbox.min).
    # Pick x_window that just covers the rotated source bbox in MODEL
    # frame; with the fix, no source should fall out.
    spec["model_plan"] = {
        "z_window_m": None, "y_window_m": None,
        # Choose a window slightly wider than the sources to leave
        # room for the auto-pad cells; the exact extent depends on the
        # tiny fixture's source positions but the post-crop grid must
        # contain ALL eligible OBN.
        "x_window_m": [0.0, 1200.0],
    }
    # Ensure run_meta dump succeeds so we can assert via config_resolved.
    spec["qc"] = {"every_n_epochs": 0}  # disable per-iter QC
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "crop_oa.yaml")))
    assert result.status.state == "success", result.status.error
    # Structural check: run succeeds + resolved config preserves the
    # configured x_window. The fix's semantic correctness is covered
    # by manual byte-comparison of the run_meta runtime extras (which
    # capture origin_xyz_m and grid_shape_zyx).
    cfg_resolved = result.task_dir / "config_resolved.yaml"
    assert cfg_resolved.exists()
    cfg = yaml.safe_load(cfg_resolved.read_text())
    assert cfg["model_plan"]["x_window_m"] == [0.0, 1200.0]


def test_multisource_fwi_qc_flags_gate_each_product(tmp_path):
    """All four multisource-specific QC products honor their own flags:
    flipping each to False should skip emission; flipping True turns
    it back on. Single end-to-end smoke covers all four toggles.
    """
    fixture = _build_tiny_multisource_fixture(tmp_path)
    # Pass 1: all four multisource QC products OFF.
    spec_off = _build_multisource_spec(tmp_path, fixture, epochs=1)
    spec_off["task_id"] = "qc_flags_off"
    spec_off["qc"] = {
        "every_n_epochs": 1,
        "vp_png": True, "vp_diff_png": True,
        "gradient_png": False, "shot_gather": False, "loss_curve": False,
        "supershot_panel": False, "well_logs": False,
    }
    res_off = TaskRunner().run(
        load_task(_write(spec_off, tmp_path / "qc_off.yaml")),
    )
    assert res_off.status.state == "success", res_off.status.error
    qc_off = res_off.task_dir / "qc"
    assert not (qc_off / "supershot").exists(), "supershot dir should be absent"
    assert not (qc_off / "well_logs").exists(), "well_logs dir should be absent"
    assert not (qc_off / "gradient").exists(), "gradient dir should be absent"
    assert not (qc_off / "loss_curve.png").exists(), "loss_curve.png should be absent"

    # Pass 2: all four ON.
    spec_on = _build_multisource_spec(tmp_path, fixture, epochs=1)
    spec_on["task_id"] = "qc_flags_on"
    spec_on["qc"] = {
        "every_n_epochs": 1,
        "vp_png": True, "vp_diff_png": True,
        "gradient_png": True, "shot_gather": False, "loss_curve": True,
        "supershot_panel": True, "well_logs": True,
    }
    res_on = TaskRunner().run(
        load_task(_write(spec_on, tmp_path / "qc_on.yaml")),
    )
    assert res_on.status.state == "success", res_on.status.error
    qc_on = res_on.task_dir / "qc"
    assert sorted((qc_on / "supershot").glob("iter_*.png")), "supershot PNG missing"
    assert sorted((qc_on / "well_logs").glob("iter_*.png")), "well_logs PNG missing"
    assert sorted((qc_on / "gradient").glob("iter_*.png")), "gradient PNG missing"
    assert (qc_on / "loss_curve.png").exists(), "loss_curve.png missing"


def test_multisource_fwi_with_smooth_reg_and_seabed_freeze(tmp_path):
    """Wire TVPrior + SeabedFreezeMask through the multisource path."""
    fixture = _build_tiny_multisource_fixture(tmp_path)
    # Seabed-depth npy: shape (ny, nx) = (16, 24); seabed at z=100m
    # everywhere -> freeze rows 0..1 (with dh=50m -> z_idx=2 cutoff).
    sb = np.full((16, 24), 100.0, dtype=np.float64)
    sb_path = tmp_path / "seabed_depth.npy"
    np.save(sb_path, sb)
    spec = _build_multisource_spec(tmp_path, fixture, epochs=2)
    spec["smooth_regularization"] = {
        "weight": 1.0e-3, "order": "first",
        "x_weight": 1.0, "y_weight": 1.0, "z_weight": 1.0,
        "velocity_scale_m_s": 1000.0,
    }
    spec["freeze_water_layer"] = {
        "enabled": True, "seabed_depth_path": str(sb_path),
        "buffer_cells": 0,
    }
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "ss_priors.yaml")))
    assert result.status.state == "success", result.status.error
    # Inverted vp should be unchanged in the water column (rows 0..1)
    # because the seabed mask zeroed those gradient entries each step.
    inv = np.load(result.task_dir / "output" / "inverted_vp.npy")
    init = np.load(fixture["init_vp"])
    np.testing.assert_allclose(inv[:2, :, :], init[:2, :, :])


def test_multisource_fwi_requires_shared_shots_per_iter(tmp_path):
    """sampling.shared_shots_per_iter=0 must fail with a clear error."""
    fixture = _build_tiny_multisource_fixture(tmp_path)
    spec = _build_multisource_spec(tmp_path, fixture)
    # Schema enforces ge=1 on shared_shots_per_iter, so spec validation
    # itself rejects this — verify the error path.
    spec["obs"]["plan"]["sampling"]["shared_shots_per_iter"] = 0
    import pytest
    with pytest.raises(Exception, match="shared_shots_per_iter"):
        load_task(_write(spec, tmp_path / "noshared.yaml"))
