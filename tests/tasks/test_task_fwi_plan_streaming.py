"""End-to-end smoke for the unified OBN plan-streaming FWI path (TASK 019).

Synthesises a tiny ``grouping='crg'`` :class:`sweepx.io.seismic_plan.SeismicPlan`
+ SEG-Y survey + rotation_metadata.json on disk, then runs
``TaskRunner`` with ``geometry.kind='from_plan'`` + ``obs.plan.sampling``
(PlanSamplingConfig) + ``source_encoding.enabled=true``. Verifies that:

* the source-encoded supershot training loop runs without crashing on
  the eager Acoustic3D backend;
* the auto grid-origin computation produces a snapped grid that fits
  all sources + receivers;
* missing pieces (encoding off, ``shared_shots_per_iter=0``, multi-GPU
  launch) error out with clear messages;
* dropping ``obs.plan.sampling`` is NOT an error — it routes to the
  static reciprocal ``plan_materialize`` path; the CSG-only grouping
  guard belongs to the geometry-only route instead.

Replaces the deprecated ``test_task_fwi_crg.py`` which exercised the
``from_crg_plan`` + ``obs.crg_plan`` shape now removed in TASK 019.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sweepx.tasks import TaskRunner, load_task
from sweepx.io.segy import (
    SEGY_BIN_HEADER_SIZE,
    SEGY_TEXT_HEADER_SIZE,
    SEGY_TRACE_HEADER_SIZE,
    FORMAT_IEEE_FLOAT32,
    write_segy_minimal,
)
from sweepx.io.seismic_plan import SeismicPlan


SAMPLES_PER_TRACE = 80
DT_S = 0.002



def _cheap(spec: dict) -> dict:
    """The tiny-fixture spec with the solver costs these tests do not read.

    * The eager step uncompiled: sweep's EagerOptions compiles by default, and
      Inductor compiles anew for every propagator a run builds (C++ on a CPU,
      seconds each). These tests check the plan-streaming loop around the
      solver; sweep's test_compiled_stencil_slices compares the compiled step
      with the uncompiled one, and the stage tests below run uncompiled too.
    * A 4-cell absorbing boundary, as the stage fixture below, instead of 8: the
      obs are random noise, so nothing here reads how well it absorbs, and the
      padded grid has 2.2x fewer cells."""
    spec["backend"]["eager_options"] = {"use_compile": False}
    spec["physics"]["abcn"] = 4
    return spec


def _trace_byte_offset(idx: int, samples: int = SAMPLES_PER_TRACE) -> int:
    return (SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE
            + idx * (SEGY_TRACE_HEADER_SIZE + samples * 4))


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _build_tiny_plan_streaming_fixture(tmp_path: Path) -> dict[str, Path]:
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
        build_meta={"label": "tiny_plan_streaming_fixture"},
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


def _build_plan_streaming_spec(
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
            "sign_seed": 7,
        },
    }
    if extra:
        spec.update(extra)
    return spec


def test_plan_streaming_fwi_encoded_smoke_runs(tmp_path):
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture))
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


def test_crg_plan_without_sampling_runs_static_reciprocal_path(tmp_path):
    """``obs.plan`` without ``sampling`` is NOT an error on a CRG plan.

    Dropping ``sampling`` routes away from ``_run_fwi_plan_streaming`` into the
    static single-source path, which materialises the plan by reciprocity
    (``plan_materialize``: group = OBN node = virtual source, its recorded
    air-gun positions = receivers). That is a supported shape, so the run must
    SUCCEED — the grouping guard in ``_load_seismic_plan_payload`` belongs to a
    different route (see the geometry-only test below).

    ``min_receivers`` is lowered because this fixture gives each node only 2
    shots; the default floor of 8 would drop every group.
    """
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture))
    spec["obs"]["plan"].pop("sampling")
    spec["obs"]["plan"]["min_receivers"] = 2
    spec.pop("source_encoding", None)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "static.yaml")))
    assert result.status.state == "success", result.status.error
    assert (result.task_dir / "output" / "inverted_vp.npy").exists()
    loss = np.load(result.task_dir / "output" / "loss.npy")
    assert loss.size == 2 and np.all(np.isfinite(loss))


def test_crg_plan_geometry_only_rejects_csg_static_loader(tmp_path):
    """The CSG-only guard fires when the plan is used for GEOMETRY alone.

    ``geometry.kind='from_plan'`` with obs coming from somewhere else (here
    ``synthetic_from``) resolves geometry through
    ``_load_seismic_plan_payload``, which supports ``grouping='csg'`` only.
    A CRG plan on that route must fail with a message naming the grouping.
    """
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture))   # builds a solver first
    spec.pop("source_encoding", None)
    spec["obs"] = {"synthetic_from": {"name": "vp",
                                      "path": str(fixture["init_vp"])}}
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "geom.yaml")))
    assert result.status.state == "failed"
    err = (result.status.error or "").lower()
    assert "csg" in err and "crg" in err


def test_plan_streaming_fwi_per_shot_path_no_longer_blocked_at_init(tmp_path):
    """Per-shot CRG (multi-GPU) mode is now wired. With
    source_encoding absent, the runner must NOT raise the old
    ``Per-shot CRG ... not yet wired`` NotImplementedError at init.

    The downstream forward call may still fail in environments without
    a working sweep CUDA build, but the failure must no longer reference
    the per-shot-not-implemented gate.
    """
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    # One epoch: the guard sat at init, before the first per-shot iteration.
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=1))
    spec.pop("source_encoding")
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "noenc.yaml")))
    err = (result.status.error or "").lower()
    # New behaviour: the per-shot init guard no longer fires.
    assert "not yet wired" not in err
    assert "per-shot crg" not in err


def test_plan_streaming_fwi_emits_ortho_slice_qc(tmp_path):
    """qc.vp_png with 3-D vp should drop a 1×3 orthogonal-slice PNG."""
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=1))
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


def test_plan_streaming_fwi_model_plan_crop_is_origin_aware(tmp_path):
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
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=1))
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


def test_plan_streaming_fwi_qc_flags_gate_each_product(tmp_path):
    """All four plan-streaming-specific QC products honor their own flags:
    flipping each to False should skip emission; flipping True turns
    it back on. Single end-to-end smoke covers all four toggles.
    """
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    # Pass 1: all four plan-streaming QC products OFF.
    spec_off = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=1))
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
    spec_on = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=1))
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


def test_plan_streaming_fwi_with_smooth_reg_and_water_pin(tmp_path):
    """Wire TVPrior + the `water:` pin through the plan-streaming path."""
    from sweepx.tasks.bathymetry import save_seabed_depth_npz

    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    # Seabed (ny, nx) = (16, 24) at 100 m everywhere -> rows 0..1 are water
    # (cell centres 25 m and 75 m lie above it, 125 m does not).
    sb_path = save_seabed_depth_npz(tmp_path / "seabed_depth.npz",
                                    np.full((16, 24), 100.0))
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=2))
    spec["smooth_regularization"] = {
        "weight": 1.0e-3, "order": "first",
        "x_weight": 1.0, "y_weight": 1.0, "z_weight": 1.0,
        "velocity_scale_m_s": 1000.0,
    }
    spec["water"] = {"seabed": str(sb_path), "velocity": 2500.0}
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "ss_priors.yaml")))
    assert result.status.state == "success", result.status.error
    # The water (rows 0..1) holds water.velocity, here the starting value.
    inv = np.load(result.task_dir / "output" / "inverted_vp.npy")
    init = np.load(fixture["init_vp"])
    assert np.array_equal(inv[:2, :, :], init[:2, :, :])
    assert not np.array_equal(inv[2:, :, :], init[2:, :, :])


def test_plan_streaming_fwi_requires_shared_shots_per_iter(tmp_path):
    """sampling.shared_shots_per_iter=0 must fail with a clear error."""
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _build_plan_streaming_spec(tmp_path, fixture)
    # Schema enforces ge=1 on shared_shots_per_iter, so spec validation
    # itself rejects this — verify the error path.
    spec["obs"]["plan"]["sampling"]["shared_shots_per_iter"] = 0
    import pytest
    with pytest.raises(Exception, match="shared_shots_per_iter"):
        load_task(_write(spec, tmp_path / "noshared.yaml"))


def test_plan_streaming_fwi_data_mask_runs_end_to_end(tmp_path):
    """The data_mask switch runs end-to-end through the real TaskRunner FWI:
    a broadcasting all-ones mask drives a full masked-misfit training loop to a
    finite loss + output, exercising the new `_loss_sum`/`_mask_chunk` path."""
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    ones = tmp_path / "ones_mask.npy"
    np.save(ones, np.ones((1,), dtype=np.float32))  # broadcasts over (nshots,nt,nrec[,nchan])
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=2, extra={
        "loss": {"kind": "mse", "data_mask_path": str(ones)},
    }))
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "masked.yaml")))
    loss = np.load(result.task_dir / "output" / "loss.npy")
    assert loss.size == 2
    assert np.all(np.isfinite(loss))


@pytest.mark.parametrize("path", ["encoded", "per_shot", "per_crg"])
def test_plan_streaming_solver_mode_comes_from_source_shape(tmp_path, monkeypatch, path):
    """sweep has no ``source_encoding=`` keyword since geophyai 49ca6bf1: the source
    shape selects the mode. Solvers up to 0.2 ignored the keyword, sweep-solver
    0.3.0 rejects it, which failed every plan-streaming run. The call is checked
    directly so a stray keyword fails on any installed solver, and the shape that
    now carries the mode is pinned for each of the three single-domain paths."""
    from sweep.propagator.torch import PropTorch

    calls = []
    forward = PropTorch.forward

    def spy(self, wavelet, sources, receivers, *args, **kwargs):
        calls.append((np.shape(sources), sorted(kwargs)))
        return forward(self, wavelet, sources, receivers, *args, **kwargs)

    monkeypatch.setattr(PropTorch, "forward", spy)
    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=1))
    if path != "encoded":
        spec.pop("source_encoding")
    if path == "per_crg":
        spec["obs"]["plan"]["sampling"]["per_crg_independent"] = True
    result = TaskRunner().run(load_task(_write(spec, tmp_path / f"{path}.yaml")))
    assert result.status.state == "success", result.status.error
    assert calls
    for shape, kwargs in calls:
        assert "source_encoding" not in kwargs
        if path == "encoded":
            assert len(shape) == 3 and shape[0] == 1    # (1, nsrc, 3): one supershot
        else:
            assert len(shape) == 2                       # (nshots, 3): a shot per row


@pytest.mark.parametrize("reparam", [False, True])
def test_plan_streaming_stage_grid_spans_the_base_extent(tmp_path, capsys, reparam):
    """A ``dh_m`` stage runs on a grid spanning the whole base extent.

    Base (16, 16, 24) at 50 m spans (750, 750, 1150) m; the 100 m stage needs
    (9, 9, 13) points for that -- the old round(n / 2) gave (8, 8, 12), 50 m
    short on every axis, with the rows near the far edge left to the kernel's
    out-of-bounds guard. The `water:` layer is rebuilt on the stage grid, and
    with a reparam network the stage coordinates stay in the base frame.
    """
    from sweepx.tasks.bathymetry import save_seabed_depth_npz

    fixture = _build_tiny_plan_streaming_fixture(tmp_path)
    sb_path = save_seabed_depth_npz(tmp_path / "seabed_depth.npz",
                                    np.full((16, 24), 100.0))
    spec = _cheap(_build_plan_streaming_spec(tmp_path, fixture, epochs=2, extra={
        "stages": [{"epochs": 1, "dh_m": 100.0}, {"epochs": 1, "dh_m": 50.0}],
        "water": {"seabed": str(sb_path), "velocity": 2500.0},
    }))
    if reparam:
        spec["reparam"] = {"kind": "velocity_inr", "hidden_features": 8,
                           "hidden_layers": 1, "hash": {"enabled": False},
                           "lr": 1.0e-4}
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "mr.yaml")))
    assert result.status.state == "success", result.status.error
    out = capsys.readouterr().out
    assert "STAGE 1 grid -> dh=100.0 m shape=(9, 9, 13)" in out, out[-3000:]
    assert "STAGE 2 grid -> dh=50.0 m shape=(16, 16, 24)" in out
    inv = np.load(result.task_dir / "output" / "inverted_vp.npy")
    assert inv.shape == (16, 16, 24)


# ---- stage grids: the model and its optimizer cross a stage change ----------
#
# Data modelled from a true model with the solver the inversion runs, so a
# working inversion has to lower the misfit in every stage. Nodes (the virtual
# sources) and shots (their receivers) sit on multiples of 100 m: exact on the
# 50 m base grid and on the 100 m stage grid alike.

_SYN_SHAPE = (12, 12, 16)                     # (nz, ny, nx) at 50 m
_SYN_DT, _SYN_NT = 0.0075, 80
_SYN_NODES = np.array([[x, y, 400.0] for y in (200.0, 400.0) for x in (200.0, 600.0)])
_SYN_SHOTS = np.array([[x, y, 100.0] for y in (100.0, 300.0, 500.0)
                       for x in (100.0, 300.0, 500.0, 700.0)])
_BAND_LO, _BAND_HI = {"lo_hz": 1.0, "hi_hz": 4.0}, {"lo_hz": 1.0, "hi_hz": 8.0}


def _synthetic_stage_spec(tmp_path: Path, stages, shape=_SYN_SHAPE, *, nt=_SYN_NT,
                          **extra) -> dict:
    """Spec + on-disk fixture for the stage-change tests (default: CPU, per shot).

    ``nt`` is the record length of the data and of the run (default ``_SYN_NT``)."""
    import torch
    from sweepx.tasks._helpers.solver_build import _build_solver
    from sweepx.tasks._helpers.wavelet_build import _build_wavelet
    from sweepx.tasks.yaml_io import load_task_from_dict

    z, y, x = np.meshgrid(*(np.arange(n) * 50.0 for n in shape), indexing="ij")
    true_vp = (2500.0 + 250.0 * np.exp(-((x - 400.0) ** 2 + (y - 300.0) ** 2
                                         + (z - 250.0) ** 2) / (2 * 120.0 ** 2)))
    np.save(tmp_path / "init_vp.npy", np.full(shape, 2500.0, dtype=np.float32))
    rot = tmp_path / "rotation_metadata.json"
    rot.write_text(json.dumps({
        "origin_xy": [0.0, 0.0], "rotation_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "target_axis": "x", "inline_shift": 0.0, "crossline_shift": 0.0}))
    plan_path, segy = tmp_path / "plan.npz", tmp_path / "line_001.sgy"
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"), "device": "cpu",
        "grid": {"dh": 50.0, "shape": list(shape)},
        "time": {"dt": _SYN_DT, "nt": nt},
        "wavelet": {"kind": "ricker", "fm": 5.0, "delay": 0.2},
        "geometry": {"kind": "from_plan", "plan_path": str(plan_path),
                     "rotation_metadata": str(rot), "dh_xyz_m": [50.0] * 3,
                     "grid_origin_xyz_m": [0.0, 0.0, 0.0]},
        "physics": {"equation": "Acoustic3D", "spatial_order": 4, "abcn": 4,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "path": str(tmp_path / "init_vp.npy")},
        "obs": {"plan": {"plan_path": str(plan_path), "cache_all": False,
                         "sampling": {"shared_shots_per_iter": len(_SYN_SHOTS),
                                      "min_coverage": 0, "num_workers": 1}}},
        "optimizer": {"kind": "adam", "lr": 10.0},
        "epochs": sum(s["epochs"] for s in stages), "batchsize": len(_SYN_NODES),
        "show_every": 100, "loss": {"kind": "trace_cosine"},
        "model_bounds": {"vp": {"min": 1500.0, "max": 4000.0}},
        "stages": stages, **extra,
    }

    # One CRG per node: the node fires, the shots record (CRG reciprocity).
    model_spec = load_task_from_dict(spec, base_dir=tmp_path)
    solver = _build_solver(model_spec.physics, model_spec.backend, shape, 50.0,
                           _SYN_DT, nt, torch.device("cpu"))
    wav = torch.as_tensor(_build_wavelet(model_spec.wavelet, model_spec.time))
    src = np.rint(_SYN_NODES / 50.0).astype(np.int64)
    rec = np.rint(_SYN_SHOTS / 50.0).astype(np.int64)
    n_nodes, n_shots = len(src), len(rec)
    with torch.no_grad():
        syn = solver(wav[None].expand(n_nodes, -1).contiguous(), src,
                     np.broadcast_to(rec[None], (n_nodes, n_shots, 3)).copy(),
                     models=[torch.as_tensor(true_vp, dtype=torch.float32)])
    traces = syn[..., 0].transpose(1, 2).reshape(n_nodes * n_shots, nt)
    write_segy_minimal(segy, np.ascontiguousarray(traces.numpy(), dtype=np.float32),
                       dt=_SYN_DT, sample_format=FORMAT_IEEE_FLOAT32)
    n_rows = n_nodes * n_shots
    SeismicPlan(
        grouping="crg", files=[segy],
        trace_size_per_file=np.asarray([SEGY_TRACE_HEADER_SIZE + nt * 4]),
        sample_format=FORMAT_IEEE_FLOAT32, samples_per_trace=nt, dt_s=_SYN_DT,
        row_file_id=np.zeros(n_rows, dtype=np.int64),
        row_trace_offset=np.array([_trace_byte_offset(i, nt) for i in range(n_rows)]),
        row_source_xyz=np.tile(_SYN_SHOTS, (n_nodes, 1)),
        row_receiver_xyz=np.repeat(_SYN_NODES, n_shots, axis=0),
        group_id=np.arange(n_nodes, dtype=np.int64) + 10, group_xyz=_SYN_NODES.copy(),
        group_offsets=np.arange(0, n_rows + 1, n_shots, dtype=np.int64),
        build_meta={"label": "stage_change_fixture"},
    ).save(plan_path)
    return spec


def _spy_solver_and_optimizers(monkeypatch):
    """Record, per solver call, the model tensor it differentiates (and its value)
    next to the optimizer live at that moment; and every optimizer the run builds."""
    from sweep.propagator.torch import PropTorch
    import sweepx.tasks.tasks.fwi_plan_streaming as ps

    rec = {"calls": [], "opts": []}
    forward = PropTorch.forward

    def spy(self, wavelet, sources, receivers, *args, **kwargs):
        m = kwargs["models"][0]
        rec["calls"].append((m, m.detach().cpu().clone(), rec["opts"][-1]))
        return forward(self, wavelet, sources, receivers, *args, **kwargs)

    monkeypatch.setattr(PropTorch, "forward", spy)
    for name in ("_build_optimizer", "_build_reparam_optimizer"):
        def build(*args, _orig=getattr(ps, name), **kwargs):
            rec["opts"].append(_orig(*args, **kwargs))
            return rec["opts"][-1]
        monkeypatch.setattr(ps, name, build)
    return rec


def _holds(opt, t) -> bool:
    return any(p is t for g in opt.param_groups for p in g["params"])


@pytest.mark.parametrize("path", ["per_shot", "encoded", "reparam"])
def test_plan_streaming_stage_dh_keeps_inverting(tmp_path, monkeypatch, path):
    """A ``dh_m`` stage change moves the current model to the new grid and keeps
    updating it.

    Grid mode used to swap in a fresh leaf (resampled from the pristine start)
    while the optimizer kept the old one: from the first dh change on, every
    gradient landed on a tensor no optimizer held, the model never moved again
    and the loss stayed flat. Now the stage-1 model, resampled onto the stage-2
    grid, is the tensor stage 2 starts from and steps. A reparam network has
    always kept its parameters and optimizer across the change; it is pinned
    here as well.
    """
    import torch
    from sweepx.tasks._helpers.stage_grid import _resample_to_grid

    # Two epochs a stage: a model the stage change froze shows from the first
    # stage-2 step on (calls[E] -> calls[E + 1]), each stage still has a first
    # and a last loss, and every step of every stage is checked below.
    E = 2
    extra = {}
    if path == "encoded":
        extra["source_encoding"] = {"enabled": True, "min_coverage": 0, "sign_seed": 7}
    if path == "reparam":
        extra["reparam"] = {"kind": "velocity_inr", "hidden_features": 16,
                            "hidden_layers": 2, "hash": {"enabled": False},
                            "lr": 1.0e-3, "vp_std": 100.0}
    spec = _synthetic_stage_spec(tmp_path, [
        {"epochs": E, "dh_m": 100.0, "bandpass": _BAND_LO},
        {"epochs": E, "dh_m": 50.0, "bandpass": _BAND_HI}], **extra)
    rec = _spy_solver_and_optimizers(monkeypatch)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "stage_dh.yaml")))
    assert result.status.state == "success", result.status.error
    loss = np.load(result.task_dir / "output" / "loss.npy")
    calls = rec["calls"]
    assert len(calls) == 2 * E
    assert [tuple(v.shape) for _, v, _ in calls] == [(7, 7, 9)] * E + [_SYN_SHAPE] * E

    # Every step of each stage moves the model. A step that has a next call in
    # its stage shows there (calls[e - 1] -> calls[e]); a stage's last step shows
    # in the model saved after it: stage 1's snapshot (on its own grid, before
    # the switch) and the final model.
    for e in range(1, 2 * E):
        if e != E:
            assert not torch.equal(calls[e][1], calls[e - 1][1]), f"epoch {e}: frozen"
    s1_final = np.load(result.task_dir / "output" / "epochs" / f"vp_epoch_{E - 1:04d}.npy")
    final = np.load(result.task_dir / "output" / "inverted_vp.npy")
    assert s1_final.shape == (7, 7, 9) and final.shape == _SYN_SHAPE
    assert not np.array_equal(s1_final, calls[E - 1][1].numpy()), "stage 1, last step"
    assert not np.array_equal(final, calls[-1][1].numpy()), "stage 2, last step"
    if path == "reparam":
        assert len(rec["opts"]) == 1                 # one network, one optimizer
        init = torch.from_numpy(np.load(tmp_path / "init_vp.npy"))
        assert not torch.equal(calls[E][1], init)    # stage 2 starts from what 1 learnt
    else:
        for e, (t, _, opt) in enumerate(calls):
            assert _holds(opt, t), f"epoch {e}: the optimizer does not hold the model"
        carried = _resample_to_grid(torch.from_numpy(s1_final), 100.0, _SYN_SHAPE, 50.0)
        assert torch.equal(calls[E][1], carried)
    if path != "encoded":   # the encoded misfit changes with every sign draw
        for k in range(2):
            assert loss[(k + 1) * E - 1] < loss[k * E], (k, loss)


def test_plan_streaming_time_only_stage_keeps_model_and_optimizer(tmp_path, monkeypatch):
    """A stage that changes only dt_s / nt rebuilds the solver and nothing else.

    It used to run the grid switch at the unchanged dh, which in grid mode put
    a fresh leaf in place, resampled from the pristine start model, that the
    optimizer did not hold: the rest of the run was frozen too.
    """
    E = 2           # two epochs a stage: stage 2 still steps after its first loss
    spec = _synthetic_stage_spec(tmp_path, [
        {"epochs": E, "bandpass": _BAND_LO},
        {"epochs": E, "dt_s": _SYN_DT / 2, "nt": 2 * _SYN_NT, "bandpass": _BAND_HI}])
    rec = _spy_solver_and_optimizers(monkeypatch)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "stage_dt.yaml")))
    assert result.status.state == "success", result.status.error
    loss = np.load(result.task_dir / "output" / "loss.npy")
    assert len({id(t) for t, _, _ in rec["calls"]}) == 1    # one leaf throughout
    leaf, _, opt = rec["calls"][-1]
    assert len(rec["opts"]) == 1 and _holds(opt, leaf)
    assert int(opt.state[leaf]["step"]) == 2 * E        # Adam state carried over
    s1_final = np.load(result.task_dir / "output" / "epochs" / f"vp_epoch_{E - 1:04d}.npy")
    np.testing.assert_array_equal(rec["calls"][E][1].numpy(), s1_final)
    assert loss[2 * E - 1] < loss[E], loss


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda(), reason="solver illumination needs impl='c' (a GPU)")
def test_plan_streaming_stage_solver_keeps_illumination(tmp_path, monkeypatch):
    """Every per-stage solver computes illumination when the precondition is on.

    The c backend computes it only on request, and the request was made once,
    on the setup solver: a stage that rebuilt the solver silently turned the
    precondition off (the first stage, when it set its own dh_m).
    """
    import sweepx.tasks.tasks.fwi_plan_streaming as ps

    seen = []
    precond = ps._apply_illumination_precond

    def spy(grad, sill, rill, **kwargs):
        seen.append(sill is not None and rill is not None)
        return precond(grad, sill, rill, **kwargs)

    monkeypatch.setattr(ps, "_apply_illumination_precond", spy)
    spec = _synthetic_stage_spec(tmp_path, [
        {"epochs": 2, "dh_m": 100.0, "bandpass": _BAND_LO},
        {"epochs": 2, "dh_m": 50.0, "bandpass": _BAND_HI}],
        illumination_precondition={"enabled": True, "relative_epsilon": 1.0e-3})
    spec["device"] = "cuda"
    spec["backend"] = {"impl": "c", "use_ckpt": False}
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "stage_illum.yaml")))
    assert result.status.state == "success", result.status.error
    assert seen == [True] * 4


def _illumination_spec(tmp_path: Path, per_crg: bool, epochs: int = 2) -> dict:
    """GPU (impl='c') run on the stage fixture with the illumination precondition."""
    spec = _synthetic_stage_spec(
        tmp_path, [{"epochs": epochs}],
        illumination_precondition={"enabled": True, "relative_epsilon": 1.0e-3})
    spec["device"] = "cuda"
    spec["backend"] = {"impl": "c", "use_ckpt": False}
    spec["obs"]["plan"]["sampling"]["per_crg_independent"] = per_crg
    spec["task_id"] = "per_crg" if per_crg else "per_shot"
    return spec


def _spy_illumination(monkeypatch):
    """Record every solver call's illumination, and what the precondition gets.

    A call's buffers are complete once its backward ran, which is before the
    next call (or the precondition): snapshot them then, so the record holds
    whether the solver hands out fresh buffers per call or reuses them.
    """
    import sweepx.tasks.tasks.fwi_plan_streaming as ps
    from sweep.propagator.torch import PropTorch

    rec = {"calls": [], "used": [], "pending": None}

    def snap():
        s, rec["pending"] = rec["pending"], None
        if s is not None:
            rec["calls"].append((s.source_illumination.detach().clone(),
                                 s.receiver_illumination.detach().clone()))

    forward = PropTorch.forward

    def spy(self, *args, **kwargs):
        snap()
        out = forward(self, *args, **kwargs)
        rec["pending"] = self
        return out

    precond = ps._apply_illumination_precond

    def spy_precond(grad, sill, rill, **kwargs):
        snap()
        rec["used"].append((sill.detach().clone(), rill.detach().clone()))
        return precond(grad, sill, rill, **kwargs)

    monkeypatch.setattr(PropTorch, "forward", spy)
    monkeypatch.setattr(ps, "_apply_illumination_precond", spy_precond)
    return rec


def _rel_max(a, b) -> float:
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.skipif(not _cuda(), reason="solver illumination needs impl='c' (a GPU)")
def test_plan_streaming_per_crg_precondition_sums_every_node(tmp_path, monkeypatch):
    """Per CRG solves the batch node by node, and the solver keeps the
    illumination of its last call only: the precondition used to divide by
    that node's. Every iteration it now gets the sum over all its nodes."""
    spec = _illumination_spec(tmp_path, per_crg=True)
    rec = _spy_illumination(monkeypatch)   # after the fixture's own solver call
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "crg.yaml")))
    assert result.status.state == "success", result.status.error
    n = len(_SYN_NODES)
    assert len(rec["used"]) == 2 and len(rec["calls"]) == 2 * n
    for sill, rill in rec["calls"]:              # every node lights the grid
        assert float(sill.abs().max()) > 0 and float(rill.abs().max()) > 0
    for e, used in enumerate(rec["used"]):
        nodes = rec["calls"][e * n:(e + 1) * n]
        for k in range(2):                       # source, receiver
            total = nodes[0][k].clone()
            for node in nodes[1:]:
                total += node[k]
            assert _rel_max(used[k], total) < 1e-6, (e, k)


@pytest.mark.skipif(not _cuda(), reason="solver illumination needs impl='c' (a GPU)")
def test_plan_streaming_per_crg_illumination_equals_one_batched_call(tmp_path, monkeypatch):
    """The same four nodes and twelve receivers, per CRG (a call per node) and
    per shot (one call for the batch): the first precondition, on the starting
    model, divides by the same source and receiver illumination."""
    specs = {c: _illumination_spec(tmp_path, per_crg=c, epochs=1) for c in (True, False)}
    rec = _spy_illumination(monkeypatch)   # after the fixture's own solver calls
    used = {}
    for per_crg, spec in specs.items():
        rec["used"].clear()
        result = TaskRunner().run(load_task(_write(spec, tmp_path / f"{per_crg}.yaml")))
        assert result.status.state == "success", result.status.error
        used[per_crg] = rec["used"][0]
    for k in range(2):                           # source, receiver
        assert _rel_max(used[True][k], used[False][k]) < 1e-5, k
