"""End to end: a coarse stage models the same traveltimes as the native grid.

The legacy stage grid (``round(n * dh0 / dh)`` points, native extent stretched
end to end) compressed the model against the geometry, so a coarse stage
predicted arrivals that drift with offset -- data the inversion then explains
with fake structure. Each test runs the real runner on a small linear-gradient
model where the first arrival is a diving wave (its traveltime integrates the
whole velocity profile), and compares the coarse stage against the native
grid. A negative control re-runs the same thing with the legacy mapping
patched back in, so the tests prove they can see the bug.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sweepx.tasks import TaskRunner, load_task

NZ, NX, DH0 = 41, 161, 10.0          # legacy 2x grid: (20, 80) -> z 5.3 % short
V0, GRAD = 1500.0, 2.0               # vp = V0 + GRAD * z


def _write(spec: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def _gradient_model(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    z = np.arange(NZ, dtype=np.float64)[:, None] * DH0
    vp = ((V0 + GRAD * z) * np.ones((1, NX))).astype(np.float32)
    path = tmp_path / "vp.npy"
    np.save(path, vp)
    return path


def _legacy_shape(native_shape, native_dh, dh):
    if abs(native_dh - dh) < 1e-12:
        return tuple(native_shape)
    return tuple(max(1, int(round(s * native_dh / dh))) for s in native_shape)


def _legacy_resample(t, src_dh, dst_shape, dst_dh):
    x = t.detach() if isinstance(t, torch.Tensor) else torch.from_numpy(np.asarray(t))
    out = torch.nn.functional.interpolate(
        x[None, None], size=tuple(dst_shape), mode="bilinear", align_corners=True)[0, 0]
    return out if isinstance(t, torch.Tensor) else out.numpy()


def _patch_legacy(monkeypatch, module):
    monkeypatch.setattr(module, "_stage_grid_shape", _legacy_shape)
    monkeypatch.setattr(module, "_resample_to_grid", _legacy_resample)


def _lag_s(a: np.ndarray, b: np.ndarray, i0: int, i1: int, dt: float) -> float:
    """Time shift of ``a`` against ``b`` (positive: ``a`` later) in [i0, i1)."""
    a, b = a[i0:i1], b[i0:i1]
    cc = np.correlate(a, b, mode="full")
    k = int(np.argmax(cc))
    frac = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        frac = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    return (k - (len(b) - 1) + frac) * dt


# ------------------------------------------------- conventional FWI ----

class _Captured(Exception):
    pass


def _run_conventional(tmp_path: Path, monkeypatch, *, abort: bool) -> dict:
    """2-stage FWI (2x dh, then native); returns stage 0's syn/obs/geometry.

    ``syn`` is a forward through the stage-0 model (init == truth) on the
    stage-0 grid; ``obs`` is the runner's stage-0 obs, i.e. the native-grid
    forward resampled from 2 to 4 ms.
    """
    from sweepx.tasks.tasks.fwi import FWIRunnerMixin

    vp = _gradient_model(tmp_path)
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"),
        "task_id": "ladder", "device": "cpu",
        "grid": {"dh": DH0}, "time": {"dt": 0.002, "nt": 625},
        "wavelet": {"kind": "ricker", "fm": 6.0, "delay": 0.2, "scale": 1.0},
        "geometry": {"kind": "line",
                     "sources": {"step": 200, "depth": 2, "start": 10, "stop": 20},
                     "receivers": {"step": 4, "depth": 2, "start": 14, "stop": 160}},
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 20,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "path": str(vp)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(vp)}},
        "optimizer": {"kind": "adam", "lr": 1.0, "eps": 1.0e-22},
        "epochs": 2, "batchsize": 1, "show_every": 100,
        "stages": [{"epochs": 1, "dh_m": 2 * DH0, "dt_s": 0.004},
                   {"epochs": 1, "dh_m": DH0, "dt_s": 0.002}],
    }
    seen: dict = {}
    original = FWIRunnerMixin._fwi_train_step

    def _spy(self, spec_, solver, wavelet, sources, receivers, inv_in_order,
             inv_by_name, obs, *args, **kwargs):
        if "syn" not in seen:
            with torch.no_grad():
                syn = solver(wavelet, sources, receivers,
                             models=[m.detach() for m in inv_in_order])
            seen.update(syn=syn.detach().cpu().numpy()[..., 0],
                        obs=obs.detach().cpu().numpy()[..., 0],
                        src=np.asarray(sources), rec=np.asarray(receivers),
                        shape=tuple(inv_in_order[0].shape))
            if abort:
                raise _Captured
        return original(self, spec_, solver, wavelet, sources, receivers,
                        inv_in_order, inv_by_name, obs, *args, **kwargs)

    monkeypatch.setattr(FWIRunnerMixin, "_fwi_train_step", _spy)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "ladder.yaml")))
    seen["result"] = result
    return seen


def _first_arrival_drift(seen: dict, dh: float, dt: float) -> tuple[float, float]:
    """(median lag near < 400 m, median lag far > 1000 m), seconds."""
    syn, obs, src, rec = seen["syn"], seen["obs"], seen["src"], seen["rec"]
    near, far = [], []
    for r in range(syn.shape[2]):
        off = abs(float(rec[0, r, 0] - src[0, 0])) * dh
        # diving-wave traveltime in v = V0 + GRAD z, plus the wavelet delay
        t_fa = (2.0 / GRAD) * np.arcsinh(GRAD * off / (2.0 * V0)) + 0.2
        i0, i1 = int((t_fa - 0.12) / dt), min(syn.shape[1], int((t_fa + 0.12) / dt))
        lag = _lag_s(syn[0, :, r], obs[0, :, r], max(0, i0), i1, dt)
        (near if off < 400.0 else far if off > 1000.0 else []).append(lag)
    return float(np.median(near)), float(np.median(far))


def test_coarse_stage_traveltimes_match_the_native_grid(tmp_path, monkeypatch):
    seen = _run_conventional(tmp_path, monkeypatch, abort=False)
    assert seen["result"].status.state == "success", seen["result"].status.error
    assert seen["shape"] == (21, 81)                  # spans 400 m x 1600 m at 20 m
    near, far = _first_arrival_drift(seen, 2 * DH0, 0.004)
    # what is left is the coarse grid's own (offset-independent) dispersion
    assert abs(far - near) < 1.0e-3, (near, far)
    assert abs(far) < 4.0e-3, far

    # negative control: the legacy squeeze puts the far diving waves ~6 ms
    # early (stage 0 is all it needs, so that run stops after capturing it)
    import sweepx.tasks._helpers.stages as stages_mod
    monkeypatch.undo()                                # drop the first spy
    _patch_legacy(monkeypatch, stages_mod)
    legacy = _run_conventional(tmp_path / "legacy", monkeypatch, abort=True)
    assert legacy["shape"] == (20, 80)
    near_l, far_l = _first_arrival_drift(legacy, 2 * DH0, 0.004)
    assert far_l - near_l < -4.0e-3, (near_l, far_l)


# ------------------------------------------------------ freqsel bands ----

def _freqsel_spec(tmp_path: Path, stages: list) -> dict:
    vp = _gradient_model(tmp_path)
    return {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"),
        "task_id": "bands", "device": "cpu",
        "grid": {"dh": DH0}, "time": {"dt": 0.002, "nt": 900},
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 16,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "path": str(vp)},
        "source_encoding": {"enabled": True, "mode": "frequency_selection",
                            "frequency": {
                                "synthesize_from_true": True,
                                "true_model_path": str(vp),
                                "synth_n_nodes": 3, "synth_rec_stride": 2,
                                "synth_node_z": 2,
                                "probe_samples": 250, "k_lo": 2, "k_hi": 4,
                                "steady_samples": 600, "slack_samples": 50,
                                "ramp_s": 0.3, "n_pools": 1}},
        "optimizer": {"kind": "adam", "lr": 1.0},
        "loss": {"kind": "steady_gcn"},
        # No water layer: the gradient starts at exactly 1500 m/s in row 0,
        # which this path used to pin silently; say it is not water.
        "water": False,
        "epochs": sum(s["epochs"] for s in stages), "batchsize": 1,
        "show_every": 100, "stages": stages,
    }


def _shard(path: Path):
    with np.load(path) as p:
        n = len(p["node_ids"])
        return (p["D"].reshape(n, -1, p["D"].shape[-1]),
                p["node_grid_xyz"].astype(np.int64),
                p["trace_grid_xyz"].reshape(n, -1, 2).astype(np.int64),
                p["freqs"].copy())


def _band_vs_native(band: Path, native: Path, dh_band: float):
    """Per node: (1 - GCN, median shift near < 400 m, median shift far > 1000 m)."""
    D0, nodes0, tr0, f0 = _shard(band)
    D1, nodes1, tr1, f1 = _shard(native)
    np.testing.assert_array_equal(f0, f1)
    # the band grid records the SAME physical acquisition as the native one
    np.testing.assert_array_equal(nodes0 * dh_band, nodes1 * DH0)
    np.testing.assert_array_equal(tr0 * dh_band, tr1 * DH0)
    out = []
    for n in range(D1.shape[0]):
        off = np.abs(tr1[n, :, 0] - nodes1[n, 0]) * DH0
        shift = -np.angle(D0[n] * np.conj(D1[n])) / (2 * np.pi * f1[None, :])
        gcn = 1 - abs(np.vdot(D1[n].ravel(), D0[n].ravel())) / (
            np.linalg.norm(D0[n]) * np.linalg.norm(D1[n]))
        far = np.median(shift[off > 1000.0]) if np.any(off > 1000.0) else np.nan
        out.append((gcn, float(np.median(shift[off < 400.0])), float(far)))
    return out


def test_freqsel_band_grid_synthesizes_the_native_data(tmp_path, monkeypatch):
    """A 2x band grid's synthetic coefficients match the native grid's, so
    obs extracted at native resolution (or from field data at round(x/dh))
    describe the model the band inverts."""
    spec = _freqsel_spec(tmp_path, [{"epochs": 1, "dh_m": 2 * DH0}, {"epochs": 1}])
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "bands.yaml")))
    assert result.status.state == "success", result.status.error
    task = Path(spec["output_dir"]) / "bands"
    assert np.load(task / "inverted_vp.npy").shape == (NZ, NX)
    rows = _band_vs_native(task / "freqsel_synth_obs_s0.npz",
                           task / "freqsel_synth_obs_s1.npz", 2 * DH0)
    edge = [r for r in rows if np.isfinite(r[2])]
    assert edge, "no node has far offsets; the layout changed"
    for gcn, near, far in rows:
        assert gcn < 1e-4, rows
    for gcn, near, far in edge:
        assert abs(far - near) < 0.5e-3, rows

    # negative control: the legacy band grid (one band only; the native shard
    # above does not depend on the mapping) drifts ~4 ms at the far offsets
    import sweepx.tasks.tasks.fwi_freqsel as fs_mod
    _patch_legacy(monkeypatch, fs_mod)
    lspec = _freqsel_spec(tmp_path / "legacy", [{"epochs": 1, "dh_m": 2 * DH0}])
    lres = TaskRunner().run(load_task(_write(lspec, tmp_path / "legacy.yaml")))
    assert lres.status.state == "success", lres.status.error
    rows_l = _band_vs_native(Path(lspec["output_dir"]) / "bands" / "freqsel_synth_obs_s0.npz",
                             task / "freqsel_synth_obs_s1.npz", 2 * DH0)
    for gcn, near, far in (r for r in rows_l if np.isfinite(r[2])):
        assert far - near < -2.0e-3, rows_l
        assert gcn > 5e-4, rows_l


# ------------------------------------------- reparam frame wiring ----

def _frame_spy(monkeypatch, module):
    """Record every stage-grid frame install and the coordinates it leaves."""
    from sweepx.tasks._helpers import stage_grid

    calls = []

    def _spy(net, *, shape, dh, frame_shape, frame_dh):
        stage_grid._set_physical_inr_frame(net, shape=shape, dh=dh,
                                           frame_shape=frame_shape,
                                           frame_dh=frame_dh)
        calls.append((tuple(shape), float(dh), tuple(frame_shape), float(frame_dh),
                      net.coords.detach().cpu().clone()))

    monkeypatch.setattr(module, "_set_physical_inr_frame", _spy)
    return calls


def _expected_coords(shape, dh, frame_shape, frame_dh):
    axes = [np.minimum(np.arange(n) * dh, (n0 - 1) * frame_dh) / (n0 * frame_dh)
            for n, n0 in zip(shape, frame_shape)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    return grid.reshape(-1, len(shape))


_TINY_REPARAM = {"kind": "velocity_inr", "hidden_features": 16, "hidden_layers": 1,
                 "hash": {"enabled": False}, "lr": 1.0e-4}


def test_conventional_reparam_ladder_keeps_the_native_frame(tmp_path, monkeypatch):
    import sweepx.tasks._helpers.stages as stages_mod

    calls = _frame_spy(monkeypatch, stages_mod)
    tmp_path.mkdir(exist_ok=True)
    z = np.arange(30)[:, None] * 10.0
    vp = ((2000.0 + 1.5 * z) * np.ones((1, 40))).astype(np.float32)
    np.save(tmp_path / "vp.npy", vp)
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"),
        "task_id": "inr", "device": "cpu",
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.06, "scale": 1.0},
        "geometry": {"kind": "line",
                     "sources": {"step": 15, "depth": 2, "start": 5, "stop": 36},
                     "receivers": {"step": 2, "depth": 2, "start": 2, "stop": 39}},
        "physics": {"equation": "Acoustic", "spatial_order": 8, "abcn": 10,
                    "free_surface": False, "pml_type": "cpmlr",
                    "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False,
                    "eager_options": {"use_compile": False}},
        "init_model": {"name": "vp", "path": str(tmp_path / "vp.npy")},
        "obs": {"synthetic_from": {"name": "vp", "path": str(tmp_path / "vp.npy")}},
        "optimizer": {"kind": "adam", "lr": 1.0, "eps": 1.0e-22},
        "reparam": _TINY_REPARAM,
        "epochs": 2, "batchsize": 3, "show_every": 100,
        "stages": [{"epochs": 1, "dh_m": 20.0}, {"epochs": 1, "dh_m": 10.0}],
    }
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "inr.yaml")))
    assert result.status.state == "success", result.status.error
    final = np.load(result.task_dir / "output" / "inverted_vp.npy")
    assert final.shape == (30, 40)
    assert [c[:4] for c in calls] == [((16, 21), 20.0, (30, 40), 10.0),
                                      ((30, 40), 10.0, (30, 40), 10.0)]
    np.testing.assert_allclose(calls[0][4].numpy(),
                               _expected_coords((16, 21), 20.0, (30, 40), 10.0),
                               rtol=0, atol=1e-7)


def test_freqsel_reparam_built_on_a_band_grid_uses_the_native_frame(tmp_path, monkeypatch):
    """Stage 0 builds the network ON the band grid; its coordinates must still
    be the native frame, which the native stage then reproduces as-is."""
    import sweepx.tasks.tasks.fwi_freqsel as fs_mod

    calls = _frame_spy(monkeypatch, fs_mod)
    spec = _freqsel_spec(tmp_path, [{"epochs": 1, "dh_m": 2 * DH0}, {"epochs": 1}])
    spec["source_encoding"]["frequency"].update(steady_samples=300, slack_samples=20)
    spec["time"]["nt"] = 570
    spec["reparam"] = _TINY_REPARAM
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "inr.yaml")))
    assert result.status.state == "success", result.status.error
    assert np.load(Path(spec["output_dir"]) / "bands" / "inverted_vp.npy").shape == (NZ, NX)
    assert [c[:4] for c in calls] == [((21, 81), 20.0, (NZ, NX), DH0),
                                      ((NZ, NX), DH0, (NZ, NX), DH0)]
    np.testing.assert_allclose(calls[0][4].numpy(),
                               _expected_coords((21, 81), 20.0, (NZ, NX), DH0),
                               rtol=0, atol=1e-7)
