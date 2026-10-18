"""Wavefield / forward guards: eager-only snapshots, (x, z) geometry, NaN / Inf.

* The bundled wavefield template shipped ``backend.impl: c``, which cannot
  capture snapshots; the C propagator then died on an unexpected-keyword
  TypeError. The template is eager now, and ``impl: c`` is rejected at load,
  as is ``use_ckpt: true`` (sweep refuses return_wavefield with checkpointing,
  but only after the models were built).
* ``snapshot_times`` come back from the solver sorted and de-duplicated, so an
  unsorted list mislabelled snapshots.npy / the figure; out-of-range ones only
  failed after the models were built. Both are load-time errors now.
* The explicit-geometry examples in the forward / fwi templates were written
  as ``[z, x]`` while the runner and sweep take ``[x, z]``.
* A forward that blew up (NaN / Inf) saved record.npy and reported success.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from sweepx.tasks import TaskRunner, WavefieldSpec, load_task, new_template
from sweepx.tasks._helpers.geometry import _build_geometry
from sweepx.tasks._helpers.util import _nonfinite_summary, _raise_if_nonfinite
from sweepx.tasks.schemas import ExplicitGeometry, LineGeometry
from sweepx.tasks.yaml_io import _read_bundled_template


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _tiny_spec(tmp_path: Path, task_type: str = "forward", *,
               vp=2200.0, dt: float = 0.001, nt: int = 200) -> dict:
    shape = (48, 48)
    if np.isscalar(vp):
        model = {"name": "vp", "constant": float(vp), "shape": list(shape)}
    else:
        np.save(tmp_path / "vp.npy", np.asarray(vp, dtype=np.float32))
        model = {"name": "vp", "path": str(tmp_path / "vp.npy")}
    spec = {
        "task_type": task_type,
        "output_dir": str(tmp_path / "tasks"),
        "device": "cpu",
        "grid": {"dh": 10.0},
        "time": {"dt": dt, "nt": nt},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 1, "depth": 4, "start": 24, "stop": 25},
            "receivers": {"step": 2, "depth": 6, "start": 4, "stop": 44},
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 12,
            "free_surface": False, "source_type": ["h1"],
            "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "models": [model],
    }
    if task_type == "wavefield":
        spec["snapshot_times"] = [50, 100, 150]
        spec["plot"] = False
    return spec


# ---------- 1. the bundled wavefield template is eager -------------------

def test_bundled_wavefield_template_is_eager_and_loads(tmp_path):
    text = _read_bundled_template("wavefield")
    path = tmp_path / "wavefield.yaml"
    path.write_text(text)          # what `sweep-tasks init/new wavefield` writes
    spec = load_task(path)
    assert isinstance(spec, WavefieldSpec)
    assert spec.backend.impl == "eager"
    # the dict path of `new_template` agrees with the file
    assert new_template("wavefield")["backend"]["impl"] == "eager"


# ---------- 2. impl c / bad snapshot_times are load-time errors ----------

@pytest.mark.parametrize("backend, match", [
    ({"impl": "c"},
     r"snapshot capture is eager-only; set backend: \{impl: eager\}"),
    ({"impl": "eager", "use_ckpt": True},
     r"use_ckpt=true cannot return snapshots .*set backend\.use_ckpt: false"),
], ids=["impl_c", "eager_use_ckpt"])
def test_wavefield_backend_without_snapshots_rejected_at_load(
        tmp_path, backend, match):
    spec = _tiny_spec(tmp_path, "wavefield")
    spec["backend"] = backend
    with pytest.raises(ValidationError, match=match):
        load_task(_write(spec, tmp_path / "wf_backend.yaml"))


@pytest.mark.parametrize("backend", [
    {"impl": "eager", "use_ckpt": False},
    {"impl": "eager", "use_ckpt": False,
     "eager_options": {"store_last_wavefield": True}},
    None,                                                # no backend block
], ids=["eager", "eager_options", "default"])
def test_wavefield_valid_backends_still_load(tmp_path, backend):
    spec = _tiny_spec(tmp_path, "wavefield")
    if backend is None:
        del spec["backend"]
    else:
        spec["backend"] = backend
    loaded = load_task(_write(spec, tmp_path / "wf_ok.yaml"))
    assert loaded.backend.impl == "eager" and not loaded.backend.use_ckpt


def test_forward_impl_c_still_allowed(tmp_path):
    spec = _tiny_spec(tmp_path, "forward")
    spec["backend"] = {"impl": "c"}
    assert load_task(_write(spec, tmp_path / "fwd_c.yaml")).backend.impl == "c"


@pytest.mark.parametrize("times, match", [
    ([50, 200], r"outside \[0, 199\]"),
    ([-1, 50], r"outside \[0, 199\]"),
    ([100, 50], r"strictly increasing"),
    ([50, 50, 100], r"strictly increasing"),
])
def test_wavefield_bad_snapshot_times_rejected(tmp_path, times, match):
    spec = _tiny_spec(tmp_path, "wavefield")
    spec["snapshot_times"] = times
    with pytest.raises(ValidationError, match=match):
        load_task(_write(spec, tmp_path / "wf_times.yaml"))


def test_new_template_wavefield_refuses_backend_c():
    with pytest.raises(ValueError, match="eager-only"):
        new_template("wavefield", backend="c")


def test_cli_new_wavefield_refuses_backend_c(capsys):
    from sweepx.tasks.cli import main
    assert main(["new", "wavefield", "--backend", "c"]) == 2
    assert "eager-only" in capsys.readouterr().out


@pytest.mark.parametrize("field, value, hint", [
    ("impl", "c", "impl: eager"),
    ("use_ckpt", True, "backend.use_ckpt: false"),
])
def test_runtime_backstop_refuses_before_building(tmp_path, field, value, hint):
    """A spec that skipped validation still gets the clear error, not the C
    propagator's unexpected-keyword TypeError or sweep's checkpointing
    ValueError raised after the models were built."""
    spec = load_task(_write(_tiny_spec(tmp_path, "wavefield"),
                            tmp_path / "wf.yaml"))
    setattr(spec.backend, field, value)  # no validate_assignment: skips the check
    result = TaskRunner().run(spec)
    assert result.status.state == "failed"
    assert result.status.error.startswith("NotImplementedError")
    assert hint in result.status.error


# ---------- 3. template explicit-geometry examples are (x, z) ------------

def _commented_explicit_geometry(template: str) -> dict:
    """The commented-out ``kind: explicit`` alternative of a bundled template,
    uncommented and parsed."""
    lines = _read_bundled_template(template).splitlines()
    for i, line in enumerate(lines):
        if (line.strip() == "# geometry:" and i + 1 < len(lines)
                and lines[i + 1].lstrip("# ").startswith("kind: explicit")):
            block = ["geometry:"]
            for nxt in lines[i + 1:]:
                if not nxt.startswith("#   "):
                    break
                block.append(nxt[2:])
            return yaml.safe_load("\n".join(block))["geometry"]
    raise AssertionError(f"no commented explicit geometry in {template}.yaml")


@pytest.mark.parametrize("template, src_z, rec_z", [
    ("forward", 1, 18),
    ("fwi", 1, 1),
])
def test_template_explicit_geometry_is_x_z(template, src_z, rec_z):
    geom = ExplicitGeometry(**_commented_explicit_geometry(template))
    shape = (281, 1361)                                  # (nz, nx)
    sources, receivers = _build_geometry(geom, shape)
    # sources at the surface row, laterally at x = 100 and 500
    np.testing.assert_array_equal(sources, [[100, src_z], [500, src_z]])
    np.testing.assert_array_equal(receivers[0],
                                  [[0, rec_z], [1, rec_z], [2, rec_z]])
    assert receivers.shape == (2, 3, 2)
    # every z sits inside nz and every x inside nx: read as (z, x) the same
    # numbers would put the sources 100 / 500 cells deep
    assert (sources[:, 1] < shape[0]).all() and (sources[:, 0] < shape[1]).all()


def test_grid_geometry_points_are_xyz_y_fastest():
    """``kind: grid`` points are (x, y, z), enumerated x-major with y varying
    fastest (the order existing shot lists and obs files rely on)."""
    from sweepx.tasks.schemas import GridGeometry
    gs = {"x": {"start": 0, "stop": 3, "step": 1},
          "y": {"start": 0, "stop": 2, "step": 1}, "depth": 1}
    sources, _ = _build_geometry(GridGeometry(sources=gs, receivers=gs),
                                 (10, 4, 6))              # (nz, ny, nx)
    np.testing.assert_array_equal(
        sources, [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [2, 0, 1],
                  [2, 1, 1]])


def test_forward_template_explicit_matches_its_line_geometry():
    """The explicit alternative picks points of the active `kind: line` block,
    which `_line_array` builds as (x, z): same convention, same points."""
    active = yaml.safe_load(_read_bundled_template("forward"))["geometry"]
    line_src, line_rec = _build_geometry(LineGeometry(**active), (281, 1361))
    geom = ExplicitGeometry(**_commented_explicit_geometry("forward"))
    ex_src, ex_rec = _build_geometry(geom, (281, 1361))
    line_src_set = {tuple(p) for p in line_src.tolist()}
    line_rec_set = {tuple(p) for p in line_rec[0].tolist()}
    assert {tuple(p) for p in ex_src.tolist()} <= line_src_set
    assert {tuple(p) for p in ex_rec[0].tolist()} <= line_rec_set


# ---------- 4. NaN / Inf in the output fails the task -------------------

def test_nonfinite_summary_locates_first_bad_value_in_chunks():
    rec = np.zeros((3, 50, 7, 1), np.float32)
    rec[1, 20, 3, 0] = np.nan
    rec[2, 0, 0, 0] = np.inf
    rec[2, 49, 6, 0] = -np.inf
    # chunk smaller than one shot: the scan must still find and count all
    msg = _nonfinite_summary(rec, "record.npy", "/x/record.npy",
                             axis_names=("shot", "time step", "receiver"),
                             chunk_elems=16)
    assert "3 non-finite value(s)" in msg and "(1 NaN, 2 Inf)" in msg
    assert "shot 1, time step 20, receiver 3" in msg
    assert "/x/record.npy" in msg
    assert _nonfinite_summary(np.ones((2, 3), np.float32), "ok") is None
    with pytest.raises(ValueError, match="record.npy contains 3"):
        _raise_if_nonfinite(None, msg)
    _raise_if_nonfinite(None, None)                      # all finite: no-op


def test_nonfinite_summary_handles_non_contiguous_input():
    a = np.zeros((4, 6, 5), np.float32)
    a[2, 1, 3] = np.nan
    t = a.transpose(1, 0, 2)                             # not C-contiguous
    msg = _nonfinite_summary(t, "a", axis_names=("i", "j", "k"), chunk_elems=4)
    assert "i 1, j 2, k 3" in msg


def test_forward_nan_model_fails_but_keeps_record(tmp_path):
    vp = np.full((48, 48), 2200.0, np.float32)
    vp[30:34, 20:24] = np.nan
    spec = _tiny_spec(tmp_path, "forward", vp=vp)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "fwd.yaml")))
    assert result.status.state == "failed"
    assert result.status.error.startswith("ValueError: record.npy contains")
    assert "non-finite" in result.status.error and "shot 0" in result.status.error
    rec = np.load(result.task_dir / "output" / "record.npy")
    assert not np.isfinite(rec).all()                    # written for inspection


def test_forward_cfl_blowup_fails(tmp_path):
    # vmax * dt / dh = 2200 * 0.01 / 10 = 2.2: far past any FD stability limit.
    spec = _tiny_spec(tmp_path, "forward", dt=0.01, nt=200)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "fwd_cfl.yaml")))
    assert result.status.state == "failed"
    assert "record.npy contains" in result.status.error
    assert (result.task_dir / "output" / "record.npy").exists()


def test_wavefield_nan_model_fails_naming_field_and_step(tmp_path):
    vp = np.full((48, 48), 2200.0, np.float32)
    vp[30:34, 20:24] = np.nan
    spec = _tiny_spec(tmp_path, "wavefield", vp=vp)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "wf.yaml")))
    assert result.status.state == "failed"
    err = result.status.error
    assert "snapshots.npy contains" in err and "record.npy contains" in err
    assert "(t-step 50)" in err and "field 0 (h1)" in err
    out = result.task_dir / "output"
    assert (out / "record.npy").exists() and (out / "snapshots.npy").exists()


@pytest.mark.parametrize("task_type", ["forward", "wavefield"])
def test_finite_run_still_succeeds(tmp_path, task_type):
    spec = _tiny_spec(tmp_path, task_type)
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "ok.yaml")))
    assert result.status.state == "success", result.status.error
    rec = np.load(result.task_dir / "output" / "record.npy")
    assert np.isfinite(rec).all() and np.abs(rec).max() > 0
