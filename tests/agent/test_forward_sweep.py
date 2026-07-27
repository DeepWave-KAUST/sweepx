"""Tests for ``run_forward_sweep`` — the solver-only forward path.

The registration test passes today. The end-to-end test is the definition of
done: it is marked xfail while the tool is a scaffold, so implementing the tool
turns it green — delete the marker once it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


def test_registered_with_a_usable_parameter_surface():
    """The tool exists and asks for the parameters the solver actually needs."""
    assert "run_forward_sweep" in registry.tools
    fields = registry.tools["run_forward_sweep"].params_model.model_fields
    for required in ("vp_path", "dh", "dt", "fm", "equation", "device"):
        assert required in fields, f"missing parameter: {required}"
    # nt and record_length_s are the two ways to say how long to record.
    assert "nt" in fields and "record_length_s" in fields


def test_missing_solver_is_reported_as_data(monkeypatch):
    """With no solver installed the tool must return an error, never raise."""
    import sweep_agent.tools.forward_sweep as fs

    monkeypatch.setattr(fs, "_require_sweep", lambda: (None, {"error": "no solver"}))
    out = fs.run_forward_sweep.fn(
        fs.RunForwardSweepParams(vp_path="/nonexistent.npy", dh=10.0, dt=1e-3, nt=100)
    )
    assert "error" in out


def test_runs_a_shot_end_to_end(tmp_path):
    """A tiny two-layer model should come back as a (nt, nrec) shot gather."""
    pytest.importorskip("sweep")
    pytest.importorskip("torch")

    import sweep_agent.tools.forward_sweep as fs

    nz, nx, nt = 80, 120, 300
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[40:, :] = 2600.0
    vp_path = tmp_path / "vp.npy"
    np.save(vp_path, vp)

    out = fs.run_forward_sweep.fn(
        fs.RunForwardSweepParams(
            vp_path=str(vp_path), dh=10.0, dt=1.0e-3, nt=nt, fm=8.0,
            abcn=20, out_dir=str(tmp_path), plot=True,
        )
    )

    assert "error" not in out, out
    record = np.load(out["record_path"])
    assert record.ndim == 2 and record.shape[0] == nt
    assert np.isfinite(record).all() and np.abs(record).max() > 0
    assert (tmp_path / "vp.npy").exists()
    if out.get("image_path"):
        assert len(open(out["image_path"], "rb").read(8)) == 8
def test_selector_includes_run_forward_sweep():
    """A forwrd shot query must surface run_forward_sweep to the model."""
    from sweep_agent.tools.selection import select_tool_names

    picked = select_tool_names("run a forward shot and plot the gather")
    assert "run_forward_sweep" in picked