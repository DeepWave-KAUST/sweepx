"""P26: run_task must not crash off the main thread.

The optional SIGALRM timeout uses signal.signal(), which ONLY works in the main
thread. Under the Gradio web UI, run_agent_turn (hence run_task) executes in a
worker thread — arming the signal there raised
'ValueError: signal only works in main thread of the main interpreter',
which failed every FWI/forward run launched from the UI at ~1s. The fix guards
the SIGALRM path to the main thread; off-thread it silently skips the timeout.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("sweep")
pytest.importorskip("sweep_tasks")

from sweep_agent.tools import registry


def _tiny_forward_yaml(tmp_path) -> str:
    vp = np.full((24, 32), 2000.0, np.float32)
    vp_path = tmp_path / "vp.npy"
    np.save(vp_path, vp)
    b = registry.get("build_forward_spec").invoke({
        "vp_path": str(vp_path), "dh": 10.0, "dt": 0.001, "nt": 20, "fm": 10.0,
        "output_dir": str(tmp_path), "task_id": "tiny", "device": "cpu",
    })
    assert b.ok and "error" not in b.value, b.value
    return b.value["yaml_path"]


def test_run_task_with_timeout_off_main_thread(tmp_path):
    """The exact UI failure mode: run_task(timeout_s=...) from a worker thread."""
    yaml_path = _tiny_forward_yaml(tmp_path)
    out: dict = {}

    def worker():
        try:
            r = registry.get("run_task").invoke({"yaml_path": yaml_path, "timeout_s": 60})
            out["ok"], out["val"] = r.ok, r.value
        except Exception as exc:  # a raised signal error would land here
            out["exc"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=worker)
    t.start(); t.join(timeout=120)

    assert "exc" not in out, f"run_task raised off the main thread: {out.get('exc')}"
    assert out.get("ok") is True, out
    assert out["val"].get("state") == "success", out["val"]


def test_run_task_with_timeout_on_main_thread_still_works(tmp_path):
    """Main-thread path (CLI) keeps the real SIGALRM timeout armed."""
    yaml_path = _tiny_forward_yaml(tmp_path)
    r = registry.get("run_task").invoke({"yaml_path": yaml_path, "timeout_s": 60})
    assert r.ok and r.value.get("state") == "success", r.value
