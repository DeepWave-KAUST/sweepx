"""P18: web-UI plumbing — figure extraction, prompt composition, and the
agent-turn streaming that feeds the chat + gallery. No gradio / LLM needed for
the core logic (run_agent_turn is driven by a fake agent)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from sweep_agent.webui import _compose_prompt, _figures_in_result, run_agent_turn


def test_figures_in_result_named_and_generic(tmp_path):
    png = tmp_path / "shot.png"
    np.save(tmp_path / "x.npy", np.zeros(3))  # a non-image path must be ignored
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot([0, 1]); plt.savefig(png); plt.close()

    # named key
    assert _figures_in_result(json.dumps({"image_path": str(png), "n": 1})) == [str(png)]
    # generic key holding an image path
    assert _figures_in_result(json.dumps({"gif_path": str(png)})) == [str(png)]
    # missing file → ignored
    assert _figures_in_result(json.dumps({"image_path": str(tmp_path / "nope.png")})) == []
    # non-dict / bad json → empty
    assert _figures_in_result("not json") == []
    assert _figures_in_result(None) == []


def test_compose_prompt_folds_in_files():
    assert _compose_prompt("做正演", []) == "做正演"
    out = _compose_prompt("做正演", ["/tmp/vp.npy", "/tmp/vs.npy"])
    assert "/tmp/vp.npy" in out and "/tmp/vs.npy" in out
    # empty text still references the files
    assert "/tmp/vp.npy" in _compose_prompt("", ["/tmp/vp.npy"])


# --- fake agent to drive run_agent_turn without an LLM ---------------------

@dataclass
class _Step:
    kind: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result_json: str | None = None

    @property
    def message(self):
        return type("M", (), {"content": "done"})()


class _FakeAgent:
    def __init__(self, steps):
        self._steps = steps

    def iter_chat(self, prompt):
        yield from self._steps


def test_run_agent_turn_streams_trace_and_inline_image(tmp_path):
    png = tmp_path / "g.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot([1, 2]); plt.savefig(png); plt.close()

    agent = _FakeAgent([
        _Step("tool", "build_forward_spec", {"vp_path": "/tmp/vp.npy"}, json.dumps({"yaml_path": "/x.yaml"})),
        _Step("tool", "plot_shot_gather", {"task_dir": "/d"}, json.dumps({"image_path": str(png)})),
        _Step("final"),
    ])
    history = []
    states = list(run_agent_turn(agent, "go", history))
    final_hist = states[-1]
    # the figure is shown INLINE in the chat as a {"path": ...} message
    assert any(isinstance(m.get("content"), dict) and m["content"].get("path") == str(png) for m in final_hist)
    assert final_hist[-1]["role"] == "assistant"
    # tool name lives in the collapsible metadata title, not the bubble text
    assert any("build_forward_spec" in str(m.get("content", "")) + str(m.get("metadata", "")) for m in final_hist)
    assert final_hist[-1]["content"] == "done"


def test_run_agent_turn_handles_backend_error():
    class _Boom:
        def iter_chat(self, prompt):
            raise RuntimeError("connection refused")
            yield  # pragma: no cover

    history = []
    states = list(run_agent_turn(_Boom(), "go", history))
    assert states, "should yield an error message"
    assert "error" in states[-1][-1]["content"].lower()
