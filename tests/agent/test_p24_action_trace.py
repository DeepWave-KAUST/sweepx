"""P24: Claude-style live action trace — human-readable tool labels, a pending
(spinner) block while the tool runs, collapsed with duration/outcome when done,
and the model's own narration surfaced as a chat message."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sweep_agent.agent import Agent, AgentStep
from sweep_agent.llm.base import BaseLLM, ChatMessage, ToolCall
from sweep_agent.webui import _describe_call, run_agent_turn


def test_describe_call_templates_and_fallback():
    assert _describe_call("make_synthetic_model", {"kind": "two_layer"}) == \
        "Making a two_layer synthetic model"
    assert _describe_call("inspect_file", {"path": "/tmp/deep/vp_init.npy"}) == \
        "Inspecting vp_init.npy"
    # missing template args → ellipsis, never a KeyError
    assert _describe_call("check_parameters", {"dh": 10.0}) == \
        "Checking stability (dh=10.0, dt=…, fm=…)"
    # unknown tool → prettified name
    assert _describe_call("frobnicate_gather", {}) == "Frobnicate gather"


class _OneToolLLM(BaseLLM):
    """Narrates, calls inspect_file once, then finishes."""

    def __init__(self) -> None:
        self.n = 0
        self._model_id = "fake"

    def chat(self, messages, tools=None, **kw) -> ChatMessage:
        self.n += 1
        if self.n == 1:
            return ChatMessage(role="assistant", content="Let me look at the file first.",
                               tool_calls=[ToolCall(id="c1", name="inspect_file",
                                                    arguments={"path": "/nonexistent/vp.npy"})])
        return ChatMessage(role="assistant", content="done")


def test_agent_emits_note_and_tool_start_before_tool():
    agent = Agent(llm=_OneToolLLM())
    kinds = [s.kind for s in agent.iter_chat("inspect it")]
    assert kinds == ["note", "tool_start", "tool", "final"]


def test_run_agent_turn_pending_then_done_with_duration():
    agent = Agent(llm=_OneToolLLM())
    history: list = []
    statuses: list = []
    for hist in run_agent_turn(agent, "inspect it", history):
        for m in hist:
            md = m.get("metadata") if isinstance(m, dict) else None
            if md and md.get("title", "").startswith("🔧"):
                statuses.append(md.get("status"))
                block_md = md
    assert "pending" in statuses and statuses[-1] == "done"   # spinner → collapsed
    assert "duration" in block_md                              # (X.Xs) annotation
    assert block_md["log"] == "⚠ error"                       # nonexistent file → error badge
    assert block_md["title"] == "🔧 Inspecting vp.npy"         # human label, not fn name
    # the narration became a normal chat message
    assert any(m.get("content") == "Let me look at the file first." for m in history)


class _NoStartAgent:
    """Driver that yields bare `tool` steps (no tool_start) — must still render."""

    def iter_chat(self, prompt):
        msg = ChatMessage(role="tool", content="{}")
        yield AgentStep(kind="tool", message=msg, tool_name="run_task",
                        tool_args={}, tool_result_json='{"task_dir": "/tmp/x"}')
        yield AgentStep(kind="final", message=ChatMessage(role="assistant", content="ok"))


def test_run_agent_turn_tolerates_missing_tool_start():
    history: list = []
    list(run_agent_turn(_NoStartAgent(), "go", history))
    blocks = [m for m in history if isinstance(m, dict) and m.get("metadata")]
    assert len(blocks) == 1 and blocks[0]["metadata"]["status"] == "done"
