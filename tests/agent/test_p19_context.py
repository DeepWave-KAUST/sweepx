"""P19: agent context-window management — tool-result trimming + history pruning
so long tool-heavy sessions don't overflow the model's context."""

from __future__ import annotations

import json

from sweep_agent.agent import (
    Agent,
    _MAX_STR_FIELD_CHARS,
    _MAX_TOOL_RESULT_CHARS,
    _trim_tool_result,
)
from sweep_agent.llm.base import BaseLLM, ChatMessage


def test_trim_drops_spec_dict():
    big = {"spec": {"x": list(range(5000))}, "yaml_path": "/runs/a.yaml", "summary": "ok"}
    out = _trim_tool_result(json.dumps(big))
    assert "/runs/a.yaml" in out          # the useful field survives
    assert "omitted from history" in out  # the bulky spec is dropped
    assert len(out) < 500                 # without the spec dict it's tiny


def test_trim_caps_long_strings_and_keeps_paths():
    d = {"image_path": "/tmp/x.png", "blob": "z" * (_MAX_STR_FIELD_CHARS + 4000)}
    out = _trim_tool_result(json.dumps(d))
    assert "/tmp/x.png" in out                       # short useful field kept
    assert len(out) <= _MAX_STR_FIELD_CHARS + 200    # the long field was capped


def test_trim_keeps_large_informational_result_whole():
    """A big list_equations result must NOT be clipped (it IS the answer the user
    asked for) — regression for the over-aggressive 2000-char cap that truncated
    the equation list."""
    import pytest
    pytest.importorskip("sweep")
    from sweep_agent.tools import registry

    full = registry.get("list_equations").invoke({}).to_json()
    assert 2000 < len(full) <= _MAX_TOOL_RESULT_CHARS  # bigger than the OLD cap, under the new one
    out = _trim_tool_result(full)
    assert out == full  # returned whole, not truncated
    assert full.count('"models"') >= 30  # all equation entries present


def test_trim_handles_non_json():
    assert _trim_tool_result("not json at all") == "not json at all"
    # only truly enormous blobs hit the backstop cap
    assert len(_trim_tool_result("y" * (_MAX_TOOL_RESULT_CHARS + 8000))) <= _MAX_TOOL_RESULT_CHARS + 1


class _DummyLLM(BaseLLM):
    def chat(self, messages, tools=None, **kw):
        return ChatMessage(role="assistant", content="ok")

    def close(self):
        pass


def _msg(role, content, **kw):
    return ChatMessage(role=role, content=content, **kw)


def test_prune_drops_oldest_exchanges_keeps_system_and_last():
    agent = Agent(llm=_DummyLLM(), history_char_budget=1000)
    # system + three exchanges, each ~600 chars → over budget.
    agent.history = [_msg("system", "S")]
    for i in range(3):
        agent.history.append(_msg("user", f"u{i} " + "x" * 600))
        agent.history.append(_msg("assistant", f"a{i}"))
    agent._prune_history()
    assert agent.history[0].role == "system"          # system kept
    users = [m for m in agent.history if m.role == "user"]
    assert users and users[-1].content.startswith("u2")  # the latest exchange survives
    assert agent._history_chars() <= 1000 or len(users) == 1


class _LoopLLM(BaseLLM):
    """Always asks for the SAME tool call — simulates a model stuck in a loop."""
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, **kw):
        self.calls += 1
        if self.calls > 8:  # safety
            return ChatMessage(role="assistant", content="done")
        from sweep_agent.llm.base import ToolCall
        return ChatMessage(role="assistant", content=None,
                           tool_calls=[ToolCall(id=f"c{self.calls}", name="inspect_file", arguments={"path": "/x.npy"})])

    def close(self):
        pass


def test_loop_guard_short_circuits_repeated_calls():
    from sweep_agent.agent import _REPEAT_LIMIT
    agent = Agent(llm=_LoopLLM(), max_steps=10)
    guard_hits = 0
    for step in agent.iter_chat("loop please"):
        if step.kind == "tool" and "Loop guard" in (step.tool_result_json or ""):
            guard_hits += 1
    # After _REPEAT_LIMIT real executions, further identical calls are guarded.
    assert guard_hits >= 1


def test_prune_keeps_last_exchange_even_if_over_budget():
    agent = Agent(llm=_DummyLLM(), history_char_budget=10)
    agent.history = [_msg("system", "S"), _msg("user", "u " + "x" * 500), _msg("assistant", "a")]
    agent._prune_history()
    # Must not strip the only/current exchange even though it exceeds budget.
    assert any(m.role == "user" for m in agent.history)
