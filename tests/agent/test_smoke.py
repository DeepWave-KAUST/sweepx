"""Smoke test for the P1 wiring: tool registry, agent loop, and OpenAI specs.

No vLLM dependency — we plug a :class:`FakeLLM` that returns canned tool calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Iterable

import numpy as np

from sweep_agent.agent import Agent
from sweep_agent.llm.base import BaseLLM, ChatMessage, ToolCall
from sweep_agent.tools import registry


class FakeLLM(BaseLLM):
    """Returns a scripted sequence of assistant messages, one per `chat()` call."""

    def __init__(self, scripted: list[ChatMessage]) -> None:
        self._queue = list(scripted)
        self.calls: list[list[ChatMessage]] = []
        self._model_id = "fake"

    def chat(
        self,
        messages: list[ChatMessage],
        tools: Iterable[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        self.calls.append(list(messages))
        if not self._queue:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self._queue.pop(0)


def test_inspect_file_tool_is_registered():
    tool = registry.get("inspect_file")
    assert tool is not None
    spec = tool.openai_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "inspect_file"
    assert "path" in spec["function"]["parameters"]["properties"]


def test_agent_loop_invokes_tool_and_returns_reply(tmp_path):
    arr_path = tmp_path / "vp.npy"
    np.save(arr_path, np.zeros((200, 500), dtype=np.float32))

    scripted = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", name="inspect_file", arguments={"path": str(arr_path)})],
        ),
        ChatMessage(role="assistant", content="velocity model 200x500 float32"),
    ]
    fake = FakeLLM(scripted)
    agent = Agent(llm=fake)

    final = agent.chat("帮我看一下 vp.npy")

    assert final.role == "assistant"
    assert "velocity" in (final.content or "").lower()

    tool_messages = [m for m in agent.history if m.role == "tool"]
    assert len(tool_messages) == 1
    result = json.loads(tool_messages[0].content)
    assert result["format"] == "npy"
    assert result["shape"] == [200, 500]
    assert result["dtype"] == "float32"


def test_agent_surfaces_tool_validation_error(tmp_path):
    scripted = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", name="inspect_file", arguments={})],  # missing path
        ),
        ChatMessage(role="assistant", content="I need the path."),
    ]
    fake = FakeLLM(scripted)
    agent = Agent(llm=fake)

    final = agent.chat("look at the file")
    assert final.content == "I need the path."

    tool_messages = [m for m in agent.history if m.role == "tool"]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0].content)
    assert "error" in payload


def test_unknown_tool_is_surfaced_to_llm():
    scripted = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="x", name="does_not_exist", arguments={})],
        ),
        ChatMessage(role="assistant", content="ok, recovered"),
    ]
    fake = FakeLLM(scripted)
    agent = Agent(llm=fake)
    final = agent.chat("trigger an unknown tool")
    assert final.content == "ok, recovered"
    tool_msg = next(m for m in agent.history if m.role == "tool")
    payload = json.loads(tool_msg.content)
    assert "error" in payload
    assert "does_not_exist" in payload["error"]


if __name__ == "__main__":
    # Allow `python tests/test_smoke.py` without pytest installed.
    import sys
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        class _Path:
            def __init__(self, p):
                self._p = p
            def __truediv__(self, other):
                import pathlib
                return pathlib.Path(self._p) / other
        tests = [
            ("test_inspect_file_tool_is_registered", lambda: test_inspect_file_tool_is_registered()),
            ("test_agent_loop_invokes_tool_and_returns_reply", lambda: test_agent_loop_invokes_tool_and_returns_reply(_Path(tmp))),
            ("test_agent_surfaces_tool_validation_error", lambda: test_agent_surfaces_tool_validation_error(_Path(tmp))),
            ("test_unknown_tool_is_surfaced_to_llm", test_unknown_tool_is_surfaced_to_llm),
        ]
        for name, fn in tests:
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
