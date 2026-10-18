"""P23: Claude-style message queueing in the web UI.

A message submitted while a turn is streaming is captured into ``_pending``,
echoed as a queued placeholder, and processed automatically (in order) by the
same ``drain_queue`` sweep. No gradio dependency — pure-logic tests.
"""

from __future__ import annotations

from sweep_agent.agent import Agent
from sweep_agent.llm.base import BaseLLM, ChatMessage
from sweep_agent.webui import _chat, _pending, _queued_lines, drain_queue, queue_message


class FakeLLM(BaseLLM):
    """Returns one canned final reply per chat() call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._model_id = "fake"

    def chat(self, messages, tools=None, **kwargs) -> ChatMessage:
        return ChatMessage(role="assistant", content=self._replies.pop(0))


def _reset_state() -> None:
    _pending.clear()
    _chat["history"] = []
    _chat["agent"] = None


def test_queue_message_captures_text_and_files():
    _reset_state()
    queue_message({"text": "hi", "files": ["/tmp/a.npy"]})
    queue_message("plain string")
    queue_message({"text": "   ", "files": []})  # blank → dropped
    assert list(_pending) == [("hi", ["/tmp/a.npy"]), ("plain string", [])]


def test_queued_lines_render_placeholders():
    _reset_state()
    queue_message({"text": "run fwi", "files": ["/tmp/v.npy"]})
    lines = _queued_lines()
    assert len(lines) == 1 and "queued" in lines[0]["content"] and "run fwi" in lines[0]["content"]


def test_drain_processes_messages_in_order():
    _reset_state()
    queue_message({"text": "first", "files": []})
    queue_message({"text": "second", "files": []})
    make = lambda: Agent(llm=FakeLLM(["reply-1", "reply-2"]))

    phases = [phase for _, phase, _ in drain_queue(make)]
    assert phases == ["begin", "step", "begin", "step", "done"]
    assert not _pending  # backlog fully consumed by ONE drain sweep

    contents = [m["content"] for m in _chat["history"]]
    assert contents == ["first", "reply-1", "second", "reply-2"]
    assert _chat["agent"] is not None  # agent created once and kept


def test_mid_turn_submit_appears_queued_then_processed():
    _reset_state()
    queue_message({"text": "first", "files": []})

    class InjectingLLM(FakeLLM):
        """Simulates the user submitting a new message while turn 1 streams."""

        def chat(self, messages, tools=None, **kwargs):
            if len(self._replies) == 2:  # during turn 1
                queue_message({"text": "late arrival", "files": []})
            return super().chat(messages, tools, **kwargs)

    make = lambda: Agent(llm=InjectingLLM(["reply-1", "reply-2"]))
    displays = list(drain_queue(make))

    # While turn 1 streams, the late message shows as a queued placeholder…
    step1_display = displays[1][0]
    assert any("queued" in str(m.get("content")) for m in step1_display)
    # …and the same sweep then processes it (order preserved, queue empty).
    assert [p for _, p, _ in displays] == ["begin", "step", "begin", "step", "done"]
    assert [m["content"] for m in _chat["history"]] == ["first", "reply-1", "late arrival", "reply-2"]
    assert not _pending


def test_drain_on_empty_queue_is_noop():
    _reset_state()
    assert list(drain_queue(lambda: (_ for _ in ()).throw(AssertionError("must not build agent")))) == []
