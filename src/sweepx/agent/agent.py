"""Multi-turn ReAct loop over a :class:`BaseLLM` and a tool :class:`Registry`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterator

from sweep_agent.llm.base import BaseLLM, ChatMessage
from sweep_agent.prompts import build_system_prompt
from sweep_agent.tools import Registry, registry as default_registry

# Big tool-result fields that must NOT be echoed back into the conversation —
# the LLM already has the path it needs; keeping the full nested dict around
# burns thousands of tokens and overflows the context window over a few turns.
# A small model sometimes re-issues the SAME successful tool call forever
# (e.g. "make a P gif and an S gif" → loops). Execute any one (name, args)
# signature at most this many times per turn; further repeats short-circuit to a
# "stop repeating" message so the model finalizes instead of spinning.
_REPEAT_LIMIT = 2

# Weak local models sometimes reply with a PLAN ("I'll run the forward modelling…")
# instead of emitting the tool call, ending the turn with nothing done. When a
# text-only reply reads like such a promise and no tool has run yet this turn, nudge
# once to actually call the tool.
_NUDGE_CALL_TOOL = (
    "You described what you would do but did not call any tool, so nothing has run. "
    "Call the appropriate tool now to actually do it (for example run_forward_sweep). "
    "If the request genuinely needs no tool, answer the user directly."
)
_ACTION_HINTS = (
    "i will", "i'll", "let me", "let's", "i'm going to", "i am going to",
    "going to run", "going to plot", "next, i", "first, i", "i can run", "i can plot",
    "我将", "我会", "我来", "让我", "接下来", "首先", "现在我", "我先", "我可以", "我准备",
)


def _looks_like_narration(text) -> bool:
    """True when a text-only reply reads like a promise to act — the model likely
    meant to call a tool but forgot to emit the call."""
    t = (text or "").strip().lower()
    return bool(t) and any(h in t for h in _ACTION_HINTS)


def _english_reply_pin(text) -> str:
    """qwen drifts to Chinese on an English message despite the standing
    system-prompt rule. If the message has NO CJK characters (so the user is
    writing English/Latin), append a terse reply-language reminder right next to
    the message — recency makes the model actually follow it. Chinese messages are
    left untouched."""
    if (text or "").strip() and not any("一" <= c <= "鿿" for c in text):
        return "\n\n(Reply in English.)"
    return ""


_BULKY_RESULT_KEYS = ("spec", "spec_attempted")
# Cap roomy enough for the big *informational* results the LLM genuinely needs in
# full (list_equations ≈ 7.2k chars, describe_task_schema sections). The real
# context hog — the nested spec dict echoed by every build_*_spec call — is
# dropped separately above, so this backstop can be generous.
_MAX_TOOL_RESULT_CHARS = 12000
_MAX_STR_FIELD_CHARS = 2000


def _trim_tool_result(result_json: str) -> str:
    """Shrink a tool result before it goes into the chat history: drop the bulky
    spec dicts (the LLM only needs yaml_path / task_dir / image_path / error) and
    cap long string fields. Returns compact text — not necessarily valid JSON,
    which is fine for the conversation log."""
    try:
        data = json.loads(result_json)
    except (ValueError, TypeError):
        return result_json if len(result_json) <= _MAX_TOOL_RESULT_CHARS else result_json[:_MAX_TOOL_RESULT_CHARS] + "…"
    if isinstance(data, dict):
        for k in list(data):
            if k in _BULKY_RESULT_KEYS:
                data[k] = "<written to YAML; omitted from history>"
            elif isinstance(data[k], str) and len(data[k]) > _MAX_STR_FIELD_CHARS:
                data[k] = data[k][:_MAX_STR_FIELD_CHARS] + "…"
        result_json = json.dumps(data, ensure_ascii=False)
    return result_json if len(result_json) <= _MAX_TOOL_RESULT_CHARS else result_json[:_MAX_TOOL_RESULT_CHARS] + "…"


@dataclass
class AgentStep:
    """One step in the inner loop.

    kind is one of:
      "note"       — the model narrated its plan alongside tool calls
                     (message.content holds the text)
      "tool_start" — a tool is ABOUT to run (lets UIs show a live spinner)
      "tool"       — the tool finished; tool_result_json holds the result
      "final"      — the assistant's final reply for this turn
    Every "tool_start" is followed by exactly one matching "tool"."""

    kind: str
    message: ChatMessage
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result_json: str | None = None


@dataclass
class Agent:
    llm: BaseLLM
    tools: Registry = field(default_factory=lambda: default_registry)
    system_prompt: str = field(default_factory=build_system_prompt)
    max_steps: int = 12
    history: list[ChatMessage] = field(default_factory=list)
    # Optional per-turn tool subsetting: (message, all_names) -> subset of names.
    # None = expose every tool. Reduces the small model's choice space + prompt.
    tool_selector: Callable[[str, list[str]], list[str]] | None = None
    _used_tools: set[str] = field(default_factory=set)
    # Char budget for the conversation (excl. the tool schemas, ~48k chars, sent
    # separately). 56k chars ≈ 14k tokens; with the ~12k-token tool schemas and
    # room for a reply this stays under a 32k context window. Oldest whole
    # exchanges are dropped when exceeded.
    history_char_budget: int = 56_000
    # Tool specs actually sent on the most recent LLM request — the UI context
    # meter reads this so it reflects the per-turn subset, not all ~30 tools.
    last_tool_specs: list[dict] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.history or self.history[0].role != "system":
            self.history.insert(0, ChatMessage(role="system", content=self.system_prompt))

    def reset(self) -> None:
        self.history = [ChatMessage(role="system", content=self.system_prompt)]

    def _history_chars(self) -> int:
        return sum(len(m.content or "") for m in self.history)

    def _prune_history(self) -> None:
        """Drop the oldest complete exchanges (a ``user`` message and everything
        up to the next ``user``) until under budget. Removing whole exchanges
        keeps assistant/tool_call pairing valid — never orphans a tool reply."""
        while self._history_chars() > self.history_char_budget:
            # Find the first user message after the system prompt.
            start = next((i for i in range(1, len(self.history)) if self.history[i].role == "user"), None)
            if start is None:
                return
            # Find the next user message (start of the exchange to keep).
            nxt = next((i for i in range(start + 1, len(self.history)) if self.history[i].role == "user"), None)
            if nxt is None:
                return  # only one exchange left — don't strip the current turn
            del self.history[start:nxt]

    def chat(self, user_message: str) -> ChatMessage:
        """Run one user→assistant turn (executing any intermediate tool calls)."""
        steps = list(self.iter_chat(user_message))
        final = steps[-1].message
        if final.role != "assistant" or final.tool_calls:
            raise RuntimeError(
                f"agent exhausted max_steps={self.max_steps} without producing a final reply"
            )
        return final

    def iter_chat(self, user_message: str) -> Iterator[AgentStep]:
        """Stream each step (tool call + result, or final reply) of one turn."""
        self.history.append(ChatMessage(role="user", content=user_message + _english_reply_pin(user_message)))
        if self.tool_selector is not None:
            names = self.tool_selector(user_message, self.tools.names())
            # Keep any tools already used this conversation so follow-ups
            # ("run again with …") still see them even if the message lacks keywords.
            names = list(dict.fromkeys(list(names) + sorted(self._used_tools)))
            tool_specs = self.tools.openai_specs(names)
        else:
            tool_specs = self.tools.openai_specs()
        self.last_tool_specs = tool_specs
        # The exact tools offered to the model this turn. A weak model sometimes
        # *hallucinates* a call to a tool it was never given — e.g. asking for an
        # elastic wavefield makes it invent `animate_wavefield`, which is gated out
        # on a base install. That tool is still in the registry, so running it
        # anyway leaks a cryptic "sweep_tasks not importable" traceback. Only run
        # what was actually offered.
        exposed_names = {s["function"]["name"] for s in tool_specs}
        call_counts: dict[str, int] = {}
        tool_used = False
        nudged = False

        for _ in range(self.max_steps):
            self._prune_history()  # keep the request under the context window
            assistant = self.llm.chat(self.history, tools=tool_specs)
            self.history.append(assistant)

            if not assistant.tool_calls:
                # Narration guard: a weak model may promise an action ("I'll run the
                # forward modelling…") without emitting the call, ending the turn
                # with nothing done. If nothing has run yet this turn and the reply
                # reads like such a promise, nudge once to actually call the tool.
                if not tool_used and not nudged and _looks_like_narration(assistant.content):
                    nudged = True
                    self.history.append(ChatMessage(role="user", content=_NUDGE_CALL_TOOL))
                    continue
                yield AgentStep(kind="final", message=assistant)
                return

            tool_used = True
            # Models often narrate the plan in `content` next to the tool calls
            # ("I'll build the spec first…") — surface it like a chat message.
            if (assistant.content or "").strip():
                yield AgentStep(kind="note", message=assistant)

            for call in assistant.tool_calls:
                self._used_tools.add(call.name)  # remember for follow-up turns' subset
                yield AgentStep(
                    kind="tool_start",
                    message=assistant,
                    tool_name=call.name,
                    tool_args=call.arguments,
                )
                sig = f"{call.name}|" + json.dumps(call.arguments or {}, sort_keys=True, ensure_ascii=False)
                call_counts[sig] = call_counts.get(sig, 0) + 1
                if call_counts[sig] > _REPEAT_LIMIT:
                    # Loop guard: stop re-running an identical call; nudge to finalize.
                    result_json = json.dumps({
                        "error": (
                            f"Loop guard: '{call.name}' was already called with these exact arguments "
                            f"{_REPEAT_LIMIT}x (and any earlier result stands). Do NOT call it again — "
                            f"use the previous result(s) and write your final answer to the user now."
                        )
                    }, ensure_ascii=False)
                elif call.name not in exposed_names:
                    # Hallucinated / gated-out tool: it was NOT offered this turn.
                    # Don't execute it (it may need an unreleased tier) — tell the
                    # model to pick a real one instead of leaking a dependency error.
                    result_json = json.dumps({
                        "error": (
                            f"'{call.name}' is not an available tool in this installation "
                            f"(it may require the unreleased sweep_tasks tier, which is not present). "
                            f"Do not call it again. Available tools: {sorted(exposed_names)}."
                        )
                    }, ensure_ascii=False)
                elif (tool := self.tools.get(call.name)) is None:
                    result_json = (
                        f'{{"error": "unknown tool: {call.name}. '
                        f'available: {self.tools.names()}"}}'
                    )
                else:
                    result = tool.invoke(call.arguments)
                    result_json = result.to_json()

                # History keeps a trimmed copy (no giant spec dicts) to protect
                # the context window; the yielded step carries the FULL result so
                # the UI / callers can still extract image paths etc.
                tool_msg = ChatMessage(
                    role="tool",
                    content=_trim_tool_result(result_json),
                    tool_call_id=call.id,
                    name=call.name,
                )
                self.history.append(tool_msg)
                yield AgentStep(
                    kind="tool",
                    message=ChatMessage(role="tool", content=result_json, tool_call_id=call.id, name=call.name),
                    tool_name=call.name,
                    tool_args=call.arguments,
                    tool_result_json=result_json,
                )

        # Fall through: out of steps. Yield the latest assistant message anyway.
        yield AgentStep(kind="final", message=self.history[-1])
