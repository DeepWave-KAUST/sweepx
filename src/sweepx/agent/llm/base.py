"""LLM backend abstraction.

A backend exposes a single ``chat()`` call that takes a conversation and an
optional set of tool descriptions, and returns one assistant turn. Tool calls
ride along inside :class:`ChatMessage.tool_calls`. The :class:`Agent` does not
care which backend is in use — vLLM, Ollama, llama.cpp, or a fake one in tests.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class ChatMessage:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out

    @classmethod
    def from_openai(cls, msg: dict[str, Any]) -> "ChatMessage":
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {"__raw__": raw_args}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
        return cls(
            role=msg.get("role", "assistant"),
            content=msg.get("content"),
            tool_calls=tool_calls,
            tool_call_id=msg.get("tool_call_id"),
            name=msg.get("name"),
        )


class BaseLLM(ABC):
    """Minimal protocol every backend must implement."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        tools: Iterable[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Send one round; return the assistant message (may include tool calls)."""

    @property
    def model_id(self) -> str:
        return getattr(self, "_model_id", "unknown")
