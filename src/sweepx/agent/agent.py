"""Multi-turn ReAct loop over a :class:`BaseLLM` and a tool :class:`Registry`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from sweep_agent.llm.base import BaseLLM, ChatMessage
from sweep_agent.prompts import SYSTEM_PROMPT
from sweep_agent.tools import Registry, registry as default_registry


@dataclass
class AgentStep:
    """One step in the inner loop: either a tool execution or a final reply."""

    kind: str  # "tool" or "final"
    message: ChatMessage
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result_json: str | None = None


@dataclass
class Agent:
    llm: BaseLLM
    tools: Registry = field(default_factory=lambda: default_registry)
    system_prompt: str = SYSTEM_PROMPT
    max_steps: int = 12
    history: list[ChatMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history or self.history[0].role != "system":
            self.history.insert(0, ChatMessage(role="system", content=self.system_prompt))

    def reset(self) -> None:
        self.history = [ChatMessage(role="system", content=self.system_prompt)]

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
        self.history.append(ChatMessage(role="user", content=user_message))
        tool_specs = self.tools.openai_specs()

        for _ in range(self.max_steps):
            assistant = self.llm.chat(self.history, tools=tool_specs)
            self.history.append(assistant)

            if not assistant.tool_calls:
                yield AgentStep(kind="final", message=assistant)
                return

            for call in assistant.tool_calls:
                tool = self.tools.get(call.name)
                if tool is None:
                    result_json = (
                        f'{{"error": "unknown tool: {call.name}. '
                        f'available: {self.tools.names()}"}}'
                    )
                else:
                    result = tool.invoke(call.arguments)
                    result_json = result.to_json()

                tool_msg = ChatMessage(
                    role="tool",
                    content=result_json,
                    tool_call_id=call.id,
                    name=call.name,
                )
                self.history.append(tool_msg)
                yield AgentStep(
                    kind="tool",
                    message=tool_msg,
                    tool_name=call.name,
                    tool_args=call.arguments,
                    tool_result_json=result_json,
                )

        # Fall through: out of steps. Yield the latest assistant message anyway.
        yield AgentStep(kind="final", message=self.history[-1])
