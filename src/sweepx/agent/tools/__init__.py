"""Tool registry.

A :class:`Tool` couples a Pydantic params model (which yields a JSON schema for
the LLM) with a callable that takes the validated model and returns any
JSON-serializable result. Tools self-register via :func:`register` so importing
``sweep_agent.tools`` brings them all online.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError


@dataclass
class ToolResult:
    ok: bool
    value: Any = None
    error: str | None = None

    def to_json(self) -> str:
        if self.ok:
            try:
                return json.dumps(self.value, ensure_ascii=False, default=str)
            except TypeError:
                return json.dumps({"repr": repr(self.value)}, ensure_ascii=False)
        return json.dumps({"error": self.error}, ensure_ascii=False)


@dataclass
class Tool:
    name: str
    description: str
    params_model: Type[BaseModel]
    fn: Callable[[BaseModel], Any]

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_model.model_json_schema(),
            },
        }

    def invoke(self, raw_args: dict[str, Any]) -> ToolResult:
        try:
            args = self.params_model.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult(ok=False, error=f"validation: {exc}")
        try:
            value = self.fn(args)
        except Exception as exc:  # surface runtime errors back to the LLM
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(ok=True, value=value)


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def openai_specs(self) -> list[dict[str, Any]]:
        return [t.openai_spec() for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)


registry = Registry()


def register(
    name: str,
    description: str,
    params_model: Type[BaseModel],
) -> Callable[[Callable[[BaseModel], Any]], Tool]:
    """Decorator: turn a function into a registered Tool."""

    def _decorate(fn: Callable[[BaseModel], Any]) -> Tool:
        desc = description or (inspect.getdoc(fn) or "")
        tool = Tool(name=name, description=desc.strip(), params_model=params_model, fn=fn)
        return registry.register(tool)

    return _decorate


# Importing the submodules triggers tool registration via the @register decorator.
# We use absolute `import` (not `from . import`) so the stdlib `inspect` module
# at the top of this file does not shadow the sweep_agent.tools.inspect submodule.
import sweep_agent.tools.inspect          # noqa: E402,F401
import sweep_agent.tools.introspect       # noqa: E402,F401
import sweep_agent.tools.build_forward    # noqa: E402,F401
import sweep_agent.tools.build_fwi        # noqa: E402,F401
import sweep_agent.tools.build_wavefield  # noqa: E402,F401
import sweep_agent.tools.build_spec       # noqa: E402,F401
import sweep_agent.tools.execute          # noqa: E402,F401
import sweep_agent.tools.status           # noqa: E402,F401
import sweep_agent.tools.visualize        # noqa: E402,F401
import sweep_agent.tools.orchestrate      # noqa: E402,F401

__all__ = ["Tool", "ToolResult", "Registry", "registry", "register"]
