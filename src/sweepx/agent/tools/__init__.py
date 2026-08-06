"""Tool registry.

A :class:`Tool` couples a Pydantic params model (which yields a JSON schema for
the LLM) with a callable that takes the validated model and returns any
JSON-serializable result. Tools self-register via :func:`register` so importing
``sweep_agent.tools`` brings them all online.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError


# Tools that work on a base install (no ``sweep_tasks``). When sweep_tasks is not
# importable, ONLY these are offered to the LLM (see ``Registry.openai_specs``), so
# it never picks a tool that fails with "sweep_tasks is not importable". Everything
# else routes through ``sweep_tasks.TaskRunner`` — the build_*_spec / run_task /
# describe_task_schema tools and the orchestrators + plotters that call them.
_CORE_SAFE_TOOLS = frozenset({
    "inspect_file", "check_parameters", "make_synthetic_model", "list_equations",
    "run_forward_sweep", "read_status", "list_artifacts",
    "plot_velocity_model", "plot_velocity_slice", "plot_wavelet", "compare_shot_gathers",
})

_sweep_tasks_ok: bool | None = None


def _sweep_tasks_available() -> bool:
    """Whether ``sweep_tasks`` (the production runner) is importable — cached."""
    global _sweep_tasks_ok
    if _sweep_tasks_ok is None:
        _sweep_tasks_ok = importlib.util.find_spec("sweep_tasks") is not None
    return _sweep_tasks_ok


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

    def openai_specs(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI tool specs. If ``names`` is given, only those tools (in registry
        order) — used for per-query tool subsetting to keep the small model's
        choice space (and the prompt) small. When ``sweep_tasks`` is not installed,
        the tools that need it are dropped so the LLM never picks a failing path."""
        core_only = not _sweep_tasks_available()
        keep = None if names is None else set(names)
        return [
            t.openai_spec()
            for n, t in self.tools.items()
            if (keep is None or n in keep) and (not core_only or n in _CORE_SAFE_TOOLS)
        ]

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
import sweep_agent.tools.synth            # noqa: E402,F401
import sweep_agent.tools.analysis         # noqa: E402,F401
import sweep_agent.tools.forward_sweep    # noqa: E402,F401

__all__ = ["Tool", "ToolResult", "Registry", "registry", "register"]
