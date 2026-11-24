"""Tool registry.

A :class:`Tool` couples a Pydantic params model (which yields a JSON schema for
the LLM) with a callable that takes the validated model and returns any
JSON-serializable result. Tools self-register via :func:`register` so importing
``sweepx.agent.tools`` brings them all online.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError

from sweepx.agent.tools._modelmeta import DH_OMIT_NOTE, MODEL_PATH_ARGS, fill_dh


def _compact_schema(node: Any) -> Any:
    """Shrink a Pydantic JSON schema for the prompt without changing what it accepts
    in practice: drop ``title`` everywhere, collapse ``Optional[X]`` (``anyOf`` with
    ``null``) to ``X``, and drop ``default: null``. Arguments are still validated by
    the full Pydantic model in :meth:`Tool.invoke`."""
    if isinstance(node, list):
        return [_compact_schema(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: _compact_schema(v) for k, v in node.items() if k != "title"}
    if "properties" in node:  # property names may themselves be "title"
        out["properties"] = {k: _compact_schema(v) for k, v in node["properties"].items()}
    any_of = out.get("anyOf")
    if isinstance(any_of, list):
        rest = [a for a in any_of if a != {"type": "null"}]
        if len(rest) < len(any_of):
            if len(rest) == 1:
                out.pop("anyOf")
                out = {**rest[0], **out}
            else:
                out["anyOf"] = rest
    if "default" in out and out["default"] is None:
        out.pop("default")
    return out


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
        params = _compact_schema(self.params_model.model_json_schema())
        props = params.get("properties", {})
        if "dh" in props and any(k in props for k in MODEL_PATH_ARGS):
            # Tool.invoke fills dh from the model's recorded metadata, so the
            # caller may omit it (and is told what happens if it does not).
            params["required"] = [r for r in params.get("required", []) if r != "dh"] or None
            if params["required"] is None:
                params.pop("required")
            props["dh"]["description"] = (props["dh"].get("description", "Grid spacing (m).") + DH_OMIT_NOTE)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }

    def invoke(self, raw_args: dict[str, Any]) -> ToolResult:
        # An argument the tool does not take is an error, not dropped: a
        # misspelt optional one would otherwise run on its default unnoticed.
        unknown = sorted(set(raw_args or {}) - set(self.params_model.model_fields))
        if unknown:
            return ToolResult(ok=False, error=(
                f"{self.name} takes no argument(s) {unknown}; its arguments are "
                f"{sorted(self.params_model.model_fields)}."))
        raw_args, dh_error = fill_dh(self.params_model.model_fields, raw_args or {})
        if dh_error is not None:
            return ToolResult(ok=False, error=dh_error)
        try:
            args = self.params_model.model_validate(raw_args)
        except ValidationError as exc:
            if any(e["loc"] == ("dh",) and e["type"] == "missing" for e in exc.errors()):
                from pathlib import Path

                missing = [raw_args[k] for k in MODEL_PATH_ARGS
                           if isinstance(raw_args.get(k), str) and not Path(raw_args[k]).is_file()]
                if missing:  # say what is actually wrong, not "dh is missing"
                    return ToolResult(ok=False, error=(
                        f"model file not found: {missing[0]} (pass the path a tool returned)."))
                return ToolResult(ok=False, error=(
                    "dh (grid spacing, m) is required: the model file has no recorded dh "
                    "(only get_benchmark_model / make_synthetic_model record one)."))
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
        choice space (and the prompt) small."""
        keep = None if names is None else set(names)
        return [t.openai_spec() for n, t in self.tools.items() if keep is None or n in keep]

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
# at the top of this file does not shadow the sweepx.agent.tools.inspect submodule.
import sweepx.agent.tools.inspect          # noqa: E402,F401
import sweepx.agent.tools.introspect       # noqa: E402,F401
import sweepx.agent.tools.build_forward    # noqa: E402,F401
import sweepx.agent.tools.build_fwi        # noqa: E402,F401
import sweepx.agent.tools.build_spec       # noqa: E402,F401
import sweepx.agent.tools.execute          # noqa: E402,F401
import sweepx.agent.tools.status           # noqa: E402,F401
import sweepx.agent.tools.visualize        # noqa: E402,F401
import sweepx.agent.tools.orchestrate      # noqa: E402,F401
import sweepx.agent.tools.synth            # noqa: E402,F401
import sweepx.agent.tools.analysis         # noqa: E402,F401
import sweepx.agent.tools.forward_sweep    # noqa: E402,F401
import sweepx.agent.tools.datasets         # noqa: E402,F401
import sweepx.agent.tools.tasks_cli        # noqa: E402,F401
import sweepx.agent.tools.slurm            # noqa: E402,F401

__all__ = ["Tool", "ToolResult", "Registry", "registry", "register"]
