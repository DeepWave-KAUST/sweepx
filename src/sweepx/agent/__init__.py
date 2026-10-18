"""sweepx.agent — offline-LLM natural-language control for the sweep stack.

Public surface kept minimal so importing the package never pulls in the heavy
optional deps (httpx for vLLM, gradio for the web UI, ...).
"""

from sweepx.agent.agent import Agent, AgentStep
from sweepx.agent.llm.base import BaseLLM, ChatMessage, ToolCall
from sweepx.agent.tools import Tool, registry

__all__ = ["Agent", "AgentStep", "BaseLLM", "ChatMessage", "ToolCall", "Tool", "registry"]

# Read the installed distribution version so this never drifts from pyproject.
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("sweepx")
except Exception:  # running from a source tree that isn't installed as a dist
    __version__ = "0.0.0+source"
