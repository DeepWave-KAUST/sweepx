"""sweep-agent — offline-LLM natural-language control for the sweep stack.

Public surface kept minimal so importing the package never pulls in the heavy
optional deps (httpx for vLLM, gradio for the web UI, ...).
"""

from sweep_agent.agent import Agent, AgentStep
from sweep_agent.llm.base import BaseLLM, ChatMessage, ToolCall
from sweep_agent.tools import Tool, registry

__all__ = ["Agent", "AgentStep", "BaseLLM", "ChatMessage", "ToolCall", "Tool", "registry"]
__version__ = "0.0.1"
