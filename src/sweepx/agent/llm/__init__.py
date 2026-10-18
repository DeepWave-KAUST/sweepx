"""LLM backend adapters (vLLM, Ollama, llama.cpp ...).

The vLLM backend lives in :mod:`sweep_agent.llm.vllm_backend` and requires the
optional ``httpx`` dependency; importing this package never pulls it in. Import
directly from the submodule when you need a concrete backend.
"""

from sweep_agent.llm.base import BaseLLM, ChatMessage, ToolCall

__all__ = ["BaseLLM", "ChatMessage", "ToolCall"]
