"""vLLM backend — talks to vLLM's OpenAI-compatible REST endpoint.

vLLM exposes ``/v1/chat/completions`` with the same wire format as OpenAI's
API, including ``tools`` + ``tool_calls``. We treat it as a plain HTTP client
(``httpx``) and do not pull the heavy ``openai`` SDK as a hard dep.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

from sweep_agent.llm.base import BaseLLM, ChatMessage

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# Model aliases — short names users actually type, mapped to full HF repo IDs
# that vLLM uses. Update here when new models are validated.
MODEL_ALIASES: dict[str, str] = {
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-7b-instruct":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-r1-distill-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1-distill-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-r1-distill-8b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


def resolve_model(name: str) -> str:
    """Map a short alias to the canonical HF repo ID; pass through unknown names."""
    return MODEL_ALIASES.get(name.lower(), name)


class VLLMBackend(BaseLLM):
    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "VLLMBackend requires httpx. Install with `pip install httpx` "
                "(or `pip install sweep-agent[vllm]`)."
            ) from exc
        self.url = (url or os.environ.get("SWEEP_AGENT_LLM_URL", DEFAULT_URL)).rstrip("/")
        raw_model = model or os.environ.get("SWEEP_AGENT_LLM_MODEL", DEFAULT_MODEL)
        self._model_id = resolve_model(raw_model)
        self.api_key = api_key or os.environ.get("SWEEP_AGENT_LLM_API_KEY", "EMPTY")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[ChatMessage],
        tools: Iterable[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        tools_list = list(tools) if tools is not None else []
        if tools_list:
            payload["tools"] = tools_list
            payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self._client.post(f"{self.url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected vLLM response shape: {json.dumps(data)[:500]}") from exc
        return ChatMessage.from_openai(msg)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VLLMBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def make_default_backend() -> VLLMBackend:
    """Build a VLLMBackend from env vars; convenience for CLI / tests."""
    return VLLMBackend()
