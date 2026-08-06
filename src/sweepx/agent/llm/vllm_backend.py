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

# Where a local LLM usually listens. Ollama is the easy cross-platform default
# (one binary, small quantized models); vLLM is the GPU/production option.
OLLAMA_URL = "http://localhost:11434/v1"
VLLM_URLS = ("http://localhost:8000/v1", "http://localhost:8001/v1")
DEFAULT_URL = OLLAMA_URL                       # used only as a last-resort fallback
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"            # ~4.4 GB pull, does tool calling
VLLM_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL = VLLM_DEFAULT_MODEL             # kept for back-compat imports

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


def is_ollama(url: str) -> bool:
    """Heuristic: an Ollama endpoint listens on 11434."""
    return ":11434" in (url or "")


def default_model_for(url: str) -> str:
    """The model to assume when the user didn't name one, given the endpoint."""
    return OLLAMA_DEFAULT_MODEL if is_ollama(url) else VLLM_DEFAULT_MODEL


def _endpoint_alive(url: str, timeout: float = 0.6) -> bool:
    """True if an OpenAI-compatible server answers GET {url}/models."""
    try:
        import httpx

        return httpx.get(url.rstrip("/") + "/models", timeout=timeout).status_code < 500
    except Exception:
        return False


def detect_endpoint() -> str:
    """Pick a local LLM endpoint with zero config: a running Ollama first, then a
    running vLLM, else Ollama by default (so an auto-pull can bring it up)."""
    for candidate in (OLLAMA_URL, *VLLM_URLS):
        if _endpoint_alive(candidate):
            return candidate
    return OLLAMA_URL


def ensure_ollama_model(url: str, model: str) -> None:
    """If ``url`` is Ollama and ``model`` isn't pulled yet, pull it once.

    Turns the manual `ollama pull ...` step into an automatic one-time download.
    A no-op for vLLM endpoints, when the Ollama API is unreachable, or when the
    `ollama` CLI isn't on PATH (the chat call will surface any real error).
    """
    if not is_ollama(url):
        return
    import shutil
    import subprocess

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        import httpx

        tags = httpx.get(base + "/api/tags", timeout=2.0).json()
    except Exception:
        return
    have = {m.get("name", "") for m in tags.get("models", [])}
    have |= {n.split(":")[0] for n in have}
    if model in have or model.split(":")[0] in have:
        return
    if shutil.which("ollama") is None:
        return
    print(f"[sweep-agent] model '{model}' isn't pulled yet — downloading it once via Ollama...", flush=True)
    subprocess.run(["ollama", "pull", model], check=False)


def backend_hint(url: str) -> str | None:
    """If no LLM server answers at ``url``, return a short, actionable message
    (how to install/start Ollama, no Homebrew needed), otherwise ``None``.

    This is what turns "connection refused" for a first-time user with nothing
    installed into a clear next step.
    """
    if _endpoint_alive(url):
        return None
    import shutil

    if is_ollama(url):
        if shutil.which("ollama") is not None:
            return (
                "Ollama is installed but not running. Start it, then re-run:\n"
                "    ollama serve            # or, with Homebrew:  brew services start ollama"
            )
        return (
            "No local LLM found. The easiest backend is Ollama, and it needs no Homebrew:\n"
            "    curl -fsSL https://ollama.com/install.sh | sh    # macOS or Linux\n"
            "    ollama serve\n"
            "  (or download the app from https://ollama.com/download)\n"
            "then re-run this command. Already have an OpenAI-compatible server? "
            "Point at it with  --url http://host:port/v1"
        )
    return (
        f"No LLM server is answering at {url}. Start one (e.g. `sweep-agent serve-llm`), "
        "run Ollama instead, or pass  --url http://host:port/v1"
    )


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
        # Zero-config: an explicit arg wins, then $SWEEP_AGENT_LLM_URL, else
        # auto-detect a running Ollama/vLLM. Same order for the model, defaulting
        # to a 7B that fits the detected backend.
        chosen_url = url or os.environ.get("SWEEP_AGENT_LLM_URL") or detect_endpoint()
        self.url = chosen_url.rstrip("/")
        raw_model = model or os.environ.get("SWEEP_AGENT_LLM_MODEL") or default_model_for(self.url)
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
