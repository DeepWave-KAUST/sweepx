"""Shared driver for LLM-driven demos.

Sends one natural-language prompt through Agent + a local LLM backend and prints
the tool-call trace. The LLM does everything — builds the spec, runs it, and
produces the figure/GIF — exactly as an end user would via `sweep-agent chat`.

Needs a running OpenAI-compatible server with a 32k context, e.g.
`bash examples/agent/start_vllm_qwen7b.sh` (vLLM) or
`OLLAMA_CONTEXT_LENGTH=32768 ollama serve`. The demo finds a server on
:8000 / :8001 (vLLM) or :11434 (Ollama) and asks it which model it serves;
SWEEP_AGENT_LLM_URL / SWEEP_AGENT_LLM_MODEL choose others.

Qwen2.5-7B finishes most demos but not every time (a rerun usually does);
Qwen2.5-14B (MODEL=Qwen/Qwen2.5-14B-Instruct, a 48 GB-class GPU) is the reliable
choice.
"""

from __future__ import annotations

import sys

from sweepx.agent.agent import Agent
from sweepx.agent.llm.vllm_backend import VLLMBackend, backend_hint
from sweepx.agent.tools.selection import select_tool_names


def drive(prompt: str, max_steps: int = 24) -> None:
    backend = VLLMBackend()
    hint = backend_hint(backend.url)
    if hint is not None:
        backend.close()
        sys.exit(f"[demo] {hint}")
    agent = Agent(llm=backend, max_steps=max_steps, tool_selector=select_tool_names)
    print(f"[demo] model={backend.model_id}  url={backend.url}\n")
    print(f"USER  > {prompt}\n")
    try:
        for step in agent.iter_chat(prompt):
            if step.kind == "tool":
                args = repr(step.tool_args)
                if len(args) > 180:
                    args = args[:180] + "..."
                res = step.tool_result_json
                if len(res) > 240:
                    res = res[:240] + "..."
                print(f"TOOL  · {step.tool_name}({args})")
                print(f"      → {res}\n")
            elif step.kind == "final":
                print(f"AGENT > {step.message.content}")
    finally:
        backend.close()
