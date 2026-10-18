"""Shared driver for LLM-driven demos.

Sends one natural-language prompt through Agent + a local vLLM backend and prints
the tool-call trace. The LLM does everything — builds the spec, runs it, and
produces the figure/GIF — exactly as an end user would via `sweep-agent chat`.

Needs a running vLLM server; see examples/start_vllm_qwen7b.sh. Override the
endpoint/model with SWEEP_AGENT_LLM_URL / SWEEP_AGENT_LLM_MODEL.
"""

from __future__ import annotations

import os

os.environ.setdefault("SWEEP_AGENT_LLM_URL", "http://localhost:8001/v1")
os.environ.setdefault("SWEEP_AGENT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

from sweep_agent.agent import Agent
from sweep_agent.llm.vllm_backend import VLLMBackend
from sweep_agent.tools.selection import select_tool_names


def drive(prompt: str, max_steps: int = 24) -> None:
    backend = VLLMBackend()
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
