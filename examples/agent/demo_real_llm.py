"""End-to-end demo: real local LLM drives a sweep forward run.

Prereqs:
    * vLLM server already running at $SWEEP_AGENT_LLM_URL (default http://localhost:8000/v1)
    * The matching model id is in $SWEEP_AGENT_LLM_MODEL (e.g. Qwen/Qwen2.5-7B-Instruct)
    * sweep-agent installed in the env where this script runs (ifwitorch)
    * /tmp/vp_demo.npy already prepared

Usage:
    python -m sweep_agent.examples.demo_real_llm
    python examples/demo_real_llm.py "your custom prompt"
"""

from __future__ import annotations

import os
import sys

# Sensible defaults so the script "just works" right after `vllm serve` is up.
os.environ.setdefault("SWEEP_AGENT_LLM_URL",   "http://localhost:8000/v1")
os.environ.setdefault("SWEEP_AGENT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

from sweep_agent.agent import Agent
from sweep_agent.llm.vllm_backend import VLLMBackend


DEFAULT_USER_MSG = (
    "I have a velocity model at /tmp/vp_demo.npy. "
    "Please run a 2-D acoustic forward modelling with an 8 Hz Ricker wavelet, "
    "grid spacing dh=12.5 m, dt=0.001 s, 400 time samples, on the CPU eager backend "
    "(set device='cpu' and backend_impl='eager'). "
    "Save outputs under /tmp/sweep_demo. "
    "After it finishes, tell me where the synthetic shot record is."
)


def main() -> int:
    user_msg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER_MSG

    backend = VLLMBackend()
    agent = Agent(llm=backend, max_steps=15)

    print(f"[demo] LLM url   = {backend.url}")
    print(f"[demo] LLM model = {backend.model_id}")
    print(f"\nUSER  > {user_msg}\n")

    try:
        for step in agent.iter_chat(user_msg):
            if step.kind == "tool":
                args_brief = repr(step.tool_args)
                if len(args_brief) > 220:
                    args_brief = args_brief[:220] + "..."
                result_brief = step.tool_result_json
                if len(result_brief) > 300:
                    result_brief = result_brief[:300] + "..."
                print(f"TOOL  · {step.tool_name}({args_brief})")
                print(f"      → {result_brief}\n")
            elif step.kind == "final":
                print(f"AGENT > {step.message.content}\n")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
