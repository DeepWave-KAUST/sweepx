"""End-to-end demo: a real local LLM drives a sweep forward run.

Prereqs:
    * an OpenAI-compatible LLM server with a 32k context, e.g.
      `bash examples/agent/start_vllm_qwen7b.sh`; found on :8000 / :8001 / :11434,
      or set $SWEEP_AGENT_LLM_URL (and $SWEEP_AGENT_LLM_MODEL)
    * sweepx installed in the environment this script runs in

Usage:
    python examples/agent/demo_real_llm.py
    python examples/agent/demo_real_llm.py "your custom prompt"

Without a prompt it writes a two-layer model to /tmp/vp_demo.npy (if absent) and
asks for a forward run on it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from sweepx.agent.agent import Agent
from sweepx.agent.llm.vllm_backend import VLLMBackend, backend_hint

VP_DEMO = "/tmp/vp_demo.npy"

DEFAULT_USER_MSG = (
    f"I have a velocity model at {VP_DEMO}. "
    "Please run a 2-D acoustic forward modelling with an 8 Hz Ricker wavelet, "
    "grid spacing dh=12.5 m, dt=0.001 s, 400 time samples, on the CPU eager backend "
    "(set device='cpu' and backend_impl='eager'). "
    "Save outputs under /tmp/sweep_demo. "
    "After it finishes, tell me where the synthetic shot record is."
)


def main() -> int:
    user_msg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER_MSG
    if len(sys.argv) <= 1 and not os.path.exists(VP_DEMO):
        vp = np.full((64, 96), 1800.0, dtype=np.float32)
        vp[32:, :] = 2600.0
        np.save(VP_DEMO, vp)

    backend = VLLMBackend()
    hint = backend_hint(backend.url)
    if hint is not None:
        backend.close()
        print(f"[demo] {hint}", file=sys.stderr)
        return 1
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
