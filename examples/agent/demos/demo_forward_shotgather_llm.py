"""LLM-driven demo: forward modelling -> shot-gather plot.

A real local LLM does everything from natural language — inspect the velocity
model, run a forward simulation, and plot the synthetic shot gather (direct +
reflected arrivals). The model never sees Python or tool names. Needs a vLLM
server (examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_forward_shotgather_llm.py

Verified end-to-end on Qwen2.5-7B with pure natural language — any path
hallucinations are absorbed by the run_task / plot_shot_gather fallbacks.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    vp_path = "/tmp/vp_demo.npy"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwd")
    # Two-layer model: a flat reflector at z=32 gives a clear reflection hyperbola.
    vp = np.full((64, 96), 1800.0, dtype=np.float32)
    vp[32:, :] = 2600.0
    np.save(vp_path, vp)

    drive(
        f"对速度模型 {vp_path} 做正演模拟(网格间距 12.5 米,记录长度 0.6 秒,主频 10Hz),"
        f"然后把炮记录画成图给我看。输出到 {out}。"
    )


if __name__ == "__main__":
    main()
