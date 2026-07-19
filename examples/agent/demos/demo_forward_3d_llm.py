"""LLM-driven demo: 3-D forward modelling -> shot gather.

A 3-D two-layer velocity model (nz, ny, nx). The model inspects the cube,
recognises it is 3-D (so the agent/tools use the 3-D acoustic kernel
automatically), runs the forward simulation, and plots one shot's gather.
Pure natural language. Needs a vLLM server (examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_forward_3d_llm.py

build_forward_spec auto-upgrades Acoustic -> Acoustic3D for a 3-D model, so the
user never has to name the 3-D equation.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    vp_path = "/tmp/vp3d_demo.npy"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwd3d")
    nz, ny, nx = 32, 32, 40
    vp = np.full((nz, ny, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :, :] = 2700.0  # flat reflector at mid-depth
    np.save(vp_path, vp)

    drive(
        f"我有一个三维速度模型 {vp_path}(网格间距 12.5 米)。帮我做正演模拟"
        f"(记录长度 0.3 秒,主频 10Hz),然后把其中一炮的炮记录画成图给我看。"
        f"输出到 {out}。"
    )


if __name__ == "__main__":
    main()
