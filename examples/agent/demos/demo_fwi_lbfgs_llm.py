"""LLM-driven demo: FWI with the L-BFGS optimizer (optimizer-variant showcase).

Same synthetic two-layer inversion as demo_fwi_llm.py, but the user asks for
L-BFGS instead of Adam — demonstrating that the optimizer is a free parameter.
The agent maps "L-BFGS" to run_fwi(optimizer='lbfgs', ...). Needs a vLLM server
(examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_fwi_lbfgs_llm.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwi_lbfgs_llm")
    nz, nx = 64, 96
    vp_true = np.full((nz, nx), 1500.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 2300.0
    vp_init = np.full((nz, nx), 1500.0, dtype=np.float32)
    np.save(f"{d}/fwilb_true.npy", vp_true)
    np.save(f"{d}/fwilb_init.npy", vp_init)

    drive(
        f"我想用 L-BFGS 优化器做全波形反演(FWI)。合成算例:真实模型 {d}/fwilb_true.npy "
        f"(生成观测数据),初始模型 {d}/fwilb_init.npy(网格间距 10 米,记录长度 0.6 秒,"
        f"主频 10Hz)。反演出速度结构,并把反演结果(与真实模型对比)和收敛曲线画出来。"
        f"输出放到 {out}。"
    )


if __name__ == "__main__":
    main()
