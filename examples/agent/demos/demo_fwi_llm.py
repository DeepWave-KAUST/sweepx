"""LLM-driven demo: hello-FWI (full-waveform inversion) from natural language.

Mirrors the notebook ``00_hello_fwi.ipynb``: a two-layer true model, a
homogeneous starting model, and an acoustic FWI that recovers the reflector.
Driven entirely by natural language — the model inspects the two inputs, runs
the inversion, and draws the result (initial vs inverted vs true + residual)
and the loss-convergence curve. Needs a vLLM server
(examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_fwi_llm.py

The intent router maps "反演 / FWI" to run_fwi, which does build + run + the two
result figures in one call — so the 7B doesn't juggle the inverted-model path or
the plotting steps.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwi_llm")
    nz, nx = 64, 96
    # True model: two layers (reflector at mid-depth). Init: homogeneous.
    vp_true = np.full((nz, nx), 1500.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 2300.0
    vp_init = np.full((nz, nx), 1500.0, dtype=np.float32)
    np.save(f"{d}/fwi_true.npy", vp_true)
    np.save(f"{d}/fwi_init.npy", vp_init)

    # Pure natural language — no tool names. The intent router maps this to
    # run_fwi (build + invert + plot_model + plot_convergence in one call).
    drive(
        f"我想做全波形反演(FWI)。这是一个合成算例:真实速度模型 {d}/fwi_true.npy "
        f"(用它生成观测数据),初始模型 {d}/fwi_init.npy(网格间距 10 米,"
        f"记录长度 0.6 秒,主频 10Hz)。帮我从初始模型反演出速度结构,"
        f"并把反演结果(和真实模型对比)以及收敛曲线画出来给我看。输出放到 {out}。"
    )


if __name__ == "__main__":
    main()
