"""LLM-driven demo: DAS (distributed acoustic sensing) elastic wavefield.

DAS records strain / strain-rate along a fibre rather than particle velocity.
sweep has several DAS equations (DASZhao, DASMu, DASElastic). This demo runs a
DAS elastic wavefield on a homogeneous model and animates it. The agent calls
list_equations to find the DAS equation (vp/vs/rho models) and animates it.
Needs a vLLM server (examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_wavefield_das_llm.py

NOTE: sweep exposes 3 DAS variants; picking one from plain language is at the
edge of Qwen2.5-7B, so this demo names "DAS" explicitly and may need a retry. A
14B model is more reliable.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_das_llm")
    nz = nx = 180
    np.save(f"{d}/vp_das.npy", np.full((nz, nx), 2500.0, dtype=np.float32))
    np.save(f"{d}/vs_das.npy", np.full((nz, nx), 1500.0, dtype=np.float32))
    np.save(f"{d}/rho_das.npy", np.full((nz, nx), 2200.0, dtype=np.float32))

    drive(
        f"我想看 DAS(分布式光纤声波传感)观测下的弹性波场。模型文件:纵波速度 {d}/vp_das.npy, "
        f"横波速度 {d}/vs_das.npy, 密度 {d}/rho_das.npy(网格间距 10 米)。请用 sweep 里的 DAS "
        f"方程,震源放在模型中心,把波场传播做成动画。输出放到 {out}。",
        max_steps=28,
    )


if __name__ == "__main__":
    main()
