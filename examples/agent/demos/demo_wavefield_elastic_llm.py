"""LLM-driven demo: elastic wavefield (P + S waves) animated.

Mirrors the notebook ``06_wavefield_elastic.ipynb`` but driven entirely by
natural language: the model inspects the three elastic parameter files
(vp / vs / rho), recognises this needs the Elastic equation, builds a wavefield
task with a centred source, runs it, and animates the snapshots into a GIF
showing the fast P-wave ring and the slower S-wave ring. Needs a vLLM server
(examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_wavefield_elastic_llm.py

The system prompt's intent routing maps "弹性 / P波 / S波 / 看波场传播" to a
wavefield task with source_at_center=True + make_wavefield_gif.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_elastic_llm")
    # Homogeneous elastic whole-space: a centred source gives clean P and S rings.
    nz = nx = 200
    np.save(f"{d}/vp_el.npy", np.full((nz, nx), 2500.0, dtype=np.float32))
    np.save(f"{d}/vs_el.npy", np.full((nz, nx), 1500.0, dtype=np.float32))
    np.save(f"{d}/rho_el.npy", np.full((nz, nx), 2200.0, dtype=np.float32))

    # Pure natural language — no tool names, no equation names, no parameters.
    drive(
        f"我有一个均匀弹性介质模型,三个参数文件:纵波速度 {d}/vp_el.npy, "
        f"横波速度 {d}/vs_el.npy, 密度 {d}/rho_el.npy(网格间距 10 米)。"
        f"震源放在模型中心,帮我看看地震波(P 波和 S 波)是怎么从震源向四周传播的,"
        f"做成一个动画。输出放到 {out}。"
    )


if __name__ == "__main__":
    main()
