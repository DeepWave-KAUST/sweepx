"""LLM-driven demo: irregular free-surface (topography) wavefield + propagation GIF.

A real local LLM does everything from natural language — inspect the inputs,
build an AcousticCurvilinear wavefield spec with the topography, run it, and
animate the snapshots into a GIF. The model never sees Python; it just calls
tools. Needs a vLLM server (examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_topography_llm.py

Verified end-to-end: Qwen2.5-7B-Instruct produces
<out>/wavefield-<ts>/output/wavefield.gif (10 frames).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    vp_path, topo_path = "/tmp/vp_topo.npy", "/tmp/topo.npy"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_topo_llm")
    nz, nx = 100, 180
    x = np.arange(nx)
    topo = np.clip(np.round(22 - 14 * np.exp(-((x - nx * 0.4) ** 2) / (2 * 25.0**2))).astype(np.int64), 0, nx)
    vp = np.full((nz, nx), 3000.0, dtype=np.float32)
    for ix in range(nx):
        s = int(topo[ix])
        vp[s : s + 8, ix] = 1800.0
        vp[s + 8 : s + 30, ix] = 2400.0
    np.save(vp_path, vp)
    np.save(topo_path, topo)

    # Pure natural language — no tool names, no equation names, no parameters.
    # The system prompt's intent routing makes the LLM pick wavefield +
    # AcousticCurvilinear + topography + make_wavefield_gif on its own.
    drive(
        f"我有一个起伏地表的速度模型 {vp_path}(网格间距 10 米),地形文件 {topo_path}。"
        f"帮我看看地震波在这个起伏地表下面是怎么传播的,做成一个动画。输出放到 {out}。"
    )


if __name__ == "__main__":
    main()
