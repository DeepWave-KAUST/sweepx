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

    drive(
        f"我想看地震波在起伏地表下的传播。速度模型 {vp_path},地形 {topo_path}。"
        f"请用曲线网格方程 AcousticCurvilinear 做波场模拟:dh=10m, dt=0.001s, nt=600, 9Hz Ricker, "
        f"abcn=30, free_surface, 在时间步 0,60,120,180,240,300,360,420,480,540 保存快照, CPU eager, "
        f"输出目录用 {out}。跑完用 make_wavefield_gif 把波场做成传播 gif(abcn=30, free_surface=true, "
        f"叠加 {topo_path} 地形)。最后告诉我 gif 路径。"
    )


if __name__ == "__main__":
    main()
