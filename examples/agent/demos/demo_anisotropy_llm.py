"""LLM-driven demo: anisotropic wavefronts (Acoustic / VTI / TTI) compared.

A real local LLM runs the same homogeneous model under three equations and then
calls compare_wavefields to plot the wavefronts side by side — the isotropic
one circular, VTI elliptical, TTI tilted. The model drives the whole thing from
natural language. Needs a vLLM server (examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_anisotropy_llm.py

NOTE: this 3-run + aggregate-compare flow is HARD for a 7B model — it tends to
nest extra_models inside `extra`, confuse forward/wavefield, or mis-pass
task_dirs, so it often needs retries on Qwen2.5-7B. Use a stronger model (14B+)
for reliable one-shot success. The single-run topography demo
(demo_topography_llm.py) IS reliable on 7B.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    nz, nx = 140, 200
    np.save(f"{d}/vp_uni.npy", np.full((nz, nx), 2500.0, dtype=np.float32))
    np.save(f"{d}/eps.npy", np.full((nz, nx), 0.25, dtype=np.float32))
    np.save(f"{d}/delta.npy", np.full((nz, nx), 0.10, dtype=np.float32))
    np.save(f"{d}/theta.npy", np.full((nz, nx), np.float32(np.deg2rad(30.0))))
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_aniso_llm")

    # Pure natural language — no tool names, no equation names, no parameters.
    # The intent router should map "各向异性/VTI/TTI" to list_equations + wavefield
    # runs and "并排对比" to compare_wavefields, all on its own.
    drive(
        f"我想对比各向同性介质和各向异性介质(VTI、TTI)里地震波波前形状的差异。"
        f"用一个均匀速度模型 {d}/vp_uni.npy。各向异性参数文件:epsilon {d}/eps.npy, "
        f"delta {d}/delta.npy, theta {d}/theta.npy。请把这三种介质的波场快照并排画出来对比,"
        f"输出到 {out}。",
        max_steps=32,
    )


if __name__ == "__main__":
    main()
