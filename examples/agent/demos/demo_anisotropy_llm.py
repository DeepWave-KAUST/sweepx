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

    drive(
        f"对比各向同性和各向异性介质的波前。均匀速度模型 {d}/vp_uni.npy。"
        f"务必用 build_wavefield_spec(波场快照,有 snapshot_times),不要用 build_forward_spec。"
        f"请分别用三个方程各做一次波场模拟,都用相同参数(dh=10m, dt=0.001s, nt=260, 12Hz Ricker, "
        f"abcn=20, CPU eager, 在时间步 259 保存一个快照, 输出目录 {out}):"
        f"(1) Acoustic; (2) AcousticVTI,epsilon={d}/eps.npy delta={d}/delta.npy; "
        f"(3) AcousticTTI,epsilon={d}/eps.npy delta={d}/delta.npy theta={d}/theta.npy。"
        f"三个都跑完后,记下每个 run_task 返回的 task_dir,用 compare_wavefields 把这三个 task_dir 的"
        f"波场并排画成对比图(labels=['Acoustic','AcousticVTI','AcousticTTI'], abcn=20, "
        f"out_path={out}/anisotropy_compare.png)。最后告诉我对比图路径。",
        max_steps=32,
    )


if __name__ == "__main__":
    main()
