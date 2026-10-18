"""LLM-driven demo: tilted-anisotropy (TTI) ELASTIC wavefield, animated.

Mirrors the notebook ``10_wavefield_elastic_tti.ipynb`` (shrunk so it runs on a
CPU in seconds) but driven by natural language. The ElasticTTI equation needs
EIGHT model files — vp0, vs0, rho and the Thomsen parameters epsilon/delta/gamma
plus the tilt angles theta/phi (radians). The model inspects them, recognises
this is elastic-TTI, and animates the tilted P/S wavefronts in one shot. Needs a
vLLM server (examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_wavefield_elastic_tti_llm.py

NOTE: ElasticTTI is the heaviest single-wavefield demo here (8 models). The
intent router maps it to animate_wavefield, which auto-detects the equation from
the model set and centres the source — but wiring 8 files is at the edge of what
Qwen2.5-7B does in one go, so it may need a retry. A 14B+ model is rock-solid.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_elastic_tti_llm")
    nz = nx = 180

    def save(name: str, val: float) -> str:
        p = f"{d}/{name}_etti.npy"
        np.save(p, np.full((nz, nx), np.float32(val)))
        return p

    vp0 = save("vp0", 2400.0)
    vs0 = save("vs0", 1200.0)
    rho = save("rho", 2200.0)
    eps = save("eps", 0.35)
    delta = save("delta", 0.05)
    gamma = save("gamma", 0.20)
    theta = save("theta", float(np.deg2rad(30.0)))  # 30° tilt, in radians
    phi = save("phi", 0.0)

    # Pure natural language — no tool names, no equation names. The intent router
    # maps elastic + anisotropy + "看波怎么传播/动画" to animate_wavefield, which
    # auto-detects ElasticTTI from this exact 8-model set.
    drive(
        f"我有一个倾斜各向异性(TTI)的弹性介质模型,倾角 30 度。模型文件:"
        f"纵波速度 {vp0}, 横波速度 {vs0}, 密度 {rho}, "
        f"Thomsen 参数 epsilon {eps}, delta {delta}, gamma {gamma}, "
        f"倾角 theta {theta}, 方位角 phi {phi}(网格间距 10 米)。"
        f"震源放在模型中心,帮我看看地震波在这种倾斜各向异性介质里是怎么传播的"
        f"(波前会是倾斜的椭圆),做成一个动画。输出放到 {out}。",
        max_steps=28,
    )


if __name__ == "__main__":
    main()
