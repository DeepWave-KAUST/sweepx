"""LLM-driven demo: FWI with the L-BFGS optimizer (optimizer-variant showcase).

Same synthetic two-layer inversion as demo_fwi_llm.py, but the user asks for
L-BFGS instead of Adam — demonstrating that the optimizer is a free parameter.
The agent maps "L-BFGS" to run_fwi(optimizer='lbfgs', ...): sweepx.opt's L-BFGS,
whose first step is lr in m/s and whose later steps come from the curvature
(a replay of the tool call: RMSE 118 -> 101 m/s, the reflector sharpened, the
deep half left at its true velocity).
Needs an LLM server (examples/agent/start_vllm_qwen7b.sh).

    python examples/agent/demos/demo_fwi_lbfgs_llm.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def _smooth(vp: np.ndarray, cells: int = 6) -> np.ndarray:
    """The true model blurred by a Gaussian of ``cells`` grid cells: a starting
    model that knows the trend but not the sharp reflector, the usual FWI start."""
    x = np.arange(-3 * cells, 3 * cells + 1)
    k = np.exp(-0.5 * (x / cells) ** 2)
    k /= k.sum()
    pad = np.pad(vp, 3 * cells, mode="edge")
    pad = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, pad)
    pad = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, pad)
    return pad[3 * cells:-3 * cells, 3 * cells:-3 * cells].astype(np.float32)


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwi_lbfgs_llm")
    nz, nx = 64, 96
    vp_true = np.full((nz, nx), 1500.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 2300.0
    vp_init = _smooth(vp_true)
    np.save(f"{d}/fwilb_true.npy", vp_true)
    np.save(f"{d}/fwilb_init.npy", vp_init)

    drive(
        f"我想用 L-BFGS 优化器做全波形反演(FWI)。合成算例:真实模型 {d}/fwilb_true.npy "
        f"(生成观测数据),初始模型 {d}/fwilb_init.npy(网格间距 10 米,记录长度 1 秒,"
        f"主频 10Hz)。反演出速度结构,并把反演结果(与真实模型对比)和收敛曲线画出来。"
        f"输出放到 {out}。"
    )


if __name__ == "__main__":
    main()
