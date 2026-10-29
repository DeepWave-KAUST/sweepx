"""LLM-driven demo: hello-FWI (full-waveform inversion) from natural language.

A two-layer true model and a smoothed copy of it as the starting model (it
knows the trend, not the sharp reflector), as in most FWI tests; an acoustic
FWI with Adam sharpens the reflector (a replay of the tool call: RMSE 118 ->
107 m/s, with some noise left below the reflector; demo_fwi_lbfgs_llm.py's
L-BFGS gets 101 m/s and a cleaner model). The 1 s record reaches the bottom of
the model: with 0.6 s the deepest rows get no reflection and Adam fills them
with stripes.
Driven entirely by natural language — the model inspects the two inputs, runs
the inversion, and draws the result (initial vs inverted vs true + residual)
and the loss-convergence curve. Needs an LLM server
(examples/agent/start_vllm_qwen7b.sh).

    python examples/agent/demos/demo_fwi_llm.py

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
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwi_llm")
    nz, nx = 64, 96
    # True model: two layers (reflector at mid-depth). Init: the true one smoothed.
    vp_true = np.full((nz, nx), 1500.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 2300.0
    vp_init = _smooth(vp_true)
    np.save(f"{d}/fwi_true.npy", vp_true)
    np.save(f"{d}/fwi_init.npy", vp_init)

    # Pure natural language — no tool names. The intent router maps this to
    # run_fwi (build + invert + plot_model + plot_convergence in one call).
    drive(
        f"我想做全波形反演(FWI)。这是一个合成算例:真实速度模型 {d}/fwi_true.npy "
        f"(用它生成观测数据),初始模型 {d}/fwi_init.npy(网格间距 10 米,"
        f"记录长度 1 秒,主频 10Hz)。帮我从初始模型反演出速度结构,"
        f"并把反演结果(和真实模型对比)以及收敛曲线画出来给我看。输出放到 {out}。"
    )


if __name__ == "__main__":
    main()
