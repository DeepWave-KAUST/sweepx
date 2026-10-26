"""LLM-driven demo (ENGLISH): hello-FWI from natural language.

English counterpart of demo_fwi_llm.py: a two-layer true model, a smoothed copy
of it as the start, and an acoustic FWI that sharpens the reflector (RMSE 118 ->
107 m/s in a replay of the tool call) — then the inverted-vs-true model
comparison and the loss-convergence curve. Needs an LLM
server (examples/agent/start_vllm_qwen7b.sh).

    python examples/agent/demos/demo_fwi_en.py
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
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwi_en")
    nz, nx = 64, 96
    vp_true = np.full((nz, nx), 1500.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 2300.0
    vp_init = _smooth(vp_true)
    np.save(f"{d}/fwi_true_en.npy", vp_true)
    np.save(f"{d}/fwi_init_en.npy", vp_init)

    drive(
        f"I'd like to run full-waveform inversion (FWI). This is a synthetic test: "
        f"the true model {d}/fwi_true_en.npy generates the observed data, and "
        f"{d}/fwi_init_en.npy is the starting model (grid spacing 10 m, record "
        f"length 1 s, peak frequency 10 Hz). Invert for the velocity structure, "
        f"and plot the inverted model against the true one plus the convergence "
        f"curve. Save everything to {out}."
    )


if __name__ == "__main__":
    main()
