"""LLM-driven demo (ENGLISH): hello-FWI from natural language.

English counterpart of demo_fwi_llm.py: a two-layer true model, a homogeneous
start, and an acoustic FWI that recovers the reflector — then the inverted-vs-
true model comparison and the loss-convergence curve. Needs a vLLM server.

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_fwi_en.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwi_en")
    nz, nx = 64, 96
    vp_true = np.full((nz, nx), 1500.0, dtype=np.float32)
    vp_true[nz // 2:, :] = 2300.0
    vp_init = np.full((nz, nx), 1500.0, dtype=np.float32)
    np.save(f"{d}/fwi_true_en.npy", vp_true)
    np.save(f"{d}/fwi_init_en.npy", vp_init)

    drive(
        f"I'd like to run full-waveform inversion (FWI). This is a synthetic test: "
        f"the true model {d}/fwi_true_en.npy generates the observed data, and "
        f"{d}/fwi_init_en.npy is the starting model (grid spacing 10 m, record "
        f"length 0.6 s, peak frequency 10 Hz). Invert for the velocity structure, "
        f"and plot the inverted model against the true one plus the convergence "
        f"curve. Save everything to {out}."
    )


if __name__ == "__main__":
    main()
