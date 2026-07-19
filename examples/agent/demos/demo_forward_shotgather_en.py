"""LLM-driven demo (ENGLISH): forward modelling -> shot-gather plot.

English-language counterpart of demo_forward_shotgather_llm.py. The local LLM
inspects the velocity model, runs a forward simulation, and plots the synthetic
shot gather — all from a plain-English request, no tool/equation names. Needs a
vLLM server (examples/start_vllm_qwen7b.sh).

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_forward_shotgather_en.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    vp_path = "/tmp/vp_demo_en.npy"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_fwd_en")
    vp = np.full((64, 96), 1800.0, dtype=np.float32)
    vp[32:, :] = 2600.0  # flat reflector → clear reflection hyperbola
    np.save(vp_path, vp)

    drive(
        f"Run a forward simulation on the velocity model {vp_path} "
        f"(grid spacing 12.5 m, record length 0.6 s, peak frequency 10 Hz), "
        f"then plot the shot gather for me. Save the output to {out}."
    )


if __name__ == "__main__":
    main()
