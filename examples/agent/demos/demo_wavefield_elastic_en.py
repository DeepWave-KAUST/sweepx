"""LLM-driven demo (ENGLISH): elastic wavefield (P + S waves) animated.

English counterpart of demo_wavefield_elastic_llm.py. From plain English the
model recognises the three elastic parameter files need the Elastic equation,
centres the source, and animates the P- and S-wave rings. Needs a vLLM server.

    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    python examples/demos/demo_wavefield_elastic_en.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _llm_demo import drive


def main() -> None:
    d = "/tmp"
    out = os.environ.get("DEMO_OUT", "/tmp/sweep_elastic_en")
    nz = nx = 200
    np.save(f"{d}/vp_el_en.npy", np.full((nz, nx), 2500.0, dtype=np.float32))
    np.save(f"{d}/vs_el_en.npy", np.full((nz, nx), 1500.0, dtype=np.float32))
    np.save(f"{d}/rho_el_en.npy", np.full((nz, nx), 2200.0, dtype=np.float32))

    drive(
        f"I have a homogeneous elastic medium given by three parameter files: "
        f"P-wave velocity {d}/vp_el_en.npy, S-wave velocity {d}/vs_el_en.npy, "
        f"density {d}/rho_el_en.npy (grid spacing 10 m). Put the source at the "
        f"centre of the model and show me how the seismic waves (P and S) "
        f"propagate outward as an animation. Save it to {out}."
    )


if __name__ == "__main__":
    main()
