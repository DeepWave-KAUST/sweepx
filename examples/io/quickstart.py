"""Minimal sweep-io demo: round-trip a velocity model and a geometry."""

from pathlib import Path

import numpy as np

from sweep_io.geometry import Geometry
from sweep_io.models import load_velocity, save_velocity


def main() -> None:
    out = Path("./_quickstart_out")
    out.mkdir(exist_ok=True)

    vp = np.linspace(1500.0, 4500.0, 200 * 400, dtype="float32").reshape(200, 400)
    save_velocity(out / "vp.npy", vp)
    vp_loaded = load_velocity(out / "vp.npy")
    assert np.array_equal(vp, vp_loaded)
    print(f"vp roundtrip ok: shape={vp_loaded.shape}, dtype={vp_loaded.dtype}")

    sources = np.array([[100, 0], [200, 0], [300, 0]], dtype="int64")
    receivers = np.broadcast_to(
        np.stack([np.arange(0, 400, 5), np.zeros(80, dtype="int64")], axis=1),
        (3, 80, 2),
    ).copy()
    geom = Geometry(sources=sources, receivers=receivers,
                    dt=0.001, nt=4000, dh=(10.0, 10.0))
    geom.save(out / "acq.json")
    geom2 = Geometry.load(out / "acq.json")
    print(f"geometry roundtrip ok: nshots={geom2.nshots}, nreceivers={geom2.nreceivers}")


if __name__ == "__main__":
    main()
