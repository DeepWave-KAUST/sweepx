"""DataPlan + ModelPlan + PhysicalGeometry.to_grid — end-to-end illustration.

No FWI engine, no real SEG-Y file — just a synthetic streamer-like
``(PhysicalGeometry, observed)`` pair so you can see what each knob does
and how the per-stage ``to_grid`` snaps + dedupes when the FWI grid is
coarser than the receiver spacing.

Run:
    python examples/plan_demo.py
"""

from __future__ import annotations

import numpy as np

from sweep_io.geometry import PhysicalGeometry
from sweep_io.plan import DataPlan, ModelPlan, apply_data_plan, apply_model_plan


def make_synthetic_streamer():
    """100 shots, 120 receivers each, towed streamer (offsets fixed per shot)."""
    nshots, nrec, nt = 100, 120, 800
    dt = 0.004
    # Sources every 25 m, x from 3000 to 5475 m
    sx = 3000.0 + np.arange(nshots) * 25.0
    sources = np.stack([sx, np.full(nshots, 6.0)], axis=1)        # depth 6 m
    # Receivers: streamer trails 250 m behind, 25 m spacing, 120 channels
    rx_off = -250.0 - np.arange(nrec) * 25.0                       # -250 .. -3225 m
    rx = sx[:, None] + rx_off[None, :]
    rz = np.full((nshots, nrec), 10.0)
    receivers = np.stack([rx, rz], axis=-1)
    pg = PhysicalGeometry(sources, receivers, dt=dt, nt=nt, meta={"line": "synthetic"})

    # Fake observed data: each shot fills its receivers with random noise
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((nshots, nt, nrec)).astype("float32") * 1e-3
    return pg, obs


def banner(text: str) -> None:
    print(f"\n{'═' * 70}\n{text}\n{'═' * 70}")


def main() -> None:
    pg, obs = make_synthetic_streamer()
    print(f"Synthetic streamer: nshots={pg.nshots}, nrec={pg.nreceivers}, "
          f"nt={pg.nt}, dt={pg.dt}s, obs.shape={obs.shape}")

    # ----- DataPlan demos -------------------------------------------------
    banner("1) shot_stride=10 — subsample shots 10× (Viking-style triage)")
    plan = DataPlan(shot_stride=10)
    pg2, obs2, mask = apply_data_plan(plan, pg, obs)
    print(f"  → {pg2.nshots} shots (was {pg.nshots}); obs.shape={obs2.shape}; "
          f"receiver mask uniform: {bool(np.all(mask == mask[0:1]))}")

    banner("2) offset_max_m=2000 — drop the far end of the streamer")
    plan = DataPlan(offset_max_m=2000.0)
    pg2, obs2, mask = apply_data_plan(plan, pg, obs)
    n_kept_per_shot = int(mask[0].sum())
    print(f"  → {n_kept_per_shot} of {pg.nreceivers} receivers survive per shot "
          f"(uniform across {pg2.nshots} shots: {bool(np.all(mask == mask[0:1]))})")

    banner("3) dt_target_s=0.008 — halve the temporal sample rate")
    plan = DataPlan(dt_target_s=0.008)
    pg2, obs2, _ = apply_data_plan(plan, pg, obs)
    print(f"  → dt {pg.dt}s → {pg2.dt}s, nt {pg.nt} → {pg2.nt}, obs.shape={obs2.shape}")

    banner("4) combo: every 5th shot, |off|≤2000 m, dt halved, t∈[0.5, 2.5]s")
    plan = DataPlan(
        shot_stride=5,
        offset_max_m=2000.0,
        dt_target_s=0.008,
        t_start_s=0.5,
        t_end_s=2.5,
    )
    pg2, obs2, mask = apply_data_plan(plan, pg, obs)
    print(f"  → shots {pg.nshots} → {pg2.nshots}, "
          f"receivers/shot kept {int(mask[0].sum())}/{pg.nreceivers}, "
          f"dt {pg.dt} → {pg2.dt}, nt {pg.nt} → {pg2.nt}")

    # ----- ModelPlan demo --------------------------------------------------
    banner("5) ModelPlan: crop a vp to a region of interest")
    vp = np.linspace(1500.0, 4500.0, 401 * 2305, dtype="float32").reshape(401, 2305)
    plan = ModelPlan(x_window_m=(5000.0, 22000.0), z_window_m=(0.0, 3500.0))
    vp_out, geom_out, src_keep = apply_model_plan(
        plan, vp, dh=(12.5, 12.5), geom=pg,
    )
    print(f"  vp: {vp.shape} → {vp_out.shape}  (dh stays 12.5 m)")
    print(f"  surviving sources: {int(src_keep.sum())} of {pg.nshots} "
          f"(x ∈ [{plan.x_window_m[0]}, {plan.x_window_m[1]}] m)")
    new_origin = geom_out.meta["model_plan_origin_m"]
    print(f"  geometry rebased to origin {new_origin}")

    # ----- PhysicalGeometry.to_grid: snap + dedupe per stage --------------
    banner("6) to_grid: per-stage snap + dedupe at different FWI dh values")
    for dh in (75.0, 37.5, 25.0, 12.5):
        gg, mask = pg.to_grid(dh=(dh, dh), dedupe=True, dedup_method="nearest")
        rate = mask.sum() / mask.size
        unique = len({tuple(int(v) for v in r) for r in gg.receivers[0, mask[0]]})
        ratio = pg.nreceivers / unique if unique else 0.0
        verdict = "OK 1:1" if abs(ratio - 1.0) < 0.05 else f"{ratio:.2f}× dedup"
        print(f"  dh={dh:5.1f} m  →  kept {int(mask[0].sum())}/{pg.nreceivers} per shot  "
              f"({rate * 100:5.1f}% global)   [{verdict}]")


if __name__ == "__main__":
    main()
