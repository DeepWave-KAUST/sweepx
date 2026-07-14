"""Quick-look PNG generators for the data-preparation stages.

`sweep-tasks build-index` and `sweep-tasks build-plan` produce only npz
files by default; this module adds a single QC PNG per command so
users see what they just produced without having to write a python
snippet.

Two functions:

* :func:`plot_index_qc` — acquisition-geometry plan view + depth /
  trace-count histograms from a :class:`sweep_io.segy_index.SEGYIndex`.
* :func:`plot_plan_qc` — group-center scatter + per-row offset
  histogram + rows-per-group histogram from a
  :class:`sweep_io.seismic_plan.SeismicPlan`. Optionally overlays the
  parent :class:`SEGYIndex` so you can see exactly which rows the
  filters dropped.

Both routines deliberately use plain matplotlib (Agg backend) so they
work on a headless cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_index_qc(index: Any, out_path: str | Path) -> Path:
    """Render an acquisition-geometry QC PNG for a SEGYIndex.

    Three panels:

      [0]  plan view (x, y) of source positions (red) + receiver
           positions (blue, alpha-blended). 2-D lines collapse to a
           single horizontal track on each side.
      [1]  source-z + receiver-z depth histograms (overlay).
      [2]  per-shot trace-count histogram (i.e. how many receivers
           recorded each shot).

    Saves to ``out_path`` and returns the path written.
    """

    plt = _setup_mpl()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sx = np.asarray(index.sx_m, dtype=np.float64)
    sy = np.asarray(index.sy_m, dtype=np.float64)
    sz = np.asarray(index.sz_m, dtype=np.float64)
    rx = np.asarray(index.rx_m, dtype=np.float64)
    ry = np.asarray(index.ry_m, dtype=np.float64)
    rz = np.asarray(index.rz_m, dtype=np.float64)
    shot_ids = np.asarray(index.shot_id, dtype=np.int64)
    n_traces = int(sx.size)

    # Per-shot trace count via np.unique(return_counts)
    _, counts = np.unique(shot_ids, return_counts=True)
    n_shots = int(counts.size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Panel 0 — plan view
    ax = axes[0]
    # Stride traces if the dataset is huge — keep the marker layer cheap.
    stride = max(1, n_traces // 50000)
    ax.scatter(rx[::stride], ry[::stride], s=2, c="#2E6FA8", alpha=0.25,
               label=f"receivers (n={n_traces:,})", linewidths=0)
    ax.scatter(sx[::stride], sy[::stride], s=8, c="#C0392B", alpha=0.8,
               label=f"sources (n={n_shots:,})", linewidths=0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Acquisition geometry — plan view")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    # Equal aspect when the geometry has any y-extent worth showing.
    if (ry.max() - ry.min()) > 1.0 or (sy.max() - sy.min()) > 1.0:
        ax.set_aspect("equal", adjustable="datalim")

    # Panel 1 — depth histograms
    ax = axes[1]
    ax.hist(sz, bins=40, color="#C0392B", alpha=0.55, label=f"source z (mean {sz.mean():.1f} m)")
    ax.hist(rz, bins=40, color="#2E6FA8", alpha=0.55, label=f"receiver z (mean {rz.mean():.1f} m)")
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("trace count")
    ax.set_title("Source / receiver depth")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # Panel 2 — per-shot trace count
    ax = axes[2]
    ax.hist(counts, bins=min(40, max(5, counts.size // 5)),
            color="#1F4E79", alpha=0.85)
    ax.axvline(float(np.median(counts)), color="black", lw=0.8, ls="--",
               label=f"median {int(np.median(counts))}")
    ax.set_xlabel("traces per shot")
    ax.set_ylabel("shot count")
    ax.set_title(f"Traces per shot (n_shots={n_shots:,})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    summary = (
        f"n_traces={n_traces:,}  n_shots={n_shots:,}  "
        f"dt={float(index.dt_s)*1000:.2f} ms  nt={int(index.n_samples):,}  "
        f"sample_format={int(index.sample_format)}"
    )
    fig.suptitle(f"SEGYIndex QC — {Path(out_path).stem}\n{summary}", fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_plan_qc(
    plan: Any,
    out_path: str | Path,
    parent_index: Any | None = None,
) -> Path:
    """Render a QC PNG showing what a SeismicPlan keeps after filtering.

    Three panels:

      [0]  Group-center plan view (x, y). For CSG the group center is
           the source position; for CRG it's the receiver-cell center.
           When ``parent_index`` is supplied, the dropped groups (= shots
           or receiver cells absent from the plan) are drawn in light
           gray underneath so the filter effect is visible.
      [1]  Per-row source-receiver offset histogram with median +
           min/max annotations — confirms an ``--offset-max-m`` cap
           visually.
      [2]  Rows-per-group histogram (= traces per shot for CSG, shots
           per receiver-cell for CRG).

    Saves to ``out_path`` and returns the path written.
    """

    plt = _setup_mpl()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grouping = str(plan.grouping)
    src_xyz = np.asarray(plan.row_source_xyz, dtype=np.float64)
    rec_xyz = np.asarray(plan.row_receiver_xyz, dtype=np.float64)
    group_xyz = np.asarray(plan.group_xyz, dtype=np.float64)
    group_offsets = np.asarray(plan.group_offsets, dtype=np.int64)
    rows_per_group = np.diff(group_offsets)

    offsets = np.hypot(rec_xyz[:, 0] - src_xyz[:, 0],
                       rec_xyz[:, 1] - src_xyz[:, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Panel 0 — group-center plan view, with dropped groups underlay
    ax = axes[0]
    if parent_index is not None:
        if grouping == "csg":
            # Each shot id in the parent index but not in the plan is a "dropped" shot.
            full_sx = np.asarray(parent_index.sx_m, dtype=np.float64)
            full_sy = np.asarray(parent_index.sy_m, dtype=np.float64)
            full_shot_ids = np.asarray(parent_index.shot_id, dtype=np.int64)
            unique_full, first_idx = np.unique(full_shot_ids, return_index=True)
            full_centers = np.column_stack([full_sx[first_idx], full_sy[first_idx]])
            kept = np.isin(unique_full, np.asarray(plan.group_id, dtype=np.int64))
            dropped = full_centers[~kept]
            if dropped.size:
                ax.scatter(dropped[:, 0], dropped[:, 1], s=10, c="0.75",
                           label=f"dropped shots (n={(~kept).sum():,})",
                           linewidths=0, alpha=0.7)
    ax.scatter(group_xyz[:, 0], group_xyz[:, 1], s=10, c="#C0392B",
               label=f"plan groups (n={plan.n_groups:,})", linewidths=0, alpha=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Plan group centers — grouping={grouping}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    if float(np.ptp(group_xyz[:, 1])) > 1.0:
        ax.set_aspect("equal", adjustable="datalim")

    # Panel 1 — per-row offset histogram
    ax = axes[1]
    ax.hist(offsets, bins=60, color="#1F4E79", alpha=0.85)
    med = float(np.median(offsets))
    ax.axvline(med, color="black", lw=0.8, ls="--",
               label=f"median {med:.0f} m")
    ax.set_xlabel("|source − receiver| offset (m)")
    ax.set_ylabel("row count")
    ax.set_title(
        f"Offsets ({float(offsets.min()):.0f} – {float(offsets.max()):.0f} m)"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # Panel 2 — rows per group histogram
    ax = axes[2]
    ax.hist(rows_per_group, bins=min(40, max(5, rows_per_group.size // 5)),
            color="#4C78A8", alpha=0.85)
    med = float(np.median(rows_per_group))
    ax.axvline(med, color="black", lw=0.8, ls="--",
               label=f"median {int(med)}")
    ax.set_xlabel(f"rows per group ({'traces/shot' if grouping == 'csg' else 'shots/receiver'})")
    ax.set_ylabel("group count")
    ax.set_title(f"Rows per group (total rows={plan.n_rows:,})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    summary = (
        f"grouping={grouping}  n_groups={plan.n_groups:,}  "
        f"n_rows={plan.n_rows:,}  dt={float(plan.dt_s)*1000:.2f} ms  "
        f"nt={int(plan.samples_per_trace):,}"
    )
    label = ""
    if isinstance(plan.build_meta, dict):
        bl = plan.build_meta.get("build_label") or plan.build_meta.get("label")
        if bl:
            label = f"  label={bl!r}"
    fig.suptitle(f"SeismicPlan QC — {Path(out_path).stem}\n{summary}{label}", fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
