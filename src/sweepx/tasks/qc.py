"""FWI quality-control products written during a run.

This module composes :mod:`sweep_viz` plotting primitives (no in-repo
matplotlib code) into per-epoch snapshots that diagnose how an FWI
inversion is behaving. The runner calls :func:`run_epoch_qc` at a
user-configurable cadence; final outputs go to ``<task_dir>/qc/``.

QC products:
  * ``vp/iter_NNNN.png`` — current vp model
  * ``vp_diff/iter_NNNN.png`` — Δvp vs the *initial* vp (resampled to
    the current stage's grid for shape parity)
  * ``gradient/iter_NNNN.png`` — ∂loss/∂vp (grid mode only)
  * ``shot_gather/iter_NNNN.png`` — N obs/syn side-by-side panels
    (triggers one extra solver forward per cadence)
  * ``loss_curve.png`` — final-only enhanced loss with per-stage shading

None of these depend on FWI internals; pass tensors / numpy arrays in
and the function writes the PNG. Errors here are caught and logged at
the call site (QC must never break the inversion).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _to_np(t):
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def save_vp_png(
    vp,
    *,
    dh: float,
    out_path: Path,
    epoch: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title_prefix: str = "vp",
) -> Path:
    """Save a single-panel velocity model PNG."""
    from sweep_viz.model import plot_vp

    vp_np = _to_np(vp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    title = f"{title_prefix} (epoch {epoch})" if epoch is not None else title_prefix
    plot_vp(vp_np, dh=(float(dh), float(dh)), ax=ax,
            vmin=vmin, vmax=vmax, title=title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_vp_diff_png(
    vp,
    vp_initial,
    *,
    dh: float,
    out_path: Path,
    epoch: int | None = None,
    perc: float = 99.0,
) -> Path:
    """Save Δvp = current − initial (initial resampled to current shape if needed)."""
    from sweep_viz.model import plot_vp_diff

    vp_np = _to_np(vp)
    init_np = _to_np(vp_initial)
    if init_np.shape != vp_np.shape:
        # Bilinear resample initial to the current shape.
        from scipy.ndimage import zoom
        zfac = vp_np.shape[0] / init_np.shape[0]
        xfac = vp_np.shape[1] / init_np.shape[1]
        init_np = zoom(init_np.astype(np.float32), (zfac, xfac), order=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    title = f"Δvp = current - initial  (epoch {epoch})" if epoch is not None else "Δvp"
    plot_vp_diff(vp_np, init_np, dh=(float(dh), float(dh)), ax=ax,
                 perc=perc, title=title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gradient_png(
    grad,
    *,
    dh: float,
    out_path: Path,
    epoch: int | None = None,
    perc: float = 99.0,
    cmap: str | None = None,
) -> Path:
    """Save the FWI gradient ∂loss/∂vp as a diverging map.

    ``cmap`` defaults to the bundled :data:`sweep_viz.colormaps.IMAGE_CMAP`
    (``"sweep_image"``) — tuned for percentile-clipped kernel/gradient
    plots. Pass ``cmap="RdBu_r"`` to recover the legacy red/blue look.
    """
    # Import here so a missing sweep-viz at install time doesn't break
    # other QC paths. The registration side-effect fires on import.
    from sweep_viz.colormaps import IMAGE_CMAP

    grad_np = _to_np(grad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = float(np.percentile(np.abs(grad_np), perc)) or 1.0
    nz, nx = grad_np.shape
    extent = (0.0, nx * float(dh), nz * float(dh), 0.0)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(grad_np, cmap=cmap or IMAGE_CMAP, vmin=-s, vmax=s,
                   aspect="auto", extent=extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    title = f"∂loss/∂vp (epoch {epoch})" if epoch is not None else "∂loss/∂vp"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="gradient")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gradient_ortho_slices_png(
    grad_zyx,
    *,
    dh: float | tuple[float, float, float],
    out_path: Path,
    epoch: int | None = None,
    perc: float = 99.0,
    cmap: str | None = None,
) -> Path:
    """3-D version of :func:`save_gradient_png` — ortho slices of the
    velocity gradient with the diverging ``sweep_image`` colormap and
    symmetric percentile-clipped ``vmin/vmax``.

    Used by the multisource runner to drop a per-QC-epoch gradient
    visualization next to the vp ortho slices (``qc/gradient/`` parallel
    to ``qc/vp/``).
    """
    from sweep_viz.colormaps import IMAGE_CMAP
    from sweep_viz.model import plot_vp_ortho_slices

    grad = _to_np(grad_zyx)
    if grad.ndim != 3:
        raise ValueError(
            f"save_gradient_ortho_slices_png expects 3-D grad; got shape {grad.shape}"
        )
    s = float(np.percentile(np.abs(grad), perc)) or 1.0
    dh_tuple = (
        (float(dh), float(dh), float(dh)) if np.isscalar(dh)
        else tuple(float(d) for d in dh)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_vp_ortho_slices(
        grad, dh_xyz=dh_tuple, vmin=-s, vmax=s,
        cmap=cmap or IMAGE_CMAP,
        title_prefix=(
            f"∂loss/∂vp (epoch {epoch})" if epoch is not None else "∂loss/∂vp"
        ),
        cbar_label="gradient",
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ortho_slices_png(
    vol_zyx,
    *,
    dh: float | tuple[float, float, float],
    out_path: Path,
    epoch: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title_prefix: str = "vp",
    cbar_label: str = "vp (m/s)",
    cmap: str | None = None,
) -> Path:
    """Save a 3-orthogonal-slice PNG for a 3-D velocity volume.

    Thin wrapper around :func:`sweep_viz.model.plot_vp_ortho_slices`;
    used by ``run_epoch_qc`` when ``state["inv_by_name"]["vp"]`` is 3-D
    (the OBN CRG-plan FWI case). ``dh`` may be a scalar (cubic dh) or a
    per-axis ``(dz, dy, dx)`` tuple.
    """
    from sweep_viz.colormaps import VP_CMAP
    from sweep_viz.model import plot_vp_ortho_slices

    vol = _to_np(vol_zyx)
    if vol.ndim != 3:
        raise ValueError(f"save_ortho_slices_png expects 3-D vol; got shape {vol.shape}")
    dh_tuple = (float(dh), float(dh), float(dh)) if np.isscalar(dh) else tuple(float(d) for d in dh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_vp_ortho_slices(
        vol, dh_xyz=dh_tuple, vmin=vmin, vmax=vmax,
        cmap=cmap or VP_CMAP,
        title_prefix=f"{title_prefix} (epoch {epoch})" if epoch is not None else title_prefix,
        cbar_label=cbar_label,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_shot_gather_png(
    obs_panel=None,
    syn_panel=None,
    *,
    dt: float | None = None,
    out_path: Path,
    epoch: int | None = None,
    perc: float = 99.0,
    shot_ids: Sequence[int] | None = None,
    normalize: str = "trace",
    payload: dict | None = None,
    interleave_block: int = 12,
) -> Path:
    """Save a multi-panel obs/syn QC figure for a sample of shots.

    Two call styles:

    1. **Legacy positional** (kept for back-compat): pass ``obs_panel`` and
       ``syn_panel`` of shape ``(n_shots, nt, nrec)``. The figure is a
       simple ``n_shots × 2`` grid of side-by-side gray-scale panels with
       trace-index as the horizontal axis. Used by the unit tests.

    2. **Rich payload** (recommended): pass ``payload`` — a dict produced
       by the runner's QC extractor — with keys

           obs                 (n_shots, nt, nrec)   raw amplitudes
           syn                 (n_shots, nt, nrec)
           shot_ids            list[int]
           src_xy_m            (n_shots, 2) sources in meters
           rec_xy_m            (n_shots, nrec, 2)
           unique_idx_per_shot list of int arrays — one trace per unique grid cell
           dh_m, dt_s

       The rich layout mirrors ``fwi_workflow-dev``'s ``save_iteration_qc``:
       per shot, an interleaved obs/syn gray-scale panel where blocks of
       ``interleave_block`` traces alternate (obs / syn / obs / syn …) so
       amplitudes can be eyeballed at matching x. Plus one row at the
       bottom with (a) acquisition geometry (sources + receivers in km
       coordinates) and (b) amplitude spectrum |FFT(obs)| vs |FFT(syn)|
       averaged across receivers, for all picked shots.

       Duplicate receivers (multiple physical receivers snapping to the
       same grid cell at coarse dh) are dropped here so that what the
       plot shows is exactly what the loss saw.

    ``normalize`` controls per-trace amplitude scaling: ``"trace"`` (each
    trace ÷ own RMS, robust to wavelet-amplitude mismatch — default),
    ``"shot"`` (single percentile per panel), ``"joint"`` (common
    percentile across obs and syn — quantitative).
    """
    # Dispatch
    if payload is not None:
        if dt is None:
            dt = float(payload.get("dt_s", 1.0))
        return _save_shot_gather_rich(
            payload=payload, out_path=out_path, epoch=epoch, perc=perc,
            normalize=normalize, interleave_block=interleave_block, dt=dt,
        )

    if obs_panel is None or syn_panel is None:
        raise TypeError("save_shot_gather_png: pass either positional panels or payload=dict")
    return _save_shot_gather_legacy(
        obs_panel=obs_panel, syn_panel=syn_panel, dt=dt, out_path=out_path,
        epoch=epoch, perc=perc, shot_ids=shot_ids, normalize=normalize,
    )


def _save_shot_gather_legacy(
    obs_panel, syn_panel, *, dt, out_path, epoch, perc, shot_ids, normalize,
) -> Path:
    """Original side-by-side ``n_shots × 2`` figure (back-compat path)."""
    obs_np = _to_np(obs_panel).astype(np.float32, copy=False)
    syn_np = _to_np(syn_panel).astype(np.float32, copy=False)
    if obs_np.shape != syn_np.shape:
        raise ValueError(
            f"shot_gather: obs.shape {obs_np.shape} != syn.shape {syn_np.shape}"
        )
    n_shots = int(obs_np.shape[0])
    if shot_ids is None:
        shot_ids = list(range(n_shots))
    elif len(shot_ids) != n_shots:
        raise ValueError(f"shot_ids length {len(shot_ids)} != n_shots {n_shots}")

    def _prep(data):
        if normalize == "trace":
            rms = np.sqrt((data ** 2).mean(axis=-2, keepdims=True)).clip(1e-20)
            d = data / rms
            s = float(np.percentile(np.abs(d), perc)) or 1.0
            return d, -s, s
        if normalize == "shot":
            s = float(np.percentile(np.abs(data), perc)) or 1.0
            return data, -s, s
        raise ValueError(f"unknown normalize mode {normalize!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_shots, 2, figsize=(11, 2.8 * n_shots), constrained_layout=True)
    if n_shots == 1:
        axes = np.array([axes])
    nt, nrec = obs_np.shape[-2], obs_np.shape[-1]
    extent = (0.0, float(nrec), nt * float(dt), 0.0)
    if normalize == "joint":
        both = np.concatenate([np.abs(obs_np).ravel(), np.abs(syn_np).ravel()])
        s_joint = float(np.percentile(both, perc)) or 1.0
    for i, sid in enumerate(shot_ids):
        for j, (data, label) in enumerate(((obs_np[i], "obs"), (syn_np[i], "syn"))):
            if normalize == "joint":
                disp, vmin, vmax = data, -s_joint, s_joint
            else:
                disp, vmin, vmax = _prep(data)
            ax = axes[i, j]
            ax.imshow(disp, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", extent=extent)
            ax.set_title(f"shot {sid} — {label}", fontsize=9)
            if j == 0:
                ax.set_ylabel("time (s)")
            ax.set_xlabel("trace")
    if epoch is not None:
        fig.suptitle(f"obs vs syn (epoch {epoch}, normalize={normalize})", fontsize=11)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_shot_gather_rich(
    *, payload, out_path, epoch, perc, normalize, interleave_block, dt,
) -> Path:
    """4-panel layout matching ``fwi_workflow-dev``: interleave + map + spectrum.

    Shows the SAME obs / syn arrays that the loss consumed — i.e. **all**
    receivers (including duplicates that snap to the same grid cell at
    coarse dh). Duplicate adjacent traces in syn are a genuine feature
    of the data the loss sees; hiding them would misrepresent the
    inversion. The horizontal axis is trace index (matches data layout);
    receiver physical x is shown as a secondary axis on top so the user
    can read where each trace sits in space.
    """
    obs_all = _to_np(payload["obs"]).astype(np.float32, copy=False)   # (n_shots, nt, nrec)
    syn_all = _to_np(payload["syn"]).astype(np.float32, copy=False)
    shot_ids = list(payload["shot_ids"])
    src_xy_m = np.asarray(payload["src_xy_m"], dtype=np.float64)      # (n_shots, 2)
    rec_xy_m = np.asarray(payload["rec_xy_m"], dtype=np.float64)      # (n_shots, nrec, 2)
    # Note: ``unique_idx_per_shot`` from the payload is INFORMATIONAL
    # only — used to annotate the number of unique grid cells but NOT
    # to filter traces. The loss saw all of them.
    unique_idx_per_shot = [
        np.asarray(u, dtype=np.int64)
        for u in payload.get("unique_idx_per_shot", [np.arange(rec_xy_m.shape[1])] * len(shot_ids))
    ]
    n_shots = len(shot_ids)
    nt = int(obs_all.shape[-2])
    nrec = int(obs_all.shape[-1])

    def _norm_trace(arr2d):
        """``arr2d`` shape (nt, nrec) → trace-normalized for display."""
        if normalize == "trace":
            rms = np.sqrt((arr2d ** 2).mean(axis=0, keepdims=True)).clip(1e-20)
            return arr2d / rms
        return arr2d

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 3.0 * n_shots + 3.5), dpi=120, constrained_layout=True)
    gs = fig.add_gridspec(n_shots + 1, 2, height_ratios=[3] * n_shots + [3],
                          width_ratios=[3, 1])

    # ---- Per-shot interleaved obs/syn (ALL traces, no dedupe) ----
    for i, sid in enumerate(shot_ids):
        obs_i = obs_all[i]                # (nt, nrec) — all 120 traces
        syn_i = syn_all[i]
        rec_x_m_all = rec_xy_m[i, :, 0]   # (nrec,)
        n_unique = int(unique_idx_per_shot[i].size)

        obs_n = _norm_trace(obs_i)
        syn_n = _norm_trace(syn_i)
        s = float(np.percentile(np.abs(np.concatenate([obs_n, syn_n], axis=1)), perc)) or 1.0

        # Build interleaved panel over ALL traces
        merged_cols: list[np.ndarray] = []
        block_kinds: list[str] = []
        block_starts: list[int] = []          # trace-index at start of each block
        j = 0
        toggle = 0
        while j < nrec:
            end = min(j + interleave_block, nrec)
            block = (obs_n if toggle == 0 else syn_n)[:, j:end]
            merged_cols.append(block)
            block_kinds.append("obs" if toggle == 0 else "syn")
            block_starts.append(j)
            toggle = 1 - toggle
            j = end
        merged = np.concatenate(merged_cols, axis=1)
        # Trace index on primary x-axis (0..nrec). Physical x via secondary axis.
        extent = [0, nrec, nt * dt, 0.0]
        ax = fig.add_subplot(gs[i, :])
        im = ax.imshow(merged, cmap="gray", vmin=-s, vmax=s, aspect="auto", extent=extent)
        # Block separators + labels
        for k, (start, kind) in enumerate(zip(block_starts, block_kinds)):
            block_end = min(start + interleave_block, nrec)
            mid = 0.5 * (start + block_end)
            color = "#1F4E79" if kind == "obs" else "#C0392B"
            ax.text(mid, 0.0, kind,
                    ha="center", va="bottom", fontsize=8, color=color,
                    transform=ax.get_xaxis_transform(), clip_on=False)
            if k < len(block_kinds) - 1:
                ax.axvline(block_end, color="white", lw=0.4, alpha=0.5)

        ax.set_title(
            f"shot {sid} — obs/syn interleaved (block={interleave_block}, "
            f"nrec={nrec}, unique cells={n_unique})",
            fontsize=10,
        )
        ax.set_xlabel("trace index (alternating obs/syn blocks)")
        ax.set_ylabel("time (s)")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

        # Secondary x-axis: physical receiver x in km. Use a function-based
        # transform so the tick labels follow the trace-index axis exactly.
        # rec_x_m_all is the x-position of each trace; we interpolate.
        def _trace_to_xkm(t, rec_x_m_all=rec_x_m_all, nrec=nrec):
            t = np.asarray(t, dtype=np.float64)
            idx = np.clip(np.round(t).astype(int), 0, nrec - 1)
            return rec_x_m_all[idx] / 1000.0

        def _xkm_to_trace(x, rec_x_m_all=rec_x_m_all):
            # Inverse: given a physical x in km, find the closest trace index.
            xm = np.asarray(x, dtype=np.float64) * 1000.0
            return np.array([int(np.argmin(np.abs(rec_x_m_all - v))) for v in xm.ravel()]).reshape(xm.shape)

        sec = ax.secondary_xaxis("top", functions=(_trace_to_xkm, _xkm_to_trace))
        sec.set_xlabel("receiver x (km)")

        # Source x marker — convert physical x to trace-index space by
        # finding the nearest trace.
        src_x_m = float(src_xy_m[i, 0])
        # Show as line at primary axis edge if source is outside trace range,
        # else interpolate within.
        rec_x_min, rec_x_max = float(rec_x_m_all.min()), float(rec_x_m_all.max())
        if rec_x_min <= src_x_m <= rec_x_max:
            t_at_src = int(np.argmin(np.abs(rec_x_m_all - src_x_m)))
            ax.axvline(t_at_src, color="lime", lw=0.8, ls=":", alpha=0.7)
            ax.text(t_at_src, nt * dt * 0.02, f"src @ {src_x_m/1000:.2f} km",
                    color="lime", fontsize=8, ha="center",
                    bbox=dict(facecolor="black", edgecolor="none", pad=1, alpha=0.5))
        else:
            side = 0 if src_x_m < rec_x_min else nrec
            ax.axvline(side, color="lime", lw=0.8, ls=":", alpha=0.7)
            ax.text(side, nt * dt * 0.02,
                    f"src @ {src_x_m/1000:.2f} km {'←' if side==0 else '→'}",
                    color="lime", fontsize=8,
                    ha="left" if side == 0 else "right",
                    bbox=dict(facecolor="black", edgecolor="none", pad=1, alpha=0.5))

    # ---- Geometry map (all receivers, all picked shots) ----
    ax_map = fig.add_subplot(gs[n_shots, 0])
    palette = plt.get_cmap("tab10")
    for i, sid in enumerate(shot_ids):
        rx_km = rec_xy_m[i, :, 0] / 1000.0
        rz_m = rec_xy_m[i, :, 1]
        sx_km = src_xy_m[i, 0] / 1000.0
        sz_m = src_xy_m[i, 1]
        ax_map.scatter(rx_km, rz_m, s=10, c=[palette(i % 10)], alpha=0.6,
                       label=f"shot {sid} recv (nrec={nrec}, unique_cells={unique_idx_per_shot[i].size})")
        ax_map.scatter([sx_km], [sz_m], s=80, marker="*",
                       facecolor=palette(i % 10), edgecolor="black", linewidth=0.5,
                       zorder=5)
    ax_map.set_xlabel("x (km)")
    ax_map.set_ylabel("depth (m)")
    ax_map.invert_yaxis()
    ax_map.set_title("Acquisition (★ source, dots receivers — duplicates overlap at coarse dh)", fontsize=9)
    ax_map.legend(fontsize=7, loc="best")
    ax_map.grid(True, alpha=0.3)

    # ---- Spectrum (over ALL traces, same as loss sees) ----
    ax_spec = fig.add_subplot(gs[n_shots, 1])
    freqs = np.fft.rfftfreq(nt, d=dt)
    for i, sid in enumerate(shot_ids):
        obs_i = obs_all[i]   # all traces
        syn_i = syn_all[i]
        obs_amp = np.abs(np.fft.rfft(obs_i, axis=0)).mean(axis=-1)
        syn_amp = np.abs(np.fft.rfft(syn_i, axis=0)).mean(axis=-1)
        obs_amp = obs_amp / (obs_amp.max() + 1e-30)
        syn_amp = syn_amp / (syn_amp.max() + 1e-30)
        c = palette(i % 10)
        ax_spec.plot(freqs, obs_amp, color=c, lw=1.0, label=f"shot {sid} obs")
        ax_spec.plot(freqs, syn_amp, color=c, lw=1.0, ls="--",
                     alpha=0.8, label=f"shot {sid} syn")
    f_nyq = 0.5 / dt
    ax_spec.set_xlim(0.0, min(60.0, f_nyq))
    ax_spec.set_xlabel("frequency (Hz)")
    ax_spec.set_ylabel("|FFT| (normalised)")
    ax_spec.set_title("Amplitude spectrum — receiver-averaged", fontsize=10)
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize=6, loc="best")

    if epoch is not None:
        fig.suptitle(
            f"obs vs syn QC — epoch {epoch}   normalize={normalize}   "
            f"(showing loss inputs — all {nrec} traces incl. duplicate grid cells)",
            fontsize=11,
        )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_loss_curve_with_stages(
    losses: Sequence[float],
    stage_boundaries: Sequence[int],
    *,
    out_path: Path,
    log_scale: bool = True,
) -> Path:
    """Loss vs epoch with per-stage alternating background shading."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    losses_arr = np.asarray(losses, dtype=np.float64)
    n = int(losses_arr.size)
    if n == 0:
        # Nothing to plot — create an empty figure with a message instead
        # of crashing.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no loss values", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    boundaries = [0, *list(stage_boundaries), n]
    boundaries = sorted(set(int(b) for b in boundaries if 0 <= int(b) <= n))
    palette = ["#f0f3ff", "#ffffff"]
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if hi > lo:
            ax.axvspan(lo, hi, facecolor=palette[i % 2], alpha=0.7, zorder=0)
            ax.text((lo + hi) / 2, losses_arr.max(), f"stage {i}",
                    ha="center", va="top", fontsize=8, color="#555",
                    alpha=0.7, zorder=1)
    ax.plot(np.arange(n), losses_arr, marker=".", lw=1.0, color="#1f77b4",
            zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if log_scale and losses_arr.min() > 0:
        ax.set_yscale("log")
    ax.set_title(f"FWI loss across {len(boundaries) - 1} stages")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Multisource supershot QC (source-encoded FWI)
# ----------------------------------------------------------------------


def save_supershot_qc_panel(
    obs_super,
    syn,
    *,
    picked_group_utm_xy: np.ndarray,
    used_shot_utm_xy: np.ndarray,
    all_groups_utm_xy: np.ndarray,
    all_shots_utm_xy: np.ndarray | None = None,
    sourceline_ids: np.ndarray | None = None,
    within_sourceline_sort_key: np.ndarray | None = None,
    frame_label: str = "UTM",
    inversion_extent_xy_m: tuple[float, float, float, float] | None = None,
    dt: float,
    out_path: Path,
    epoch: int,
    interleave_block: int = 50,
    perc: float = 99.0,
    f_hi_hz: float | None = None,
) -> Path:
    """Three-panel QC for one source-encoded supershot iter.

    The multisource loss compares ONE encoded supershot
    ``obs_super`` (sum of ±1 signed obs across ``B`` picked OBN nodes)
    against the solver's ``syn`` for the same encoded source. The legacy
    per-shot ``_save_shot_gather_rich`` doesn't fit (there's no
    per-physical-shot pairing); this panel mirrors its *layout* but on
    the encoded pair.

    Layout (single figure, ``out_path``):

      ┌─────────────────────────────────────────────────────────────┐
      │  obs / syn interleave (alternating blocks of `interleave_   │
      │  block` columns; "receiver" axis = the n_shared physical    │
      │  shot positions that all picked OBN nodes saw in common)    │
      ├──────────────────────────┬──────────────────────────────────┤
      │ Survey map (UTM)         │ Amplitude spectrum               │
      │  • light dots = ALL OBN  │  • obs (solid), syn (dashed)     │
      │  • light X    = all phys │  • receiver-averaged             │
      │    shots                 │  • optional vertical line @ f_hi │
      │  • red ★      = picked   │                                  │
      │    OBN nodes (B)         │                                  │
      │  • blue •     = used     │                                  │
      │    physical shots        │                                  │
      └──────────────────────────┴──────────────────────────────────┘

    Parameters
    ----------
    obs_super, syn
        Tensors/ndarrays of shape ``(1, nt, n_shared, 1)`` or
        ``(1, n_shared, nt)`` (either supported — squeezed/permuted to
        ``(nt, n_shared)``). The two are compared trace-for-trace —
        ``n_shared`` MUST match.
    picked_group_utm_xy
        ``(B, 2)`` (x, y) of the B picked OBN nodes. Despite the legacy
        ``_utm_`` suffix, the helper plots **whatever frame the caller
        provides** — UTM raw, model-frame (rotated, in meters), or any
        2-D Cartesian frame. ``frame_label`` sets the axis text. All
        five coord arrays must be in the SAME frame.
    used_shot_utm_xy
        ``(n_shared, 2)`` (x, y) of the physical shots used in this
        iter's supershot.
    all_groups_utm_xy
        ``(n_groups_total, 2)`` (x, y) of EVERY OBN node in the survey
        — gives the "where do my picked nodes sit" context.
    all_shots_utm_xy
        Optional ``(K, 2)`` (x, y) positions of every physical shot in
        the survey (any K). When provided, plotted as faint X markers
        underneath the used-shot dots. Usually a deduped down-sample of
        ``plan.row_source_xyz[:, :2]`` (the full set can be huge).
    dt
        Time step in seconds — sets the time axis on the gather and the
        Nyquist on the spectrum.
    out_path
        Where to write the PNG.
    epoch
        For the figure title.
    interleave_block
        Number of adjacent traces per obs/syn alternation. 50 → reading
        any 100-trace window contains both an obs block and a syn block
        of the same physical shots (eye sees the cycle-skip directly).
    perc
        Percentile (e.g. 99) used as the gather amplitude clip.
    f_hi_hz
        Optional vertical guide on the spectrum at the current bandpass
        upper edge. Useful to check syn isn't leaking energy above the
        bandpass.
    frame_label
        Short label shown on the survey-map axes — e.g. ``"UTM"``
        (default) or ``"model"`` when the caller passed rotated coords.
    inversion_extent_xy_m
        Optional ``(xmin, xmax, ymin, ymax)`` in METERS of the active
        inversion grid extent — drawn as a dashed yellow rectangle on
        the survey map so the user can see which picked OBN nodes /
        physical shots actually sit inside the active model window.
        Should be in the same frame as the coord arrays.
    sourceline_ids
        Optional ``(n_shared,)`` integer array of source-line / file-id
        per physical-shot column. When provided, traces are sorted by
        ``(sourceline_id, within_sourceline_sort_key)`` (defaults to
        the x-coordinate of ``used_shot_utm_xy`` when no secondary key
        is given) so adjacent gather columns belong to the same sail
        line — much easier to read than the sampler's natural
        (sx, sy)-key order, which scatters lines randomly. Thin
        vertical lines + line-id labels mark the boundaries. The sort
        permutation is applied to obs / syn / used_shot_utm_xy in
        lock-step so the survey-map dots still correspond to the
        gather columns.
    within_sourceline_sort_key
        Optional ``(n_shared,)`` array used as the secondary sort key
        WITHIN each source-line partition. The sensible default for
        SEG-Y data is the absolute plan-row index of each column — that
        recovers the original SEG-Y file order (= along-sail-line
        order) regardless of how the survey is oriented in space. Using
        the x-coordinate as secondary (the fallback when this is None)
        only works when the sail line runs along x in the chosen
        frame; rotated model frames where lines run along y will
        produce chaos otherwise.
    """
    obs_np = _to_np(obs_super).astype(np.float32, copy=False)
    syn_np = _to_np(syn).astype(np.float32, copy=False)

    def _to_nt_n_shared(arr):
        if arr.ndim == 4:
            # (1, ?, ?, 1) — squeeze leading/trailing singletons.
            arr = np.squeeze(arr, axis=(0, -1))
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(
                f"save_supershot_qc_panel: expected obs/syn reducible "
                f"to 2-D (nt, n_shared); got shape {arr.shape}"
            )
        # Canonicalise to (nt, n_shared). Heuristic: usually nt >> n_shared.
        if arr.shape[0] < arr.shape[1]:
            # Likely (n_shared, nt) — transpose.
            arr = arr.T
        return arr

    obs2 = _to_nt_n_shared(obs_np)
    syn2 = _to_nt_n_shared(syn_np)
    if obs2.shape != syn2.shape:
        raise ValueError(
            f"save_supershot_qc_panel: obs.shape {obs2.shape} != "
            f"syn.shape {syn2.shape}"
        )
    nt, n_shared = obs2.shape
    B = int(picked_group_utm_xy.shape[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Sort columns by source-line so gather rows make geological sense.
    # Sampler returns rows in (sx, sy)-key order which scatters source
    # lines across the gather. With sourceline_ids in hand we sort by
    # (line_id ASC, sx_utm ASC) — within a line, sx increases along
    # the sail direction (approximately), giving a clean
    # "shotpoint vs time" panel per line + a thin divider between lines.
    used_shot_utm_xy = np.asarray(used_shot_utm_xy, dtype=np.float64)
    if sourceline_ids is not None:
        sl_ids = np.asarray(sourceline_ids, dtype=np.int64).reshape(-1)
        if sl_ids.size != n_shared:
            raise ValueError(
                f"save_supershot_qc_panel: sourceline_ids length "
                f"{sl_ids.size} != n_shared {n_shared}"
            )
        # Secondary key: caller-supplied along-line key (preferred — e.g.
        # absolute plan-row index recovers original SEG-Y file order
        # regardless of survey orientation). Fallback: x-coord, which is
        # only valid when the sail line runs along x in the chosen frame.
        if within_sourceline_sort_key is not None:
            sk = np.asarray(within_sourceline_sort_key).reshape(-1)
            if sk.size != n_shared:
                raise ValueError(
                    f"save_supershot_qc_panel: "
                    f"within_sourceline_sort_key length {sk.size} != "
                    f"n_shared {n_shared}"
                )
        else:
            sk = used_shot_utm_xy[:, 0]
        sort_order = np.lexsort((sk, sl_ids))
        obs2 = obs2[:, sort_order]
        syn2 = syn2[:, sort_order]
        used_shot_utm_xy = used_shot_utm_xy[sort_order]
        sl_ids_sorted = sl_ids[sort_order]
        # Boundary column-indices where the line-id flips.
        line_boundaries = np.flatnonzero(
            np.diff(sl_ids_sorted) != 0
        ) + 1
        # Per-line spans = (start_col, end_col, line_id) for labelling.
        seg_edges = np.concatenate((
            [0], line_boundaries, [n_shared],
        )).astype(np.int64)
        line_spans = [
            (int(seg_edges[i]), int(seg_edges[i + 1]),
             int(sl_ids_sorted[seg_edges[i]]))
            for i in range(seg_edges.size - 1)
        ]
    else:
        sl_ids_sorted = None
        line_boundaries = np.empty(0, dtype=np.int64)
        line_spans = [(0, n_shared, -1)]

    # Trace-normalise so weak far-offset traces are visible. Joint clip
    # (same vmin/vmax for obs and syn) — necessary so cycle-skips read
    # as colour differences, not normalisation artefacts.
    def _trace_norm(a):
        rms = np.sqrt((a ** 2).mean(axis=0, keepdims=True)).clip(1e-20)
        return a / rms
    obs_n = _trace_norm(obs2)
    syn_n = _trace_norm(syn2)
    s = float(np.percentile(
        np.abs(np.concatenate([obs_n, syn_n], axis=1)), perc,
    )) or 1.0

    # Build interleaved (nt, n_shared) panel: alternating obs/syn blocks
    # of `interleave_block` consecutive physical-shot columns.
    blk = max(1, int(interleave_block))
    merged_cols: list[np.ndarray] = []
    block_kinds: list[str] = []
    block_starts: list[int] = []
    j, toggle = 0, 0
    while j < n_shared:
        end = min(j + blk, n_shared)
        merged_cols.append((obs_n if toggle == 0 else syn_n)[:, j:end])
        block_kinds.append("obs" if toggle == 0 else "syn")
        block_starts.append(j)
        toggle ^= 1
        j = end
    merged = np.concatenate(merged_cols, axis=1)

    # Figure: top = interleave (full width); bottom = map | spectrum.
    fig = plt.figure(figsize=(14, 9), dpi=120, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[3, 2])

    # ---- Interleave gather (full width row 0) ----
    ax_g = fig.add_subplot(gs[0, :])
    extent = [0, n_shared, nt * float(dt), 0.0]
    im = ax_g.imshow(merged, cmap="gray", vmin=-s, vmax=s,
                     aspect="auto", extent=extent, interpolation="nearest")
    for k, (start, kind) in enumerate(zip(block_starts, block_kinds)):
        block_end = min(start + blk, n_shared)
        mid = 0.5 * (start + block_end)
        color = "#1F4E79" if kind == "obs" else "#C0392B"
        ax_g.text(mid, 0.0, kind, ha="center", va="bottom",
                  fontsize=8, color=color,
                  transform=ax_g.get_xaxis_transform(), clip_on=False)
        if k < len(block_kinds) - 1:
            ax_g.axvline(block_end, color="white", lw=0.4, alpha=0.5)
    # Source-line boundaries: thick yellow vertical lines + line-id
    # labels just above the top axis. Drawn AFTER the obs/syn dividers
    # so they sit on top.
    for b in line_boundaries:
        ax_g.axvline(float(b), color="#FFD400", lw=0.8, alpha=0.9)
    if sl_ids_sorted is not None:
        for s_col, e_col, lid in line_spans:
            ax_g.text(0.5 * (s_col + e_col), 1.02, f"L{lid}",
                      ha="center", va="bottom", fontsize=7,
                      color="#996600",
                      transform=ax_g.get_xaxis_transform(),
                      clip_on=False)
        title_extra = f", {len(line_spans)} source lines"
    else:
        title_extra = ""
    ax_g.set_title(
        f"encoded supershot — obs vs syn interleaved "
        f"(block={blk}, B={B} nodes × n_shared={n_shared} phys shots"
        f"{title_extra})",
        fontsize=10,
    )
    xlabel = "physical-shot index (alternating obs/syn blocks"
    if sl_ids_sorted is not None:
        xlabel += "; sorted by source-line + sx)"
    else:
        xlabel += ")"
    ax_g.set_xlabel(xlabel)
    ax_g.set_ylabel("time (s)")
    plt.colorbar(im, ax=ax_g, fraction=0.02, pad=0.01)

    # ---- Survey map (caller-chosen frame; default UTM) ----
    ax_map = fig.add_subplot(gs[1, 0])
    if all_shots_utm_xy is not None and all_shots_utm_xy.size:
        ax_map.scatter(all_shots_utm_xy[:, 0] / 1000.0,
                       all_shots_utm_xy[:, 1] / 1000.0,
                       s=2, c="#cccccc", marker="x", alpha=0.5,
                       label=f"all shots ({len(all_shots_utm_xy):,})")
    ax_map.scatter(all_groups_utm_xy[:, 0] / 1000.0,
                   all_groups_utm_xy[:, 1] / 1000.0,
                   s=4, c="#999999", marker="o", alpha=0.5,
                   label=f"all OBN nodes ({len(all_groups_utm_xy):,})")
    ax_map.scatter(used_shot_utm_xy[:, 0] / 1000.0,
                   used_shot_utm_xy[:, 1] / 1000.0,
                   s=18, c="#1F77B4", marker="o",
                   edgecolor="black", linewidth=0.3, alpha=0.85,
                   label=f"used phys shots ({n_shared})", zorder=4)
    ax_map.scatter(picked_group_utm_xy[:, 0] / 1000.0,
                   picked_group_utm_xy[:, 1] / 1000.0,
                   s=90, c="#D62728", marker="*",
                   edgecolor="black", linewidth=0.5,
                   label=f"picked OBN nodes (B={B})", zorder=5)
    # Inversion-active grid extent rectangle (dashed yellow). Useful
    # for confirming that picked OBN + used shots fall inside the
    # crop / origin-padded model grid the solver actually uses.
    if inversion_extent_xy_m is not None:
        x0, x1, y0, y1 = (float(v) / 1000.0 for v in inversion_extent_xy_m)
        ax_map.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color="#E5B400", lw=1.4, ls="--", alpha=0.95,
            label=(f"inversion grid "
                   f"({(x1 - x0):.1f}×{(y1 - y0):.1f} km)"),
            zorder=6,
        )
    ax_map.set_xlabel(f"x ({frame_label}, km)")
    ax_map.set_ylabel(f"y ({frame_label}, km)")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_title(
        f"Acquisition footprint for this iter ({frame_label} frame)",
        fontsize=10,
    )
    ax_map.legend(fontsize=7, loc="best")
    ax_map.grid(True, alpha=0.3)

    # ---- Spectrum (receiver-averaged) ----
    ax_spec = fig.add_subplot(gs[1, 1])
    freqs = np.fft.rfftfreq(nt, d=float(dt))
    obs_amp = np.abs(np.fft.rfft(obs2, axis=0)).mean(axis=-1)
    syn_amp = np.abs(np.fft.rfft(syn2, axis=0)).mean(axis=-1)
    obs_amp_n = obs_amp / (obs_amp.max() + 1e-30)
    syn_amp_n = syn_amp / (syn_amp.max() + 1e-30)
    ax_spec.plot(freqs, obs_amp_n, color="#1F4E79", lw=1.2, label="obs")
    ax_spec.plot(freqs, syn_amp_n, color="#C0392B", lw=1.2, ls="--",
                 label="syn")
    if f_hi_hz is not None and f_hi_hz > 0:
        ax_spec.axvline(float(f_hi_hz), color="#888888", lw=0.6, ls=":")
        ax_spec.text(float(f_hi_hz), 1.02, f"f_hi={f_hi_hz:.2f} Hz",
                     fontsize=7, color="#666666", ha="center", va="bottom",
                     transform=ax_spec.get_xaxis_transform())
    f_nyq = 0.5 / float(dt)
    ax_spec.set_xlim(0.0, min(2.5 * float(f_hi_hz) if (f_hi_hz and f_hi_hz > 0)
                               else 30.0, f_nyq))
    ax_spec.set_xlabel("frequency (Hz)")
    ax_spec.set_ylabel("|FFT| (normalised, recv-averaged)")
    ax_spec.set_title("Amplitude spectrum — obs vs syn", fontsize=10)
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"multisource supershot QC — epoch {epoch}", fontsize=11,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_vp_well_logs_png(
    vp_current,
    vp_initial,
    *,
    well_grid_idx: np.ndarray,
    well_labels: list[str],
    dz_m: float,
    out_path: Path,
    epoch: int,
    nrows: int = 2,
    ncols: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    z_top_m: float = 0.0,
    water_vp: float | None = 1500.0,
) -> Path:
    """Plot ``vp(z)`` 1-D profiles at a set of "pseudo-wells".

    The supershot gather + survey map tell you WHERE you have data;
    well logs tell you WHAT the model looks like at points of interest
    — especially the shallow water column (`should` be ~1500 m/s) and
    the seabed-to-1st-reflector transition. A flat line at 1500 m/s
    across the water depth is a strong sanity signal; SIREN init noise
    or freeze-water-layer mis-config show up here immediately.

    Parameters
    ----------
    vp_current
        Current vp tensor / ndarray of shape ``(nz, ny, nx)`` (m/s).
    vp_initial
        Reference (init) vp of the same shape — plotted as a dashed
        background curve in every well subplot.
    well_grid_idx
        ``(n_wells, 2)`` ``int`` array of ``(iy, ix)`` grid indices per
        well. The runner converts (model_x, model_y) in meters to these
        indices via ``round((x - origin_x) / dx_m)``.
    well_labels
        ``n_wells`` strings shown as subplot titles
        (e.g. ``"x=14.0 y=6.0 km"``).
    dz_m
        Vertical cell size in meters — sets the depth axis.
    out_path
        Where to write the PNG.
    epoch
        For the figure suptitle.
    nrows, ncols
        Subplot grid — must satisfy ``nrows * ncols >= n_wells``.
    vmin, vmax
        Optional fixed x-axis limits (m/s) for every subplot. Default:
        per-figure min/max across all displayed wells (current + init).
    z_top_m
        Depth coordinate of grid row ``iz=0`` (usually 0 = sea surface).
    water_vp
        Optional reference line drawn vertically on every subplot
        (default 1500). Pass ``None`` to disable.
    """
    vc = _to_np(vp_current).astype(np.float32, copy=False)
    vi = _to_np(vp_initial).astype(np.float32, copy=False)
    if vc.shape != vi.shape:
        raise ValueError(
            f"save_vp_well_logs_png: vp_current.shape {vc.shape} != "
            f"vp_initial.shape {vi.shape}"
        )
    if vc.ndim != 3:
        raise ValueError(
            f"save_vp_well_logs_png: expected 3-D vp (nz, ny, nx); "
            f"got shape {vc.shape}"
        )
    nz, ny, nx = vc.shape

    well_idx = np.asarray(well_grid_idx, dtype=np.int64).reshape(-1, 2)
    n_wells = int(well_idx.shape[0])
    if len(well_labels) != n_wells:
        raise ValueError(
            f"save_vp_well_logs_png: well_labels length {len(well_labels)} "
            f"!= n_wells {n_wells}"
        )
    if nrows * ncols < n_wells:
        raise ValueError(
            f"save_vp_well_logs_png: nrows*ncols ({nrows*ncols}) < "
            f"n_wells ({n_wells})"
        )

    # Extract each well's z-profile; clip indices into the grid to avoid
    # IndexError on rounding boundaries. Bad-index wells get NaN'd out.
    iy_all = np.clip(well_idx[:, 0], 0, ny - 1)
    ix_all = np.clip(well_idx[:, 1], 0, nx - 1)
    inside = (
        (well_idx[:, 0] >= 0) & (well_idx[:, 0] < ny)
        & (well_idx[:, 1] >= 0) & (well_idx[:, 1] < nx)
    )

    depths = z_top_m + np.arange(nz, dtype=np.float32) * float(dz_m)

    if vmin is None or vmax is None:
        all_vals = []
        for i in range(n_wells):
            if inside[i]:
                all_vals.append(vc[:, iy_all[i], ix_all[i]])
                all_vals.append(vi[:, iy_all[i], ix_all[i]])
        if all_vals:
            stack = np.concatenate(all_vals)
            if vmin is None:
                vmin = float(stack.min() - 50)
            if vmax is None:
                vmax = float(stack.max() + 50)
        else:
            vmin = vmin if vmin is not None else 1400.0
            vmax = vmax if vmax is not None else 5500.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.8 * ncols + 0.8, 3.5 * nrows + 0.7),
        dpi=120, constrained_layout=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        if i >= n_wells:
            ax.set_axis_off()
            continue
        label = well_labels[i]
        if not inside[i]:
            ax.text(0.5, 0.5, f"{label}\nOUT OF GRID",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#C0392B", fontsize=9)
            ax.set_axis_off()
            continue
        iy, ix = int(iy_all[i]), int(ix_all[i])
        cur = vc[:, iy, ix]
        ini = vi[:, iy, ix]
        ax.plot(ini, depths, color="#888888", lw=1.0, ls="--",
                label="init", alpha=0.85)
        ax.plot(cur, depths, color="#1F77B4", lw=1.3,
                label=f"epoch {epoch}", alpha=0.95)
        if water_vp is not None:
            ax.axvline(float(water_vp), color="#3CB371", lw=0.5,
                       ls=":", alpha=0.7)
        ax.set_xlim(float(vmin), float(vmax))
        ax.set_ylim(depths[-1], depths[0])  # invert: depth grows down
        ax.set_title(label, fontsize=9)
        if c == 0:
            ax.set_ylabel("depth (m)")
        if r == nrows - 1:
            ax.set_xlabel("vp (m/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"vp well logs — epoch {epoch}   "
        f"({n_wells} pseudo-wells, water_vp_ref={water_vp})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Runner orchestrator
# ----------------------------------------------------------------------


def run_epoch_qc(
    *,
    qc_spec,
    qc_dir: Path,
    epoch: int,
    state: dict,
    spec,
    extract_obs_syn_panels=None,
) -> list[Path]:
    """Run all configured per-epoch QC plots. Returns the paths written.

    ``extract_obs_syn_panels`` (callable) is provided by the runner when
    ``qc.shot_gather`` is enabled; it returns ``(obs_panel, syn_panel,
    shot_ids)`` with shapes ``(n_shots, nt, nrec)`` in time-major order.
    QC must never crash the inversion — exceptions bubble up to the
    runner, which logs them and continues.
    """
    written: list[Path] = []
    vp = state.get("inv_by_name", {}).get("vp")
    dh = float(state.get("dh", 1.0))
    if vp is None:
        return written
    iter_tag = f"iter_{epoch:04d}.png"
    bounds = spec.model_bounds.get("vp") if hasattr(spec, "model_bounds") else None
    if bounds is not None and not getattr(bounds, "enabled", True):
        bounds = None  # ``enabled: false`` → no fixed colormap range; let matplotlib auto-scale
    vmin = bounds.min if bounds is not None and bounds.min is not None else None
    vmax = bounds.max if bounds is not None and bounds.max is not None else None

    # 3-D path: emit three orthogonal slices (XY @ mid-depth, XZ @ mid-y,
    # YZ @ mid-x) for vp, Δvp, and the gradient. The 2-D helpers retain
    # the simple single-panel layout.
    if vp.ndim == 3:
        init_3d = state.get("qc_initial_vp")
        grad_3d = getattr(vp, "grad", None)
        # Per-axis dh: the runner state may carry a 3-tuple under "dh_xyz"
        # when a non-cubic grid is in use; fall back to the scalar.
        dh_xyz = state.get("dh_xyz", (dh, dh, dh))
        if qc_spec.vp_png:
            written.append(save_ortho_slices_png(
                vp, dh=dh_xyz, out_path=qc_dir / "vp" / iter_tag,
                epoch=epoch, vmin=vmin, vmax=vmax,
                title_prefix="vp", cbar_label="vp (m/s)",
            ))
        if qc_spec.vp_diff_png and init_3d is not None:
            init_np = _to_np(init_3d)
            if init_np.ndim == 3 and init_np.shape == vp.shape:
                diff = _to_np(vp) - init_np
                s = float(np.percentile(np.abs(diff), 99.0)) or 1.0
                from sweep_viz.colormaps import IMAGE_CMAP
                written.append(save_ortho_slices_png(
                    diff, dh=dh_xyz, out_path=qc_dir / "vp_diff" / iter_tag,
                    epoch=epoch, vmin=-s, vmax=s,
                    title_prefix="Δvp", cbar_label="Δvp (m/s)",
                    cmap=IMAGE_CMAP,
                ))
        if qc_spec.gradient_png and grad_3d is not None:
            grad_np = _to_np(grad_3d)
            s = float(np.percentile(np.abs(grad_np), 99.0)) or 1.0
            from sweep_viz.colormaps import IMAGE_CMAP
            written.append(save_ortho_slices_png(
                grad_np, dh=dh_xyz, out_path=qc_dir / "gradient" / iter_tag,
                epoch=epoch, vmin=-s, vmax=s,
                title_prefix="∂loss/∂vp", cbar_label="gradient",
                cmap=IMAGE_CMAP,
            ))
        # Shot-gather + reparam-precond plots remain 2-D-only for now;
        # the per-shot OBN QC layout is §3.6 polish work tracked separately.
        return written

    if qc_spec.vp_png:
        written.append(save_vp_png(
            vp, dh=dh, out_path=qc_dir / "vp" / iter_tag, epoch=epoch,
            vmin=vmin, vmax=vmax,
        ))

    init_for_plot = state.get("qc_initial_vp")
    if qc_spec.vp_diff_png and init_for_plot is not None:
        written.append(save_vp_diff_png(
            vp, init_for_plot,
            dh=dh, out_path=qc_dir / "vp_diff" / iter_tag, epoch=epoch,
        ))

    grad_for_plot = getattr(vp, "grad", None)
    if qc_spec.gradient_png:
        # Grid mode: vp leaf has .grad. Reparam mode: vp is non-leaf
        # (rendered) so .grad is None — skip silently.
        if grad_for_plot is not None:
            written.append(save_gradient_png(
                grad_for_plot, dh=dh, out_path=qc_dir / "gradient" / iter_tag,
                epoch=epoch,
            ))

    if qc_spec.shot_gather and extract_obs_syn_panels is not None:
        if vp.ndim == 3:
            # Rich shot-gather payload expects 2-D src/rec xy arrays; the
            # 3-D acq layout is handled by the orthogonal-slice QC (§3.6),
            # not this 2-D-only helper. Skip silently for now.
            return written
        try:
            result = extract_obs_syn_panels()
            # Back-compat: support both old 3-tuple and new payload dict.
            if isinstance(result, dict):
                written.append(save_shot_gather_png(
                    payload=result,
                    out_path=qc_dir / "shot_gather" / iter_tag,
                    epoch=epoch,
                    perc=float(qc_spec.shot_gather_perc),
                    normalize=str(getattr(qc_spec, "shot_gather_normalize", "trace")),
                    interleave_block=int(getattr(qc_spec, "shot_gather_interleave_block", 12)),
                ))
            else:
                obs_panel, syn_panel, shot_ids = result
                written.append(save_shot_gather_png(
                    obs_panel, syn_panel,
                    dt=float(state.get("dt", 1.0)),
                    out_path=qc_dir / "shot_gather" / iter_tag,
                    epoch=epoch,
                    perc=float(qc_spec.shot_gather_perc),
                    shot_ids=shot_ids,
                    normalize=str(getattr(qc_spec, "shot_gather_normalize", "trace")),
                ))
        except Exception as err:  # noqa: BLE001
            import traceback
            print(f"[qc] shot_gather failed at epoch {epoch}: {err}")
            traceback.print_exc()

    return written


def save_final_qc(
    *,
    qc_spec,
    qc_dir: Path,
    losses: Sequence[float],
    stage_epoch_boundaries: Sequence[int],
) -> list[Path]:
    """Final-only QC products (loss curve)."""
    written: list[Path] = []
    if qc_spec.loss_curve:
        written.append(save_loss_curve_with_stages(
            losses, stage_epoch_boundaries,
            out_path=qc_dir / "loss_curve.png",
        ))
    return written


__all__ = [
    "save_vp_png",
    "save_vp_diff_png",
    "save_gradient_png",
    "save_gradient_ortho_slices_png",
    "save_ortho_slices_png",
    "save_shot_gather_png",
    "save_supershot_qc_panel",
    "save_vp_well_logs_png",
    "save_loss_curve_with_stages",
    "run_epoch_qc",
    "save_final_qc",
]
