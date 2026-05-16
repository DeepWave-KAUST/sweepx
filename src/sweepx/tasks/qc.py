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
) -> Path:
    """Save the FWI gradient ∂loss/∂vp as a diverging map."""
    grad_np = _to_np(grad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = float(np.percentile(np.abs(grad_np), perc)) or 1.0
    nz, nx = grad_np.shape
    extent = (0.0, nx * float(dh), nz * float(dh), 0.0)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(grad_np, cmap="RdBu_r", vmin=-s, vmax=s,
                   aspect="auto", extent=extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    title = f"∂loss/∂vp (epoch {epoch})" if epoch is not None else "∂loss/∂vp"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="gradient")
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
    vmin = bounds.min if bounds is not None and bounds.min is not None else None
    vmax = bounds.max if bounds is not None and bounds.max is not None else None

    if qc_spec.vp_png:
        written.append(save_vp_png(
            vp, dh=dh, out_path=qc_dir / "vp" / iter_tag, epoch=epoch,
            vmin=vmin, vmax=vmax,
        ))

    if qc_spec.vp_diff_png and state.get("qc_initial_vp") is not None:
        written.append(save_vp_diff_png(
            vp, state["qc_initial_vp"],
            dh=dh, out_path=qc_dir / "vp_diff" / iter_tag, epoch=epoch,
        ))

    if qc_spec.gradient_png:
        # Grid mode: vp leaf has .grad. Reparam mode: vp is non-leaf
        # (rendered) so .grad is None — skip silently.
        grad = getattr(vp, "grad", None)
        if grad is not None:
            written.append(save_gradient_png(
                grad, dh=dh, out_path=qc_dir / "gradient" / iter_tag,
                epoch=epoch,
            ))

    if qc_spec.shot_gather and extract_obs_syn_panels is not None:
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
    "save_shot_gather_png",
    "save_loss_curve_with_stages",
    "run_epoch_qc",
    "save_final_qc",
]
