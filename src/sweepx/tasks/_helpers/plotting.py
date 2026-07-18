"""Loss/wavefield/RTM-QC plotting + illumination save. Verbatim from runner.py."""
from pathlib import Path

import numpy as np

def _save_illumination(solver, snapshots_dir: Path, epoch: int) -> list[Path]:
    artifacts: list[Path] = []
    for attr_name, label in (("source_illumination", "src"),
                             ("receiver_illumination", "rec")):
        tensor = getattr(solver, attr_name, None)
        if tensor is None:
            continue
        try:
            arr = tensor.detach().cpu().numpy()
        except AttributeError:
            continue
        path = snapshots_dir / f"{label}_illumination_epoch_{epoch:04d}.npy"
        np.save(path, arr)
        artifacts.append(path)
    return artifacts


def _dump_diving_window_qc(obs4, syn4, mask4, off_m, dt, epoch, node_id, qc_dir,
                           max_traces: int = 900, rec_xy=None, node_xy=None,
                           order: str = "position"):
    """Dump EXACTLY what the diving-wave misfit sees for one node, this iteration.

    The window is fitted to the first-break picks, so scoring it against those
    same picks is circular — the independent question is whether it brackets the
    real first-break energy in the real data. So this plots obs exactly as the
    runner hands it to the loss — whatever has (or has not) been done to it —
    rather than a side reconstruction, which would answer a different question.
    That is not academic: this dump is what revealed obs reaching the per-CRG
    misfit unfiltered, against a band-limited syn.

    ``order='position'`` (default) shows traces by REAL acquisition position, not
    by offset. This matters: the window is a function of |offset| and is therefore
    azimuthally SYMMETRIC, while a real 3-D first break is not. Offset-sorting
    interleaves every azimuth into one monotonic curve, so the window always looks
    like it hugs the data and an azimuthal error cannot show up — a synthetic with
    a deliberate +-0.45 s azimuthal term still read as "window contains the event"
    when offset-sorted, and only showed up as a smeared band.

    Snapped to the model grid the shots are a dense areal CARPET (hundreds of
    distinct x by hundreds of distinct y ~= one node's full shot set), so real
    acquisition lines are not resolvable and there is nothing to group by. Instead
    this takes a few x=const transects and keeps each WHOLE: every transect is a
    spatially continuous profile with its own V-shaped moveout (offset minimum
    where it passes the node), so a window that only tracks |offset| visibly
    drifts off the arrivals wherever the structure is azimuthally variable.
    ``order='offset'`` restores the old view.

    Args:
      obs4, syn4, mask4: (1, nt, nrec, 1) tensors, taken at the loss call site.
      off_m: (nrec,) source-receiver offset in metres.
      rec_xy: (nrec, 2) receiver (=survey shot) position in model-frame metres.
              Required for order='position'.
      node_xy: (2,) the node's own model-frame xy, for the plot annotation.
    Writes ``qc/diving_window_iter<E>_node<N>.npz`` (data + positions, for
    re-plotting) and ``.png``. Traces are decimated to ``max_traces``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = np.asarray(off_m, dtype=np.float64)
    xy = None if rec_xy is None else np.asarray(rec_xy, dtype=np.float64)
    lines_kept = 0
    if order == "position" and xy is not None:
        # The shots are a DENSE areal carpet once snapped to the model grid (on
        # the scale of hundreds of distinct x by hundreds of distinct y for a
        # node), so real acquisition lines are not resolvable here. Take x=const
        # transects instead — each is a spatially continuous profile through the
        # carpet with its own V-shaped moveout (offset minimum where the transect
        # passes the node).
        # Keep WHOLE transects and drop others. Decimating uniformly across all
        # traces instead would leave ~7 traces per transect and destroy the very
        # moveout this view exists to show — which is exactly what it did.
        xb = xy[:, 0]
        ux = np.unique(xb)
        n_lines = int(min(8, len(ux)))
        pick = ux[np.linspace(0, len(ux) - 1, n_lines).astype(np.int64)]
        per = max(2, int(max_traces) // max(n_lines, 1))
        sel = []
        for x0 in pick:
            idx = np.flatnonzero(xb == x0)
            idx = idx[np.argsort(xy[idx, 1])]              # along the transect
            if idx.size > per:
                idx = idx[np.linspace(0, idx.size - 1, per).astype(np.int64)]
            sel.append(idx)
        srt = np.concatenate(sel) if sel else np.argsort(o)
        lines_kept = n_lines
    else:
        order = "offset"
        srt = np.argsort(o)
        if srt.size > max_traces:  # decimate ACROSS offset so the span survives
            srt = srt[np.linspace(0, srt.size - 1, max_traces).astype(np.int64)]

    def _np(t):
        return t.detach().float()[0, :, :, 0].cpu().numpy()[:, srt]

    obs = _np(obs4); syn = _np(syn4); msk = _np(mask4)
    o = o[srt] / 1000.0
    if xy is not None:
        xy = xy[srt]
    nt = obs.shape[0]
    t_end = nt * float(dt)

    qc_dir = Path(qc_dir); qc_dir.mkdir(parents=True, exist_ok=True)
    stem = str(Path(qc_dir) / f"diving_window_iter{int(epoch):04d}_node{int(node_id)}")
    np.savez_compressed(stem + ".npz", obs=obs.astype(np.float32),
                        syn=syn.astype(np.float32), mask=msk.astype(np.float32),
                        offset_km=o.astype(np.float32), dt=float(dt),
                        epoch=int(epoch), node_id=int(node_id), order=order,
                        rec_xy=(np.zeros((0, 2), np.float32) if xy is None
                                else xy.astype(np.float32)),
                        node_xy=(np.zeros(2, np.float32) if node_xy is None
                                 else np.asarray(node_xy, np.float32)))

    # Edges read back OUT of the mask, not recomputed — this is what was applied.
    top = np.full(msk.shape[1], np.nan); bot = np.full(msk.shape[1], np.nan)
    for c in range(msk.shape[1]):
        nz = np.flatnonzero(msk[:, c] > 0.5)
        if nz.size:
            top[c] = nz[0] * dt; bot[c] = nz[-1] * dt

    def _agc(d, win_s=0.5):
        """Sliding-RMS AGC — DISPLAY ONLY (the npz above holds the raw data).
        Without it the amplitude decay swamps everything past ~4 km, which is
        exactly where the diving wave this window exists to isolate lives."""
        n = max(3, int(round(win_s / float(dt))) | 1)          # odd window
        q = np.pad(d.astype(np.float64) ** 2, ((n // 2, n // 2), (0, 0)))
        c = np.concatenate([np.zeros((1, q.shape[1])), np.cumsum(q, axis=0)], axis=0)
        rms = np.sqrt(np.maximum(c[n:] - c[:-n], 0.0) / n)     # (nt, nrec)
        return d / (rms + 1e-6 * (float(np.abs(rms).max()) or 1.0))

    ntr = obs.shape[1]
    xax = np.arange(ntr)                       # position order -> index axis
    lines = []                                 # transect boundaries
    if order == "position" and xy is not None:
        lines = list(np.flatnonzero(np.diff(xy[:, 0]) != 0) + 1)
    if order == "offset":
        xax = o

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharey=True)
    for ax, (name, d) in zip(axes, [("obs (as the loss gets it)", obs),
                                    ("obs x mask", obs * msk),
                                    ("syn x mask", syn * msk)]):
        d = _agc(d) * (msk if "mask" in name else 1.0)   # AGC re-inflates muted noise
        v = float(np.percentile(np.abs(d), 96)) or 1.0   # signed, real range
        ax.imshow(d, aspect="auto", cmap="gray", vmin=-v, vmax=v,
                  extent=[xax[0], xax[-1], t_end, 0.0], interpolation="nearest")
        ax.plot(xax, top, "-", lw=0.7, color="#e8112d", alpha=0.85)
        ax.plot(xax, bot, "-", lw=0.7, color="#e8112d", alpha=0.85)
        for b in lines:                        # one boundary per source line
            ax.axvline(b, color="#0a84ff", lw=0.4, alpha=0.45)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("trace  (x=const transects, each sorted along y)" if order == "position"
                      else "offset (km)")
        ax.set_ylim(t_end, 0.0)
    axes[0].set_ylabel("time (s)")
    if order == "position":
        # keep offset readable — it is what the window is actually a function of
        axt = axes[0].twiny()
        axt.set_xlim(axes[0].get_xlim())
        tk = np.linspace(0, ntr - 1, 8).astype(int)
        axt.set_xticks(tk); axt.set_xticklabels([f"{o[i]:.1f}" for i in tk], fontsize=8)
        axt.set_xlabel("offset (km) at that trace", fontsize=9)
    fig.suptitle(f"Diving-wave window — iter {int(epoch)}, node {int(node_id)}   "
                 f"(red = applied window edges, kept {100.0*float(msk.mean()):.1f}%"
                 f"{f', blue = transect boundary ({lines_kept} x=const transects)' if lines else ''}) "
                 f"[{order} order]", fontsize=13)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(stem + ".png")


def _plot_loss_curve(losses, path: Path, title: str) -> Path:
    """Loss curve via `sweep_tasks.viz.convergence.plot_loss`."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sweep_tasks.viz.convergence import plot_loss

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss(list(losses), ax=ax, logy=True, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_wavefield_snapshots(snapshots_np, snapshot_times, abcn, shape, path,
                              free_surface) -> Path:
    """Wavefield snapshot grid via `sweep_tasks.viz.wavefield.plot_snapshot` (one per panel).

    The PML / absorbing-boundary cropping logic (and the `(nsnap, 1, 1, 1, ...)`
    sweep-binding tensor layout) is sweep-tasks-specific, so it stays here;
    each cropped panel is then rendered by sweep_tasks.viz.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sweep_tasks.viz.wavefield import plot_snapshot

    nz, nx = int(shape[0]), int(shape[1])
    nsnap = snapshots_np.shape[0]
    fig, axes = plt.subplots(1, nsnap, figsize=(4 * nsnap, 4), squeeze=False)
    for i in range(nsnap):
        panel = snapshots_np[i, 0, 0, 0]
        if free_surface:
            panel = panel[:nz, abcn: abcn + nx]
        else:
            panel = panel[abcn: abcn + nz, abcn: abcn + nx]
        plot_snapshot(panel, ax=axes[0, i], perc=100.0,
                      cmap="seismic", title=f"t-step {snapshot_times[i]}")
        axes[0, i].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_rtm_qc_pngs(
    *,
    gradient_sum,
    rtm_sum,
    source_illum_sum,
    receiver_illum_sum,
    gradient_per_shot_norm_sum=None,
    rtm_per_shot_norm_sum=None,
    qc_dir: Path,
    suffix: str = "",
    eps: float = 1.0e-6,
    normalize: bool = True,
) -> list[Path]:
    """Write the standard RTM QC PNGs to ``qc_dir``.

    ``suffix`` lets the live-update path emit ``*_latest.png`` without
    overwriting the final products.

    Returns the list of written paths (so the caller can register them as
    artifacts in the task status).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    # Side-effect import: registers ``sweep_image`` / ``sweep_vp`` so the
    # RTM panels below resolve. Tolerate sweep_tasks.viz being missing — caller
    # gets a clear matplotlib error in that case.
    try:
        from sweep_tasks.viz.colormaps import IMAGE_CMAP as _RTM_CMAP
    except Exception:  # noqa: BLE001
        _RTM_CMAP = "seismic"

    qc_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def _imshow(ax, array, title, *, symmetric: bool, cmap: str, perc=(1.0, 99.0)):
        finite = array[np.isfinite(array)]
        if finite.size:
            vmin, vmax = np.percentile(finite, list(perc))
            if symmetric:
                m = max(abs(float(vmin)), abs(float(vmax)))
                vmin, vmax = -m, m
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None
        im = ax.imshow(array, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        return im

    def _single_png(array, title, path, *, symmetric, cmap):
        fig, ax = plt.subplots(1, 1, figsize=(11, 3.8))
        try:
            im = _imshow(ax, array, title, symmetric=symmetric, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        return path

    # Raw RTM image
    paths.append(_single_png(
        rtm_sum, f"RTM image{suffix}", qc_dir / f"rtm_image{suffix}.png",
        symmetric=True, cmap=_RTM_CMAP,
    ))
    # Raw FWI gradient image
    paths.append(_single_png(
        gradient_sum, f"FWI gradient image{suffix}",
        qc_dir / f"gradient_image{suffix}.png",
        symmetric=True, cmap=_RTM_CMAP,
    ))

    # Normalised image (only when normalize_by_illumination is on; otherwise
    # the raw RTM is already the deliverable)
    if normalize:
        denom = np.sqrt(np.maximum(source_illum_sum * receiver_illum_sum, 0.0) + float(eps))
        rtm_norm = (rtm_sum / denom).astype(np.float32)
        paths.append(_single_png(
            rtm_norm,
            f"RTM image (global illumination-normalised){suffix}",
            qc_dir / f"rtm_image_normalised{suffix}.png",
            symmetric=True, cmap=_RTM_CMAP,
        ))

    # Per-shot illumination-normalised image — the canonical Viking RTM
    # display (legacy `illumination_normalized_rtm_per_shot.png`). Each
    # shot's contribution was divided by its own illumination BEFORE the
    # stack, so deep / weakly-illuminated cells survive instead of being
    # drowned by the strong shallow energy in the global-normalised view.
    if rtm_per_shot_norm_sum is not None:
        paths.append(_single_png(
            np.asarray(rtm_per_shot_norm_sum),
            f"RTM image (per-shot illumination-normalised){suffix}",
            qc_dir / f"rtm_image_per_shot_normalised{suffix}.png",
            symmetric=True, cmap=_RTM_CMAP,
        ))
    if gradient_per_shot_norm_sum is not None:
        paths.append(_single_png(
            np.asarray(gradient_per_shot_norm_sum),
            f"FWI gradient (per-shot illumination-normalised){suffix}",
            qc_dir / f"gradient_image_per_shot_normalised{suffix}.png",
            symmetric=True, cmap=_RTM_CMAP,
        ))

    # Illumination panel: source / receiver / product
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), squeeze=False)
        _imshow(axes[0, 0], source_illum_sum, "Source illumination", symmetric=False, cmap="magma")
        _imshow(axes[0, 1], receiver_illum_sum, "Receiver illumination", symmetric=False, cmap="magma")
        prod = source_illum_sum * receiver_illum_sum
        _imshow(axes[0, 2], prod, "S * R (illumination product)", symmetric=False, cmap="magma")
        fig.tight_layout()
        ill_path = qc_dir / f"illumination{suffix}.png"
        fig.savefig(ill_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(ill_path)
    except Exception:  # noqa: BLE001
        plt.close("all")

    return paths


def _dump_receiver_rotation_qc(
    *,
    task_dir: Path,
    recv_model_xy: np.ndarray,
    recv_utm_xy: np.ndarray,
    recv_z: np.ndarray,
    grid_origin_xyz: tuple[float, float, float],
    grid_shape: tuple[int, int, int],
    dh_xyz: tuple[float, float, float],
) -> Path:
    """Save a receiver-layout PNG + npz showing the UTM→model rotation.

    Two-panel PNG (model frame on left, raw UTM on right) for visual
    confirmation that the ``rotation_metadata.json`` projects the OBN
    nodes the same way the legacy pipeline did. Companion npz holds the
    underlying float64 arrays so the user can do ``np.allclose`` against
    a legacy reference.

    Inputs are POST-filter (after any ModelPlan crop) — exactly the
    receivers the FWI run will actually see.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    qc_dir = task_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, _nz = (int(v) for v in grid_shape)
    dx_m, dy_m, _dz_m = (float(v) for v in dh_xyz)
    origin_x, origin_y, _origin_z = (float(v) for v in grid_origin_xyz)
    recv_model_xy = np.asarray(recv_model_xy, dtype=np.float64)
    recv_utm_xy = np.asarray(recv_utm_xy, dtype=np.float64)
    recv_z = np.asarray(recv_z, dtype=np.float64)

    # Origin-shifted model coords so (0, 0) = grid corner; matches the
    # propagator's local frame. Helpful when overlaying the grid bbox.
    shifted = recv_model_xy - np.asarray([origin_x, origin_y])[None, :]
    bbox_x = [0.0, nx * dx_m, nx * dx_m, 0.0, 0.0]
    bbox_y = [0.0, 0.0, ny * dy_m, ny * dy_m, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Left — model frame, origin-shifted, grid bbox overlay.
    sc0 = axes[0].scatter(
        shifted[:, 0], shifted[:, 1], c=recv_z, cmap="viridis",
        s=18, edgecolors="k", linewidths=0.4,
    )
    axes[0].plot(bbox_x, bbox_y, "k--", lw=1, label="grid bbox")
    axes[0].set_xlabel("x (m, model frame; origin at 0)")
    axes[0].set_ylabel("y (m, model frame; origin at 0)")
    axes[0].set_title(
        f"Receivers in model frame (n={len(shifted)})  "
        f"grid={nx}×{ny} @ ({dx_m:.1f}, {dy_m:.1f}) m"
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.04, pad=0.02)
    cb0.set_label("receiver z (m)")

    # Right — raw UTM (pre-rotation) for direct comparison with legacy outputs.
    sc1 = axes[1].scatter(
        recv_utm_xy[:, 0], recv_utm_xy[:, 1], c=recv_z, cmap="viridis",
        s=18, edgecolors="k", linewidths=0.4,
    )
    axes[1].set_xlabel("UTM x (m)")
    axes[1].set_ylabel("UTM y (m)")
    axes[1].set_title(f"Receivers in UTM (raw, n={len(recv_utm_xy)})")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(alpha=0.3)
    cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.04, pad=0.02)
    cb1.set_label("receiver z (m)")

    fig.suptitle(
        "Receiver layout — rotation QC "
        "(left = post-rotation model frame, right = raw UTM)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = qc_dir / "receiver_layout_rotation.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Companion npz — full float64 arrays for np.allclose vs. legacy.
    out_npz = qc_dir / "receiver_layout.npz"
    np.savez(
        out_npz,
        recv_model_xy=recv_model_xy.astype(np.float64, copy=False),
        recv_utm_xy=recv_utm_xy.astype(np.float64, copy=False),
        recv_z=recv_z.astype(np.float64, copy=False),
        grid_origin_xyz=np.asarray(grid_origin_xyz, dtype=np.float64),
        grid_shape=np.asarray(grid_shape, dtype=np.int64),
        dh_xyz=np.asarray(dh_xyz, dtype=np.float64),
    )
    print(f"[multisource] receiver layout QC -> {out_png}")
    print(f"[multisource] receiver layout npz -> {out_npz}")
    return out_png


def _lsrtm_background_equation(lsrtm_name: str) -> str:
    mapping = {
        "AcousticLSRTM": "Acoustic",
        "AcousticLSRTM3D": "Acoustic3D",
    }
    if lsrtm_name not in mapping:
        raise ValueError(
            f"No background equation known for LSRTM variant '{lsrtm_name}'. "
            f"Known: {sorted(mapping)}"
        )
    return mapping[lsrtm_name]


# ---------- lightweight plotting helpers ---------------------------------


def _crop_padded_volume_to_model(volume, target_shape: tuple, physics) -> "np.ndarray":
    """Crop a sweep c-backend RTM/illumination volume back to the model shape.

    The c-backend pads each model side by ``M = spatial_order // 2`` (for FD
    stencil) and ``abcn`` (for the absorbing layer). For free-surface runs,
    the top z-side is padded only by ``M`` instead of ``M + abcn``.

    Mirrors ``fwi_workflow.imaging.rtm._crop_solver_volume_2d``. 2-D-only;
    3-D RTM cropping is not yet implemented (raise a clear error).
    """
    import torch as _torch

    if isinstance(volume, _torch.Tensor):
        values = volume.detach().cpu()
        while values.ndim > 2:
            # Squeeze leading shot / channel axes (typical c-backend output is
            # (B, 1, nz, nx) for 2-D acoustic; sum over the batch axis to merge
            # contributions from multiple shots in the same batch).
            if int(values.shape[0]) == 1:
                values = values.squeeze(0)
            else:
                values = values.sum(dim=0)
        array = values.numpy().astype(np.float32, copy=False)
    else:
        array = np.asarray(volume, dtype=np.float32)
        while array.ndim > 2:
            if int(array.shape[0]) == 1:
                array = array[0]
            else:
                array = array.sum(axis=0)

    if tuple(array.shape) == tuple(target_shape):
        return np.ascontiguousarray(array)
    if array.ndim != 2:
        raise NotImplementedError(
            f"_crop_padded_volume_to_model: only 2-D crops are supported; "
            f"got array shape {tuple(array.shape)} (3-D RTM cropping not yet "
            "implemented)."
        )
    margin = max(0, int(physics.spatial_order) // 2)
    pad = max(0, int(physics.abcn)) + margin
    nz_pad, nx_pad = int(array.shape[0]), int(array.shape[1])
    nz_target, nx_target = int(target_shape[0]), int(target_shape[1])
    x_slice = slice(pad, nx_pad - pad if pad > 0 else None)
    z_slice = slice(
        margin if physics.free_surface else pad,
        nz_pad - pad if pad > 0 else None,
    )
    cropped = array[z_slice, x_slice]
    # Some backends return a slightly larger volume than the strict crop;
    # if the strict slice is wrong by 1-2 cells, take the leading sub-array
    # that matches the target shape (matches the legacy fallback path).
    if cropped.shape != (nz_target, nx_target):
        if (cropped.shape[0] >= nz_target and cropped.shape[1] >= nx_target):
            cropped = cropped[:nz_target, :nx_target]
        else:
            raise ValueError(
                f"_crop_padded_volume_to_model: cannot crop from {array.shape} "
                f"to {target_shape}; got {cropped.shape}."
            )
    return np.ascontiguousarray(cropped, dtype=np.float32)
