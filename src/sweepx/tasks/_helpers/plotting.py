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


def _plot_loss_curve(losses, path: Path, title: str) -> Path:
    """Loss curve via `sweep_viz.convergence.plot_loss`."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sweep_viz.convergence import plot_loss

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss(list(losses), ax=ax, logy=True, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_wavefield_snapshots(snapshots_np, snapshot_times, abcn, shape, path,
                              free_surface) -> Path:
    """Wavefield snapshot grid via `sweep_viz.wavefield.plot_snapshot` (one per panel).

    The PML / absorbing-boundary cropping logic (and the `(nsnap, 1, 1, 1, ...)`
    sweep-binding tensor layout) is sweep-tasks-specific, so it stays here;
    each cropped panel is then rendered by sweep_viz.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from sweep_viz.wavefield import plot_snapshot

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
    # RTM panels below resolve. Tolerate sweep_viz being missing — caller
    # gets a clear matplotlib error in that case.
    try:
        from sweep_viz.colormaps import IMAGE_CMAP as _RTM_CMAP
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
