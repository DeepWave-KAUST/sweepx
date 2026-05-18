"""Depth-tapered z-axis low-cut filter for RTM / FWI gradient images.

Port of ``fwi_workflow.imaging.filter_imaging`` (legacy script
``07_filter_imaging.py``). Removes the slowly-varying-with-depth drift
that contaminates the shallow part of a stacked RTM image and tapers
to zero below ``depth_m + taper_m`` so the deep part is untouched.

Two entry points:

* :func:`filter_shallow_low_frequency` — the pure algorithm; takes a
  ``(nz, nx)`` array and returns ``(filtered, removed, weights)``.
* :func:`filter_image_file` — load → filter → save (npy + comparison
  PNG). Used by ``sweep-tasks filter-image`` (standalone CLI) and the
  ``RTMImagingSpec.post_filter`` block inside the runner.

The two paths share the same algorithm and the same on-disk filename
convention (``<input_stem>_shallow_zlowcut.npy``), so any artefact
produced by route B can be re-run via route A with adjusted params
without re-doing the underlying RTM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def cosine_taper(nz: int, dz_m: float, depth_m: float, taper_m: float) -> np.ndarray:
    """Per-z weight vector: 1 near surface, cosine ramp to 0 below ``depth_m``.

    The "shallow low-cut" subtraction is multiplied by this weight, so
    the filter is full-strength in 0..depth_m, smoothly ramps off across
    depth_m..depth_m+taper_m, and is a no-op below.

    ``taper_m=0`` makes it a hard step (cliff at depth_m).
    """
    z_m = np.arange(int(nz), dtype=np.float64) * float(dz_m)
    depth_m = float(depth_m)
    taper_m = max(float(taper_m), 0.0)
    weights = np.ones(int(nz), dtype=np.float64)
    if taper_m <= 0.0:
        weights[z_m > depth_m] = 0.0
        return weights.astype(np.float32)
    stop_m = depth_m + taper_m
    weights[z_m >= stop_m] = 0.0
    ramp = (z_m > depth_m) & (z_m < stop_m)
    phase = (z_m[ramp] - depth_m) / taper_m
    weights[ramp] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return weights.astype(np.float32)


def gaussian_lowpass_z(values: np.ndarray, dz_m: float, wavelength_m: float) -> np.ndarray:
    """Extract the z-low-frequency content via a Gaussian Fourier LP.

    Wavelengths > ``wavelength_m`` survive. Cutoff is the corresponding
    spatial frequency; Gaussian σ chosen so the response is 1/√2 (≈ -3 dB)
    at the cutoff. Operates per-x-column; no horizontal smoothing.
    """
    array = np.asarray(values, dtype=np.float32)
    nz = int(array.shape[0])
    wavelength_m = max(float(wavelength_m), float(dz_m) * 2.0)
    freqs = np.fft.rfftfreq(nz, d=float(dz_m))
    cutoff = 1.0 / wavelength_m
    sigma = max(cutoff / math.sqrt(2.0 * math.log(2.0)), 1.0e-12)
    response = np.exp(-0.5 * (freqs / sigma) ** 2).astype(np.float32)
    spectrum = np.fft.rfft(array, axis=0)
    low = np.fft.irfft(spectrum * response[:, None], n=nz, axis=0)
    return np.asarray(low, dtype=np.float32)


def filter_shallow_low_frequency(
    values: np.ndarray,
    *,
    dz_m: float,
    wavelength_m: float = 300.0,
    depth_m: float = 600.0,
    taper_m: float = 400.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove the shallow slowly-varying-with-depth drift from a 2-D image.

    Parameters
    ----------
    values
        ``(nz, nx)`` float array (RTM image, FWI gradient image, …).
    dz_m
        Vertical grid spacing in meters.
    wavelength_m
        z-direction features with wavelength > this are considered the
        slow drift the filter targets. Default 300 m.
    depth_m
        The filter is full-strength in ``0 .. depth_m``. Default 600 m.
    taper_m
        Cosine ramp from full-strength at ``depth_m`` to zero at
        ``depth_m + taper_m``. Default 400 m.

    Returns
    -------
    (filtered, removed, weights)
        ``filtered`` is the cleaned image (shape preserved). ``removed``
        is the subtracted drift (``original - filtered``). ``weights``
        is the per-z taper vector (shape ``(nz,)``).
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"filter_shallow_low_frequency: expected 2-D image, got shape "
            f"{array.shape}"
        )
    low = gaussian_lowpass_z(array, dz_m=dz_m, wavelength_m=wavelength_m)
    weights = cosine_taper(
        array.shape[0], dz_m=dz_m, depth_m=depth_m, taper_m=taper_m
    )
    filtered = array - weights[:, None] * low
    removed = array - filtered
    return (filtered.astype(np.float32),
            removed.astype(np.float32),
            weights.astype(np.float32))


def _save_png(
    path: Path,
    values: np.ndarray,
    *,
    title: str,
    clip_percentile: float,
    cmap: str,
    dx_m: float | None,
    dz_m: float | None,
    x_origin_m: float,
    z_origin_m: float,
    x_max_m: float | None,
    display_scale: float,
) -> None:
    """Save a single 2-D image as PNG with signed-percentile clipping."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    try:  # registers ``sweep_image`` / ``sweep_vp`` as matplotlib cmaps
        import sweep_viz.colormaps  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    array = np.asarray(values, dtype=np.float32)
    pct = float(clip_percentile)
    vmin, vmax = np.nanpercentile(array, [pct, 100.0 - pct])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmin) == float(vmax):
        vmin, vmax = None, None
    elif float(display_scale) > 0.0:
        c = 0.5 * (float(vmin) + float(vmax))
        hw = 0.5 * (float(vmax) - float(vmin)) * float(display_scale)
        vmin, vmax = c - hw, c + hw
    nz, nx = (int(v) for v in array.shape)
    extent = None
    if dx_m is not None and dz_m is not None:
        extent = [
            float(x_origin_m), float(x_origin_m) + (nx - 1) * float(dx_m),
            float(z_origin_m) + (nz - 1) * float(dz_m), float(z_origin_m),
        ]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.0))
    try:
        im = ax.imshow(array, cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("x (m)" if extent else "x index")
        ax.set_ylabel("z (m)" if extent else "z index")
        if x_max_m is not None:
            ax.set_xlim(float(x_origin_m), float(x_max_m))
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def _save_comparison_png(
    path: Path,
    *,
    original: np.ndarray,
    filtered: np.ndarray,
    removed: np.ndarray,
    clip_percentile: float,
    cmap: str,
    dx_m: float | None,
    dz_m: float | None,
    x_origin_m: float,
    z_origin_m: float,
    x_max_m: float | None,
    display_scale: float,
) -> None:
    """3-row before / after / removed comparison PNG."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    try:  # registers ``sweep_image`` / ``sweep_vp`` as matplotlib cmaps
        import sweep_viz.colormaps  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    panels = [
        ("original", np.asarray(original, dtype=np.float32)),
        ("filtered", np.asarray(filtered, dtype=np.float32)),
        ("removed shallow low-z", np.asarray(removed, dtype=np.float32)),
    ]
    nz, nx = (int(v) for v in panels[0][1].shape)
    extent = None
    if dx_m is not None and dz_m is not None:
        extent = [
            float(x_origin_m), float(x_origin_m) + (nx - 1) * float(dx_m),
            float(z_origin_m) + (nz - 1) * float(dz_m), float(z_origin_m),
        ]
    pct = float(clip_percentile)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.0), sharex=True, sharey=True)
    try:
        for ax, (title, array) in zip(axes, panels):
            vmin, vmax = np.nanpercentile(array, [pct, 100.0 - pct])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmin) == float(vmax):
                vmin, vmax = None, None
            elif float(display_scale) > 0.0:
                c = 0.5 * (float(vmin) + float(vmax))
                hw = 0.5 * (float(vmax) - float(vmin)) * float(display_scale)
                vmin, vmax = c - hw, c + hw
            ax.imshow(array, cmap=cmap, aspect="auto",
                      vmin=vmin, vmax=vmax, extent=extent)
            ax.set_title(title)
            ax.set_ylabel("z (m)" if extent else "z index")
            if x_max_m is not None:
                ax.set_xlim(float(x_origin_m), float(x_max_m))
        axes[-1].set_xlabel("x (m)" if extent else "x index")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def filter_image_file(
    input_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_name: str | None = None,
    dz_m: float,
    dx_m: float | None = None,
    wavelength_m: float = 300.0,
    depth_m: float = 600.0,
    taper_m: float = 400.0,
    clip_percentile: float = 1.0,
    display_scale: float = 1.0,
    x_origin_m: float = 0.0,
    z_origin_m: float = 0.0,
    x_max_m: float | None = None,
    cmap: str = "sweep_image",
    save_png: bool = True,
) -> dict:
    """Load a 2-D image npy, run :func:`filter_shallow_low_frequency`, save.

    Mirrors ``fwi_workflow-dev/scripts/07_filter_imaging.py``. Returns a
    metadata dict (also written to ``<stem>_metadata.json`` next to the
    output npy) so the caller can introspect what was applied.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir is not None else input_path.parent
    stem = output_name or f"{input_path.stem}_shallow_zlowcut"
    output_dir.mkdir(parents=True, exist_ok=True)

    original = np.asarray(np.load(input_path), dtype=np.float32)
    filtered, removed, weights = filter_shallow_low_frequency(
        original,
        dz_m=float(dz_m),
        wavelength_m=float(wavelength_m),
        depth_m=float(depth_m),
        taper_m=float(taper_m),
    )
    out_npy = output_dir / f"{stem}.npy"
    np.save(out_npy, filtered)
    np.save(output_dir / f"{stem}_removed.npy", removed)
    np.save(output_dir / f"{stem}_z_taper.npy", weights)

    if save_png:
        _save_png(
            output_dir / f"{stem}.png", filtered,
            title=f"{stem} (signed {clip_percentile}/{100 - clip_percentile} percentile clip)",
            clip_percentile=float(clip_percentile),
            cmap=cmap,
            dx_m=dx_m, dz_m=dz_m,
            x_origin_m=x_origin_m, z_origin_m=z_origin_m,
            x_max_m=x_max_m, display_scale=display_scale,
        )
        _save_comparison_png(
            output_dir / f"{stem}_comparison.png",
            original=original, filtered=filtered, removed=removed,
            clip_percentile=float(clip_percentile),
            cmap=cmap,
            dx_m=dx_m, dz_m=dz_m,
            x_origin_m=x_origin_m, z_origin_m=z_origin_m,
            x_max_m=x_max_m, display_scale=display_scale,
        )

    metadata = {
        "input": str(input_path),
        "output": str(out_npy),
        "removed": str(output_dir / f"{stem}_removed.npy"),
        "z_taper": str(output_dir / f"{stem}_z_taper.npy"),
        "dz_m": float(dz_m),
        "dx_m": (None if dx_m is None else float(dx_m)),
        "x_origin_m": float(x_origin_m),
        "z_origin_m": float(z_origin_m),
        "display_x_max_m": (None if x_max_m is None else float(x_max_m)),
        "lowpass_wavelength_m": float(wavelength_m),
        "filter_full_depth_m": float(depth_m),
        "taper_m": float(taper_m),
        "clip_percentile": float(clip_percentile),
        "display_scale": float(display_scale),
        "display_percentile_mode": "signed",
        "input_shape": list(original.shape),
    }
    with (output_dir / f"{stem}_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    return metadata


__all__ = [
    "cosine_taper",
    "gaussian_lowpass_z",
    "filter_shallow_low_frequency",
    "filter_image_file",
]
