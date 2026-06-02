"""Convert a Viking-style ASCII FarField wavelet into a sweep-compatible npz.

Ported from ``fwi_workflow-dev/scripts/03_convert_farfield_wavelet.py``.
The Viking Graben distribution (and several other public marine
benchmarks) ships a far-field source signature as a plain text file
(one sample per line). Other sweep-tasks workflow stages expect a
wavelet npz with ``wavelet`` and ``time_s`` arrays.

This module reads the text file, applies optional cosine tapering /
initial-sample zeroing, and writes the matching npz alongside a JSON
metadata sidecar and an optional QC figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sweep_torch import apply_wavelet_postprocessing


def _load_text_wavelet(path: Path) -> np.ndarray:
    """Read a one-column ASCII wavelet into a 1-D float32 array."""

    samples = np.loadtxt(path).astype(np.float32)
    if samples.ndim != 1:
        raise ValueError(
            f"{path} must contain a single column of samples; got shape {samples.shape}"
        )
    return samples


def convert_farfield(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dt_s: float = 0.004,
    zero_initial_samples: int = 0,
    taper_start_s: float | None = None,
    taper_end_s: float | None = None,
    save_plot: bool = True,
) -> dict:
    """Convert ASCII farfield file -> sweep-compatible npz.

    Parameters
    ----------
    input_path
        ASCII FarField file (one sample per line).
    output_path
        Destination npz path. Parent directories are created if missing.
    dt_s
        Sample interval in seconds. Viking FarField.dat default is 4 ms.
    zero_initial_samples
        Force the first N samples to zero (sometimes useful when the
        manufacturer adds a small lead-in delay).
    taper_start_s, taper_end_s
        Optional cosine taper window — start full-amplitude at
        ``taper_start_s`` and ramp to zero by ``taper_end_s``.
    save_plot
        Write a QC PNG (time + spectrum) next to the npz.

    Returns
    -------
    Metadata dict (also written to ``<output_path>.json``).
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"FarField file not found: {input_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_wavelet = _load_text_wavelet(input_path)
    final_wavelet = apply_wavelet_postprocessing(
        raw_wavelet,
        dt_s=dt_s,
        taper_start_s=taper_start_s,
        taper_end_s=taper_end_s,
        zero_initial_samples=zero_initial_samples,
    )
    time_s = np.arange(final_wavelet.size, dtype=np.float32) * float(dt_s)

    payload = {
        "wavelet": final_wavelet.astype(np.float32),
        "time_s": time_s.astype(np.float32),
        "raw_estimated_wavelet": raw_wavelet.astype(np.float32),
    }
    np.savez_compressed(output_path, **payload)

    metadata = {
        "input_text_file": str(input_path),
        "output_npz": str(output_path),
        "dt_s": float(dt_s),
        "n_samples": int(final_wavelet.size),
        "duration_s": float(final_wavelet.size * dt_s),
        "zero_initial_samples": int(zero_initial_samples),
        "taper_start_s": None if taper_start_s is None else float(taper_start_s),
        "taper_end_s": None if taper_end_s is None else float(taper_end_s),
        "raw_min": float(np.min(raw_wavelet)),
        "raw_max": float(np.max(raw_wavelet)),
        "final_min": float(np.min(final_wavelet)),
        "final_max": float(np.max(final_wavelet)),
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    if save_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            axes[0].plot(time_s, raw_wavelet, color="0.55", label="raw far-field")
            axes[0].plot(time_s, final_wavelet, color="black", label="post-processed")
            axes[0].set_xlabel("time (s)")
            axes[0].set_ylabel("amplitude")
            axes[0].set_title(
                f"Far-field wavelet — {input_path.name} (dt={dt_s*1000:.2f} ms)"
            )
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)

            spec = np.abs(np.fft.rfft(final_wavelet))
            spec_raw = np.abs(np.fft.rfft(raw_wavelet))
            freqs = np.fft.rfftfreq(final_wavelet.size, dt_s)
            axes[1].plot(freqs, spec_raw / np.max(spec_raw + 1e-30), color="0.55", label="raw")
            axes[1].plot(freqs, spec / np.max(spec + 1e-30), color="black", label="post-processed")
            axes[1].set_xlim(0.0, min(80.0, float(freqs.max())))
            axes[1].set_xlabel("frequency (Hz)")
            axes[1].set_ylabel("normalized amplitude")
            axes[1].set_title("amplitude spectrum")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right")

            fig.tight_layout()
            fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 — best-effort plotting
            print(f"warning: failed to save wavelet QC figure: {exc}")

    return metadata
