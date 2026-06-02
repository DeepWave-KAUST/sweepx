"""Plot the four-step wavelet inversion pipeline as one comparison figure.

Ported from ``fwi_workflow-dev/scripts/03_plot_wavelet_pipeline_steps.py``.
The Step-03 wavelet pipeline produces four wavelets that share the same
time grid:

  step 1 — rank-1 robust average resampled onto the SIREN grid
           (``initial_wavelet_target.npz`` :: ``wavelet``)
  step 2 — SIREN output after the LBFGS MSE prefit
           (``prefit_siren_wavelet.npz`` :: ``wavelet``)
  step 3 — SIREN output at the end of the wave-equation main loop,
           before tapering / zeroing
           (``estimated_wavelet.npz`` :: ``raw_estimated_wavelet``)
  step 4 — post-processed final wavelet used by FWI / imaging
           (``estimated_wavelet.npz`` :: ``wavelet``; same array
           referenced by the FWI / RTM YAML's ``wavelet.path``)

This module reads those three npz files from one ``siren_pipeline``
output directory and writes two PNGs: a 4-row stack + a 1-row overlay.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing pipeline artifact: {path}")
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(
                f"{path} does not contain key {key!r}; available: {list(data.files)}"
            )
        return np.asarray(data[key], dtype=np.float64).reshape(-1)


def _load_time_axis(path: Path, fallback_dt_s: float | None) -> tuple[np.ndarray, float]:
    with np.load(path) as data:
        if "time_s" in data.files:
            time_s = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
            if time_s.size > 1:
                return time_s, float(np.median(np.diff(time_s)))
    if fallback_dt_s is None:
        raise ValueError(f"{path} has no time_s; pass dt_s explicitly")
    n = _load_array(path, "wavelet").size
    time_s = np.arange(n, dtype=np.float64) * float(fallback_dt_s)
    return time_s, float(fallback_dt_s)


def plot_wavelet_pipeline_steps(
    pipeline_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    max_freq_hz: float = 80.0,
    dt_s: float | None = None,
) -> tuple[Path, Path]:
    """Render the 4-step wavelet pipeline comparison PNGs.

    Returns ``(stack_png_path, overlay_png_path)``.
    """

    pipeline_dir = Path(pipeline_dir).resolve()
    if not pipeline_dir.exists():
        raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")

    target_npz = pipeline_dir / "initial_wavelet_target.npz"
    prefit_npz = pipeline_dir / "prefit_siren_wavelet.npz"
    final_npz = pipeline_dir / "estimated_wavelet.npz"

    time_s, eff_dt_s = _load_time_axis(target_npz, dt_s)
    target_w = _load_array(target_npz, "wavelet")
    prefit_w = _load_array(prefit_npz, "wavelet")
    raw_w = _load_array(final_npz, "raw_estimated_wavelet")
    final_w = _load_array(final_npz, "wavelet")

    nt = min(time_s.size, target_w.size, prefit_w.size, raw_w.size, final_w.size)
    time_s = time_s[:nt]
    target_w = target_w[:nt]
    prefit_w = prefit_w[:nt]
    raw_w = raw_w[:nt]
    final_w = final_w[:nt]

    output_path = Path(output_path) if output_path else (pipeline_dir / "wavelet_pipeline_steps.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"matplotlib is required: {exc}") from exc

    steps = [
        ("step 1: rank-1 robust average (analysis)", target_w, "tab:blue"),
        ("step 2: SIREN after LBFGS prefit",         prefit_w, "tab:orange"),
        ("step 3: SIREN after wave-equation main loop", raw_w, "tab:green"),
        ("step 4: post-processed wavelet (used by FWI)", final_w, "tab:red"),
    ]

    freqs = np.fft.rfftfreq(nt, float(eff_dt_s))

    def _norm_spectrum(values: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(values))
        scale = float(spec.max())
        return spec / scale if scale > 0.0 else spec

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    overall_absmax = max(float(np.max(np.abs(w))) for _, w, _ in steps)
    overall_absmax = max(overall_absmax, 1.0e-12)
    fig.suptitle(f"Source wavelet inversion pipeline — {pipeline_dir.name}", fontsize=13)

    for irow, (label, wavelet, color) in enumerate(steps):
        ax_t = axes[irow, 0]
        ax_t.plot(time_s, wavelet, color=color, linewidth=1.5)
        ax_t.axvline(0.0, color="0.2", linewidth=0.6, alpha=0.6)
        ax_t.set_xlabel("time (s)")
        ax_t.set_ylabel("amplitude")
        ax_t.set_title(label)
        ax_t.grid(alpha=0.25)
        ax_t.set_ylim(-1.05 * overall_absmax, 1.05 * overall_absmax)

        ax_f = axes[irow, 1]
        ax_f.plot(freqs, _norm_spectrum(wavelet), color=color, linewidth=1.5)
        ax_f.set_xlim(0.0, min(float(max_freq_hz), float(freqs.max())))
        ax_f.set_ylim(0.0, 1.05)
        ax_f.set_xlabel("frequency (Hz)")
        ax_f.set_ylabel("normalised amplitude")
        ax_f.set_title(label)
        ax_f.grid(alpha=0.25)

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    overlay_path = output_path.with_name(output_path.stem + "_overlay.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for label, wavelet, color in steps:
        axes[0].plot(time_s, wavelet, color=color, linewidth=1.4, label=label.split(":", 1)[0])
        axes[1].plot(freqs, _norm_spectrum(wavelet), color=color, linewidth=1.4, label=label.split(":", 1)[0])
    axes[0].axvline(0.0, color="0.2", linewidth=0.6, alpha=0.6)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Steps 1-4 overlay (time)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="best", fontsize=9)
    axes[1].set_xlim(0.0, min(float(max_freq_hz), float(freqs.max())))
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("normalised amplitude")
    axes[1].set_title("Steps 1-4 overlay (spectrum)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="best", fontsize=9)
    fig.savefig(overlay_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return output_path, overlay_path
