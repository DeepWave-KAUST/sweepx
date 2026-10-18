"""Rank-1 robust direct-wave wavelet analysis (per-group + average).

Ported from ``fwi_workflow-dev/scripts/03_analyze_direct_wavelet_batch.py``.
Reads a :class:`sweep_io.seismic_plan.SeismicPlan` (CSG or CRG) instead
of the legacy ``csg_index_v2`` npz; the inner rank-1 + robust-average
algorithm lives in :mod:`sweep_tasks.wavelet.direct_arrival` and is
reused verbatim. Under CSG one group is a shot with N receivers; under
CRG (OBN reciprocity) one group is a receiver with N shots. The rank-1
SVD only sees the physical offset ``|rec_xyz - src_xyz|`` either way,
so the algorithm is fully dual.

For each selected group the algorithm:

1. Picks the N nearest-offset rows.
2. Reads those traces from SEG-Y via :class:`PlanReader`.
3. Optionally Butterworth-filters them.
4. Computes predicted direct-arrival times ``tau = |offset| / v_water``.
5. Runs rank-1 alternation in a window around each predicted arrival.

Then across all per-shot wavelets:

6. Masks shots whose direct-window residual-to-observed RMS ratio is
   above ``min_residual_ratio``.
7. Polarity-aligns the survivors.
8. Masks again on shape-correlation to the preliminary mean.
9. Averages the survivors and writes the causal wavelet npz.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sweep_io.seismic_plan import PlanReader, SeismicPlan

from .direct_arrival import (
    causal_event_time_s,
    estimate_direct_wavelet_rank1,
    robust_average_wavelets,
    synthesize_direct_record,
    to_causal_wavelet,
)


@dataclass(frozen=True)
class AnalyzeWaveletConfig:
    """Inputs for :func:`analyze_direct_wavelet_batch`."""

    plan_path: Path
    output_dir: Path
    shot_start: int = 1
    shot_stop: int = 200
    shot_stride: int = 1
    nearest_receivers: int = 10
    max_abs_offset_m: float | None = None
    water_velocity_m_s: float = 1500.0
    rel_t_min_s: float = -0.05
    rel_t_max_s: float = 0.40
    iterations: int = 20
    filter_lowcut_hz: float | None = None
    filter_highcut_hz: float | None = None
    filter_order: int = 4
    prepad_s: float = 0.0
    min_residual_ratio: float = 0.6
    min_shape_corr_to_average: float = 0.9
    no_plot: bool = False
    save_per_shot_npz: bool = False
    command: str | None = None


def _butterworth_filter(traces, dt_s, lowcut_hz, highcut_hz, order):
    """Apply zero-phase Butterworth filter along the last axis (scipy)."""

    if lowcut_hz is None and highcut_hz is None:
        return np.asarray(traces, dtype=np.float32)
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:  # pragma: no cover - scipy is in env
        raise SystemExit("scipy is required for --filter-* options") from exc
    nyq = 0.5 / float(dt_s)
    low = max(0.0, float(lowcut_hz)) if lowcut_hz is not None else 0.0
    high = min(float(highcut_hz), nyq * 0.999) if highcut_hz is not None else nyq * 0.999
    if low <= 0.0 and (highcut_hz is None or high >= nyq * 0.999):
        return np.asarray(traces, dtype=np.float32)
    if low <= 0.0:
        b, a = butter(int(max(1, order)), high, btype="lowpass", fs=1.0 / float(dt_s))
    elif highcut_hz is None:
        b, a = butter(int(max(1, order)), low, btype="highpass", fs=1.0 / float(dt_s))
    else:
        if low >= high:
            raise ValueError("filter_lowcut_hz must be less than filter_highcut_hz")
        b, a = butter(int(max(1, order)), [low, high], btype="bandpass", fs=1.0 / float(dt_s))
    filtered = filtfilt(b, a, np.asarray(traces, dtype=np.float64), axis=-1)
    return np.asarray(filtered, dtype=np.float32)


def _select_one_shot_near_offset_rows(
    plan: SeismicPlan,
    shot_ordinal: int,
    nearest_receivers: int,
    max_abs_offset_m: float | None,
) -> dict | None:
    """Pick the N nearest-offset rows of one group (CSG or CRG).

    The offset ``|rec_xyz - src_xyz|`` is symmetric under reciprocity, so
    this selector is grouping-agnostic. Returns ``None`` when the row
    count is insufficient after filtering; the caller should skip the
    group in that case.
    """

    i = int(shot_ordinal) - 1
    if i < 0 or i >= plan.n_groups:
        return None
    sl = plan.group_slice(i)
    rows = np.arange(int(sl.start), int(sl.stop), dtype=np.int64)
    if rows.size == 0:
        return None

    src_xyz = np.asarray(plan.row_source_xyz, dtype=np.float64)
    rec_xyz = np.asarray(plan.row_receiver_xyz, dtype=np.float64)
    abs_offsets = np.hypot(
        rec_xyz[rows, 0] - src_xyz[rows, 0],
        rec_xyz[rows, 1] - src_xyz[rows, 1],
    )

    if max_abs_offset_m is not None:
        keep = abs_offsets <= float(max_abs_offset_m)
        rows = rows[keep]
        abs_offsets = abs_offsets[keep]
    if rows.size == 0:
        return None
    order = np.argsort(abs_offsets, kind="mergesort")
    rows = rows[order]
    abs_offsets = abs_offsets[order]
    if nearest_receivers > 0:
        if rows.size < nearest_receivers:
            return None
        rows = rows[: int(nearest_receivers)]
        abs_offsets = abs_offsets[: int(nearest_receivers)]

    return {
        "shot_id": int(plan.group_id[i]),
        "rows": rows,
        "source_x": src_xyz[rows, 0],
        "source_z": src_xyz[rows, 2],
        "receiver_x": rec_xyz[rows, 0],
        "receiver_z": rec_xyz[rows, 2],
        "abs_offset_m": abs_offsets,
    }


def _save_overlay_plot(output_path, rel_t, aligned, final_mask, average, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = aligned[final_mask]
    n_used = used.shape[0]
    n_total = aligned.shape[0]
    if used.size == 0:
        return
    sample_count = min(150, n_used)
    sample_idx = np.linspace(0, n_used - 1, sample_count, dtype=int)
    mean = np.mean(used, axis=0)
    std = np.std(used, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.plot(rel_t, used[sample_idx].T, color="0.72", lw=0.5, alpha=0.25)
    ax.fill_between(
        rel_t, mean - std, mean + std, color="#4C78A8", alpha=0.18,
        label=f"used wavelets ±1 std (n={n_used})",
    )
    ax.plot(rel_t, average, color="#1F4E79", lw=2.0, label="robust average")
    ax.axvline(0.0, color="0.15", lw=0.8, label="predicted direct arrival")
    ax.set_xlabel("Time relative to direct arrival (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"{title} (used {n_used} / {n_total})")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_spectrum_plot(output_path, aligned, final_mask, average, dt_s, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = aligned[final_mask]
    if used.size == 0:
        return
    spec = np.abs(np.fft.rfft(used, axis=-1))
    spec = spec / np.maximum(np.max(spec, axis=-1, keepdims=True), 1.0e-12)
    freq = np.fft.rfftfreq(used.shape[1], float(dt_s))
    mean_s = np.mean(spec, axis=0)
    std_s = np.std(spec, axis=0)
    avg_spec = np.abs(np.fft.rfft(average))
    avg_spec = avg_spec / max(float(np.max(avg_spec)), 1.0e-12)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.fill_between(freq, mean_s - std_s, mean_s + std_s, color="#4C78A8", alpha=0.18, label="used wavelets ±1 std")
    ax.plot(freq, mean_s, color="#4C78A8", lw=1.4, label="mean spectrum")
    ax.plot(freq, avg_spec, color="#1F4E79", lw=2.0, label="robust average spectrum")
    ax.set_xlim(0.0, min(80.0, float(freq.max())))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_residual_overlay_plot(output_path, rel_t, aligned, polarity, average, final_mask):
    """Optional residual-after-average overlay for sanity checking."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = aligned[final_mask]
    if used.size == 0:
        return
    residual = used - average[None, :]
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    sample_count = min(150, residual.shape[0])
    sample_idx = np.linspace(0, residual.shape[0] - 1, sample_count, dtype=int)
    ax.plot(rel_t, residual[sample_idx].T, color="0.55", lw=0.5, alpha=0.4)
    rms = np.sqrt(np.mean(residual ** 2, axis=0))
    ax.plot(rel_t, rms, color="#C0392B", lw=1.6, label="per-sample residual RMS")
    ax.set_xlabel("Time relative to direct arrival (s)")
    ax.set_ylabel("Amplitude residual")
    ax.set_title("Per-shot wavelets minus robust average")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_direct_wavelet_batch(config: AnalyzeWaveletConfig) -> Path:
    """Run the full rank-1 + robust-average pipeline.

    Returns the path to ``average/average_direct_wavelet.npz`` (which is
    what :class:`sweep_tasks.wavelet.sweep_torch.WaveletInversionConfig`
    expects as ``initial_wavelet_path``).
    """

    plan = SeismicPlan.load(config.plan_path)
    if plan.grouping not in ("csg", "crg"):
        raise ValueError(
            f"analyze-wavelet expects a CSG or CRG plan; got grouping={plan.grouping!r}"
        )
    dt_s = float(plan.dt_s)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    average_dir = output_dir / "average"
    average_dir.mkdir(parents=True, exist_ok=True)
    per_shot_dir = output_dir / "per_shot" if config.save_per_shot_npz else None
    if per_shot_dir is not None:
        per_shot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze-wavelet] plan={config.plan_path}", flush=True)
    print(f"[analyze-wavelet] output_dir={output_dir}", flush=True)

    shot_ordinals = list(range(
        int(config.shot_start),
        int(config.shot_stop) + 1,
        int(config.shot_stride),
    ))
    if not shot_ordinals:
        raise SystemExit("No shots selected; check shot range / stride")

    rel_t_template = None
    wavelet_stack: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    # Cache the entire plan in RAM for Viking-scale CSG plans (~700 MB
    # at most). For very large 3-D plans the caller should adjust.
    with PlanReader(plan, cache_all=False) as reader:
        for ordinal in shot_ordinals:
            sel = _select_one_shot_near_offset_rows(
                plan, ordinal,
                int(config.nearest_receivers),
                config.max_abs_offset_m,
            )
            if sel is None:
                print(f"shot_ordinal={ordinal}: skip (offset/receiver constraints)", flush=True)
                continue

            traces = reader.read_rows(sel["rows"]).astype(np.float32)
            if config.filter_lowcut_hz is not None or config.filter_highcut_hz is not None:
                traces = _butterworth_filter(
                    traces, dt_s,
                    config.filter_lowcut_hz, config.filter_highcut_hz,
                    int(config.filter_order),
                )

            dx = sel["receiver_x"] - sel["source_x"]
            dz = sel["receiver_z"] - sel["source_z"]
            offset_2d = np.sqrt(dx * dx + dz * dz).astype(np.float64)
            order = np.argsort(offset_2d, kind="stable")
            traces = traces[order]
            offset_2d = offset_2d[order]
            travel_times = offset_2d / float(config.water_velocity_m_s)

            result = estimate_direct_wavelet_rank1(
                traces, dt_s, travel_times,
                rel_t_min_s=float(config.rel_t_min_s),
                rel_t_max_s=float(config.rel_t_max_s),
                iterations=int(config.iterations),
            )
            if rel_t_template is None:
                rel_t_template = result.rel_t_s.astype(np.float32)
            elif result.rel_t_s.shape != rel_t_template.shape:
                print(f"shot_ordinal={ordinal}: skip (rel_t length mismatch)", flush=True)
                continue

            synth = synthesize_direct_record(
                traces.shape, dt_s, travel_times,
                result.rel_t_s, result.wavelet, result.amplitudes,
            )
            sample_t = np.arange(traces.shape[1], dtype=np.float64) * dt_s
            direct_mask = np.zeros_like(traces, dtype=bool)
            for itrace, tau in enumerate(travel_times):
                direct_mask[itrace] = (
                    (sample_t >= float(tau) + float(config.rel_t_min_s))
                    & (sample_t <= float(tau) + float(config.rel_t_max_s))
                )
            if direct_mask.any():
                obs_rms = float(np.sqrt(np.mean(traces[direct_mask] ** 2)))
                res_rms = float(np.sqrt(np.mean((traces - synth)[direct_mask] ** 2)))
            else:
                obs_rms = 0.0
                res_rms = 0.0
            residual_ratio = res_rms / max(obs_rms, 1.0e-12)

            wavelet_stack.append(result.wavelet.astype(np.float32))
            rows.append({
                "shot_ordinal": int(ordinal),
                "shot_id": int(sel["shot_id"]),
                "source_x_2d_m": float(sel["source_x"][0]),
                "source_z_m": float(sel["source_z"][0]),
                "receiver_count": int(traces.shape[0]),
                "abs_offset_min_m": float(np.min(offset_2d)),
                "abs_offset_max_m": float(np.max(offset_2d)),
                "direct_window_obs_rms": obs_rms,
                "direct_window_residual_rms": res_rms,
                "residual_ratio": residual_ratio,
                "wavelet_absmax": float(np.max(np.abs(result.wavelet))),
                "iterations": int(result.iterations),
            })

            if per_shot_dir is not None:
                shot_dir = per_shot_dir / f"shot_{int(ordinal):05d}_id_{int(sel['shot_id']):05d}"
                shot_dir.mkdir(parents=True, exist_ok=True)
                time_s, causal = to_causal_wavelet(
                    result.rel_t_s, result.wavelet, dt_s, prepad_s=float(config.prepad_s),
                )
                np.savez_compressed(
                    shot_dir / "direct_wavelet.npz",
                    wavelet=causal.astype(np.float32),
                    time_s=time_s.astype(np.float32),
                    wavelet_relative=result.wavelet.astype(np.float32),
                    time_relative_s=result.rel_t_s.astype(np.float32),
                    amplitudes=result.amplitudes.astype(np.float32),
                    travel_times_s=travel_times.astype(np.float32),
                    offsets_2d_m=offset_2d.astype(np.float32),
                    shot_id=np.asarray(int(sel["shot_id"]), dtype=np.int64),
                )
            if (len(wavelet_stack)) % 25 == 0:
                print(f"  ...estimated {len(wavelet_stack)} per-shot wavelets so far", flush=True)

    if not wavelet_stack:
        raise SystemExit("No per-shot wavelets were produced; check selection/filter parameters")
    rel_t = rel_t_template
    wavelets = np.stack(wavelet_stack)
    print(f"per-shot wavelet stack: {wavelets.shape}", flush=True)

    residual_ratios = np.asarray([row["residual_ratio"] for row in rows], dtype=np.float64)
    initial_mask = (residual_ratios <= float(config.min_residual_ratio)) & np.all(
        np.isfinite(wavelets), axis=1,
    )
    if not np.any(initial_mask):
        print(
            "warning: residual-ratio mask removed every shot; falling back to finiteness mask only",
            flush=True,
        )
        initial_mask = np.all(np.isfinite(wavelets), axis=1)

    average_pkg = robust_average_wavelets(
        wavelets,
        initial_mask=initial_mask,
        min_shape_corr_to_average=float(config.min_shape_corr_to_average),
    )
    aligned = average_pkg["aligned"]
    polarity = average_pkg["polarity"]
    average = average_pkg["average"]
    mean = average_pkg["mean"]
    std = average_pkg["std"]
    final_mask = average_pkg["final_mask"]
    shape_corr = average_pkg["shape_corr_to_average"]

    time_s, causal_average = to_causal_wavelet(rel_t, average, dt_s, prepad_s=float(config.prepad_s))
    event_time_s = causal_event_time_s(rel_t, prepad_s=float(config.prepad_s))
    average_npz_path = average_dir / "average_direct_wavelet.npz"
    np.savez_compressed(
        average_npz_path,
        wavelet=causal_average.astype(np.float32),
        time_s=time_s.astype(np.float32),
        wavelet_relative=average.astype(np.float32),
        time_relative_s=rel_t.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        polarity=polarity.astype(np.float32),
        shape_corr_to_average=shape_corr.astype(np.float32),
        final_mask=final_mask.astype(np.bool_),
        aligned_wavelets=aligned.astype(np.float32),
        rel_t_min_s=np.asarray(float(config.rel_t_min_s), dtype=np.float32),
        rel_t_max_s=np.asarray(float(config.rel_t_max_s), dtype=np.float32),
        prepad_s=np.asarray(float(config.prepad_s), dtype=np.float32),
        event_time_s=np.asarray(event_time_s, dtype=np.float32),
        water_velocity_m_s=np.asarray(float(config.water_velocity_m_s), dtype=np.float32),
    )

    fields = [
        "shot_ordinal", "shot_id", "source_x_2d_m", "source_z_m",
        "receiver_count", "abs_offset_min_m", "abs_offset_max_m",
        "direct_window_obs_rms", "direct_window_residual_rms",
        "residual_ratio", "wavelet_absmax", "iterations",
    ]
    stats_path = analysis_dir / "wavelet_stats.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields + ["initial_mask", "shape_corr_to_average", "polarity", "used_in_average"],
        )
        writer.writeheader()
        for i, row in enumerate(rows):
            payload = {key: row[key] for key in fields}
            payload["initial_mask"] = bool(initial_mask[i])
            payload["shape_corr_to_average"] = float(shape_corr[i])
            payload["polarity"] = float(polarity[i])
            payload["used_in_average"] = bool(final_mask[i])
            writer.writerow(payload)

    title_prefix = "per-shot direct-wave wavelet"
    if not config.no_plot:
        try:
            _save_overlay_plot(
                analysis_dir / "wavelet_overlay_mean_std.png",
                rel_t, aligned, final_mask, average, title_prefix,
            )
            _save_spectrum_plot(
                analysis_dir / "wavelet_spectra_mean_std.png",
                aligned, final_mask, average, dt_s,
                f"{title_prefix} amplitude spectrum",
            )
            _save_residual_overlay_plot(
                analysis_dir / "wavelet_residual_after_average.png",
                rel_t, aligned, polarity, average, final_mask,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"warning: failed to save analysis plots: {exc}")

    summary = {
        "plan_path": str(Path(config.plan_path).resolve()),
        "shot_start": int(config.shot_start),
        "shot_stop": int(config.shot_stop),
        "shot_stride": int(config.shot_stride),
        "shot_count_attempted": len(shot_ordinals),
        "shot_count_estimated": int(wavelets.shape[0]),
        "shot_count_initial_mask": int(np.sum(initial_mask)),
        "shot_count_used_in_average": int(np.sum(final_mask)),
        "nearest_receivers": int(config.nearest_receivers),
        "max_abs_offset_m": None if config.max_abs_offset_m is None else float(config.max_abs_offset_m),
        "water_velocity_m_s": float(config.water_velocity_m_s),
        "rel_t_min_s": float(config.rel_t_min_s),
        "rel_t_max_s": float(config.rel_t_max_s),
        "iterations": int(config.iterations),
        "filter_lowcut_hz": None if config.filter_lowcut_hz is None else float(config.filter_lowcut_hz),
        "filter_highcut_hz": None if config.filter_highcut_hz is None else float(config.filter_highcut_hz),
        "prepad_s": float(config.prepad_s),
        "min_residual_ratio": float(config.min_residual_ratio),
        "min_shape_corr_to_average": float(config.min_shape_corr_to_average),
        "dt_s": float(dt_s),
        "sample_count": int(plan.samples_per_trace),
        "wavelet_npz": str(average_npz_path),
        "shape_corr_used_min": float(np.min(shape_corr[final_mask])) if np.any(final_mask) else None,
        "shape_corr_used_median": float(np.median(shape_corr[final_mask])) if np.any(final_mask) else None,
        "shape_corr_used_max": float(np.max(shape_corr[final_mask])) if np.any(final_mask) else None,
        "residual_ratio_median": float(np.median(residual_ratios)),
        "residual_ratio_p10": float(np.percentile(residual_ratios, 10)),
        "residual_ratio_p90": float(np.percentile(residual_ratios, 90)),
        "wavelet_event_time_s_in_causal": float(event_time_s),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if config.command:
        (output_dir / "command.txt").write_text(config.command.rstrip() + "\n")

    print(f"[analyze-wavelet] average_wavelet_npz: {average_npz_path}")
    print(f"[analyze-wavelet] summary_json: {output_dir / 'summary.json'}")
    return average_npz_path
