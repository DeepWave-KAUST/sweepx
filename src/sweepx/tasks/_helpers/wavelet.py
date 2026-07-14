"""Wavelet construction + source-delay lookup. Verbatim from runner.py."""
import numpy as np

from sweep.signal import ricker

def _build_wavelet(wavelet_spec, time_spec, *, override_dt: float | None = None,
                   override_nt: int | None = None) -> "np.ndarray":
    """Dispatch on wavelet.kind. Returned array has length nt.

    ``override_dt`` / ``override_nt`` let the runner pass in effective
    values when DataPlan / per-stage dt-sync changes the solver grid.
    """

    nt = int(override_nt) if override_nt is not None else int(time_spec.nt)
    dt = float(override_dt) if override_dt is not None else float(time_spec.dt)
    kind = getattr(wavelet_spec, "kind", None)
    if kind == "ricker":
        t = np.arange(nt, dtype=np.float32) * dt
        wave = ricker(t - float(wavelet_spec.delay), f=float(wavelet_spec.fm)).astype(np.float32)
        return float(wavelet_spec.scale) * wave
    if kind == "from_npy":
        arr = np.load(wavelet_spec.path).astype(np.float32)
        if arr.ndim != 1:
            raise ValueError(
                f"FromNpyWavelet expects a 1D array, got shape {arr.shape}."
            )
        if arr.shape[0] != nt:
            raise ValueError(
                f"FromNpyWavelet length {arr.shape[0]} != time.nt {nt}."
            )
        return float(wavelet_spec.scale) * arr
    if kind == "siren_pipeline_npz":
        from sweep_io.wavelet import load_wavelet_npz
        loaded = load_wavelet_npz(
            wavelet_spec.path, explicit_key=wavelet_spec.explicit_key,
        )
        samples = loaded.samples
        # Note: the SIREN-pipeline npz also reports ``source_delay_s``
        # (the zero-prepad applied during SIREN training); consumers
        # that need to align obs to the wavelet's frame should call
        # :func:`_get_wavelet_source_delay_s` and left-shift obs by
        # the same amount.
        # Resample to solver dt when the wavelet was sampled at a different
        # rate (typical for SIREN wavelets fit at SEG-Y dt while the solver
        # runs at a finer step). Uses sweep_tasks.preproc.resample_time.
        if abs(loaded.dt_s - dt) > 1.0e-12:
            from sweep_tasks.preproc.resample import resample_time
            samples = resample_time(
                samples.astype(np.float32), loaded.dt_s, dt, axis=0,
            ).astype(np.float32)
        # Length-align: truncate excess samples, zero-pad short tails.
        if samples.shape[0] > nt:
            samples = samples[:nt]
        elif samples.shape[0] < nt:
            samples = np.concatenate(
                [samples, np.zeros(nt - samples.shape[0], dtype=np.float32)]
            )
        return float(wavelet_spec.scale) * samples.astype(np.float32, copy=False)
    raise ValueError(f"Unknown wavelet.kind '{kind}'.")


def _get_wavelet_source_delay_s(wavelet_spec) -> float:
    """Return the wavelet's zero-prepad delay in seconds, or 0.0.

    Only the SIREN-pipeline npz format carries this metadata
    (``source_delay_s`` scalar). Consumers should left-shift observed
    traces by ``int(round(source_delay_s / dt_obs))`` samples so the
    main wavelet bang lines up with the actual event in obs — without
    this, syn and obs are off by the prepad (a systematic
    cycle-skip).

    For any other wavelet kind returns ``0.0``.
    """
    kind = getattr(wavelet_spec, "kind", None)
    if kind != "siren_pipeline_npz":
        return 0.0
    from sweep_io.wavelet import load_wavelet_npz
    loaded = load_wavelet_npz(
        wavelet_spec.path,
        explicit_key=getattr(wavelet_spec, "explicit_key", None),
    )
    return float(loaded.source_delay_s or 0.0)
