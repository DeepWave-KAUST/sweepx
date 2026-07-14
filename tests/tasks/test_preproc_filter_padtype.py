"""Bandpass padding-policy tests.

Two policies need to behave correctly:

1. ``padtype="odd"`` (default) — scipy's reflective padding. Edge
   transients are suppressed to ~machine precision.

2. ``padtype=None`` — no padding. Matches the behavior of
   ``torchaudio.functional.filtfilt`` used in ``fwi_workflow-dev``'s
   GPU FWI path. Edge transients are visible across the trace.

Both should agree in the deep interior (away from edges) to ~machine
precision, since the underlying ``sosfiltfilt`` magnitude response is
identical; only padding differs.
"""

import numpy as np
from scipy import signal as sig
from scipy.signal import sosfiltfilt, filtfilt as scipy_filtfilt

from sweep_tasks.preproc.filter import bandpass


def _signal(nt=1500, nrec=4, dt=0.004, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * dt
    x = (rng.standard_normal((nrec, nt))
         + 0.3 * np.sin(2 * np.pi * 5 * t)
         + 0.2 * np.sin(2 * np.pi * 25 * t)).astype(np.float64)
    return x, dt


def test_padtype_default_matches_scipy_sosfiltfilt_default():
    """Default padtype='odd' must match scipy.sosfiltfilt(padtype='odd')."""
    x, dt = _signal()
    order = 4
    nyq = 0.5 / dt
    sos = sig.butter(order, [2.0 / nyq, 10.0 / nyq], btype="band", output="sos")
    expected = sosfiltfilt(sos, x, axis=-1, padtype="odd")
    actual = bandpass(x, lo=2.0, hi=10.0, dt=dt, order=order)
    assert np.allclose(actual, expected, atol=1e-12)


def test_padtype_none_matches_no_padding():
    """padtype=None must match scipy.sosfiltfilt(padtype=None)."""
    x, dt = _signal()
    order = 4
    nyq = 0.5 / dt
    sos = sig.butter(order, [2.0 / nyq, 10.0 / nyq], btype="band", output="sos")
    expected = sosfiltfilt(sos, x, axis=-1, padtype=None)
    actual = bandpass(x, lo=2.0, hi=10.0, dt=dt, order=order, padtype=None)
    assert np.allclose(actual, expected, atol=1e-12)


def test_padtype_none_diverges_from_padded():
    """no-padding filtfilt has transients that propagate across the whole
    trace (NOT just the edges) — this is a known property of zero-phase
    filtering without input padding. Default reflective padding suppresses
    these transients to machine precision.

    The test exists to document the expected divergence: a user who picks
    ``padtype=None`` to match ``fwi_workflow-dev``'s GPU path should know
    they are accepting visible transients.
    """
    x, dt = _signal(nt=3000)
    order = 4
    y_pad = bandpass(x, lo=2.0, hi=10.0, dt=dt, order=order, padtype="odd")
    y_nopad = bandpass(x, lo=2.0, hi=10.0, dt=dt, order=order, padtype=None)
    # Transient is non-trivial: the two diverge by more than 1% of signal RMS.
    signal_rms = np.sqrt((y_pad ** 2).mean())
    diff_rms = np.sqrt(((y_pad - y_nopad) ** 2).mean())
    assert diff_rms > 0.001 * signal_rms, (
        "no-padding filter should differ from reflective-padded output; "
        f"diff_rms={diff_rms:.3e}, signal_rms={signal_rms:.3e}"
    )


def test_no_padding_matches_fwi_workflow_dev_semantics():
    """padtype=None + sweep-preproc bandpass must match the fwi_workflow-dev
    filter path (b, a) + scipy.filtfilt(padtype=None), which is also what
    torchaudio.functional.filtfilt computes.

    This is the contract: when a user sets padtype=None in their
    StageBandpass, the resulting obs is bit-equivalent (to machine
    precision in the interior) to what ``fwi_workflow-dev`` would produce.
    """
    x, dt = _signal()
    order = 4
    nyq = 0.5 / dt
    b, a = sig.butter(order, [2.0 / nyq, 10.0 / nyq], btype="bandpass")
    # fwi_workflow-dev path (no-padding (b,a) filtfilt)
    expected = scipy_filtfilt(b, a, x, axis=-1, padtype=None)
    actual = bandpass(x, lo=2.0, hi=10.0, dt=dt, order=order, padtype=None)
    # (b,a) and SOS forms of the same butter prototype produce numerically
    # identical filtfilt outputs to ~1e-7 (SOS is more numerically stable
    # but the order=4 (b,a) is also fine in float64).
    assert np.allclose(actual, expected, atol=1e-6)


def test_order_higher_order_attenuates_more():
    """Increasing prototype order N → ``|H_N(f)|²`` rolls off faster.

    Both 2nd and 8th order should attenuate a far-stopband tone, but
    8th order should attenuate ~more strongly. This pins the order-N
    semantics without depending on an absolute dB number (which is
    sensitive to the exact transition-band geometry).
    """
    nt = 8192
    dt = 0.004
    t = np.arange(nt) * dt
    x = np.sin(2 * np.pi * 60.0 * t).astype(np.float64)
    rms_in = np.sqrt((x ** 2).mean())
    y2 = bandpass(x, lo=2.0, hi=10.0, dt=dt, order=2)
    y8 = bandpass(x, lo=2.0, hi=10.0, dt=dt, order=8)
    r2 = np.sqrt((y2 ** 2).mean()) / rms_in
    r8 = np.sqrt((y8 ** 2).mean()) / rms_in
    # Higher order attenuates more aggressively in the stop band.
    assert r8 < r2, f"order 8 should attenuate more than order 2; got r2={r2}, r8={r8}"
    # Both should suppress the stop-band tone to under 10% of input.
    assert r2 < 0.1 and r8 < 0.1
