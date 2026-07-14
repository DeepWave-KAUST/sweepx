"""Signal processing for seismic FWI — bandpass filtering, muting, trace
normalization, time-axis resampling, source-wavelet estimation.

Absorbed verbatim from the standalone ``sweep-preproc`` package (now retired).
Import paths moved ``sweep_preproc.<mod>`` -> ``sweep_tasks.preproc.<mod>``.
"""

from __future__ import annotations

from . import filter, mute, normalize, resample, wavelet

__all__ = ["filter", "mute", "normalize", "resample", "wavelet"]
