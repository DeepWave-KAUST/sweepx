"""Signal processing for seismic FWI — bandpass filtering + time-axis resampling.

The ``filter`` and ``resample`` modules were absorbed from the retired
``sweep-preproc`` package (the only ones sweep-tasks consumes). The unused
``mute`` / ``normalize`` / ``wavelet`` modules were dropped; they remain in the
archived ``sweep-preproc`` repo if ever needed.
"""

from __future__ import annotations

from . import filter, resample

__all__ = ["filter", "resample"]
