"""Wavelet analysis + inversion utilities for sweep-tasks.

Ported from ``fwi_workflow-dev`` (see
``docs/datasets/viking/README.md`` Step 3 for the call chain). Reads
:class:`sweep_io.seismic_plan.SeismicPlan` plans built by
``sweep-tasks build-plan`` instead of the legacy ``csg_index_v2`` npz.

Public surface
--------------

Library helpers (pure numpy, no torch / sweep dependencies):

* :func:`sweep_tasks.wavelet.estimation.ricker`
* :func:`sweep_tasks.wavelet.estimation.normalize_wavelet`
* :func:`sweep_tasks.wavelet.estimation.apply_fft_filter`
* :func:`sweep_tasks.wavelet.direct_arrival.estimate_direct_wavelet_rank1`
* :func:`sweep_tasks.wavelet.direct_arrival.robust_average_wavelets`
* :func:`sweep_tasks.wavelet.direct_arrival.to_causal_wavelet`
* :func:`sweep_tasks.wavelet.direct_arrival.synthesize_direct_record`

Stage drivers (one per CLI subcommand):

* :func:`sweep_tasks.wavelet.analyze.analyze_direct_wavelet_batch`
  (``sweep-tasks analyze-wavelet``)
* :func:`sweep_tasks.wavelet.sweep_torch.run_wavelet_inversion`
  (``sweep-tasks estimate-wavelet``)
* :func:`sweep_tasks.wavelet.convert_farfield.convert_farfield`
  (``sweep-tasks convert-farfield``)
* :func:`sweep_tasks.wavelet.plot_steps.plot_wavelet_pipeline_steps`
  (``sweep-tasks plot-wavelet-steps``)

Config dataclasses:

* :class:`sweep_tasks.wavelet.analyze.AnalyzeWaveletConfig`
* :class:`sweep_tasks.wavelet.sweep_torch.WaveletInversionConfig`
"""

from .analyze import AnalyzeWaveletConfig, analyze_direct_wavelet_batch
from .convert_farfield import convert_farfield
from .plot_steps import plot_wavelet_pipeline_steps
from .sweep_torch import (
    PreparedWaveletData,
    WaveletInversionConfig,
    apply_wavelet_postprocessing,
    run_wavelet_inversion,
)

__all__ = [
    "AnalyzeWaveletConfig",
    "PreparedWaveletData",
    "WaveletInversionConfig",
    "analyze_direct_wavelet_batch",
    "apply_wavelet_postprocessing",
    "convert_farfield",
    "plot_wavelet_pipeline_steps",
    "run_wavelet_inversion",
]
