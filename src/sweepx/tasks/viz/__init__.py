"""Visualization for seismic FWI / migration — colormaps, seismic gathers,
wavefield snapshots, velocity-model panels, convergence curves, metrics.

Absorbed verbatim from the standalone ``sweep-viz`` package (now retired).
Import paths moved ``sweep_viz.<mod>`` -> ``sweepx.tasks.viz.<mod>``.

Importing ``sweepx.tasks.viz.colormaps`` registers the bundled diverging LUTs
(e.g. ``sweep_image``) as matplotlib colormaps as a side effect.
"""

from __future__ import annotations

from . import colormaps, convergence, metrics, model, seismic, wavefield

__all__ = ["colormaps", "convergence", "metrics", "model", "seismic", "wavefield"]
