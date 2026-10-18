# API reference

Generated from the source docstrings. Every module is importable on its own, e.g.
`from sweepx.io.seismic_plan import PlanReader`.

## SEG-Y

::: sweepx.io.segy.SEGYReader
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.segy.MultiFileSEGYReader
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.segy.read_segy
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.segy.write_segy
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.segy.write_segy_minimal
    options:
      show_root_heading: true
      docstring_style: numpy

## Header index

::: sweepx.io.segy_index.build_segy_index
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.segy_index.SEGYIndex
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.segy_index.IndexedShotGatherDataset
    options:
      show_root_heading: true
      docstring_style: numpy

## Plans and readers

::: sweepx.io.seismic_plan.build_seismic_plan
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.seismic_plan.SeismicPlan
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.seismic_plan.PlanReader
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.seismic_plan.sample_shared_shots_from_plan
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.seismic_plan.sample_percrg_independent
    options:
      show_root_heading: true
      docstring_style: numpy

## Data and model windows

::: sweepx.io.plan.DataPlan
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.plan.apply_data_plan
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.plan.ModelPlan
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.plan.apply_model_plan
    options:
      show_root_heading: true
      docstring_style: numpy

## Geometry

::: sweepx.io.geometry.Geometry
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.geometry.PhysicalGeometry
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.geometry.RotatedFrame
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.geometry.load_rotation_metadata
    options:
      show_root_heading: true
      docstring_style: numpy

## Velocity models

::: sweepx.io.models.load_velocity
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.models.save_velocity
    options:
      show_root_heading: true
      docstring_style: numpy

## Prefetching and datasets

::: sweepx.io.prefetch.Prefetcher
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.prefetch.ThreadPoolPrefetcher
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.prefetch.TimingPrefetcher
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.cuda_prefetch.CUDAPrefetcher
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.datasets.ShotGatherDataset
    options:
      show_root_heading: true
      docstring_style: numpy

## CRG plan cache

::: sweepx.io.crg_build.build_crg_plan_from_segy
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.crg_build.build_crg_plan_from_index
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.crg_plan.CRGPlan
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.crg_plan.load_crg_shot_plan_cache
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.crg_dataset.CRGBatchDataset
    options:
      show_root_heading: true
      docstring_style: numpy

## Wavelets

::: sweepx.io.wavelet.load_wavelet_npz
    options:
      show_root_heading: true
      docstring_style: numpy

::: sweepx.io.wavelet.WaveletNPZ
    options:
      show_root_heading: true
      docstring_style: numpy
