# Examples

Runnable scripts in the repository's
[`examples/`](https://github.com/DeepWave-KAUST/sweep-io/tree/main/examples). Each one
makes its own synthetic data and runs on a CPU in seconds:

```bash
git clone https://github.com/DeepWave-KAUST/sweep-io
cd sweep-io
python examples/segy_to_plan.py
```

| Script | What it shows |
|---|---|
| [`segy_to_plan.py`](https://github.com/DeepWave-KAUST/sweep-io/blob/main/examples/segy_to_plan.py) | The field-data path: write a SEG-Y with coordinates in its trace headers, scan it into a `SEGYIndex`, build and save a shot-gather `SeismicPlan`, read the gathers through a `PlanReader` with a `Prefetcher` |
| [`plan_demo.py`](https://github.com/DeepWave-KAUST/sweep-io/blob/main/examples/plan_demo.py) | `DataPlan`, `ModelPlan` and `PhysicalGeometry.to_grid` on a synthetic streamer line: shot and offset selection, resampling, a model window, snapping to the grid of each FWI stage |
| [`prefetch_demo.py`](https://github.com/DeepWave-KAUST/sweep-io/blob/main/examples/prefetch_demo.py) | Blocking versus prefetched SEG-Y reads, timed: how much I/O wait the prefetcher hides |
| [`quickstart.py`](https://github.com/DeepWave-KAUST/sweep-io/blob/main/examples/quickstart.py) | Saving and loading a velocity model and an acquisition geometry |
