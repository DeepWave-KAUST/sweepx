# Quickstart

## From SEG-Y files to shot gathers

Scan the trace headers once, organise the traces as gathers, and read them back by
number:

```python
from sweep_io.segy_index import build_segy_index
from sweep_io.seismic_plan import PlanReader, SeismicPlan

index = build_segy_index(["line_001.sgy", "line_002.sgy"])    # headers only, in parallel
plan = index.to_seismic_plan(grouping="csg", offset_min_m=200.0)
plan.save("plan.npz")                                         # later runs: SeismicPlan.load

reader = PlanReader(plan)
gather = reader.read_group(0)                                 # (n_traces, n_samples) float32
print(plan.n_groups, gather.shape, reader.dt_s)
```

`grouping="csg"` gives one group per shot; `grouping="crg"` gives one group per
receiver position, binning receivers onto a `receiver_quantize_m` grid, which is what
node (OBN) data wants. The plan stores, for every trace, which file it is in and
at which byte offset, so `read_group` seeks straight to the traces and merges adjacent
reads.

[`examples/segy_to_plan.py`](https://github.com/DeepWave-KAUST/sweep-io/blob/main/examples/segy_to_plan.py)
runs this on a synthetic file it writes first, so you can try it without data.

## Read ahead while the GPU works

Wrap any iterator in a `Prefetcher` and the next item is read in a background thread
while the loop body runs:

```python
from sweep_io.prefetch import Prefetcher

gathers = (reader.read_group(g) for g in range(plan.n_groups))
with Prefetcher(gathers, queue_depth=2) as pf:
    for obs in pf:
        pred = solver(wavelet, sources, receivers, models=[vp])    # your FWI step
        loss = misfit(pred, obs)
        loss.backward()
```

`ThreadPoolPrefetcher` reads several items in parallel, for wide-striped or many-file
storage; `CUDAPrefetcher` also copies each item to the GPU on a side stream.

## Pick the data and the model window

`DataPlan` selects shots, receivers, offsets and a time window; `ModelPlan` crops the
model and rebases the geometry onto it. Both work on a `PhysicalGeometry` (positions in
metres) and the observed data:

```python
from sweep_io.plan import DataPlan, ModelPlan, apply_data_plan, apply_model_plan

data_plan = DataPlan(shot_stride=5, offset_max_m=2000.0, dt_target_s=0.008)
geom_m, obs, receiver_mask = apply_data_plan(data_plan, geom_m, obs)

vp, geom_m, shot_mask = apply_model_plan(ModelPlan(x_window_m=(5000.0, 22000.0)),
                                         vp, dh=(12.5, 12.5), geom=geom_m)
```

`PhysicalGeometry.to_grid(dh)` then snaps the positions onto the grid of an FWI stage,
dropping receivers that land on the same cell.

## Velocity models and geometry files

```python
import numpy as np
from sweep_io.geometry import Geometry
from sweep_io.models import load_velocity, save_velocity

save_velocity("vp.npy", vp)                 # .npy, .npz, .h5 or raw binary, by extension
vp = load_velocity("vp.npy")

geom = Geometry(sources=sources, receivers=receivers, dt=0.001, nt=4000, dh=(10.0, 10.0))
geom.save("acq.json")                       # grid indices, dt, nt, dh and metadata
geom = Geometry.load("acq.json")
```
