# sweepx.io

Seismic I/O without the solver: SEG-Y header index, gather plans, byte-offset
readers, acquisition geometry, data/model selection, model files, wavelets,
prefetchers. `import sweepx.io` does not import torch (only `datasets`,
`cuda_prefetch`, `crg_dataset` do, lazily). Full-file `read_segy` /
`write_segy` need `pip install "sweepx[segy]"`; the index, plan and readers
do not.

## Contents

- [Data layouts](#data-layouts)
- [SEG-Y → index → plan → gathers](#seg-y--index--plan--gathers) (runnable)
- [Geometry](#geometry)
- [Data and model plans](#data-and-model-plans)
- [Models, wavelets, prefetch](#models-wavelets-prefetch)

## Data layouts

| object | layout |
|---|---|
| solver output, `record.npy`, `sweepx.loss` input, FWI `obs.npy_path` | `(nshots, nt, nrec, nchan)` — time on axis 1 (= -3) |
| `PlanReader.read_group(g)`, `SEGYReader`, `IndexedShotGatherDataset` obs | `(n_traces, n_samples)` per gather; stacked `(nshots, nrec, nt)` |
| `write_segy` / `write_segy_minimal` input | `(n_traces, n_samples)` |
| models | `(nz, nx)` / `(nz, ny, nx)`, float32, m/s |
| grid geometry (`Geometry`, task `sources.npy`) | sources `(nshots, ndim)`, receivers `(nshots, nrec, ndim)`, int grid indices `(x, z)` / `(x, y, z)` |
| physical geometry (`PhysicalGeometry`) | same shapes, float metres |

Gather → solver layout: `obs = np.stack(gathers).transpose(0, 2, 1)[..., None]`.

## SEG-Y → index → plan → gathers

Trace-header byte offsets are **0-based** positions in the 240-byte header,
keyed as in `SEGY_REV1_BYTES`:

| key | offset | SEG-Y rev1 bytes (1-based) |
|---|---|---|
| `shot` | 8 | 9-12 field record |
| `trace_in_shot` | 12 | 13-16 |
| `source_elev` | 40 | 41-44 |
| `source_depth` | 48 | 49-52 |
| `receiver_depth` | 52 | 53-56 |
| `coord_scalar` | 70 | 71-72 (negative = divisor) |
| `sx`, `sy` | 72, 76 | 73-76, 77-80 |
| `rx`, `ry` | 80, 84 | 81-84 (group x), 85-88 |
| `receiver_elev` | 104 | 105-108 |

Override with `byte_map={...}` (Python), `byte_map:` (task YAML) or
`--byte-map KEY=BYTE` (`build-index`). Unknown keys (`gx`, `gy`, `fldr`, ...)
are ignored without an error, and a 1-based value reads the wrong bytes:

```python
import struct
import numpy as np
from sweepx.io.segy import (SEGY_BIN_HEADER_SIZE, SEGY_TEXT_HEADER_SIZE,
                            SEGY_TRACE_HEADER_SIZE, SEGYReader, write_segy_minimal)
from sweepx.io.segy_index import SEGY_REV1_BYTES as B, build_segy_index, SEGYIndex
from sweepx.io.seismic_plan import PlanReader, SeismicPlan

nshots, nrec, nt, dt = 3, 48, 500, 0.004
data = np.random.default_rng(0).standard_normal((nshots * nrec, nt)).astype("float32")
write_segy_minimal("line.sgy", data, dt=dt)             # blank trace headers, IEEE floats
with open("line.sgy", "r+b") as f:                     # fill what the index reads
    for s in range(nshots):
        for r in range(nrec):
            pos = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE + (s * nrec + r) * (SEGY_TRACE_HEADER_SIZE + nt * 4)
            f.seek(pos); h = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
            struct.pack_into(">i", h, B["shot"], s + 1)
            struct.pack_into(">i", h, B["trace_in_shot"], r + 1)
            struct.pack_into(">h", h, B["coord_scalar"], 1)
            struct.pack_into(">i", h, B["sx"], 1000 + 50 * s)
            struct.pack_into(">i", h, B["rx"], 1000 + 50 * s + 100 + 25 * r)
            f.seek(pos); f.write(h)

print(SEGYReader("line.sgy").dt, SEGYReader("line.sgy").n_samples)   # binary header: 0.004 500

idx = build_segy_index(["line.sgy"], num_workers=1)    # headers only; parallel over files
print(idx.n_shots, idx.n_traces, idx.sx_m[[0, nrec]], idx.rx_m[:2])  # metres, scalar applied
bad = build_segy_index(["line.sgy"], byte_map={"sx": 73}, num_workers=1)
print("1-based sx reads:", bad.sx_m[0])                # garbage, no error
idx.save("index.npz"); idx = SEGYIndex.load("index.npz")

plan = idx.to_seismic_plan(grouping="csg", offset_min_m=200.0)   # one group per shot
plan.save("plan.npz"); plan = SeismicPlan.load("plan.npz")
with PlanReader(plan) as reader:
    g0 = reader.read_group(0)                          # (n_traces, n_samples) float32
    print(plan.n_groups, g0.shape, reader.dt_s)
    assert np.array_equal(g0, data[4:48])              # offsets 100..1275 m: first 4 dropped
```

`grouping="crg"` groups by receiver position (node data), binning receivers
with `receiver_quantize_m`. CLI equivalents: `sweepx tasks build-index` /
`build-plan` (`sweepx tasks build-plan --help`). Many files: `build_segy_index(paths,
num_workers=N)`; SEG-Y paths stored in a plan can be redirected at read time
with `FWI_SEGY_ROOT` / `FWI_SEGY_REMAP`. `read_segy(path)` returns
`(data, {"dt", "n_samples", "ilines", "xlines"})` (a cube for structured 3-D
files, `(n_traces, ns)` otherwise).

## Geometry

```python
import numpy as np
from sweepx.io.geometry import Geometry, PhysicalGeometry

pg = PhysicalGeometry(
    sources_xyz_m=np.array([[100.0, 10.0], [400.0, 10.0]]),          # (nshots, 2): x, z in m
    receivers_xyz_m=np.stack([np.c_[np.arange(0, 600, 12.5), np.full(48, 20.0)]] * 2),
    dt=0.001, nt=1000)
geom, keep = pg.to_grid(25.0, dedupe=True)    # snap to a 25 m grid (round half to even)
print(geom.sources.tolist(), geom.receivers.shape, keep.sum(axis=1))   # [[4, 0], [16, 0]] (2, 48, 2) [25 25]
geom.save("acq.json"); print(Geometry.load("acq.json").nshots)
```

`to_grid(dh, origin_xyz_m=None, dedupe=True, dedup_method="nearest")` returns
the grid `Geometry` with **all** receivers (same-cell duplicates included) and
a `(nshots, nrec)` keep mask, True where a receiver survived dedupe: slice the
receivers and the data with it yourself (`obs[s][..., keep[s], :]`). A source
10 m deep snaps to row 0 on a 25 m grid: check depths after snapping to a coarse grid. Rotated surveys:
`RotatedFrame` / `load_rotation_metadata` (UTM ↔ model frame), and
`PhysicalGeometry.apply_rotation(frame)`.

## Data and model plans

`DataPlan` (shots, receiver stride, offsets, time window / resampling) and
`ModelPlan` (crop windows in metres) are the library form of the task keys
`data_plan:` / `model_plan:`.

```python
import numpy as np
from sweepx.io.geometry import PhysicalGeometry
from sweepx.io.plan import DataPlan, ModelPlan, apply_data_plan, apply_model_plan

pg = PhysicalGeometry(sources_xyz_m=np.c_[np.arange(0, 1000, 100.0), np.full(10, 10.0)],
                      receivers_xyz_m=np.stack([np.c_[np.arange(0, 1000, 10.0), np.full(100, 10.0)]] * 10),
                      dt=0.002, nt=500)
obs = np.zeros((10, 500, 100, 1), np.float32)          # (nshots, nt, nrec, 1)
dp = DataPlan(shot_stride=2, offset_max_m=300.0, dt_target_s=0.004)
pg2, obs2, keep = apply_data_plan(dp, pg, obs, time_axis=1, receiver_axis=2)  # axes for 4-D!
print(obs2.shape, keep.shape, keep.sum(axis=1)[:3])

vp = np.full((50, 100), 2000.0, np.float32)
vp2, pg3, src_keep = apply_model_plan(ModelPlan(x_window_m=(200.0, 700.0)), vp, dh=(10.0, 10.0), geom=pg)
print(vp2.shape, pg3.nshots)
```

`apply_data_plan`'s defaults are `time_axis=-2, receiver_axis=-1`, i.e. a 3-D
`(nshots, nt, nrec)` array; for the 4-D solver layout pass `time_axis=1,
receiver_axis=2`. `dt_target_s` and `time_decimate` are mutually exclusive.
The receiver mask is per shot (offset windows differ), so the caller applies it.
`ModelPlan` drops sources outside the window (`drop_outside_sources=True`) but
keeps receivers by default, and rebases positions to the cropped origin.

## Models, wavelets, prefetch

| API | does |
|---|---|
| `sweepx.io.models.load_velocity(path)` / `save_velocity(path, arr)` | `.npy`, `.npz`, `.h5` (needs `[hdf5]`), raw binary by extension |
| `sweepx.io.wavelet.load_wavelet_npz(path)` | the wavelet `.npz` that `sweep-tasks estimate-wavelet` writes (key auto-detected, `dt_s` or `time_s`); the task YAML form is `wavelet: {kind: siren_pipeline_npz, path: ...}`, resampled to `time.dt` and cut/padded to `nt` |
| `sweepx.io.prefetch.Prefetcher(iterable, queue_depth=2)` | read the next item in a background thread (context manager) |
| `ThreadPoolPrefetcher`, `sweepx.io.cuda_prefetch.CUDAPrefetcher` | parallel reads; copy to GPU on a side stream |
| `sweepx.io.datasets.ShotGatherDataset(geometry, obs)` | torch `Dataset`, one shot per item |
| `sweepx.io.segy_index.IndexedShotGatherDataset(index, shot_ids=...)` | lazy per-shot reads from an index; items have `"obs"` `(nrec, nt)` |
