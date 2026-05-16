# sweep-io

File and dataset I/O utilities for seismic full-waveform inversion (FWI).
Designed to be **independent** of the `sweep` core — you can use it in any
FWI / migration / imaging project, with or without the wave-equation engine.

The emphasis is on **fast hot-path reads with overlap-with-compute**:
your FWI loop should never sit blocking on a SEG-Y read when the GPU has
useful work to do.

## What's in it

| Module | Purpose | Extra deps |
|---|---|---|
| `sweep_io.models` | Load/save velocity models (`.npy`, `.npz`, raw binary, `.h5`) | none / `h5py` |
| `sweep_io.geometry` | Acquisition geometry dataclass (sources, receivers, `dt`, `dh`) | none |
| `sweep_io.prefetch` | Background-thread prefetchers (`Prefetcher`, `ThreadPoolPrefetcher`, `TimingPrefetcher`) | stdlib only |
| `sweep_io.segy` | SEG-Y byte-offset readers (`SEGYReader`, `MultiFileSEGYReader`) + IBM↔IEEE codecs; high-level `read_segy` / `write_segy` defer ``segyio`` import | numpy / `segyio` (lazy) |
| `sweep_io.datasets` | `torch.utils.data.Dataset` for shot gathers, with `iter_prefetched(...)` | `torch` |
| `sweep_io.cuda_prefetch` | CUDA-aware `CUDAPrefetcher` with pinned memory + side-stream H2D | `torch` |

All optional backends are **lazy-imported** — you only need what you use.

## Install

```bash
pip install sweep-io                  # numpy core + prefetch + low-level SEG-Y
pip install sweep-io[segy]            # + segyio (for read_segy / write_segy)
pip install sweep-io[hdf5]            # + h5py
pip install sweep-io[torch]           # + torch (Dataset + CUDAPrefetcher)
pip install sweep-io[segy,hdf5,torch]
```

Or via the ecosystem meta-package: `pip install sweep[full]`.

## Quick example — fast SEG-Y reads

```python
from sweep_io.segy import MultiFileSEGYReader

reader = MultiFileSEGYReader(["line_001.sgy", "line_002.sgy"])
print(reader.n_samples, reader.dt, reader.sample_format)

# Pre-built index: which (file_id, byte_offset) belongs to which shot.
file_ids, byte_offsets = my_shot_index[shot_idx]
traces = reader.read_traces(file_ids, byte_offsets, coalesce_gap=4096)
# -> (n_receivers, n_samples) float32, in caller order
```

`SEGYReader` uses `mmap` + sorted-offset coalescing so a batch of
random-looking offsets turns into a small number of large sequential
reads — typical 5-10× speedup on Lustre vs. one `pread` per trace.

## Quick example — prefetched FWI loop

```python
from sweep_io.prefetch import Prefetcher

def load_shot(i):
    return reader.read_traces(*my_index[i])

with Prefetcher((load_shot(i) for i in range(n_shots)), queue_depth=2) as pf:
    for shot in pf:
        # I/O for the next shot happens in a background thread
        # while we run forward + backward + optimizer.step() here.
        pred = solver(wavelet, sources, receivers, models=[vp])
        loss = misfit(pred, shot)
        loss.backward()
        optim.step(); optim.zero_grad()
```

Want parallel reads (many independent files, NVMe, Lustre wide stripe)?
Use `ThreadPoolPrefetcher`:

```python
from sweep_io.prefetch import ThreadPoolPrefetcher

with ThreadPoolPrefetcher(load_shot, range(n_shots),
                          num_workers=4, queue_depth=4) as pf:
    for shot in pf:
        ...
```

## Quick example — GPU pipeline with pinned memory + side stream

For multi-GB-per-batch workloads the host→device transfer is itself a
bottleneck. `CUDAPrefetcher` overlaps it with both I/O and compute:

```python
from sweep_io.cuda_prefetch import CUDAPrefetcher

def load_shot(i):
    arr = reader.read_traces(*my_index[i])    # CPU numpy
    return {"obs": arr, "shot_index": i}

with CUDAPrefetcher((load_shot(i) for i in range(N)),
                    device="cuda:0", queue_depth=2) as pf:
    for sample in pf:
        # sample["obs"] is already on cuda:0; the H2D was issued on a
        # side stream, and the current stream waits on its event before
        # the next kernel touches it.
        loss = forward_and_loss(sample["obs"])
        loss.backward()
        ...
```

## Quick example — torch Dataset with prefetch baked in

```python
from sweep_io.datasets import ShotGatherDataset

ds = ShotGatherDataset(geometry, obs=load_shot)   # callable for OOC
for sample in ds.iter_prefetched(queue_depth=2, num_workers=2):
    pred = solver(sample["sources"], sample["receivers"], ...)
    ...
```

## Demo

`examples/prefetch_demo.py` writes a few synthetic SEG-Y files and
benchmarks blocking vs. prefetched reads:

```
$ python examples/prefetch_demo.py
[blocking ] total   751.6 ms   io_wait   107.6 ms  compute 640.0 ms
[prefetch ] total   652.8 ms   io_wait     7.4 ms  compute 640.0 ms
speedup ≈ 1.15x; io_wait dropped from 107.6 ms to 7.4 ms
```

(Real FWI numbers depend on shot size and storage — the I/O wait
reduction is what matters; it's the wall-clock you get back when
compute would otherwise be twiddling its thumbs.)

## Design principles

- **Framework-agnostic core.** `numpy` is the only required dep. `segyio`,
  `h5py`, `torch` are optional extras and lazy-imported.
- **Byte-offset over headers.** The hot path takes a `(file_id, byte_offset)`
  catalog you build once and reuse forever — no per-iteration header parse,
  no Python loop per trace.
- **Sort-and-coalesce reads.** Adjacent reads merge into one `pread`; random
  reads become sequential. Bandwidth, not seek time, dominates.
- **Overlap I/O with compute.** Prefetchers run a worker that reads ahead
  by `queue_depth` items; consumer's `next()` only blocks if the worker
  hasn't caught up. Pair with `CUDAPrefetcher` for GPU pipelines.
- **Thread-safe readers.** `SEGYReader` uses `pread` / `mmap`-views, so one
  reader shared across N prefetch threads is safe.

## Caveats

- `SEGYReader` assumes **fixed-length traces** (all traces have the same
  `n_samples`). Variable-length SEG-Y is rare in production datasets but
  not supported here.
- `mmap_mode=True` (default) can misbehave on flaky network filesystems
  (`SIGBUS` if a node disappears mid-read). Pass `mmap_mode=False` to fall
  back to `pread`.
- The `Prefetcher` worker is a *thread*, not a process — fine for I/O-bound
  work (most of our hot path) but not for CPU-bound transforms that don't
  release the GIL. For heavy decoding, do it inside numpy / C extensions.

## License

MIT.
