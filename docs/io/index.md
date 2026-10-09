# sweep-io

Seismic data and model I/O for full-waveform inversion: SEG-Y readers, a header
catalog built once over many files, shot- or receiver-grouped data plans, velocity
models, acquisition geometry, and prefetchers that read the next gather while the GPU
works on the current one. It needs only NumPy; `segyio`, `h5py` and `torch` are
optional and imported only by the code that uses them. There is no dependency on the
wave solver.

Installed with `pip install sweep-io` (also bundled by `pip install sweepx`)
→ `import sweep_io`.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __[Getting started](getting-started/installation.md)__

    ---

    Install, then go from a SEG-Y file to shot gathers in a few lines.

-   :material-file-code-outline: __[Examples](examples.md)__

    ---

    Runnable scripts: plans, prefetched reads, model and geometry files.

-   :material-api: __[API reference](api/index.md)__

    ---

    Readers, header index, plans, geometry, prefetchers.

</div>

## The data path

```
SEG-Y files ──build_segy_index──▶ SEGYIndex ──build_seismic_plan──▶ SeismicPlan ──PlanReader──▶ gathers
             headers, read once    one row per trace    shot or receiver groups,   byte-offset reads
                                                         saved as .npz
```

Scan the trace headers once into a `SEGYIndex`. Organise the traces as shot gathers
(CSG) or receiver gathers (CRG) in a `SeismicPlan`, filtering by shot, offset or trace
count, and save it. Then read gathers by number through a `PlanReader`: it seeks
straight to each trace's byte offset and merges adjacent reads, so a gather costs a few
large reads rather than one per trace. [sweep-tasks](../tasks/index.md) runs this path
from its CLI (`sweep-tasks build-index`, `sweep-tasks build-plan`) and its YAML specs.

## What is in it

| Module | What it gives you | Extra deps |
|---|---|---|
| `sweep_io.segy` | `SEGYReader`, `MultiFileSEGYReader`: byte-offset trace reads with sorted, coalesced `pread`/`mmap`; IBM ↔ IEEE codecs; `read_segy` / `write_segy` for whole files | none / `segyio` |
| `sweep_io.segy_index` | `build_segy_index` → `SEGYIndex`, a per-trace header catalog over many files; a lazy shot-gather dataset | none |
| `sweep_io.seismic_plan` | `build_seismic_plan` → `SeismicPlan` (CSG or CRG groups), `PlanReader`, and the shared-shot and per-receiver samplers used for source-encoded 3-D FWI | none |
| `sweep_io.plan` | `DataPlan` / `ModelPlan`: pick shots, receivers, offsets, time windows and the model window before FWI sees the data | none |
| `sweep_io.geometry` | `Geometry` (grid indices) and `PhysicalGeometry` (metres) acquisition dataclasses; `RotatedFrame`, the UTM ↔ model-frame rotation | none |
| `sweep_io.models` | Load and save velocity models: `.npy`, `.npz`, raw binary, `.h5` | none / `h5py` |
| `sweep_io.prefetch` | `Prefetcher`, `ThreadPoolPrefetcher`, `TimingPrefetcher`: read ahead in background threads | none |
| `sweep_io.cuda_prefetch` | `CUDAPrefetcher`: pinned memory and a side-stream host-to-device copy | `torch` |
| `sweep_io.datasets` | `ShotGatherDataset`, a `torch` Dataset with prefetched iteration | `torch` |
| `sweep_io.crg_build`, `crg_plan`, `crg_dataset` | Build, load and iterate the CRG plan cache (`crg_fwi_plan_v1`) | none / `mpi4py`, `torch` |
| `sweep_io.wavelet` | `load_wavelet_npz`: a source wavelet from an `.npz` | none |
