# Backends and memory

```bash
sweep-tasks run examples/synthetic/08_fwi_marmousi_backends.yaml
```

The same FWI as [02](fwi-single.md) cut to 10 epochs, with every `backend:`
variant written out and commented so you can swap one block at a time and watch
wall time and `nvidia-smi`. **Reference: 43 s** for the default (fused CUDA +
boundary saving). The physics and the gradient are identical across the
variants — only speed and peak memory move.

## The adjoint-memory ladder

A gradient needs the forward wavefield again during the backward pass.
`cuda_options.memory.strategy` decides how it gets it back:

| `strategy` | what it keeps for the adjoint |
|---|---|
| `boundary` (default) | only the PML-zone slab; the interior is reconstructed backwards. Exact. `storage:` sends the slab to `gpu` / `cpu` / `disk`. |
| `full` | every time step. Fastest, and out of memory well before a 3-D grid — 276 GiB already for 2-D Marmousi at 10,000 steps. |
| `ckpt` | gradient checkpointing — recompute instead of store. |

Leaving `cuda_options` out on `impl: c` gives boundary saving on the GPU; the
resolved block is written to the run's `config_resolved.yaml`.

| `boundary.storage` | |
|---|---|
| `gpu` | slab stays on device — fastest |
| `cpu` | slab in host RAM; frees device memory at the cost of PCIe traffic (`transfer_interval` batches the copies) |
| `disk` | slab on disk, for runs whose slab does not fit in host RAM either |

[Frequency-selection FWI](freqsel-2d.md) is where this stops being an academic
choice: at 60,500 time steps, `full` needs about 50 GB where `boundary` needs
2.2 GB.

## Other backends in the file

`impl: eager` runs the same equation in plain PyTorch — on a CPU, or on a GPU
without the fused kernels — at a large cost in speed. The file also shows
`torch.compile` for the eager path. Run [15](introspect.md) to see which
equations have the fused `c` kernels in your build.
