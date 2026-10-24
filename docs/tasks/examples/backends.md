# Backends and memory

```bash
sweep-tasks run examples/tasks/synthetic/08_fwi_marmousi_backends.yaml
```

[02](fwi-single.md)'s FWI at a batch of 4 shots per step (02 draws 16), cut
to 10 epochs, with the main `backend:` variants written out as commented
blocks so you can swap one in at a time and watch wall time and `nvidia-smi`.
**Reference: about 70 s on a 32 GB V100** for the default (fused CUDA +
boundary saving); see variant A in the table below.

Every block in the file, swapped in as its header says and run on one V100
(32 GB):

| variant | wall | `nvidia-smi` peak | epoch-0 loss |
|---|---|---|---|
| A — `c`, boundary saving, slab on `gpu` (active) | 71 s | 9.8 GiB | 1.457908e-2 |
| B — `c`, boundary saving, slab on `cpu` | 84 s | 7.3 GiB | 1.457908e-2 |
| C — `c`, `ckpt`, 64 chunks | 71 s | 10.5 GiB | 1.457908e-2 |
| D — `eager` + `torch.compile`, `use_ckpt: true` | 265 s | 13.7 GiB | 1.445130e-2 |

The three `impl: c` variants compute the same gradient: B reproduced A bit
for bit, and C's loss stayed within 1e-5 (relative) of A's at every epoch,
its final model within 0.2 m/s rms. Only speed and peak memory move.

`impl: eager` solves the same equation but not with exactly the same discrete
absorbing boundary: its loss sat 0.3–0.9 % away from A's at every epoch, and
after 10 epochs its model differed from A's by 4 m/s rms. The gap lives in the
boundary, not the interior: with `physics.abcn: 60` instead of this file's 20,
the epoch-0 losses of A and D (taken before any update, so forward modelling
only) differ by 0.013 % instead of 0.88 %.

## The adjoint-memory ladder

A gradient needs the forward wavefield again during the backward pass.
`cuda_options.memory.strategy` decides how it gets it back:

| `strategy` | what it keeps for the adjoint |
|---|---|
| `boundary` (default) | only the PML-zone slab; the interior is reconstructed backwards. Exact. `storage:` sends the slab to `gpu` / `cpu` / `disk`. |
| `full` | every time step. Fastest, and out of memory well before a 3-D grid — this file's 4-shot batch already asks for 69 GiB at 10,000 steps (276 GiB at 02's 16), and stops with an out-of-memory error on a 32 GB card. |
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
without the fused kernels. Variant D keeps `use_ckpt: true`: with it off,
autograd keeps all 10,000 steps for the backward pass, and the same block with
`use_ckpt: false` ran out of memory on the V100 in its first training forward.
Leaving the line out does not help, because the schema default is `false`.
On the V100, D as written (compiled) took 3.7× A's wall time; with
`use_compile: false` it took 36× (2418 s against 66 s for A on the same node)
and peaked at 16.6 GiB. Both numbers include the checkpoint recompute.

`eager` is also the only backend that captures wavefield snapshots, so a
[wavefield task](wavefield.md) must use it. Run [15](introspect.md) to see
which equations have the fused `c` kernels in your build.
