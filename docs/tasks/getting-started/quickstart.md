# Quickstart

## One command, nothing to download

The models come from `sweep.datasets`, so the example YAMLs are self-contained —
no `.npy` to prepare, no env vars, no SEG-Y:

```bash
sweep-tasks run examples/synthetic/02_fwi_marmousi_single.yaml
```

That is a full FWI on Marmousi-II: the observed data are synthesised from the
true model on the fly, and 200 epochs from a smoothed start take about 30 min
on one RTX 6000 Ada. The result lands in
`examples/synthetic/sweep_runs/fwi_marmousi_single/output/inverted_vp.npy`:

![inverted vp](../examples/figures/02_vp_final.png)

To see the data rather than invert it, run the forward on its own:

```bash
sweep-tasks run examples/synthetic/01_forward_marmousi.yaml
```

The [Examples](../examples/index.md) cover every task type the same way, with
reference numbers to check your run against.

## A task YAML by hand

The same FWI as above, cut to 30 epochs and written out with only the keys it
needs:

```yaml
task_type: fwi
task_id: my_fwi
seed: 0
grid: {dh: 12.5}
time: {dt: 0.001, nt: 10000}
wavelet: {kind: ricker, fm: 8.0, delay: 2.0}
geometry:
  kind: line
  sources:   {step: 12, depth: 1}
  receivers: {step: 1,  depth: 18}
physics: {equation: Acoustic}
backend:
  impl: c
  cuda_options: {memory: {strategy: boundary, boundary: {storage: gpu}}}
init_model: {name: vp, dataset: "marmousi:2d-demo", preset: vp_smooth}
obs:
  synthetic_from: {name: vp, dataset: "marmousi:2d-demo", preset: vp_true}
loss: {kind: mse}
optimizer: {kind: adam, lr: 25.0}
epochs: 30
batchsize: 16
```

The `cuda_options` line spells out the default for `impl: c`, boundary saving:
the adjoint keeps only the PML slab and reconstructs the rest exactly.
`strategy: full` would store every time step instead, about 276 GiB on this
grid (see [Backends and memory](../examples/backends.md)).

Run it from the shell, or drive it from Python:

```bash
sweep-tasks run task.yaml
```

```python
from sweep_tasks import TaskRunner, load_task

spec = load_task("task.yaml")
result = TaskRunner().run(spec)
print(result.status.state, result.task_dir)
```

Prefer to start from an annotated template?

```bash
sweep-tasks init fwi -o my_fwi.yaml     # every key commented
```
