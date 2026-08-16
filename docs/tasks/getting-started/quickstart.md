# Quickstart

## Two commands, nothing to download

The models come from `sweep.datasets`, so the example YAMLs are self-contained —
no `.npy` to prepare, no env vars, no SEG-Y:

```bash
sweep-tasks run examples/synthetic/01_forward_marmousi.yaml      # synthesise obs
sweep-tasks run examples/synthetic/02_fwi_marmousi_single.yaml   # invert (100 epochs)
```

## A task YAML by hand

```yaml
task_type: fwi
equation: Acoustic
backend: {kind: eager}
physics: {dh: 12.5, free_surface: false}
time: {dt: 0.001, nt: 2000}
wavelet: {kind: ricker, peak_hz: 4.0}
geometry: {kind: line, sources: {start: 0, stop: 680, step: 10}, receivers: {start: 0, stop: 680, step: 1}}
init_model: {name: vp, path: marmousi_init.npy}
obs: {kind: synthetic_from, model: {name: vp, path: marmousi_true.npy}}
loss: {kind: mse}
optimizer: {kind: adam, lr: 25.0}
epochs: 30
```

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
