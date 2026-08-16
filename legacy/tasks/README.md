# sweep-tasks

Typed YAML task layer for [sweep](https://github.com/DeepWave-KAUST/sweep) —
the front door for **CLI-** and **LLM-driven** FWI / LSRTM / forward / wavefield
runs.

If you only want to write a Python loop around the wave solver, use `sweep`
directly. **`sweep-tasks` is for when you want to describe a job in YAML
(or JSON, from an LLM) and have it executed reproducibly.**

## What's in it

| Module | Purpose |
|---|---|
| `sweep_tasks.schemas` | Pydantic models for every task type (`FWISpec`, `LSRTMSpec`, `RTMSpec`, `ForwardSpec`, `WavefieldSpec`, `IntrospectSpec`) + every sub-spec (`LossSpec`, `OptimizerAdam/SGD/LBFGS`, `Scheduler*`, `ModelRef`, `LineGeometry`/`GridGeometry`/`ExplicitGeometry`/`FromFileGeometry`/`FromPlanGeometry`, …) |
| `sweep_tasks.runner` | `TaskRunner` — synchronous local executor. One process or torchrun multi-rank; shot-parallel by default; writes `status.json` + `checkpoint.pt` + figures per run |
| `sweep_tasks.yaml_io` | `load_task` / `dump_task` / `new_template` — YAML ↔ Pydantic ↔ canonical template strings |
| `sweep_tasks.registry` | The `task_type -> spec class` map (used by `new_template` and `load_task`) |
| `sweep_tasks._distributed` | torchrun bootstrap + all-reduce / broadcast helpers used by the runner |
| `sweep_tasks.freqsel` | Frequency-selection (steady-state comb) encoding: `extract_shard` turns recorded gathers into DTFT coefficients, `FreqSelTargets` / `SteadyGCNLoss` drive the wavelet-free inversion |
| `sweep_tasks.cli` | `sweep-tasks run / init / new / build-index / build-plan / extract-coeff / tasks {list,status,logs}` (+ the wavelet tools) |

## Install

```bash
pip install sweep-tasks    # also pulls sweep (the solver) and pydantic+torch+yaml
```

Or via the ecosystem meta-package: `pip install sweep[full]`.

## Quick example — Marmousi synthetic (2 commands, nothing to download)

The models come from `sweep.datasets`, so the YAMLs are self-contained —
no `.npy` to prepare, no env vars, no SEG-Y:

```bash
sweep-tasks run examples/synthetic/01_forward_marmousi.yaml   # synthesise obs
sweep-tasks run examples/synthetic/02_fwi_marmousi_single.yaml  # invert (100 epochs)
```

### Models without a file

A task YAML never has to point at an `.npy`. `ModelRef` takes four sources:

```yaml
models:
  - {name: vp, dataset: marmousi:2d-demo, preset: vp_true}   # a benchmark
  - {name: vp, constant: 2200.0, shape: [120, 180]}          # a uniform box
  - {name: vp, path: my_vp.npy}                              # your own file
  - name: vp                                                  # a 1-D cold start,
    shape: [281, 1361]                                        # built in memory
    linear_gradient: {vmin: 1500.0, vmax: 4000.0,
                      water_rows: 37, water_vp: 1500.0}
```

Any of them can be post-processed with `smooth_sigma_cells`. When the start is
built in memory there is no input file to point at afterwards, so an FWI run
writes the resolved array to `output/initial_vp.npy` before training.

Or hand-write your own task YAML using the bundled templates:

```bash
sweep-tasks init forward -o my_forward.yaml   # annotated forward template
sweep-tasks init fwi     -o my_fwi.yaml       # annotated FWI template
sweep-tasks init rtm     -o my_rtm.yaml       # annotated RTM template
sweep-tasks init viking  -o viking.yaml       # full pipeline config (build-index → wavelet)
```

Every template comes with per-parameter comments explaining what each
key controls. See `examples/README.md` for the example index —
`examples/synthetic/` runs with no downloads, `examples/field/` works on
real SEG-Y — and `docs/datasets/viking/README.md` for the full Viking
workflow walkthrough.

## Quick example — task YAML by hand

Write a task YAML:

```yaml
task_type: fwi
equation: Acoustic
backend: {kind: eager}
physics:
  dh: 12.5
  free_surface: false
time: {dt: 0.001, nt: 2000}
wavelet: {kind: ricker, peak_hz: 4.0}
geometry: {kind: line, sources: {start: 0, stop: 680, step: 10}, receivers: {start: 0, stop: 680, step: 1}}
init_model: {name: vp, path: marmousi_init.npy}
obs: {kind: synthetic_from, model: {name: vp, path: marmousi_true.npy}}
loss: {kind: mse}
optimizer: {kind: adam, lr: 25.0}
epochs: 30
```

Run it from the shell:

```bash
sweep-tasks run task.yaml
```

Or in Python:

```python
from sweep_tasks import TaskRunner, load_task

spec = load_task("task.yaml")
result = TaskRunner().run(spec)
print(result.status.state, result.task_dir)
```

When `sweep` is installed alongside, the same names are reachable through
the unified namespace — any of these work:

```python
from sweep_tasks import TaskRunner             # direct dist name
from sweep.tasks import TaskRunner             # via sweep's companion alias
import sweep
sweep.tasks.TaskRunner                          # attribute access
```

The aliasing lives in `sweep/__init__.py` (PEP 562 `__getattr__` plus a
meta-path finder) — `sweep-tasks` itself doesn't know it's aliased; it
just publishes the `sweep_tasks` distribution as usual.

## Why a separate repo?

- **sweep stays clean as a solver** — equations, propagator, CUDA bindings, nothing else.
- **The task layer is where LLM / web / API integration lives**, and it evolves on its own cadence (new task types, new schemas, MCP tooling, …) without ever touching solver internals.
- The schemas themselves *are* the LLM contract: emit a JSON object matching `FWISpec`, hand it to `TaskRunner`, done.

## Distributed

```bash
torchrun --nproc_per_node=2 --standalone -m sweep_tasks.cli run task.yaml
```

Shot-parallel by default; gradient all-reduce + scalar loss reduction handled
internally. `device: auto` becomes `cuda:LOCAL_RANK`. LBFGS is blocked under
torchrun (closure semantics conflict with shot-parallel all-reduce).

## License

MIT.
