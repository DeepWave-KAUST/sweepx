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
| `sweep_tasks.schemas` | Pydantic models for every task type (`FWISpec`, `LSRTMSpec`, `ForwardSpec`, `WavefieldSpec`, `IntrospectSpec`) + every sub-spec (`LossSpec`, `OptimizerAdam/SGD/LBFGS`, `Scheduler*`, `ModelRef`, `LineGeometry`/`ExplicitGeometry`/`FromFileGeometry`, …) |
| `sweep_tasks.runner` | `TaskRunner` — synchronous local executor. One process or torchrun multi-rank; shot-parallel by default; writes `status.json` + `checkpoint.pt` + figures per run |
| `sweep_tasks.yaml_io` | `load_task` / `dump_task` / `new_template` — YAML ↔ Pydantic ↔ canonical template strings |
| `sweep_tasks.registry` | The `task_type -> spec class` map (used by `new_template` and `load_task`) |
| `sweep_tasks._distributed` | torchrun bootstrap + all-reduce / broadcast helpers used by the runner |
| `sweep_tasks.cli` | `sweep-tasks run / new / tasks {list,status,logs}` |

## Install

```bash
pip install sweep-tasks    # also pulls sweep (the solver) and pydantic+torch+yaml
```

Or via the ecosystem meta-package: `pip install sweep[full]`.

## Quick example

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
