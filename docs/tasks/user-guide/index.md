# User guide

## Task types

Every job is one `task_type`, each backed by a Pydantic spec in
`sweep_tasks.schemas`:

| `task_type` | Spec | Does |
|---|---|---|
| `forward` | `ForwardSpec` | Synthesise shot gathers from a model |
| `wavefield` | `WavefieldSpec` | Save wavefield snapshots |
| `fwi` | `FWISpec` | Full-waveform inversion |
| `lsrtm` | `LSRTMSpec` | Least-squares RTM |
| `rtm` | `RTMSpec` | Reverse-time migration |
| `introspect` | `IntrospectSpec` | Report what a config resolves to |

## Models without a file — `ModelRef`

A task YAML never has to point at an `.npy`. `ModelRef` takes four sources:

```yaml
models:
  - {name: vp, dataset: marmousi:2d-demo, preset: vp_true}   # a benchmark
  - {name: vp, constant: 2200.0, shape: [120, 180]}          # a uniform box
  - {name: vp, path: my_vp.npy}                              # your own file
  - name: vp                                                  # a 1-D cold start,
    shape: [281, 1361]                                        # built in memory
    linear_gradient: {vmin: 1500.0, vmax: 4000.0, water_rows: 37, water_vp: 1500.0}
```

Any source can be post-processed with `smooth_sigma_cells`.

## The water layer — `water:`

`water_rows` above only builds a starting model. What an inversion holds fixed
is one top-level block, read by every inversion path:

```yaml
water:
  seabed: seabed.npz     # key "seabed_depth", metres below z = 0; or a number
  # mask: water.npy      # or a boolean mask of the starting model's grid
  velocity: 1500.0
  mode: pin              # or reset_each_stage
```

Details, and how the retired `freeze_top_n_rows` / `freeze_water_layer` /
`reparam.mask_water_layer` keys map onto it (`sweep-tasks migrate-water`), are
in [CLI workflow](../cli_workflow.md#the-water-layer-water).

## The CLI

```
sweep-tasks run <task.yaml>          # execute a task
sweep-tasks init <type> -o out.yaml  # write an annotated template
sweep-tasks new  <type>              # print a minimal template
sweep-tasks tasks list | status | logs   # inspect runs
```

The full **SEG-Y → FWI data pipeline** — `build-index → build-plan → run`, plus
the `extract-coeff` frequency-selection branch for node data — is documented
step by step, with measured numbers, in **[CLI workflow](../cli_workflow.md)**.

## Distributed

```bash
torchrun --nproc_per_node=2 --standalone -m sweep_tasks.cli run task.yaml
```

Shot-parallel by default; gradient all-reduce + scalar-loss reduction are
handled internally, and `device: auto` becomes `cuda:LOCAL_RANK`. (L-BFGS is
blocked under `torchrun` — its closure semantics conflict with shot-parallel
all-reduce.)

## The schema *is* the LLM contract

Emit a JSON object matching `FWISpec`, hand it to `TaskRunner`, done — which is
exactly what [`sweep-agent`](../../agent/) does under the hood.
