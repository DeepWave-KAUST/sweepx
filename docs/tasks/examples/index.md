# Examples

Every example is one YAML file under `examples/tasks/synthetic/`, run with one
command. The models come from `sweep.datasets` (Marmousi-II is embedded in the
`sweep` package), so there is nothing to download and no path to set. Each
page below covers one task: what it teaches, the command, and the result the
reference run produced, so you can check your own run against something.

| page | examples | what it covers |
|---|---|---|
| [Forward modelling](forward.md) | 01, 09, 10 | the task-YAML anatomy, a shot record, the geometry kinds |
| [FWI, single-scale](fwi-single.md) | 02 | the FWI task shape, synthetic obs, boundary-saving adjoint, QC |
| [FWI, multiscale](fwi-multiscale.md) | 03 | `stages:` — frequency continuation from a cold 1-D start |
| [iFWI](ifwi.md) | 07 | the model as a coordinate network (`reparam:`) |
| [RTM and LSRTM](imaging.md) | 04, 05 | imaging at a fixed velocity |
| [Wavefield snapshots](wavefield.md) | 06 | `task_type: wavefield`, and seeing a physics flag |
| [Backends and memory](backends.md) | 08 | the adjoint-memory ladder: `full` / `boundary` / `ckpt` |
| [Frequency-selection FWI, 2-D](freqsel-2d.md) | 11, 12 | wavelet-free FWI on pre-extracted frequency coefficients |
| [Frequency-selection FWI, 3-D](freqsel-3d.md) | 13, 14 | the same pipeline on 3-D Overthrust |
| [Anisotropic and elastic](multiparameter.md) | 16, 17 | multi-parameter equations: VTI, and P + S in one record |
| [Introspection](introspect.md) | 15 | ask the installed solver what it can do |
| [Viking (field data)](../datasets/viking/README.md) | `examples/tasks/field/viking/` | real SEG-Y: build-index → build-plan → extract-coeff → frequency-selection FWI |

New to task YAMLs? Read [09](forward.md#09-the-smallest-task-there-is) first:
it annotates every field. On a new machine, run [15](introspect.md) first: it
tells you which equations and backends your build has.

## Before you start

The examples default to `backend.impl: c` — sweep's fused CUDA kernels — so a
CUDA GPU is recommended. On a CPU-only machine replace the whole backend block,
`--override 'backend={impl: eager, use_ckpt: true}'`: overriding only
`backend.impl=eager` is refused when the file carries a `cuda_options:` block,
and without `use_ckpt` the eager gradient keeps every time step in memory.
Expect hours — 05 took 3 h 53 min and 40 GiB of RAM on 16 CPU cores, against
3 minutes with `c` on a V100 — and run 04 on a GPU: RTM needs the illumination
only `c` computes. 06, 09, 10, 16 and 17 already run on `eager`.

Relative paths in a task YAML resolve against the YAML **file**, not your
shell's working directory, so every run lands in
`examples/tasks/synthetic/sweep_runs/<task_id>/` (gitignored; a file without
`task_id` gets `<task_type>-<timestamp>`). Pass
`--override output_dir=/somewhere/else` to put it elsewhere.

## The Marmousi velocity presets

01–08 and 11–12 share the Marmousi-II model (281 × 1361 cells at 12.5 m).
`ModelRef.dataset` names the `sweep.datasets` entry in the YAML and `preset`
picks the model within it:

| preset | role |
|---|---|
| `vp_true` | true Marmousi-II vp — used to synthesise obs |
| `vp_smooth` | low-pass-smoothed true model — the easy FWI start ([02](fwi-single.md)) |
| `vp_linear` | 1-D gradient, zero lateral structure |

[03](fwi-multiscale.md) deliberately does **not** use `vp_linear`. It builds
its own 1-D ramp with `ModelRef.linear_gradient`, because the preset gets the
water column wrong (it ramps to 1797 m/s where Marmousi is a flat 1500) and
caps at 3812 m/s, below the true deep section. `sweep datasets list` shows the
full catalogue.

## Checking your own run

Every reference number on these pages is reproducible: `seed` is set
explicitly in each YAML, the models come from the embedded dataset, and two
runs of the same file on the same machine come out **bit-identical**
(verified — matching misfit to every digit, max model difference 0.0 m/s).

Each run directory is self-describing:

```
sweep_runs/<task_id>/
  config_resolved.yaml     every default filled in — the exact spec that ran
  task_source.yaml         the task file as written (comments kept)
  run_meta.json            host, CUDA, package versions of the code that ran,
                           git state, labels, inputs, the Slurm job
  status.json              state, times, artifacts (also relative to the run dir)
  manifest.json            every file of the run dir at the end, with its size
  logs/                    the run's stdout / stderr
  launch/slurm_<job>.sh    the batch script of each Slurm job (a resumed run keeps all)
  provenance/<pkg>.diff    uncommitted changes of a dirty checkout
  output/initial_vp.npy    the resolved STARTING model
  output/inverted_vp.npy   the result
  output/loss.npy          misfit per epoch
  output/epochs/           per-`show_every` model snapshots
  qc/                      vp, vp_diff, shot_gather, loss_curve figures
```

The run dir keeps what it takes to read it once it is moved away from where
it was launched (archived): the task file, the logs, the batch script, where
each input really was (links resolved) with its size and a fingerprint, and
the labels it ran under -- set `SWEEP_LABEL_<KEY>=value` in the environment
(e.g. `SWEEP_LABEL_LINE=<experiment line>`, `SWEEP_LABEL_PROJECT=<project>`)
and `run_meta.json` records them as `labels: {line: ..., project: ...}`.
A resumed run appends to `logs/` and lists the launches before it in
`previous_launches`. A variable named like a credential (`*KEY*`, `*TOKEN*`,
`*SECRET*`, `*PASSW*`) is recorded as set, not with its value, also in the
copied batch script. `SWEEP_RUN_LOG=0` stops the copy of stdout / stderr to `logs/`.

`output/initial_vp.npy` matters when the start is built in memory
(`dataset:` or `linear_gradient:`): there is no input file to point at
afterwards, so the runner writes the resolved array before training begins.

**Scope of "reproducible":** bit-identical is a same-machine, same-build
claim. A different GPU model or a rebuilt CUDA extension can reorder floating
point atomics, so expect last-digit differences there. The numbers should still
land within a fraction of a percent, and none of the conclusions move.

## Start your own

The examples are worked cases. For your own task, start from the bundled
annotated template rather than by copying an example:

```bash
sweep-tasks init --list                       # every template
sweep-tasks init fwi -o my_fwi.yaml           # annotated FWI reference
sweep-tasks run my_fwi.yaml                   # runs as written
```

Every task template (`introspect`, `forward`, `wavefield`, `fwi`, `rtm`,
`lsrtm`) runs as it comes out of `init`: a small Marmousi case built from
`sweep.datasets`, nothing to download and no variable to set. Each documents
every key inline, the field-data alternatives (`from_plan` geometry,
`obs.plan`, an estimated wavelet) included. On one V100 the `fwi` template
(a three-stage multiscale run) takes about 5 minutes, `lsrtm` 4, `rtm` 2 and
the others under a minute. `viking` is not a task: it is the pipeline config
for `build-index` / `build-plan` and the wavelet tools.
