# Examples

Every example is one YAML file under `examples/synthetic/`, run with one
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
| [Viking (field data)](../datasets/viking/README.md) | `examples/field/viking/` | real SEG-Y: build-index → wavelet → FWI → RTM |

New to task YAMLs? Read [09](forward.md#09-the-smallest-task-there-is) first:
it annotates every field. On a new machine, run [15](introspect.md) first: it
tells you which equations and backends your build has.

## Before you start

The examples default to `backend.impl: c` — sweep's fused CUDA kernels — so a
CUDA GPU is recommended. On a CPU-only machine set `backend.impl: eager` (or
pass `--override backend.impl=eager`) and expect roughly 30× the wall time.
09, 10, 16 and 17 already run on `eager`.

Relative paths in a task YAML resolve against the YAML **file**, not your
shell's working directory, so every run lands in
`examples/synthetic/sweep_runs/<task_id>/` (gitignored). Pass
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
  run_meta.json            host, CUDA, package versions, git state
  output/initial_vp.npy    the resolved STARTING model
  output/inverted_vp.npy   the result
  output/loss.npy          misfit per epoch
  output/epochs/           per-`show_every` model snapshots
  qc/                      vp, vp_diff, shot_gather, loss_curve figures
```

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
sweep-tasks run my_fwi.yaml
```
