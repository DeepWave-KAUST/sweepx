# sweep-tasks examples

Everything here is a YAML file run by the CLI. There are no helper scripts to
read first and nothing to prepare:

```bash
sweep-tasks run examples/synthetic/01_forward_marmousi.yaml
```

```
examples/
  synthetic/   nothing to download — models come from sweep.datasets
  field/       real SEG-Y you supply (or fetch, for the public Viking set)
  sbatch/      SLURM wrappers for the long runs
```

---

## Start your own: `sweep-tasks init`

The examples are worked cases. When you want your own, start from the bundled
annotated template rather than by copying an example:

```bash
sweep-tasks init --list                       # every template
sweep-tasks init fwi -o my_fwi.yaml           # annotated FWI reference
sweep-tasks run my_fwi.yaml
```

There is a template for every task type, and each one documents every key
inline — what it does, what the alternatives are, what the default means:

| `sweep-tasks init <name>` | what you get |
|---|---|
| `introspect` | ask the build what equations/models/backends it has. Run this first on a new machine. |
| `forward` | synthesise a shot record |
| `wavefield` | forward + pressure-field snapshots |
| `fwi` | full waveform inversion — the largest template, covers stages, reparam, QC |
| `rtm` | reverse-time migration |
| `lsrtm` | least-squares RTM |
| `viking` | *not* a task spec — the pipeline config for `build-index` / `build-plan` |

`sweep-tasks init` and `sweep-tasks new` read the same files, so the two can
never drift apart.

---

## synthetic/ — runnable as-is

Models are named in the YAML with `ModelRef.dataset`, so there is no `.npy` to
prepare and no environment variable to set. `marmousi:2d-demo` and
`overthrust:2d-demo` are embedded in the `sweep` package (zero network);
`sweep datasets list` shows the whole catalogue.

Read in order — 01 → 03 build on each other and share one grid, wavelet and
geometry so they are directly comparable:

| example | teaches |
|---|---|
| [`01_forward_marmousi`](synthetic/01_forward_marmousi.yaml) | task-YAML anatomy; `models:` straight from a dataset |
| [`02_fwi_marmousi_single`](synthetic/02_fwi_marmousi_single.yaml) | FWI shape, `obs.synthetic_from`, boundary-saving adjoint, per-epoch QC |
| [`03_fwi_marmousi_multiscale`](synthetic/03_fwi_marmousi_multiscale.yaml) | `stages:` — per-stage bandpass + `lr_scale`, and why a 1-D start needs them |
| [`04_rtm_marmousi`](synthetic/04_rtm_marmousi.yaml) | imaging at fixed velocity; illumination normalisation and the depth-tapered post-filter |
| [`05_lsrtm_marmousi`](synthetic/05_lsrtm_marmousi.yaml) | inverting reflectivity instead of velocity — where LSRTM sits between RTM and FWI |
| [`06_wavefield_snapshots`](synthetic/06_wavefield_snapshots.yaml) | `task_type: wavefield`; flip `free_surface` and watch the ghost and multiples appear |
| [`07_fwi_marmousi_inr`](synthetic/07_fwi_marmousi_inr.yaml) | `reparam:` — the model as a coordinate network (SIREN + hash encoding) |
| [`08_fwi_marmousi_backends`](synthetic/08_fwi_marmousi_backends.yaml) | `backend:` — the adjoint-memory ladder (full / boundary / ckpt), CPU eager, `torch.compile` |
| [`09_forward_constant_box`](synthetic/09_forward_constant_box.yaml) | the smallest possible task: a constant-velocity box |
| [`10_forward_explicit_geometry`](synthetic/10_forward_explicit_geometry.yaml) | `geometry.kind: explicit` — hand-placed sources and receivers |
| [`11_forward_freqsel_nodes`](synthetic/11_forward_freqsel_nodes.yaml) | OBN reciprocity: nodes as sources, so `extract-coeff` has gathers to read |
| [`12_fwi_marmousi_freqsel`](synthetic/12_fwi_marmousi_freqsel.yaml) | `source_encoding.mode: frequency_selection` — a whole node pool per forward, wavelet-free |
| [`13_forward_freqsel_nodes_3d`](synthetic/13_forward_freqsel_nodes_3d.yaml) | `geometry.kind: grid` — a 3-D source/receiver patch without 1089 YAML lines |
| [`14_fwi_overthrust_freqsel_3d`](synthetic/14_fwi_overthrust_freqsel_3d.yaml) | the same encoding in 3-D, where it changes what is affordable |
| [`15_introspect_equations`](synthetic/15_introspect_equations.yaml) | `task_type: introspect` — ask the INSTALLED solver what it registers, and whether the fused binding is compiled |
| [`16_forward_anisotropic`](synthetic/16_forward_anisotropic.yaml) | a multi-parameter equation (`AcousticVTI`): three models, and moveout that depends on direction |
| [`17_forward_elastic`](synthetic/17_forward_elastic.yaml) | the elastic equation: P and S in one record, and why `pml_type` has to match the equation's grid |

11 → 14 are two pairs, and each pair takes three commands rather than one:
record the node gathers, turn them into coefficients with `sweep-tasks
extract-coeff`, then invert. The headers of 11 (2-D) and 13 (3-D) spell out the
exact commands. In a field project the first command disappears — real node
gathers replace it and `extract-coeff` reads those instead.

Wall times are not filled in yet; they get measured and written into each YAML
header once the set has been run on a known GPU.

**No GPU?** The examples default to `backend.impl: c` (fused CUDA kernels).
Switch to `impl: eager` to run on CPU and expect roughly 30× the wall time —
08 explains the trade in full.

**Multiple GPUs?** Shot-parallel is a flag, not a different file:

```bash
sweep-tasks run examples/synthetic/02_fwi_marmousi_single.yaml --nproc-per-node 4
```

---

## field/ — real data

| example | data |
|---|---|
| [`field/viking/`](field/viking/) | Mobil AVO Viking Graben — public, ~1.4 GB SEG-Y. Full walkthrough in [`docs/datasets/viking/README.md`](../docs/datasets/viking/README.md) |

These read paths from environment variables (`$VIKING_HOME`,
`$SWEEP_RUNS_ROOT`) rather than hardcoding anyone's directory layout.
`pipeline_config.yaml` is not a task spec — it drives the `build-index` /
`build-plan` CLI stage that turns raw SEG-Y into a `SeismicPlan`.

---

## Adding an example

Two tests keep this directory honest, both globbed rather than hardcoded:

- `test_shipped_example_yaml_validates` schema-validates every
  `examples/**/*.yaml` carrying a `task_type`.
- `test_init_template_validates` round-trips every `sweep-tasks init` template.

So a new example or template is covered the moment you add it, and a stale one
goes red instead of rotting quietly. Pipeline configs (no `task_type`) are
skipped.
