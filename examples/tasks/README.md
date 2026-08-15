# sweep-tasks examples

Every example is one YAML plus one command:

```bash
sweep-tasks run examples/<path>.yaml
```

The tree splits by what an example needs from you:

```
examples/
  synthetic/   nothing to download — models come from sweep.datasets
  field/       real SEG-Y you supply (or fetch, for the public Viking set)
  sbatch/      SLURM wrappers for the long runs
```

---

## synthetic/ — runnable as-is

Models are named in the YAML via `ModelRef.dataset`, so there is no `.npy` to
prepare and no environment variable to set. `marmousi:2d-demo` and
`overthrust:2d-demo` are embedded in the `sweep` package (zero network);
`sweep datasets list` shows the whole catalogue.

| example | teaches | wall time |
|---|---|---|
| [`01_forward_marmousi.yaml`](synthetic/01_forward_marmousi.yaml) | task-YAML anatomy; `models:` from a dataset | see note |
| [`02_fwi_marmousi_single.yaml`](synthetic/02_fwi_marmousi_single.yaml) | FWI shape, `obs.synthetic_from`, boundary-saving adjoint, per-epoch QC | see note |
| [`03_fwi_marmousi_multiscale.yaml`](synthetic/03_fwi_marmousi_multiscale.yaml) | `stages:` — per-stage bandpass + `lr_scale`, and why a 1-D start needs them | see note |
| [`forward_acoustic_constant.yaml`](synthetic/forward_acoustic_constant.yaml) | the smallest possible task: a constant-velocity box | seconds |
| [`forward_explicit_geometry.yaml`](synthetic/forward_explicit_geometry.yaml) | `geometry.kind: explicit` — hand-placed sources/receivers | seconds |
| [`wavefield_free_surface.yaml`](synthetic/wavefield_free_surface.yaml) | `task_type: wavefield` — snapshots, and what a free surface does to them | seconds |

Wall times are not filled in yet — they get measured and written into each
YAML header once the examples are run on a known GPU.

01 → 02 → 03 share one grid, wavelet and geometry on purpose, so the three
are directly comparable.

**No GPU?** Every example defaults to `backend.impl: c` (fused CUDA kernels).
Switch to `impl: eager` to run on CPU and expect roughly 30× the wall time.

---

## field/ — real data

| example | data |
|---|---|
| [`field/viking/`](field/viking/) | Mobil AVO Viking Graben — public, ~1.4 GB SEG-Y. Full walkthrough in [`docs/datasets/viking/README.md`](../docs/datasets/viking/README.md) |

The Viking YAMLs read paths from environment variables (`$VIKING_HOME`,
`$SWEEP_RUNS_ROOT`) rather than hardcoding anyone's directory layout.
`pipeline_config.yaml` is not a task spec — it drives the `build-index` /
`build-plan` CLI stage that turns SEG-Y into a `SeismicPlan`.

---

## Other files

- [`quickstart.py`](quickstart.py) — the same idea from Python instead of YAML.
- [`cli_walkthrough.py`](cli_walkthrough.py) — drives every CLI subcommand
  end-to-end in one script.
- [`sbatch/`](sbatch/) — SLURM submission wrappers for the multi-hour runs.

---

## Adding an example

`tests/test_task_schemas.py::test_shipped_example_yaml_validates` globs
`examples/**/*.yaml` and schema-validates everything carrying a `task_type`,
so a new example is covered the moment you add it — and a stale one goes red
instead of rotting quietly. Pipeline configs (no `task_type`) are skipped.
