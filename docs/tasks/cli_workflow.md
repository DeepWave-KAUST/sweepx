# sweep-tasks CLI workflow — SEG-Y to FWI in three steps

This document is the canonical reference for the unified data pipeline.
The same three-step flow drives 2-D streamer (Viking-style) and 3-D OBN
(production-OBN-style) inversions — the only thing that changes is the
`--grouping` flag at plan-build time.

There is one branch off it, documented below: frequency-selection encoding
reduces the data to DTFT coefficients ONCE (`extract-coeff`) and then inverts
with no SEG-Y I/O at all, at the cost of needing the acquisition to be node-like.

```
┌──────────────────────────────────────────────────────────────────────┐
│  STEP 1  ▸  sweep-tasks build-index                                  │
│            scan SEG-Y trace headers → SEGYIndex npz                  │
│            (per-trace catalog; no sample data; one-time per dataset) │
├──────────────────────────────────────────────────────────────────────┤
│  STEP 2  ▸  sweep-tasks build-plan                                   │
│            SEGYIndex npz + grouping + filters → SeismicPlan npz      │
│            (cheap, repeatable; declares WHICH traces FWI consumes)   │
├──────────────────────────────────────────────────────────────────────┤
│  STEP 3  ▸  sweep-tasks run <fwi.yaml>                               │
│            YAML's `geometry: from_plan` + `obs: { plan: ... }`       │
│            FWI reads trace bytes lazily via PlanReader               │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │  ALTERNATIVE for OBN, from STEP 2 on:
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  BRANCH  ▸  sweep-tasks extract-coeff                                │
│            common-node gathers → DTFT coefficients on a comb         │
│            (one complex number per trace per frequency)              │
├──────────────────────────────────────────────────────────────────────┤
│          ▸  sweep-tasks run <fwi.yaml>                               │
│            `source_encoding.mode: frequency_selection`               │
│            reads ONLY the shards — zero SEG-Y I/O per iteration      │
└──────────────────────────────────────────────────────────────────────┘
```

Behind these CLI commands sit the modular sweep-io primitives:

```
SEG-Y bytes
    │  sweep_io.segy_index.build_segy_index        (parallel scan)
    ▼
sweep_io.segy_index.SEGYIndex                       ← STEP 1 output
    │  sweep_io.seismic_plan.build_seismic_plan    (filter + group)
    ▼
sweep_io.seismic_plan.SeismicPlan                   ← STEP 2 output
    │  sweep_io.seismic_plan.PlanReader            (lazy SEG-Y read)
    ▼
np.ndarray  (n_in_group, nt) float32                ← consumed by STEP 3
```

## STEP 1 — `sweep-tasks build-index`

Scans SEG-Y trace headers in parallel (threads or MPI) and saves a
**lossless** per-trace catalog as npz. Run this **once per dataset**;
the same index serves any number of plans.

### Minimum invocation (single-process, threaded)

```bash
sweep-tasks build-index \
    --segy-root /path/to/segy_dir \
    --glob '*.sgy' \
    --num-workers 16 \
    --out /path/to/dataset_index.npz
```

### Production (multi-rank MPI for large surveys)

```bash
# Submit via sbatch with `--ntasks=50 --cpus-per-task=1 --mem=192G`:
sweep-tasks build-index -n "$SLURM_NTASKS" \
    --segy-root /path/to/segy_dir \
    --glob '*.sgy' \
    --out /path/to/dataset_index.npz \
    --source-depth-m 9.0 \
    --receiver-z-byte 40
```

`-n N` is the convenience shortcut for `mpiexec -n N sweep-tasks
build-index --mpi …` — it re-execs the same command under `mpiexec`
internally and adds `--mpi` automatically, so no explicit `mpiexec`
wrapper is needed. The explicit `mpiexec -n N … --mpi` form still
works for environments where you want to launch ranks yourself.

### Per-dataset header overrides

| Flag | What it sets | When to use |
|---|---|---|
| `--source-depth-m FLOAT` | Constant source depth override (m) | Marine source above water (airgun, streamer) |
| `--receiver-depth-m FLOAT` | Constant receiver depth override (m) | When SEG-Y depth bytes are unreliable |
| `--receiver-z-byte INT` | 0-based byte for gz | OBN nodes whose depth lives in non-standard byte |
| `--source-z-byte INT` | 0-based byte for source-z | Same idea for source side |
| `--byte-map KEY=BYTE` | Override any rev1 standard byte | Custom acquisition formats |
| `--coord-scalar FLOAT` | Override SEG-Y coord scalar | When per-trace scalar is wrong |
| `--files-from PATH` | Whitelist file (one path per line) | Restrict to a curated subset |

### Output

Small (typically 0.1-1 % of the SEG-Y volume): for Viking 1001-shot 2-D
that's ~300 KB; a large 3-D OBN survey's index runs to tens or
hundreds of MB.

## STEP 2 — `sweep-tasks build-plan`

Loads the index, applies filters, and groups rows by either common-shot
(CSG) or common-receiver (CRG). Saves a `seismic_plan_v1` npz that
declares **exactly which trace bytes FWI will read**. Plans are cheap to
rebuild — re-run any time you want a different AOI / offset window /
shot subset.

### CSG (2-D / 3-D streamer)

```bash
sweep-tasks build-plan \
    --index /path/to/dataset_index.npz \
    --grouping csg \
    --out /path/to/csg_plan.npz
```

With filters:
```bash
sweep-tasks build-plan \
    --index /path/to/dataset_index.npz \
    --grouping csg \
    --shot-ids 0,100,200,300 \
    --offset-min-m 100 \
    --offset-max-m 5000 \
    --max-traces-per-group 100 \
    --out /path/to/csg_plan_filtered.npz \
    --label "near-offset 100m only"
```

### CRG (3-D OBN)

`--grouping crg` requires `--receiver-quantize-m` to define the cell
size that collapses physical receivers into virtual sources.

```bash
sweep-tasks build-plan \
    --index /path/to/obn3d_index.npz \
    --grouping crg \
    --receiver-quantize-m 0.5 \
    --out /path/to/obn3d_crg_plan.npz
```

CRG plans go through the same `geometry.kind: from_plan` + `obs.plan:`
YAML as CSG; the only OBN-specific thing is the `obs.plan.sampling`
sub-block (`PlanSamplingConfig`) that enables the per-iter
plan-streaming supershot loop with ±1 source encoding. The runner auto-
detects this and dispatches to the dedicated plan-streaming training loop;
no separate `from_crg_plan` / `obs.crg_plan` / `build-crg-plan` symbols
exist anymore. See `src/sweep_tasks/templates/fwi.yaml` for the canonical
shape.

### Filter knobs

| Flag | Effect |
|---|---|
| `--shot-ids "1,2,3"` | Keep only these shot FFIDs |
| `--offset-min-m FLOAT` | Drop traces with offset < FLOAT meters |
| `--offset-max-m FLOAT` | Drop traces with offset > FLOAT meters |
| `--max-traces-per-group INT` | Cap rows per group (random sub-sample) |
| `--receiver-quantize-m FLOAT` | (CRG only) receiver-cell tolerance |
| `--seed INT` | RNG seed for the per-group cap sub-sampling |
| `--label STR` | Provenance tag stored in `plan.build_meta` |

### Output

Bigger than the index (must store per-row file_id + byte_offset + xyz)
but still small relative to the SEG-Y: Viking full CSG ~8 MB; a large
3-D CRG plan can reach several GB.

## STEP 3 — `sweep-tasks run` with `from_plan`

The FWI YAML now references the plan directly. Geometry resolver builds
sources / receivers from `plan.row_*_xyz`, obs is materialised via
`PlanReader` (cached in RAM for 2-D, lazy per-iter for 3-D).

```yaml
# minimal Viking-like FWI YAML
task_type: fwi
output_dir: /path/to/runs
task_id: my_fwi_run

grid: {dh: 12.5, shape: [401, 2305]}
time: {dt: 0.001, nt: 6000}

geometry:
  kind: from_plan
  plan_path: /path/to/csg_plan.npz
  dedupe: true                 # per-stage grid-snap dedupe (typical streamer)

obs:
  plan:
    plan_path: /path/to/csg_plan.npz   # usually same file as geometry.plan_path
    cache_all: true            # 2-D: load to RAM once; 3-D: leave False

wavelet:    {kind: siren_pipeline_npz, path: ..., scale: 1.0}
init_model: {name: vp, path: /path/to/vp_init.npy}
# ...physics, backend, optimizer, loss, reparam, stages... (see benchmark YAML)
```

Submit with the usual sbatch:

```bash
ssh glogin.ibex.kaust.edu.sa
sbatch your_fwi.sbatch
```

## End-to-end example (Viking 2-D)

Quick summary below assuming **one local GPU, no SLURM**. The full
walkthrough — including the public S3 download URLs, SEG-Y header
specs, the legacy wavelet / initial-model bridge, the post-filter
recipe, and an optional appendix on SLURM / ibex sbatch entry points —
lives in [`docs/datasets/viking/README.md`](datasets/viking/README.md).

```bash
# Pick a writable work folder (any local path).
export VIKING_HOME=$HOME/viking
mkdir -p $VIKING_HOME/{raw,run}

# Step 0: one-time download of seismic.segy + Farfield.dat (~1.4 GB).
# Full URLs in docs/datasets/viking/README.md § Step 0.

# Step 1: scan SEG-Y → index (~10 s for Viking 1001 shots)
sweep-tasks build-index \
    --segy-root $VIKING_HOME/raw \
    --glob 'seismic.segy' \
    --out $VIKING_HOME/run/viking_index.npz \
    --num-workers 8 --progress \
    --source-depth-m 6.0 --receiver-depth-m 10.0

# Step 2: build CSG plan with no filter (= use all 1001 shots × 120 receivers)
sweep-tasks build-plan \
    --index $VIKING_HOME/run/viking_index.npz \
    --grouping csg \
    --out $VIKING_HOME/run/viking_csg_plan.npz \
    --label "viking_full_csg"

# Step 3 (legacy bridge, until ported): produce the SIREN-pipeline
# wavelet and the 12.5 m initial vp via fwi_workflow-dev.
#   fwi analyze-wavelet -d viking
#   fwi estimate-wavelet -d viking
#   fwi build-initial-model -d viking

# Step 4 — single-GPU FWI (6-stage production, ~1 h on one V100).
# Copy the example YAML and edit the wavelet / plan / init_model paths in it.
sweep-tasks run $VIKING_HOME/run/viking_6stage.yaml
# (multi-GPU on the same box: append --nproc-per-node N)

# Step 5 — RTM + depth-tapered z-low-cut on the inverted vp.
sweep-tasks run $VIKING_HOME/run/viking_rtm.yaml
sweep-tasks filter-image \
    $VIKING_HOME/run/viking_rtm/viking_rtm_v1/output/rtm_image_per_shot_normalised.npy \
    --wavelength-m 300 --depth-m 600 --taper-m 400
```

`sweep-tasks run` defaults to one local rank (in-process). For
multi-GPU on the same box pass `--nproc-per-node N`; sweep-tasks
re-execs itself under `torchrun --standalone --nproc_per_node=N`
automatically when needed.

For SLURM / ibex submission, see
[§ "Running on a cluster"](datasets/viking/README.md#running-on-a-cluster-optional)
in the Viking walkthrough — same `sweep-tasks run` invocations,
just wrapped in `sbatch` templates that consume `SWEEP_*_ROOT` /
`PROJECT_DATA_ROOT` environment variables.

## Branch — frequency-selection encoding (no per-iteration I/O)

### `sweep-tasks extract-coeff`

Steps 1–3 stream trace bytes on every iteration. The frequency-selection path
does the data reduction ONCE instead: it turns each common-node gather into a
handful of DTFT coefficients, and the inversion never opens the SEG-Y again.

```bash
sweep-tasks extract-coeff <forward_run_dir> \
    --n-p 24000 --k-lo 24 --k-hi 72 -o coeff_1_3hz.npz
```

The positional argument is a directory holding `output/record.npy`,
`sources.npy` and `receivers.npy` — the layout a `task_type: forward` run
leaves behind, with the NODES as its sources (reciprocity). `--dt` defaults to
`time.dt` read out of that run's `config_resolved.yaml`.

For real data, skip the CLI and call the library directly with your own arrays,
which is the same code path:

```python
from sweep_tasks.freqsel import FrequencyComb, extract_shard
import numpy as np

comb = FrequencyComb(dt=0.004, n_p=8000, ks=np.arange(64, 129))   # 1-2 Hz
extract_shard("coeff.npz", record, node_grid_xyz, trace_grid_xyz, comb)
```

`record` is `(n_nodes, nt, n_rec)`, the grids are integer grid indices — `(x, z)`
in 2-D, `(x, y, z)` in 3-D. Absolute scale and time origin do not matter: the
inversion's misfit is invariant to a per-(node, bin) complex factor, which is
exactly what a source spectrum, an excitation delay or a coupling constant
contribute.

#### Sizing the comb

Two relations govern every flag:

```
bins = bandwidth × window length          (window = n_p × dt)
pool size ≤ bins                          (hard constraint)
```

The pool is how many nodes fire simultaneously in ONE forward, so **comb bins
are the currency, not shots** — extra sources ride the same forward for free,
and the way to fire more of them is to lengthen the analysis window. The
consequence is worth internalising: the window SHORTENS as a frequency ladder
climbs, because a wider band reaches the same bin count in less time. The
expensive rung is the lowest one.

Extraction is a DTFT of a record that already exists, so **one recording feeds
an entire ladder** — extract once per rung, which costs seconds. The `--n-p`
given here MUST equal `frequency.probe_samples` in the FWI YAML; the runner
rejects a shard whose `n_p`, `dt` or `ks` disagree with the stage reading it.

#### Output

An npz with `node_ids`, `node_grid_xyz`, `node_ptr`, `D` (complex64,
`n_items × n_bins`), `fold`, `trace_grid_xyz`, `freqs`, `ks`, `meta`. Tiny
next to the SEG-Y: 100 nodes × 677 traces × 101 bins is ~55 MB.

### `sweep-tasks run` with `frequency_selection`

```yaml
task_type: fwi
# no `wavelet:`, no `geometry:`, no `obs:` — the schema makes all three
# optional in this mode. Obs is the shard, the geometry rides inside it, and
# the source spectrum cancels in the misfit.
source_encoding:
  enabled: true
  mode: frequency_selection
  frequency:
    coeff_shards: ./runs/coeff_1_3hz.npz   # glob; resolved RELATIVE TO THIS FILE
    probe_samples: 24000
    k_lo: 24                               # 1.0 Hz at dt = 1 ms
    k_hi: 72                               # 3.0 Hz
    steady_samples: 10000                  # ring-up before the analysis window
    slack_samples: 500
    n_pools: 1                             # 1 = fire every node each iteration
```

`steady_samples` is dead time while the continuous sources ring up. The runner
prints a two-window check at every stage entry — it extracts the coefficients
twice, `slack_samples` apart, and reports the median relative difference.
**Under 1e-2 means the window really is steady**; raise `steady_samples` if it
is not.

Two things that cost real runs to learn:

* **Size the PML by the longest wavelength in the ladder.** At 1 Hz in 1500 m/s
  the wavelength is ~1500 m, and the default `abcn: 20` on a 12.5 m grid is
  only 250 m — λ/6. A continuous-wave field in a leaky box builds standing
  waves, and unlike a transient that does *not* cancel between a transient obs
  and a steady-state syn.
* **`freeze_top_n_rows` is rejected in this mode**, not ignored. The water is
  pinned by VALUE instead: every cell whose starting value is exactly the water
  velocity (1500 m/s, or `reparam.water_vp_m_s`) has its gradient zeroed.

A worked synthetic end-to-end — forward, three extractions, three-rung
inversion, with measured numbers — is examples 11 + 12 (2-D) and 13 + 14 (3-D),
walked through in [`docs/datasets/marmousi/README.md`](datasets/marmousi/README.md).

## Post-processing — depth-tapered z-axis low-cut on RTM images

A stacked RTM (or FWI gradient) image typically has a slowly-varying
"shallow drift" — energy that comes from the source / receiver
half-space rather than reflectivity — which masks the deep structure on
default colour scales. The standard fix is a depth-tapered z-axis
low-cut filter: low-pass the image along z, then subtract that low
content with a cosine-tapered weight that ramps from full-strength near
the surface to zero below a target depth. This is the port of the
legacy `fwi_workflow.imaging.filter_imaging`
(`scripts/07_filter_imaging.py`) into sweep-tasks.

Two equivalent routes are provided — they share the same algorithm and
the same `<input_stem>_shallow_zlowcut.npy` output naming, so anything
produced by route B can be re-run via route A with different parameters
without redoing the underlying RTM.

### Route A — standalone CLI (`sweep-tasks filter-image`)

Use this when the RTM run already finished and you want to iterate on
filter parameters without paying the GPU cost again.

```bash
# Default (matches the legacy Viking recipe):
sweep-tasks filter-image \
    /path/to/runs/my_rtm/output/rtm_image_per_shot_normalised.npy \
    --dz-m 12.5
# -> writes:
#    rtm_image_per_shot_normalised_shallow_zlowcut.npy
#    rtm_image_per_shot_normalised_shallow_zlowcut_removed.npy
#    rtm_image_per_shot_normalised_shallow_zlowcut_z_taper.npy
#    rtm_image_per_shot_normalised_shallow_zlowcut.png
#    rtm_image_per_shot_normalised_shallow_zlowcut_comparison.png
#    rtm_image_per_shot_normalised_shallow_zlowcut_metadata.json
```

`--dz-m` is auto-inferred from a sibling `rtm_result.npz` (the RTM
runner writes the grid `dh` there), so for sweep-tasks-produced images
the bare `sweep-tasks filter-image <npy>` works.

Full knob surface (all optional except the input + `--dz-m` fallback):

| Flag | Default | Effect |
|---|---|---|
| `--dz-m FLOAT` | (from `rtm_result.npz`) | Vertical grid spacing (m) |
| `--dx-m FLOAT` | `=--dz-m` | Horizontal grid spacing (m) — PNG axes only |
| `--wavelength-m FLOAT` | `300` | z-direction wavelength threshold for "slow drift" |
| `--depth-m FLOAT` | `600` | Filter is full-strength in `0..depth_m` (m) |
| `--taper-m FLOAT` | `400` | Cosine ramp full-strength → 0 across `depth_m..depth_m+taper_m` |
| `--clip-percentile FLOAT` | `1.0` | Signed percentile for PNG colour clip |
| `--display-scale FLOAT` | `1.0` | Scale factor on vmin/vmax half-width |
| `--cmap STR` | `sweep_image` | Matplotlib colormap (bundled `sweep_viz` diverging LUT; pass `seismic` / `gray` for legacy display) |
| `--output-dir PATH` | (alongside input) | Where to write outputs |
| `--output-name STR` | `<input_stem>_shallow_zlowcut` | Override the output stem |
| `--x-origin-m FLOAT` | `0` | x-axis origin (m) for PNG extent |
| `--z-origin-m FLOAT` | `0` | z-axis origin (m) for PNG extent |
| `--x-max-m FLOAT` | (full width) | Crop PNG x-axis to `[x_origin, x_max]` |
| `--no-png` | (off) | Skip PNG generation (npy + metadata only) |

### Route B — RTM YAML `post_filter` block (auto-apply)

Use this when you already know roughly what filter you want and want
the imaging products + filtered products written in a single run. Add a
`post_filter:` block to `imaging:` in any RTM task YAML — the runner
applies the same algorithm to every saved imaging product right after
the main RTM finishes (rank 0 only).

```yaml
task_type: rtm
# ... grid / time / wavelet / geometry / obs / physics / velocity_model ...
imaging:
  shots_per_batch: 1
  filter_lowcut_hz: 3.0
  filter_highcut_hz: 25.0
  filter_target: syn
  normalize_by_illumination: true
  save_per_shot: false
  live_update_every_batches: 10
  loss_kind: mse
  post_filter:
    enabled: true             # set false to skip without removing the block
    wavelength_m: 300.0       # same defaults as `sweep-tasks filter-image`
    depth_m: 600.0
    taper_m: 400.0
    clip_percentile: 1.0
    display_scale: 1.0
    cmap: sweep_image           # bundled sweep_viz LUT; pass "seismic" for legacy red/blue
    # Which imaging products to filter. "all" (default) covers all six:
    #   fwi_gradient_image                 rtm_image
    #   fwi_gradient_image_normalised      rtm_image_normalised
    #   fwi_gradient_image_per_shot_normalised  rtm_image_per_shot_normalised
    # Pass an explicit list of basenames (without ".npy") to be selective.
    targets: all
    save_png: true
```

When enabled, the runner adds these files per target to `artifacts`:

```
<output>/<stem>_shallow_zlowcut.npy           # cleaned image
<output>/<stem>_shallow_zlowcut_removed.npy   # subtracted shallow drift
<output>/<stem>_shallow_zlowcut_z_taper.npy   # per-z taper weights
<output>/<stem>_shallow_zlowcut.png           # PNG (optional)
<output>/<stem>_shallow_zlowcut_comparison.png # 3-panel before/after/removed
<output>/<stem>_shallow_zlowcut_metadata.json # exact knobs + shape
```

### When to use which

| Situation | Route |
|---|---|
| Just finished RTM, defaults look right | B (rerun RTM with `post_filter` to get filtered products inline) |
| Already finished RTM, want to try new wavelength / depth | A (cheap; no GPU) |
| Sweeping filter params for a paper figure | A (loop bash over `--wavelength-m`) |
| Production batch jobs with a known recipe | B (single sbatch produces raw + filtered) |
| QC during dev (1 input, dozens of param combos) | A |

## Cheat sheet

| Need | Command |
|---|---|
| First-time scan a survey | `sweep-tasks build-index --segy-root ... --out idx.npz` |
| Use all data, default grouping | `sweep-tasks build-plan --index idx.npz --grouping csg --out plan.npz` |
| Restrict to a shot subset | `... --shot-ids 0,1,2,...` |
| Drop far-offset noise | `... --offset-max-m 5000` |
| Random sub-sample per group | `... --max-traces-per-group 100 --seed 42` |
| OBN receiver-gather build | `sweep-tasks build-plan --index idx.npz --grouping crg --receiver-quantize-m 0.5 --out crg_plan.npz` |
| OBN plan-streaming supershot FWI | YAML: `geometry.kind: from_plan` (with `rotation_metadata`) + `obs.plan.sampling.shared_shots_per_iter > 0` |
| Inspect a plan | `python -c "from sweep_io.seismic_plan import SeismicPlan; p=SeismicPlan.load('plan.npz'); print(p.grouping, p.n_groups, p.n_rows, p.build_meta)"` |
| Run FWI on a plan | `sweep-tasks run fwi.yaml` (with `geometry.kind=from_plan` + `obs.plan.plan_path`) |
| Extract freqsel coefficients | `sweep-tasks extract-coeff <forward_run_dir> --n-p 24000 --k-lo 24 --k-hi 72 -o coeff.npz` |
| Wavelet-free encoded FWI | YAML: `source_encoding: {enabled: true, mode: frequency_selection, frequency: {coeff_shards: ...}}` |
| Ask the build what it can solve | `sweep-tasks run examples/synthetic/15_introspect_equations.yaml` |
| Post-filter a saved RTM npy | `sweep-tasks filter-image runs/.../rtm_image_per_shot_normalised.npy` |
| Bake the post-filter into the RTM run | Add `imaging.post_filter: {enabled: true}` to the RTM YAML |

## Why three steps (not one)

The first time someone runs FWI on a new dataset they often want to:

1. Try the full dataset (no filter)
2. Look at QC, decide "this offset is mostly noise / these shots are
   junk / I only need the middle 200 shots for a near-surface test"
3. Re-run with the narrower scope
4. Iterate on FWI hyperparameters (lr, wavelet, stages) without
   re-scanning SEG-Y

Step 1 (the slow part — header parse on every trace) happens **once**.
Step 2 is seconds. Step 3 is the iteration loop you actually care about.

With the legacy single-step pipeline every FWI submission re-scanned
the entire SEG-Y, which is why Viking startup dominated wall time
for short FWI runs.
