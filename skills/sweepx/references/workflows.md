# Workflows

Task recipes beyond the minimal forward run of SKILL.md. Every runnable block
here was executed against sweepx 0.4.1; the field list of every block is in
[spec-schema.md](spec-schema.md).

## Contents

- [Task types and required keys](#task-types-and-required-keys)
- [Single-band FWI and the Python API](#single-band-fwi-and-the-python-api) (runnable)
- [Run directory](#run-directory)
- [Multiscale stages](#multiscale-stages) (runnable, with a water layer)
- [The water layer](#the-water-layer)
- [Wavefield snapshots](#wavefield-snapshots) (runnable)
- [RTM and LSRTM](#rtm-and-lsrtm) (LSRTM runnable)
- [Field data: SEG-Y → plan → FWI](#field-data-seg-y--plan--fwi) (runnable up to validation)
- [Frequency-selection FWI](#frequency-selection-fwi) (runnable)
- [Multi-GPU, domain decomposition, resume](#multi-gpu-domain-decomposition-resume)
- [What the validator refuses](#what-the-validator-refuses)

## Task types and required keys

| `task_type` | required keys (besides `task_type`) | notes |
|---|---|---|
| `introspect` | `action` (`list_equations`, `describe_equation`, `describe_field`, `describe_model`, `list_supported_bindings`); `target` for `describe_*` | writes `output/introspect.json` |
| `forward` | `grid`, `time`, `wavelet`, `geometry`, `physics`, `models` | |
| `wavefield` | as forward + `snapshot_times` | `impl: eager` only |
| `fwi` | `grid`, `time`, `physics`, `optimizer`, `epochs`, one of `init_model` / `init_models`; `wavelet`, `geometry`, `obs` unless frequency selection | |
| `rtm` | `grid`, `time`, `wavelet`, `geometry`, `physics`, `velocity_model`, `obs` | needs `backend.impl: c` (fails at run time on eager) |
| `lsrtm` | `grid`, `time`, `wavelet`, `geometry`, `physics` (`AcousticBorn` or `AcousticBorn3D`), `background_model`, `true_model`, `optimizer`, `epochs` | synthetic only: obs are modelled from `true_model` |

Common optional keys: `task_id` (default `<task_type>-<YYYYmmdd-HHMMSS>`),
`output_dir` (default `./sweep_runs`), `seed` (0), `device` (`auto`),
`backend` (default `{impl: eager}`; `impl: c` gets boundary saving written in).

`models` / `init_models` list one `ModelRef` per model the equation declares,
named as the equation names them (Acoustic `[vp]`, Elastic `[vp, vs, rho]`,
AcousticVTI `[vp, epsilon, delta]`, ...). Get the list from the installed solver:

```python
from sweepx.tasks.introspect import describe_equation, equation_catalog
print(describe_equation("Elastic")["models"])
usable = [n for n, r in equation_catalog().items() if not r["is_facade"] and r["impl"]["eager"]]
print(len(usable), usable[:6])
```

A `ModelRef` takes exactly one source: `path` (.npy), `constant` (+ `shape`),
`dataset` (+ `preset` / `downsample`, see [datasets.md](datasets.md)) or
`linear_gradient` (+ `shape`); any of them can add `smooth_sigma_cells`.

## Single-band FWI and the Python API

The model of SKILL.md's forward run:

```python
import numpy as np
from scipy.ndimage import gaussian_filter
vp = np.full((60, 120), 1800.0, np.float32)
vp[25:, :] = 2400.0
vp[35:45, 50:70] = 2800.0
np.save("vp_true.npy", vp)
np.save("vp_init.npy", gaussian_filter(vp, 8).astype(np.float32))
```

```yaml
# file: fwi.yaml
task_type: fwi
task_id: demo_fwi
output_dir: ./runs
grid: {dh: 10.0}
time: {dt: 0.001, nt: 600}
wavelet: {kind: ricker, fm: 10.0, delay: 0.12}
geometry:
  kind: line
  sources:   {step: 30, depth: 1, start: 15}
  receivers: {step: 2,  depth: 1}
physics: {equation: Acoustic}
backend: {impl: eager, eager_options: {use_compile: false}}
init_model: {name: vp, path: vp_init.npy}     # init_models: [...] for multi-parameter equations
obs:                                          # exactly one source
  synthetic_from: {name: vp, path: vp_true.npy}
  # npy_path: obs.npy                         # (nshots, nt, nrec, 1), like record.npy
optimizer: {kind: adam, lr: 10.0}             # lr ~ m/s per step for a grid model
loss: {kind: mse}
epochs: 3
batchsize: 4                                  # shots per optimizer step
model_bounds:
  vp: {min: 1500.0, max: 3200.0}              # clamp after every step
```

```bash
sweepx tasks run fwi.yaml
sweepx tasks run fwi.yaml --override epochs=2 --override task_id=demo_fwi_short   # dotted-key overrides
```

```python
import numpy as np
out = "runs/demo_fwi/output"
vp = np.load(f"{out}/inverted_vp.npy")           # on the last stage's grid
loss = np.load(f"{out}/loss.npy")                # one value per epoch
print(vp.shape, loss.round(6).tolist())
assert np.load(f"{out}/initial_vp.npy").shape == vp.shape
```

The same from Python; a spec can also be built from a dict:

```python
import yaml
from sweepx.tasks import load_task, load_task_from_dict, TaskRunner

spec = load_task("fwi.yaml")                      # raises pydantic.ValidationError on a bad key
raw = yaml.safe_load(open("fwi.yaml"))
raw.update(task_id="demo_fwi_py", epochs=1)
spec = load_task_from_dict(raw, base_dir=".")     # relative paths resolve against base_dir
res = TaskRunner().run(spec)
print(res.status.state, res.task_dir, res.status.summary)
assert res.status.state == "success", res.status.error
```

## Run directory

| file | what |
|---|---|
| `status.json` | `state` (running/success/failed), `error`, `summary`, `artifacts`, `artifacts_rel` |
| `config_resolved.yaml` | the validated spec with every default; `sweepx tasks run config_resolved.yaml` re-runs it |
| `task_source.yaml` | the YAML as written (comments kept) + `run_meta.json["source"]["overrides"]` |
| `run_meta.json` | host, argv, cwd, CUDA devices, package versions (`sweepx`, `sweep`, `torch`, `numpy`) of the code that ran, `installed_versions_differ`, git SHA/describe/dirty per package checkout, `inputs` (real path, size, quick sha256 of every existing absolute path in the config), `labels` from `SWEEP_LABEL_<KEY>` env vars, Slurm job info, `previous_launches` |
| `provenance/<pkg>.diff` | uncommitted changes of a dirty checkout that ran |
| `metrics.jsonl` | live progress events (`start`, `iter` with loss/lr, `end`) |
| `logs/stdout.log`, `logs/stderr.log` | the run's console output |
| `manifest.json` | every file in the run dir at the end |
| forward | `output/record.npy`, `sources.npy` (nshots, 2) `[x, z]`, `receivers.npy` (nshots, nrec, 2) |
| wavefield | `output/record.npy`, `snapshots.npy`, `snapshots.png` |
| fwi (time domain) | `output/inverted_<name>.npy`, `initial_<name>.npy`, `loss.npy`, `loss.png`, `epochs/<name>_epoch_NNNN.npy` (every `show_every` epochs + last), `checkpoint.pt`, `qc/` when `qc:` is set |
| fwi (frequency selection) | at the run-dir root: `inverted_vp.npy`, `vp_iterNNNN.npy`, `curves.npz`, `summary.json` (`output/` stays empty) |
| rtm | `output/rtm_image*.npy`, `fwi_gradient_image*.npy`, `source_illumination.npy`, `receiver_illumination.npy` |
| lsrtm | `output/reflectivity.npy`, `loss.npy`, `loss.png` |

## Multiscale stages

`stages:` is a list; each stage runs `epochs` and may override the run's
settings. The conventional (static-obs) FWI path reads: `epochs`, `wavelet`,
`lr_scale`, `inr_lr_scale` (reparam only), `optimizer_reset`, `dh_m`, `dt_s`,
`nt`, `batch_size`, `bandpass`. It refuses `boundary` and `frequency`
(frequency-selection only).

Rules that are easy to get wrong:

- top-level `epochs` must equal the sum of the stage epochs;
- a stage that omits `dh_m` / `dt_s` / `nt` keeps the **previous stage's**
  value (a new `dt_s` without `nt` keeps the record length). Return to the
  native grid explicitly, or `inverted_vp.npy` comes out on the coarse grid;
- `bandpass` is not inherited; each stage filters the pristine data
  (filters never compose). `target: syn` (default) filters obs once and the
  synthetic every iteration; `target: wavelet` filters obs and the wavelet once;
- `lr_scale` multiplies `optimizer.lr` for the stage; the `scheduler` runs over
  the global epoch across all stages and is not restarted;
- the optimizer state carries across stages unless `optimizer_reset: true`;
  a grid model's Adam is rebuilt anyway when `dh_m` changes.

A runnable two-band example on a marine-style model (water on top, free
surface on, flat seabed at 80 m):

```python
import numpy as np
from scipy.ndimage import gaussian_filter
nz, nx = 60, 120
vp = np.full((nz, nx), 2000.0, np.float32)
vp[8:, :] = np.linspace(1900.0, 2600.0, nz - 8)[:, None]
vp[35:45, 50:70] = 2900.0
vp[:8, :] = 1500.0                                   # water rows 0..7 (seabed at 80 m)
np.save("vp_true.npy", vp)
init = gaussian_filter(vp, 6).astype(np.float32)
init[:8, :] = 1500.0
np.save("vp_init.npy", init)
```

```yaml
# file: multiscale.yaml
task_type: fwi
task_id: demo_multiscale
output_dir: ./runs
grid: {dh: 10.0}
time: {dt: 0.001, nt: 700}
wavelet: {kind: ricker, fm: 10.0, delay: 0.25}      # long lead-in: the low band stretches it
geometry:
  kind: line
  sources:   {step: 30, depth: 2, start: 15}       # 20 m: off the free surface on both grids
  receivers: {step: 2,  depth: 2}                   # even indices stay on the 20 m grid
physics: {equation: Acoustic, free_surface: true}   # marine: free surface on
backend: {impl: eager, eager_options: {use_compile: false}}
init_model: {name: vp, path: vp_init.npy}
obs: {synthetic_from: {name: vp, path: vp_true.npy}}
optimizer: {kind: adam, lr: 10.0}
water:
  seabed: 80.0          # flat seabed, metres below z=0 (or an .npz with key seabed_depth)
  velocity: 1500.0
  mode: pin             # held for the whole run, gradient zeroed, re-stamped each step
model_bounds:
  vp: {min: 1450.0, max: 3200.0}
batchsize: 4
epochs: 4               # = 2 + 2
stages:
  - epochs: 2
    dh_m: 20.0          # coarse grid ...
    dt_s: 0.002         # ... and time step (nt follows: record length kept)
    bandpass: {lo_hz: 1.0, hi_hz: 6.0}
  - epochs: 2
    dh_m: 10.0          # back to the native grid -- NOT inherited from grid.dh
    dt_s: 0.001
    lr_scale: 0.5
```

```bash
sweepx tasks run multiscale.yaml
```

```python
import numpy as np
out = "runs/demo_multiscale/output"
inv = np.load(f"{out}/inverted_vp.npy")
assert inv.shape == (60, 120), inv.shape        # native grid because the last stage set dh_m: 10
assert np.all(inv[:8] == 1500.0)                # water pinned: rows with centre above 80 m
print(np.load(f"{out}/loss.npy").round(6).tolist())
```

## The water layer

One top-level `water:` block serves every inversion path (conventional,
plan streaming, frequency selection, reparam, LSRTM).

| key | meaning |
|---|---|
| `seabed` | a number (flat seabed, m below z=0) or an `.npz` with key `seabed_depth`, shape `(nx,)` or `(ny, nx)` spanning the model's lateral extent |
| `mask` | a boolean `.npy` on the **native** grid (`grid.dh`, starting-model shape), True = water |
| `velocity` | value held in the water (default 1500); not allowed on LSRTM (reflectivity is held at 0) |
| `buffer_cells` | extend a `seabed` downward by N native cells (not with `mask`) |
| `mode` | `pin` (held all run) or `reset_each_stage` (reset at each stage entry, free within) |

Give exactly one of `seabed` / `mask`. `water: false` declares "no water
layer"; frequency selection refuses a starting model with cells at exactly
1500 m/s unless `water:` is given or `false`. A seabed file from a model
(explicit, one-off step):

```python
import numpy as np
from sweepx.tasks.bathymetry import (derive_seabed_depth_from_vp, save_seabed_depth_npz,
                                     water_mask_from_seabed_depth)
vp = np.load("vp_init.npy")                                   # (nz, nx)
depth = derive_seabed_depth_from_vp(vp, dh_z_m=10.0, water_vp=1500.0)   # (nx,) metres
save_seabed_depth_npz("seabed.npz", depth)                    # -> water: {seabed: seabed.npz}
print(depth[:3])
```

Retired keys (`freeze_top_n_rows`, `freeze_water_layer`,
`reparam.mask_water_layer`, `reparam.water_vp_m_s`, `reparam.seabed_depth_path`,
`reparam.water_reset_each_stage`) are refused with a message that spells out
the `water:` replacement; `sweepx tasks migrate-water old.yaml -o new.yaml`
rewrites a file.

## Wavefield snapshots

```yaml
# file: wavefield.yaml
task_type: wavefield
task_id: demo_wavefield
output_dir: ./runs
grid: {dh: 10.0}
time: {dt: 0.001, nt: 400}
wavelet: {kind: ricker, fm: 12.0, delay: 0.1}
geometry:
  kind: explicit                 # [x, z] grid indices
  sources:   [[60, 30]]          # model centre: shows the full wavefront
  receivers: [[0, 2], [60, 2], [119, 2]]
physics: {equation: Acoustic}
backend: {impl: eager, eager_options: {use_compile: false}}   # impl: c is refused here
models:
  - {name: vp, constant: 2000.0, shape: [60, 120]}
snapshot_times: [100, 200, 300]  # step indices, strictly increasing, < nt
plot: true                       # also writes output/snapshots.png
```

```bash
sweepx tasks run wavefield.yaml
python -c "import numpy as np; print(np.load('runs/demo_wavefield/output/snapshots.npy').shape)"
```

## RTM and LSRTM

RTM images a given `velocity_model` against `obs` in one pass (no
iterations). It needs `backend.impl: c` (an eager run loads and then fails:
the eager backend fills no illumination). The misfit is `imaging.loss_kind`
(`trace_cosine` default, `mse`, `l1`); top-level `loss.kind` is ignored.

<!-- skill-test: skip -->
```yaml
# file: rtm.yaml   (needs an NVIDIA GPU)
task_type: rtm
task_id: demo_rtm
output_dir: ./runs
grid: {dh: 10.0}
time: {dt: 0.001, nt: 700}
wavelet: {kind: ricker, fm: 10.0, delay: 0.25}
geometry:
  kind: line
  sources:   {step: 30, depth: 2, start: 15}
  receivers: {step: 2,  depth: 2}
physics: {equation: Acoustic, free_surface: true}
backend: {impl: c}
velocity_model: {name: vp, path: vp_init.npy}
obs: {synthetic_from: {name: vp, path: vp_true.npy}}   # or npy_path / segy / plan
imaging:
  shots_per_batch: 4
  loss_kind: trace_cosine
  # filter_lowcut_hz / filter_highcut_hz, normalize_by_illumination, post_filter: {...}
```

LSRTM inverts a reflectivity `mp` with `AcousticBorn` (`AcousticBorn3D` in
3-D; before sweepx 0.4.1 / sweep 0.3.6 these were `AcousticLSRTM*`); obs are
modelled from `true_model`. A stage reads only `epochs`, `lr_scale` and `wavelet`.

```yaml
# file: lsrtm.yaml
task_type: lsrtm
task_id: demo_lsrtm
output_dir: ./runs
grid: {dh: 10.0}
time: {dt: 0.001, nt: 700}
wavelet: {kind: ricker, fm: 10.0, delay: 0.25}
geometry:
  kind: line
  sources:   {step: 30, depth: 1, start: 15}
  receivers: {step: 2,  depth: 2}
physics:
  equation: AcousticBorn         # background equation (Acoustic) is derived
  receiver_type: [sh1]           # record the scattered field
backend: {impl: eager, eager_options: {use_compile: false}}
background_model: {name: vp, path: vp_init.npy}
true_model: {name: vp, path: vp_true.npy}
optimizer: {kind: adam, lr: 0.002}
epochs: 2
batchsize: 4
```

```bash
sweepx tasks run lsrtm.yaml
python -c "import numpy as np; print(np.load('runs/demo_lsrtm/output/reflectivity.npy').shape)"
```

## Field data: SEG-Y → plan → FWI

1. `sweepx tasks build-index` scans trace headers once into an index `.npz`.
2. `sweepx tasks build-plan` groups traces (`csg` = shot gathers, `crg` =
   receiver/node gathers with `--receiver-quantize-m`) and applies offset /
   shot filters; the plan stores file id + byte offset per trace.
3. The FWI YAML uses `geometry: {kind: from_plan, plan_path: ...}` and
   `obs: {plan: {plan_path: ...}}` (usually the same file). Coordinates in
   metres are snapped to `grid.dh` (and re-snapped per stage `dh_m`).

A small SEG-Y with coordinates in the headers (random traces, for the
mechanics only):

```python
import struct
import numpy as np
from sweepx.io.segy import (SEGY_BIN_HEADER_SIZE, SEGY_TEXT_HEADER_SIZE,
                            SEGY_TRACE_HEADER_SIZE, write_segy_minimal)
from sweepx.io.segy_index import SEGY_REV1_BYTES as B     # 0-based header offsets

nshots, nrec, nt = 3, 40, 500
data = np.random.default_rng(0).standard_normal((nshots * nrec, nt)).astype("float32")
write_segy_minimal("line.sgy", data, dt=0.002)            # headers left blank
with open("line.sgy", "r+b") as f:
    for s in range(nshots):
        for r in range(nrec):
            pos = SEGY_TEXT_HEADER_SIZE + SEGY_BIN_HEADER_SIZE + (s * nrec + r) * (SEGY_TRACE_HEADER_SIZE + nt * 4)
            f.seek(pos); h = bytearray(f.read(SEGY_TRACE_HEADER_SIZE))
            struct.pack_into(">i", h, B["shot"], s + 1)
            struct.pack_into(">i", h, B["trace_in_shot"], r + 1)
            struct.pack_into(">h", h, B["coord_scalar"], 1)          # 1 = metres
            struct.pack_into(">i", h, B["sx"], 200 + 300 * s)
            struct.pack_into(">i", h, B["rx"], 100 + 25 * r)
            f.seek(pos); f.write(h)
```

```bash
sweepx tasks build-index --segy-root . --glob 'line.sgy' --out index.npz --num-workers 1 --no-qc-png
sweepx tasks build-plan --index index.npz --grouping csg --offset-max-m 800 --out plan.npz --no-qc-png
```

```yaml
# file: field_fwi.yaml
task_type: fwi
output_dir: ./runs
grid: {dh: 10.0, shape: [60, 120]}
time: {dt: 0.001, nt: 1000}           # solver axis; plan traces are resampled to it
wavelet: {kind: ricker, fm: 8.0, delay: 0.2}   # field data: siren_pipeline_npz from `estimate-wavelet`
geometry: {kind: from_plan, plan_path: plan.npz, dedupe: true}   # plan_path: relative to the cwd
obs:
  plan: {plan_path: plan.npz, cache_all: true}    # no `sampling:` -> one static dataset
physics: {equation: Acoustic, free_surface: true}
init_model: {name: vp, path: vp_init.npy}
optimizer: {kind: adam, lr: 10.0}
loss: {kind: trace_cosine}            # amplitude-robust: field wavelet/scale unknown
epochs: 1
```

```python
from sweepx.tasks import load_task
spec = load_task("field_fwi.yaml")
print(spec.geometry.kind, spec.obs.plan.plan_path, spec.loss.kind)
```

With `obs.plan` and no `sampling`, a `data_plan:` may only set `dt_target_s`
(other fields are refused): select shots/offsets in `build-plan` or with
`obs.plan.max_shots` / `n_receivers_per_shot`. Adding `obs.plan.sampling:`
switches to the 3-D node (CRG) plan-streaming loop, which refuses `scheduler`,
`resume`, `train_shot_batchsize`, `local_model_window`, `data_plan`,
`modeling_override`, `grad_smooth`, `diffusion_prior`, and a few others.
SEG-Y paths stored inside a plan can be remapped with the env vars
`FWI_SEGY_ROOT` / `FWI_SEGY_REMAP`. Wavelet estimation and QC:
`analyze-wavelet`, `estimate-wavelet`, `plot-wavelet-steps` (`--help` on each);
the Viking Graben walkthrough in the sweepx docs runs the whole chain on a
public line.

## Frequency-selection FWI

Every node (source) emits one exclusive DFT-comb bin continuously; the DFT
of a steady-state window separates them exactly, so all nodes share one
forward per iteration. The misfit (`steady_gcn`) is invariant to a per-node
complex scale, so **no `wavelet`, `geometry`, `obs` or `loss`** are given.
Data come as coefficient shards (`coeff_shards`, from `extract-coeff`) or, for
tests, `synthesize_from_true`.

```yaml
# file: freqsel.yaml
task_type: fwi
task_id: demo_freqsel
output_dir: ./runs
grid: {dh: 10.0}
time: {dt: 0.001, nt: 1700}         # required by the schema; the stage nt comes from the comb
physics: {equation: Acoustic, abcn: 30}
backend: {impl: eager, eager_options: {use_compile: false}}
init_model: {name: vp, path: vp_init.npy}
water: {seabed: 80.0}
source_encoding:
  enabled: true
  mode: frequency_selection
  frequency:
    synthesize_from_true: true      # test path; field data: coeff_shards: <glob>
    true_model_path: vp_true.npy
    synth_n_nodes: 6
    synth_rec_stride: 2
    synth_node_z: 2
    probe_samples: 1000             # window = 1000 * dt = 1 s -> bins 1 Hz apart
    k_lo: 5                         # 5 Hz
    k_hi: 12                        # 12 Hz -> 8 bins >= 6 nodes in the pool
    steady_samples: 600             # ring-up; record = steady + slack + probe
    slack_samples: 100
    n_pools: 1                      # all nodes fire every iteration
optimizer: {kind: adam, lr: 10.0}   # Adam only, torch eps/betas; no scheduler
epochs: 2
```

```bash
sweepx tasks run freqsel.yaml
python -c "import json; print(json.load(open('runs/demo_freqsel/summary.json'))['loss_kind'])"
```

Outputs go to the run-dir root (`inverted_vp.npy`, `vp_iterNNNN.npy`,
`curves.npz`, `summary.json`), not `output/`. Rules:

- `bins = (k_hi - k_lo + 1)`, frequencies `k / (probe_samples * dt)`; the pool
  size (nodes firing together) must be ≤ bins. Lower bands need longer windows.
- Check the printed "steady-state two-window check": below 1e-2 means the
  window is steady; otherwise raise `steady_samples`.
- Size the PML (`abcn`) by the longest wavelength (continuous waves in a leaky
  box build standing waves).
- Per-band ladder: give each stage its own `frequency:` block (+ `dh_m`,
  `dt_s`, `lr_scale`, `optimizer_reset`, `boundary`); a stage's `wavelet`,
  `nt`, `batch_size`, `bandpass` are refused. Unlike the conventional path, a
  stage without `dh_m` / `dt_s` runs at `grid.dh` / `time.dt`, not at the
  previous stage's values.
- `loss:`, `scheduler` other than constant, `optimizer.kind` other than adam,
  `eps`/`betas`, and (under `reparam`) a stage `lr_scale` are refused.
- Truncated backward: `bwd_tail_margin_s` (seconds, preferred in multi-rate
  ladders) or `bwd_tail_margin` (steps), not both.

Shards from a recorded or modelled node gather (positional dir = a forward run
whose sources are the nodes; `--n-p` = `probe_samples`, `--k-lo/--k-hi` = the
stage's `k_lo/k_hi`, `--dt` = the stage's dt):

<!-- skill-test: skip -->
```bash
sweepx tasks extract-coeff runs/forward_nodes --n-p 1000 --k-lo 5 --k-hi 12 -o coeff_5_12hz.npz
sweepx tasks extract-coeff --plan plan.npz --dh-m 25 --dt 0.002 --n-p 2500 --k-lo 10 --k-hi 30 -o coeff.npz
```

## Multi-GPU, domain decomposition, resume

| need | how |
|---|---|
| shots over N GPUs (time-domain fwi, rtm, lsrtm; forward has no shot splitting) | `sweepx tasks run x.yaml --nproc-per-node N` (re-execs under `torchrun --standalone`); gradients all-reduced, `device: auto` → `cuda:LOCAL_RANK`. `lbfgs` is refused under torchrun |
| model split over GPUs (3-D) | `--dd-py PY --dd-px PX` (sets `SWEEP_DD_*`, world = PY*PX) |
| frequency selection on several GPUs | only via domain decomposition; `--nproc-per-node N` alone runs N duplicate inversions |
| continue an interrupted fwi/lsrtm | fixed `task_id` + `resume: true` (or `--resume`); Ctrl-C saves `checkpoint.pt` at the next iteration boundary. `resume_from: <task_id>` starts from another run's checkpoint |
| label runs | `SWEEP_LABEL_<KEY>=value` env vars land in `run_meta.json["labels"]` |

## What the validator refuses

Beyond unknown keys and type errors, load-time validators refuse (each error
message says what to write instead):

- `ModelRef` with zero or several sources; `constant`/`linear_gradient` without `shape`; `preset`/`downsample`/`dataset_field` without `dataset`.
- `ObsSpec` with zero or several sources; `fwi` without `wavelet`/`geometry`/`obs` (outside frequency selection); both or neither of `init_model`/`init_models`; empty `stages`.
- `backend.cuda_options` with `impl: eager`, `eager_options` with `impl: c`; `memory.boundary` unless `strategy: boundary`, `memory.ckpt` unless `strategy: ckpt`.
- `wavefield` with `impl: c`, `use_ckpt: true`, or `snapshot_times` out of range / unsorted.
- `ModelBounds` enabled without `min`/`max`, or `min >= max`.
- `optimizer.kind: lbfgs` with `batchsize` below the shot count (each batch is a new misfit, which breaks the curvature pairs); `lbfgs` `initial_step` under `reparam`.
- `water` with neither or both of `seabed`/`mask`, `buffer_cells` with `mask`, `water.velocity` on LSRTM, `free_params[].water_value` without `water`.
- stage keys the active path never reads; `epochs` ≠ sum of stage epochs; `inr_lr_scale` without `reparam`.
- `loss.kind: steady_gcn` outside frequency selection; any loss/optimizer/scheduler setting frequency selection would ignore.
- plan-streaming (`obs.plan.sampling`) leftovers; `shared_shots_per_iter: 1`; `reparam.backward_mode: single_step` with illumination preconditioning or smoothing on that path.
- `hash.backend: triton` with `hash.c2f.growing`.
- An unset `$VAR` in any string.

At run time, a fwi/lsrtm epoch whose loss is not finite stops the run with
`FloatingPointError` instead of carrying a NaN model on.

Not refused (silently ignored or wrong): see the Pitfalls in SKILL.md, and
run `scripts/validate_spec.py --preflight`, which flags most of them.
