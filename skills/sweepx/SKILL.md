---
name: sweepx
description: Writes and runs sweepx seismic workflows on top of the sweep-solver wave-equation engine. Use when writing or fixing a sweepx task YAML (forward, wavefield, fwi, rtm, lsrtm, introspect), running `sweepx tasks` / `sweep-tasks` commands (run, init, build-index, build-plan, extract-coeff, migrate-water), setting up FWI (multiscale stages, water layer, frequency-selection encoding, INR/hash reparameterization, multi-GPU), reading SEG-Y or building acquisition plans with sweepx.io, choosing a misfit (sweepx.loss), optimizer, bound parameterization or preconditioner (sweepx.opt), or a neural reparameterization (sweepx.nn), loading results from a run directory, or driving the natural-language sweep-agent. Covers which keys each task type accepts, what the validator refuses, and settings that load but silently do the wrong thing.
---

# sweepx

`sweepx` is the toolkit around the `sweep` engine (`pip install sweep-solver`,
`import sweep`): `sweepx.tasks` (typed YAML task runner + CLI), `sweepx.io`
(SEG-Y, geometry, plans), `sweepx.loss` (misfits), `sweepx.opt` (optimizers,
bounds, preconditioners), `sweepx.nn` (INR / hash / SIREN reparameterizations,
priors), `sweepx.agent` (local-LLM chat front end).

For equations, propagators, boundary conditions, memory strategies and
gradients of the engine itself, use the **sweep-solver** skill. This skill only
covers what a sweepx user needs from it.

## Workflow

1. Start from `sweepx tasks init <task_type> -o x.yaml`, an annotated template
   that runs as written on the embedded Marmousi demo.
2. Validate without running: `python scripts/validate_spec.py x.yaml --preflight`
   (sweepx's own loader, then models, geometry, CFL, points per wavelength and
   settings that load but are ignored).
3. Run: `sweepx tasks run x.yaml` (= `sweep-tasks run`; Python: `TaskRunner().run(load_task("x.yaml"))`).
4. Read `<output_dir>/<task_id>/` (below); its `config_resolved.yaml` re-runs the task.

## Install

<!-- skill-test: skip -->
```bash
pip install sweepx            # + sweep-solver (prebuilt CUDA core), torch, pydantic, scipy, matplotlib
pip install "sweepx[segy]"    # + segyio for SEG-Y; other extras: hdf5, animate, agent-ui, mcp, all
# torch from PyPI is a CUDA 13 build. NVIDIA driver older than 580: install a CUDA 12.8 torch first
pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install sweepx
```

`backend: {impl: c}` needs an NVIDIA GPU (no compile step); `impl: eager` is
pure PyTorch and runs anywhere, including CPU.

## Intent → workflow

| user wants | do |
|---|---|
| synthetic shot gathers | `task_type: forward` → `output/record.npy` |
| wavefield snapshots / movie | `task_type: wavefield` (`backend.impl: eager` only, `snapshot_times` are step indices) |
| which equations exist, which models each needs | `sweepx.tasks.introspect.equation_catalog()` / `describe_equation(name)`, or `task_type: introspect, action: list_equations` |
| FWI on synthetic data | `task_type: fwi`, `obs: {synthetic_from: <true ModelRef>}` ([workflows.md](references/workflows.md#single-band-fwi-and-the-python-api)) |
| FWI on recorded data | SEG-Y → `sweepx tasks build-index` → `build-plan` → `geometry: {kind: from_plan}` + `obs: {plan: ...}` ([workflows.md](references/workflows.md#field-data-seg-y--plan--fwi)) |
| avoid cycle skipping | `stages:` low → high band, each with `bandpass:` ([workflows.md](references/workflows.md#multiscale-stages)) |
| marine data / water column | `physics.free_surface: true` + a `water:` block |
| network (INR) instead of grid | `reparam:` block ([nn.md](references/nn.md)) |
| node/OBN data, wavelet unknown, cheap many-source FWI | frequency selection: `source_encoding.mode: frequency_selection` ([workflows.md](references/workflows.md#frequency-selection-fwi)) |
| image a model | `task_type: rtm` (recorded or synthetic obs); `task_type: lsrtm` (synthetic only: needs `true_model`) |
| custom loop, other optimizers (L-BFGS-B, Newton, SOAP ...) | Python: `sweep` solver + `sweepx.loss` + `sweepx.opt` ([opt.md](references/opt.md)) |
| plain-language control | `sweepx agent chat` with a local LLM ([agent.md](references/agent.md)) |

## Minimal forward run

A small velocity model, made with numpy (no download):

```python
import numpy as np
from scipy.ndimage import gaussian_filter
nz, nx = 60, 120                       # model arrays are (nz, nx) in 2-D, (nz, ny, nx) in 3-D
vp = np.full((nz, nx), 1800.0, np.float32)
vp[25:, :] = 2400.0                    # a flat reflector at 250 m
vp[35:45, 50:70] = 2800.0              # a fast box
np.save("vp_true.npy", vp)
np.save("vp_init.npy", gaussian_filter(vp, 8).astype(np.float32))
```

```yaml
# file: forward.yaml
task_type: forward
task_id: demo_forward               # omit for a timestamped dir; a fixed id reuses (overwrites) the dir
output_dir: ./runs                  # relative to THIS file
grid: {dh: 10.0}                    # shape inferred from models[0]
time: {dt: 0.001, nt: 600}          # vmax*dt/dh = 0.28
wavelet: {kind: ricker, fm: 10.0, delay: 0.12}
geometry:
  kind: line                        # grid indices, points are (x, z)
  sources:   {step: 30, depth: 1, start: 15}
  receivers: {step: 2,  depth: 1}
physics:
  equation: Acoustic                # models: [vp]
  free_surface: false               # true for marine data
backend:
  impl: eager                       # `c` = CUDA kernels (GPU only)
  eager_options: {use_compile: false}   # default compiles: slow first call on tiny runs
models:
  - {name: vp, path: vp_true.npy}
```

```bash
sweepx tasks run forward.yaml
python -c "import numpy as np; print(np.load('runs/demo_forward/output/record.npy').shape)"   # (nshots, nt, nrec, 1)
```

An FWI on the same model, the Python API (`load_task`, `TaskRunner`) and
every file of a run directory: [workflows.md](references/workflows.md#single-band-fwi-and-the-python-api).

## Run directory

`<output_dir>/<task_id>/` holds `status.json` (`state`, `error`, `summary`),
`config_resolved.yaml` (re-runs the task), `run_meta.json` (versions, git
state, input fingerprints), `metrics.jsonl`, `logs/` and `output/`:

| task | `output/` |
|---|---|
| forward | `record.npy` (nshots, nt, nrec, ncomp), `sources.npy`, `receivers.npy` |
| wavefield | `record.npy`, `snapshots.npy`, `snapshots.png` |
| fwi | `inverted_<name>.npy`, `initial_<name>.npy`, `loss.npy`, `epochs/`, `checkpoint.pt` |
| fwi, frequency selection | nothing: `inverted_vp.npy`, `curves.npz`, `summary.json` are at the run-dir root |
| rtm | `rtm_image*.npy`, `source_illumination.npy`, `receiver_illumination.npy` |
| lsrtm | `reflectivity.npy`, `loss.npy` |

## Pitfalls

Unknown keys and many keys a path never reads are refused at load time
([workflows.md](references/workflows.md#what-the-validator-refuses)). These
load fine and are wrong; `scripts/validate_spec.py --preflight` flags most:

- **`obs.npy_path` layout** must be `(nshots, nt, nrec, 1)`, like `record.npy`.
  Only axis 0 is checked: `(nshots, nrec, nt)` is broadcast and the FWI
  "succeeds" on garbage. `sweepx.io` SEG-Y readers return `(nshots, nrec, nt)`:
  use `.transpose(0, 2, 1)[..., None]`.
- **Stages inherit the previous stage's grid**: a stage without `dh_m` / `dt_s`
  / `nt` keeps the previous stage's values, not the top-level ones. Set them
  on the stage that returns to the native grid. `bandpass` is not inherited.
- **Dataset models do not set `grid.dh`**: `marmousi:2d-demo` is at 12.5 m
  (`downsample: 2` doubles it); set `grid.dh` to match.
- **SEG-Y `byte_map`** offsets are 0-based (`sx: 72`, not 73); `gx`/`gy` are
  unknown keys and silently ignored (use `rx`/`ry`). The bundled templates
  show the wrong form.
- **Marine data**: `physics.free_surface: true` (default `false`) plus a
  `water:` block from the seabed. Never hand-roll a water mask or threshold
  1500 m/s: the runner rebuilds it on every stage grid.
- **Under `reparam:`** the network trains at `reparam.lr`; `optimizer.lr` is
  required but unused. `reparam.vp_std` (m/s) sets the effective step size.
- **RTM** needs `backend.impl: c` and uses `imaging.loss_kind`; `loss.kind` is ignored.
- **Loss options are per kind** (`huber_delta` only with `huber`, `cc_*` only
  with `cc_traveltime`); for another kind they are silently unused.
- **`--nproc-per-node N` with frequency selection** starts N identical
  inversions into one run dir; only domain decomposition splits it.
- **CFL**: fwi/rtm refuse only `vmax*dt/dh > 1.0`; forward and wavefield do
  not check. The 8th-order 2-D Acoustic blows up above ~0.55: keep ≤ 0.5
  (less in 3-D) and points per wavelength `vmin / (2.5 fm dh)` ≥ 5.
- **`impl: eager` on a GPU runs in TF32**: ~1e-2 relative error in gradients.
  Use `impl: c` when accuracy matters.
- **Low-frequency stages need a long wavelet `delay`**: band-passing a Ricker
  to 0.5–2 Hz stretches it ~2 s backwards.
- **Relative paths**: model, obs, output, water and geometry files resolve
  against the YAML; `plan_path`, `index_path`, `data_mask_path`,
  `topography`, `reparam.init_from` against the process cwd.
- **Order**: points are `(x, z)` / `(x, y, z)` grid indices, models
  `(nz, nx)` / `(nz, ny, nx)`; `FromPlanGeometry.dh_xyz_m`,
  `grid_origin_xyz_m` and `auto_origin_pad_cells` are `(z, y, x)`.
- **`device`** is `auto`, `cpu` or `cuda:N`; a bare `"0"` fails at run time.

## References

| file | contents |
|---|---|
| [references/spec-schema.md](references/spec-schema.md) | every task type and sub-model: fields, types, defaults (generated) |
| [references/workflows.md](references/workflows.md) | FWI and the Python API, run directory, multiscale, water, wavefield, RTM/LSRTM, field data, frequency selection, multi-GPU, resume, what the validator refuses |
| [references/io.md](references/io.md) | SEG-Y index/plan/reader, geometry, data and model plans, data layouts |
| [references/loss.md](references/loss.md) | YAML loss kinds and the `sweepx.loss` library |
| [references/opt.md](references/opt.md) | optimizers, step-size units, bounds, preconditioners (YAML vs Python) |
| [references/nn.md](references/nn.md) | `VelocityINR`, hash grid, `reparam:` block settings |
| [references/datasets.md](references/datasets.md) | benchmark models (`ModelRef.dataset`), embedded demos, Viking field line |
| [references/agent.md](references/agent.md) | `sweep-agent` chat/ui/tools, LLM backends, tools as Python functions |

Command options: `sweepx tasks <command> --help` (`run`, `init`,
`build-index`, `build-plan`, `extract-coeff`, `migrate-water`, ...).
Scripts: `scripts/validate_spec.py` (validate + preflight) and
`scripts/gen_spec_reference.py` (regenerate `spec-schema.md` after upgrading).
