# Datasets

Benchmark models come from the solver's catalog (`sweep.datasets`) and are
referenced from any `ModelRef` (`models:`, `init_model:`, `obs.synthetic_from:`,
`velocity_model:`, ...), so a task YAML needs no `.npy`.

## In a task YAML

```yaml
init_model:
  name: vp
  dataset: marmousi:2d-demo     # "<name>" or "<name>:<variant>"
  preset: vp_smooth             # embedded demos only
  # downsample: 2               # full-size entries only; scalar or per-axis
  # dataset_field: vp           # which payload key; defaults to `name`
  # smooth_sigma_cells: 8       # any source: Gaussian smoothing in cells
```

| entry | kind | shape (nz, nx) | dh (m) | presets / notes |
|---|---|---|---|---|
| `marmousi:2d-demo` | embedded (no network) | (281, 1361) | 12.5 | `vp_true`, `vp_smooth`, `vp_linear`; seabed flat at 462.5 m |
| `overthrust:2d-demo` | embedded (no network) | (187, 801) | 25.0 | `true`, `smooth` |
| others (`marmousi`, `marmousi2`, `overthrust:3d-acoustic`, `bp-2004`, `bp-2007-tti`, `hess-vti`, `seg-eage-salt`, `chevron-fwi-2014`, ... depending on the installed sweep-solver) | download on first use into `$SWEEP_DATASETS_CACHE` (default `~/.cache/sweep-datasets`) | see `sweep datasets list` | per entry | each prints its license and citation once per process |

- **`grid.dh` is not taken from the dataset.** Set it to the entry's dh
  (×`downsample`). `scripts/validate_spec.py --preflight` warns on a mismatch.
- `water:` is still needed for a marine model: the 1500 m/s rows of a
  dataset are not treated as water (Marmousi demo: `water: {seabed: 462.5}`).
- `preset` / `downsample` / `dataset_field` without `dataset` are refused.
- The catalog belongs to the solver; list it with `sweep datasets list`
  (sweep-solver CLI) or the agent tool `list_benchmark_models`.

## In Python

```python
from sweepx.tasks.introspect import load_model_array

vp = load_model_array({"name": "vp", "dataset": "marmousi:2d-demo", "preset": "vp_smooth"}, base_dir=".")
print(vp.shape, vp.dtype, float(vp.min()), float(vp.max()))
ot = load_model_array({"name": "vp", "dataset": "overthrust:2d-demo", "preset": "true"}, base_dir=".")
print(ot.shape)
```

`load_model_array` runs the runner's own loader on a `ModelRef` dict (path
resolution, `smooth_sigma_cells`, `linear_gradient` included), so it shows
exactly what a task would load.

## Field data: Viking Graben line 12

The sweepx documentation walks the public Mobil AVO Viking Graben line 12
(Keys & Foster 1998, doi:10.1190/1.9781560802082.ch1; archived copy
doi:10.5281/zenodo.23481108, ~760 MB) from raw SEG-Y to an inverted model:
`build-index → build-plan → extract-coeff` (6 bands) `→ run`, using
`sweepx tasks init viking -o pipeline.yaml` as the `--config` for the
pipeline steps and the example task `examples/tasks/field/viking/` in the
sweepx repository. One GPU with ≥ 16 GB. Do not download it unless the user
asks for it.
