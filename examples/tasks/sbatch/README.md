# sweep-tasks SLURM templates

Reference sbatch scripts for ibex. Pair each one with the matching YAML
in `examples/tasks/`.

## `build_crg_plan_obn3d.sbatch`

Generate an OBN CRG plan cache (the `crg_fwi_plan_v1` npz consumed
by `geometry.kind=from_crg_plan` / `obs.kind=crg_plan` in FWI YAMLs).

### Submission

```bash
ssh glogin.ibex.kaust.edu.sa
cd ${SWEEP_RUNS_ROOT}/crg_plan_build
sbatch ${SWEEP_STACK_ROOT}/sweep-tasks/examples/sbatch/build_crg_plan_obn3d.sbatch
```

### Single-process (smaller dataset, local box)

```bash
sweep-tasks build-crg-plan \
    --segy-root <segy-dir> \
    --glob '*.sgy' \
    --out <output>.npz \
    --num-workers 16 \
    --source-depth-m <SDEPTH_const> \
    --receiver-z-byte <Z_BYTE_OFFSET> \
    --receiver-quantize-m 0.5
```

### Dataset-specific overrides

| Dataset | source-depth-m | source-z-byte | receiver-z-byte | receiver-quantize-m | notes |
|---|---|---|---|---|---|
| Viking  | 6.0 (streamer) | n/a | n/a (default ok) | 0.1 | 2-D streamer, no CRG |
| Volve   | -            | n/a | n/a | -      | 2-D OBC; quantize TBD |

sweep-tasks reuses the legacy per-file scan logic with an in-memory MPI
gather (simpler than the legacy "sharded-merge" path; it works while the
merged per-trace arrays fit in rank-0 RAM).

### Output schema (matches legacy)

The output npz has format `crg_fwi_plan_v1` and is interchangeable with
the legacy build at
the reference CRG plan npz on disk.
Validated against the legacy build on a subset of the files.
