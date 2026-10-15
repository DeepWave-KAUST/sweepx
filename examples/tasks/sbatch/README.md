# SLURM templates (Viking)

Reference sbatch scripts for ibex, one per step of the Viking workflow. Each
runs a YAML from `examples/tasks/field/viking/` in the sweepx checkout:

| Template | YAML |
|---|---|
| `viking_6stage_30hz.sbatch` | `viking_siren_hash_6stage_30hz.yaml` |
| `viking_6stage_30hz_from_plan.sbatch` | `viking_siren_hash_6stage_30hz_from_plan.yaml` |
| `viking_dump_loss_inputs.sbatch` | `viking_dump_loss_inputs.yaml` |
| `viking_from_plan_smoke.sbatch` | `viking_from_plan_smoke.yaml` |
| `viking_rtm.sbatch` | `viking_rtm_finegrid.yaml` |
| `viking_rtm_legacy_match.sbatch` | `viking_rtm_legacy_match.yaml` |

They read a few environment roots (`SWEEP_USER_ROOT`, `SWEEP_REPOS_ROOT`,
`SWEEP_RUNS_ROOT`, `SWEEP_CONDA_ROOT`, `PROJECT_DATA_ROOT`) set in your login
shell; the Viking page (`docs/tasks/datasets/viking/README.md`, "Running on a
cluster") shows a working ibex layout. Then, for example:

```bash
sbatch ${SWEEP_REPOS_ROOT}/sweepx/examples/tasks/sbatch/viking_rtm.sbatch
```
