# SLURM template (Viking)

[`viking_freqsel.sbatch`](viking_freqsel.sbatch) runs Steps 1-5 of the Viking
walkthrough ([`docs/tasks/datasets/viking/README.md`](../../../docs/tasks/datasets/viking/README.md))
in one job on one GPU: `build-index` and `build-plan` with
[`../field/viking/pipeline_config.yaml`](../field/viking/pipeline_config.yaml),
`extract-coeff` for the six bands, then `sweep-tasks run` on
[`../field/viking/viking_freqsel_allteeth.yaml`](../field/viking/viking_freqsel_allteeth.yaml).
On a resubmit, Steps 1-4 are skipped when their output exists and a finished
inversion is not repeated; an unfinished inversion restarts from the first
band.

Do Step 0 of the walkthrough (download) first, activate the environment that
has sweepx (sbatch passes your shell's environment on to the job), then:

```bash
export VIKING_HOME=$HOME/viking SWEEPX=$HOME/sweepx   # SWEEPX: a checkout of this repo
sbatch $SWEEPX/examples/tasks/sbatch/viking_freqsel.sbatch
```

The log goes to `viking_freqsel-<jobid>.out` in the directory you submit from.
Partition, account and GPU type differ between clusters; pass them on the
sbatch command line (for example `--partition=…`, or `--gres=gpu:<type>:1` on
clusters that name GPU types).
