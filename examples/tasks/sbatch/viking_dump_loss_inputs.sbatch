#!/bin/bash
#
# One-iter Viking debug: dump exact obs+syn loss inputs to npz, then exit.
#
# Output: ${SWEEP_RUNS_ROOT}/viking_dump/loss_inputs.npz
#   keys = obs (n_chunk, nrec, nt), syn (...), chunk_indices,
#          stage_dt_s, bandpass_lo_hz, bandpass_hi_hz, bandpass_order
#
#SBATCH --job-name=viking_dump_loss
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=${SWEEP_RUNS_ROOT}/viking_dump/slurm-logs/%j.out
#SBATCH --error=${SWEEP_RUNS_ROOT}/viking_dump/slurm-logs/%j.err

set -eo pipefail

mkdir -p ${SWEEP_RUNS_ROOT}/viking_dump/slurm-logs

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)   Started: $(date -Iseconds)"
echo "============================================================"
nvidia-smi -L

export PATH=${SWEEP_CONDA_ROOT}/bin:$PATH
source ${SWEEP_CONDA_ROOT}/etc/profile.d/conda.sh
conda activate base
export MPLCONFIGDIR=/tmp/mpl_viking_dump
export PYTHONUNBUFFERED=1

# Trigger one-shot dump + early-exit in runner._maybe_dump_loss_inputs.
export DUMP_LOSS_INPUTS=${SWEEP_RUNS_ROOT}/viking_dump/loss_inputs_nodedupe.npz
export DUMP_LOSS_INPUTS_EXIT=1

YAML=${SWEEP_STACK_ROOT}/sweep-tasks/examples/tasks/viking_dump_loss_inputs.yaml

cd ${SWEEP_USER_ROOT}/sweep-stack
torchrun --standalone --nproc_per_node=1 -m sweep_tasks run "$YAML" || true

echo "============================================================"
echo "Finished: $(date -Iseconds)"
echo "============================================================"
ls -la ${SWEEP_RUNS_ROOT}/viking_dump/loss_inputs.npz 2>&1 || true
