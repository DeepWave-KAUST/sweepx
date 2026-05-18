#!/bin/bash
#
# Viking RTM (legacy-matching: max_abs_offset=3000, 3-25Hz, shots/batch=1, mse).
# Expected wall ~45 min on 1× V100 (legacy reference: 42 min on same hardware).
#
#SBATCH --job-name=viking_rtm_lm
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
# add --exclude=<node> here if a specific node has bad GPUs
#SBATCH --output=${SWEEP_RUNS_ROOT}/viking_rtm_legacy_match/slurm-logs/%j.out
#SBATCH --error=${SWEEP_RUNS_ROOT}/viking_rtm_legacy_match/slurm-logs/%j.err

set -eo pipefail

mkdir -p ${SWEEP_RUNS_ROOT}/viking_rtm_legacy_match/slurm-logs

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)   Started: $(date -Iseconds)"
echo "============================================================"
nvidia-smi -L

export PATH=${SWEEP_CONDA_ROOT}/bin:$PATH
source ${SWEEP_CONDA_ROOT}/etc/profile.d/conda.sh
conda activate base
export MPLCONFIGDIR=/tmp/mpl_viking_rtm_lm
export PYTHONUNBUFFERED=1

YAML=${SWEEP_STACK_ROOT}/sweep-tasks/examples/tasks/viking_rtm_legacy_match.yaml

cd ${SWEEP_USER_ROOT}/sweep-stack
torchrun --standalone --nproc_per_node=1 -m sweep_tasks run "$YAML"

echo "============================================================"
echo "Finished: $(date -Iseconds)"
echo "============================================================"
