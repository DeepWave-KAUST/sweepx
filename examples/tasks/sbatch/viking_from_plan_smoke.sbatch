#!/bin/bash
#
# Viking unified-plan smoke test — exercises `from_plan` + `obs.plan`
# on 1 V100, 5 epochs. Pre-requires:
#   sweep-tasks build-index ...   →  ${SWEEP_RUNS_ROOT}/viking_unified/viking_index.npz
#   sweep-tasks build-plan ...    →  ${SWEEP_RUNS_ROOT}/viking_unified/viking_csg_plan.npz
#
#SBATCH --job-name=viking_plan_smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=${SWEEP_RUNS_ROOT}/viking_from_plan_smoke/slurm-logs/%j.out
#SBATCH --error=${SWEEP_RUNS_ROOT}/viking_from_plan_smoke/slurm-logs/%j.err

set -eo pipefail

mkdir -p ${SWEEP_RUNS_ROOT}/viking_from_plan_smoke/slurm-logs

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)   Started: $(date -Iseconds)"
echo "============================================================"
nvidia-smi -L

export PATH=${SWEEP_CONDA_ROOT}/bin:$PATH
source ${SWEEP_CONDA_ROOT}/etc/profile.d/conda.sh
conda activate base
export MPLCONFIGDIR=/tmp/mpl_viking_plan_smoke
export PYTHONUNBUFFERED=1

YAML=${SWEEP_STACK_ROOT}/sweep-tasks/examples/tasks/viking_from_plan_smoke.yaml

cd ${SWEEP_USER_ROOT}/sweep-stack
torchrun --standalone --nproc_per_node=1 -m sweep_tasks run "$YAML"

echo "============================================================"
echo "Finished: $(date -Iseconds)"
echo "============================================================"
