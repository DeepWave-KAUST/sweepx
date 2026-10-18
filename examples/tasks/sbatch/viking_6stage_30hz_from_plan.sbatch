#!/bin/bash
#
# Viking 6-stage FWI (300 epochs, 2-5 → 2-30 Hz) — UNIFIED PLAN PATH.
# Same hardware footprint as the benchmark for direct wall
# and final_loss comparison. Pre-requires viking_csg_plan.npz under
# ${SWEEP_RUNS_ROOT}/viking_unified/ (sweep-tasks build-index/plan).
#
#SBATCH --job-name=viking_6stage_plan
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:4
# add --exclude=<node> here if a specific node has bad GPUs
#SBATCH --output=${SWEEP_RUNS_ROOT}/viking_sweep_6stage_from_plan/slurm-logs/fwi_%j.out
#SBATCH --error=${SWEEP_RUNS_ROOT}/viking_sweep_6stage_from_plan/slurm-logs/fwi_%j.err

set -eo pipefail

mkdir -p ${SWEEP_RUNS_ROOT}/viking_sweep_6stage_from_plan/slurm-logs

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)   Started: $(date -Iseconds)"
echo "============================================================"
nvidia-smi -L

export PATH=${SWEEP_CONDA_ROOT}/bin:$PATH
source ${SWEEP_CONDA_ROOT}/etc/profile.d/conda.sh
conda activate base
export MPLCONFIGDIR=/tmp/mpl_viking_6stage_plan
export PYTHONUNBUFFERED=1

YAML=${SWEEP_STACK_ROOT}/sweep-tasks/examples/tasks/viking_siren_hash_6stage_30hz_from_plan.yaml

cd ${SWEEP_USER_ROOT}/sweep-stack
torchrun --standalone --nproc_per_node=4 -m sweep_tasks run "$YAML"

echo "============================================================"
echo "Finished: $(date -Iseconds)"
echo "============================================================"
