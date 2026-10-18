#!/bin/bash
#
# Viking 2-D RTM — single V100, one-pass post-FWI imaging at the fine 12.5 m
# grid. Reads the final inverted vp from the SIREN/hash 6-stage FWI run and
# produces RTM + FWI-gradient images + illumination products under
#   ${SWEEP_RUNS_ROOT}/viking_rtm/viking_rtm_finegrid_v1/
#
# Inputs prebuilt by the same workflow that ran the FWI smoke (the unified
# seismic_plan_v1 cache + the estimated wavelet npz are shared).
#
#SBATCH --job-name=viking_rtm
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=${SWEEP_RUNS_ROOT}/viking_rtm/slurm-logs/%j.out
#SBATCH --error=${SWEEP_RUNS_ROOT}/viking_rtm/slurm-logs/%j.err

set -eo pipefail

mkdir -p ${SWEEP_RUNS_ROOT}/viking_rtm/slurm-logs

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)   Started: $(date -Iseconds)"
echo "============================================================"
nvidia-smi -L

export PATH=${SWEEP_CONDA_ROOT}/bin:$PATH
source ${SWEEP_CONDA_ROOT}/etc/profile.d/conda.sh
conda activate base
export MPLCONFIGDIR=/tmp/mpl_viking_rtm
export PYTHONUNBUFFERED=1

YAML=${SWEEP_REPOS_ROOT}/sweep-tasks/examples/tasks/viking_rtm_finegrid.yaml

cd ${SWEEP_REPOS_ROOT}/sweep-tasks
torchrun --standalone --nproc_per_node=1 -m sweep_tasks run "$YAML"

echo "============================================================"
echo "Finished: $(date -Iseconds)"
echo "============================================================"
