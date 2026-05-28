#!/usr/bin/env bash
#SBATCH --job-name=sweep_loss_fwi
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"   # submit from tests/usability

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
set -u

echo "=== Node: $(hostname)"
nvidia-smi -L
echo "=== Python: $(which python) ($(python --version 2>&1))"
echo "=== Started: $(date)"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Re-run FWI only (with lr=4, 60 epoch).
echo
echo "########## 03_fwi_layer.py (lr=4, 60 ep) ##########"
python -u 03_fwi_layer.py

echo "=== Finished: $(date)"
