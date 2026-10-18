#!/usr/bin/env bash
#SBATCH --job-name=sweep_loss_grad
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"   # submit from tests/usability
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
set -u

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

nvidia-smi -L

echo "########## 05_fwi_gradient_alignment.py ##########"
python -u 05_fwi_gradient_alignment.py
